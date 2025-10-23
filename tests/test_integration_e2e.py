"""
End-to-end integration tests covering full ingestion to query pipeline.

Tests the complete workflow from document ingestion through to question answering,
validating that all components work together correctly in both AWS and local modes.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sop_qa_tool.api.main import app
from sop_qa_tool.config.settings import Settings
from sop_qa_tool.models.sop_models import SOPDocument
from sop_qa_tool.services.document_ingestion import DocumentIngestionService
from sop_qa_tool.services.rag_chain import RAGChain


class TestEndToEndIntegration:
    """End-to-end integration tests for the complete SOP Q&A pipeline."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_sop_content(self):
        """Sample SOP document content for testing."""
        return """
        SOP-FILL-001: Bottle Filling Procedure
        Revision: 2.1
        Effective Date: 2024-01-15
        
        1. Purpose and Scope
        This procedure describes the standard operating procedure for bottle filling operations.
        
        2. Roles and Responsibilities
        - Operator: Responsible for machine setup and operation
        - QA Inspector: Responsible for quality checks and validation
        
        3. Equipment Required
        - Filler-01 (Main filling machine)
        - Temperature Probe TP-001
        - Flow Meter FM-002
        
        4. Procedure Steps
        4.1 Pre-operation Setup
        4.1.1 Verify filler temperature is between 18-22°C
        4.1.2 Check flow meter calibration date (must be within 6 months)
        4.1.3 Inspect bottles for defects
        
        4.2 Filling Process
        4.2.1 Start filling sequence
        4.2.2 Monitor fill levels continuously
        4.2.3 Record batch information every 30 minutes
        
        5. Quality Controls
        - Fill level must be 95-105% of target volume
        - Temperature must remain stable within ±2°C
        - Visual inspection required for every 100th bottle
        
        6. Risk Assessment
        - Risk R-001: Overfilling leading to product waste
        - Control C-001: Automated fill level monitoring
        - Risk R-002: Temperature deviation affecting product quality
        - Control C-002: Continuous temperature monitoring with alarms
        """
    
    @pytest.fixture
    def sample_pdf_file(self, sample_sop_content):
        """Create a temporary PDF-like file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_sop_content)
            return f.name
    
    @pytest.fixture
    def mock_settings_local(self):
        """Mock settings for local mode testing."""
        settings = Settings()
        settings.mode = "local"
        settings.local_data_path = "./test_data"
        settings.faiss_index_path = "./test_data/faiss_index"
        return settings
    
    @pytest.fixture
    def mock_settings_aws(self):
        """Mock settings for AWS mode testing."""
        settings = Settings()
        settings.mode = "aws"
        settings.aws_region = "us-east-1"
        settings.s3_raw_bucket = "test-sop-raw"
        settings.s3_chunks_bucket = "test-sop-chunks"
        return settings
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_local_mode(self, client, sample_pdf_file, mock_settings_local):
        """Test complete pipeline from ingestion to query in local mode."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Step 1: Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            ingest_result = response.json()
            assert ingest_result["status"] == "success"
            assert len(ingest_result["processed_documents"]) == 1
            
            doc_id = ingest_result["processed_documents"][0]["doc_id"]
            
            # Step 2: Wait for processing to complete
            await asyncio.sleep(2)
            
            # Step 3: Verify document is indexed
            response = client.get("/sources")
            assert response.status_code == 200
            sources = response.json()
            assert len(sources["documents"]) >= 1
            assert any(doc["doc_id"] == doc_id for doc in sources["documents"])
            
            # Step 4: Test question answering
            test_questions = [
                "What is the temperature range for the filler?",
                "Who is responsible for quality checks?",
                "What equipment is required for this procedure?",
                "What are the quality controls for fill levels?",
                "What risks are identified in this SOP?"
            ]
            
            for question in test_questions:
                response = client.post(
                    "/ask",
                    json={"question": question, "filters": {}}
                )
                
                assert response.status_code == 200
                answer_result = response.json()
                
                # Validate answer structure
                assert "answer" in answer_result
                assert "confidence" in answer_result
                assert "citations" in answer_result
                assert "sources" in answer_result
                
                # Validate answer quality
                assert len(answer_result["answer"]) > 10
                assert 0.0 <= answer_result["confidence"] <= 1.0
                assert len(answer_result["citations"]) > 0
                
                # Validate citations have required fields
                for citation in answer_result["citations"]:
                    assert "doc_id" in citation
                    assert "text" in citation
                    assert "page_no" in citation
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_aws_mode(self, client, sample_pdf_file, mock_settings_aws):
        """Test complete pipeline from ingestion to query in AWS mode."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws), \
             patch('boto3.client') as mock_boto_client, \
             patch('opensearchpy.OpenSearch') as mock_opensearch:
            
            # Mock AWS services
            mock_bedrock = MagicMock()
            mock_s3 = MagicMock()
            mock_textract = MagicMock()
            
            mock_boto_client.side_effect = lambda service, **kwargs: {
                'bedrock-runtime': mock_bedrock,
                's3': mock_s3,
                'textract': mock_textract
            }[service]
            
            # Mock Bedrock responses
            mock_bedrock.invoke_model.return_value = {
                'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": "The temperature range is 18-22°C as specified in step 4.1.1."}]
                }).encode())
            }
            
            # Mock embedding responses
            mock_bedrock.invoke_model.side_effect = [
                # Ontology extraction response
                {'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": json.dumps({
                        "title": "SOP-FILL-001: Bottle Filling Procedure",
                        "process_name": "Bottle Filling",
                        "revision": "2.1",
                        "procedure_steps": [
                            {"step_id": "4.1.1", "description": "Verify filler temperature is between 18-22°C"}
                        ]
                    })}]
                }).encode())},
                # Q&A response
                {'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": "The temperature range is 18-22°C as specified in step 4.1.1."}]
                }).encode())}
            ]
            
            # Mock Titan embeddings
            mock_titan_response = {
                'body': MagicMock(read=lambda: json.dumps({
                    "embedding": [0.1] * 768
                }).encode())
            }
            
            # Step 1: Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            ingest_result = response.json()
            assert ingest_result["status"] == "success"
            
            # Step 2: Test question answering
            response = client.post(
                "/ask",
                json={"question": "What is the temperature range?", "filters": {}}
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            assert "answer" in answer_result
            assert "18-22°C" in answer_result["answer"]
    
    @pytest.mark.asyncio
    async def test_filtering_functionality(self, client, sample_pdf_file, mock_settings_local):
        """Test filtering by role, equipment, and document type."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            await asyncio.sleep(2)
            
            # Test role filtering
            response = client.post(
                "/ask",
                json={
                    "question": "What are my responsibilities?",
                    "filters": {"roles": ["Operator"]}
                }
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            assert "operator" in answer_result["answer"].lower()
            
            # Test equipment filtering
            response = client.post(
                "/ask",
                json={
                    "question": "How do I use this equipment?",
                    "filters": {"equipment": ["Filler-01"]}
                }
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            assert len(answer_result["sources"]) > 0
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, client, sample_pdf_file, mock_settings_local):
        """Test confidence scoring for different types of questions."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            await asyncio.sleep(2)
            
            # Test high-confidence question (specific fact in document)
            response = client.post(
                "/ask",
                json={"question": "What is the revision number of SOP-FILL-001?", "filters": {}}
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            assert answer_result["confidence"] > 0.7
            assert "2.1" in answer_result["answer"]
            
            # Test low-confidence question (not in document)
            response = client.post(
                "/ask",
                json={"question": "What is the weather forecast for tomorrow?", "filters": {}}
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            assert answer_result["confidence"] < 0.4
            assert "don't know" in answer_result["answer"].lower() or "not found" in answer_result["answer"].lower()
    
    @pytest.mark.asyncio
    async def test_citation_accuracy(self, client, sample_pdf_file, mock_settings_local):
        """Test that citations accurately reference source content."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            await asyncio.sleep(2)
            
            # Ask question with specific answer in document
            response = client.post(
                "/ask",
                json={"question": "What temperature range is required for the filler?", "filters": {}}
            )
            
            assert response.status_code == 200
            answer_result = response.json()
            
            # Validate citations
            assert len(answer_result["citations"]) > 0
            
            for citation in answer_result["citations"]:
                # Citation should contain relevant text
                citation_text = citation["text"].lower()
                assert any(keyword in citation_text for keyword in ["temperature", "18-22", "filler"])
                
                # Citation should have valid structure
                assert citation["doc_id"]
                assert isinstance(citation["page_no"], int)
                assert len(citation["text"]) > 10
    
    def test_health_check_integration(self, client):
        """Test system health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "components" in health_data
        assert "mode" in health_data
        assert "uptime_seconds" in health_data
        
        # Validate component health
        components = health_data["components"]
        expected_components = ["database", "embeddings", "llm", "storage"]
        
        for component in expected_components:
            if component in components:
                assert "status" in components[component]
                assert components[component]["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_sources_management(self, client, sample_pdf_file, mock_settings_local):
        """Test document source management functionality."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Ingest document
            with open(sample_pdf_file, 'rb') as f:
                response = client.post(
                    ("/ingest/files","/ingest/files",$1files={"files": ("test_sop.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            ingest_result = response.json()
            doc_id = ingest_result["processed_documents"][0]["doc_id"]
            
            # List sources
            response = client.get("/sources")
            assert response.status_code == 200
            sources = response.json()
            assert len(sources["documents"]) >= 1
            
            # Delete source
            response = client.delete(f"/sources/{doc_id}")
            assert response.status_code == 200
            
            # Verify deletion
            response = client.get("/sources")
            assert response.status_code == 200
            sources = response.json()
            assert not any(doc["doc_id"] == doc_id for doc in sources["documents"])
    
    @pytest.mark.asyncio
    async def test_reindex_functionality(self, client, mock_settings_local):
        """Test index rebuilding functionality."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            response = client.post("/reindex")
            assert response.status_code == 200
            
            reindex_result = response.json()
            assert "status" in reindex_result
            assert reindex_result["status"] in ["success", "in_progress"]
    
    def teardown_method(self, method):
        """Clean up test files after each test."""
        # Clean up any temporary files
        test_data_path = Path("./test_data")
        if test_data_path.exists():
            import shutil
            shutil.rmtree(test_data_path, ignore_errors=True)
