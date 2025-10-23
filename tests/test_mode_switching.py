"""
Mode switching tests to verify AWS/local compatibility.

Tests that the system can operate correctly in both AWS and local modes,
and that switching between modes maintains functionality and data consistency.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sop_qa_tool.api.main import app
from sop_qa_tool.config.settings import Settings


class TestModeSwitching:
    """Tests for AWS/local mode compatibility and switching."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_sop_content(self):
        """Sample SOP content for testing."""
        return """
        SOP-MODE-001: Mode Switching Test Procedure
        Revision: 1.0
        Effective Date: 2024-01-01
        
        1. Purpose
        This procedure tests mode switching functionality.
        
        2. Equipment
        - Test Equipment A
        - Test Equipment B
        
        3. Steps
        3.1 Initialize system in current mode
        3.2 Verify functionality works correctly
        3.3 Test query responses are consistent
        
        4. Quality Controls
        - Response accuracy must be maintained
        - Citations must be preserved
        - Performance should meet mode-specific requirements
        """
    
    @pytest.fixture
    def mock_settings_local(self):
        """Mock settings for local mode."""
        settings = Settings()
        settings.mode = "local"
        settings.local_data_path = "./test_data_local"
        settings.faiss_index_path = "./test_data_local/faiss_index"
        settings.hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
        return settings
    
    @pytest.fixture
    def mock_settings_aws(self):
        """Mock settings for AWS mode."""
        settings = Settings()
        settings.mode = "aws"
        settings.aws_region = "us-east-1"
        settings.s3_raw_bucket = "test-sop-raw"
        settings.s3_chunks_bucket = "test-sop-chunks"
        settings.opensearch_endpoint = "https://test-search.us-east-1.aoss.amazonaws.com"
        settings.bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        settings.titan_embeddings_id = "amazon.titan-embed-text-v2:0"
        return settings
    
    def test_local_mode_functionality(self, client, sample_sop_content, mock_settings_local):
        """Test complete functionality in local mode."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_sop_content)
                temp_file = f.name
            
            try:
                # Test ingestion
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        "/ingest/files",
                        files={"files": ("mode_test_local.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 200
                ingest_result = response.json()
                assert ingest_result["status"] == "success"
                
                # Wait for processing
                time.sleep(2)
                
                # Test query functionality
                response = client.post(
                    "/ask",
                    json={"question": "What is the purpose of this procedure?", "filters": {}}
                )
                
                assert response.status_code == 200
                answer_result = response.json()
                
                # Validate local mode response structure
                assert "answer" in answer_result
                assert "confidence" in answer_result
                assert "citations" in answer_result
                assert "sources" in answer_result
                
                # Validate content
                assert "mode switching" in answer_result["answer"].lower()
                assert len(answer_result["citations"]) > 0
                
                # Test sources endpoint
                response = client.get("/sources")
                assert response.status_code == 200
                sources = response.json()
                assert len(sources["documents"]) >= 1
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_aws_mode_functionality(self, client, sample_sop_content, mock_settings_aws):
        """Test complete functionality in AWS mode with mocked services."""
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
                    "content": [{"text": "This procedure tests mode switching functionality as stated in the purpose section."}]
                }).encode())
            }
            
            # Mock Titan embeddings
            mock_bedrock.invoke_model.side_effect = [
                # Ontology extraction
                {'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": json.dumps({
                        "title": "SOP-MODE-001: Mode Switching Test Procedure",
                        "purpose": "This procedure tests mode switching functionality",
                        "equipment": ["Test Equipment A", "Test Equipment B"]
                    })}]
                }).encode())},
                # Embeddings
                {'body': MagicMock(read=lambda: json.dumps({
                    "embedding": [0.1] * 768
                }).encode())},
                # Q&A response
                {'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": "This procedure tests mode switching functionality as stated in the purpose section."}]
                }).encode())}
            ]
            
            # Mock OpenSearch
            mock_os_instance = MagicMock()
            mock_opensearch.return_value = mock_os_instance
            mock_os_instance.search.return_value = {
                'hits': {
                    'hits': [{
                        '_source': {
                            'chunk_text': 'This procedure tests mode switching functionality.',
                            'doc_id': 'test_doc_001',
                            'metadata': {'page_no': 1}
                        },
                        '_score': 0.95
                    }]
                }
            }
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_sop_content)
                temp_file = f.name
            
            try:
                # Test ingestion
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("mode_test_aws.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 200
                ingest_result = response.json()
                assert ingest_result["status"] == "success"
                
                # Test query functionality
                response = client.post(
                    "/ask",
                    json={"question": "What is the purpose of this procedure?", "filters": {}}
                )
                
                assert response.status_code == 200
                answer_result = response.json()
                
                # Validate AWS mode response structure
                assert "answer" in answer_result
                assert "confidence" in answer_result
                assert "citations" in answer_result
                
                # Validate AWS-specific behavior
                assert "mode switching" in answer_result["answer"].lower()
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_mode_specific_configurations(self, mock_settings_local, mock_settings_aws):
        """Test that mode-specific configurations are correctly applied."""
        # Test local mode settings
        assert mock_settings_local.mode == "local"
        assert mock_settings_local.LOCAL_DATA_PATH == "./test_data_local"
        assert mock_settings_local.FAISS_INDEX_PATH == "./test_data_local/faiss_index"
        assert "sentence-transformers" in mock_settings_local.HF_MODEL_PATH
        
        # Test AWS mode settings
        assert mock_settings_aws.mode == "aws"
        assert mock_settings_aws.AWS_REGION == "us-east-1"
        assert mock_settings_aws.S3_RAW_BUCKET == "test-sop-raw"
        assert mock_settings_aws.OPENSEARCH_ENDPOINT.startswith("https://")
        assert "anthropic.claude" in mock_settings_aws.BEDROCK_MODEL_ID
        assert "amazon.titan" in mock_settings_aws.TITAN_EMBEDDINGS_ID
    
    def test_embedding_dimension_consistency(self, client, sample_sop_content):
        """Test that embedding dimensions are consistent within each mode."""
        # Test local mode embeddings (384 dimensions)
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            # Mock sentence transformer to return 384-dim embeddings
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1] * 384]
            mock_st.return_value = mock_model
            
            from sop_qa_tool.services.embedder import EmbeddingService
            embedder = EmbeddingService()
            
            embeddings = embedder.embed_texts(["test text"])
            assert embeddings.shape[1] == 384, f"Local mode should use 384-dim embeddings, got {embeddings.shape[1]}"
        
        # Test AWS mode embeddings (768 dimensions)
        mock_settings_aws = Settings()
        mock_settings_aws.mode = "aws"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws), \
             patch('boto3.client') as mock_boto:
            
            mock_bedrock = MagicMock()
            mock_boto.return_value = mock_bedrock
            
            # Mock Titan embeddings response (768 dimensions)
            mock_bedrock.invoke_model.return_value = {
                'body': MagicMock(read=lambda: json.dumps({
                    "embedding": [0.1] * 768
                }).encode())
            }
            
            from sop_qa_tool.services.embedder import EmbeddingService
            embedder = EmbeddingService()
            
            embeddings = embedder.embed_texts(["test text"])
            assert embeddings.shape[1] == 768, f"AWS mode should use 768-dim embeddings, got {embeddings.shape[1]}"
    
    def test_storage_backend_switching(self, client, sample_sop_content):
        """Test that storage backends work correctly in each mode."""
        # Test local mode with FAISS
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        mock_settings_local.faiss_index_path = "./test_faiss"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('faiss.IndexFlatIP') as mock_faiss:
            
            mock_index = MagicMock()
            mock_faiss.return_value = mock_index
            
            from sop_qa_tool.services.vectorstore import VectorStoreService
            vectorstore = VectorStoreService()
            
            # Test that FAISS is used in local mode
            assert vectorstore.mode == "local"
        
        # Test AWS mode with OpenSearch
        mock_settings_aws = Settings()
        mock_settings_aws.mode = "aws"
        mock_settings_aws.opensearch_endpoint = "https://test.aoss.amazonaws.com"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws), \
             patch('opensearchpy.OpenSearch') as mock_os:
            
            mock_client = MagicMock()
            mock_os.return_value = mock_client
            
            from sop_qa_tool.services.vectorstore import VectorStoreService
            vectorstore = VectorStoreService()
            
            # Test that OpenSearch is used in AWS mode
            assert vectorstore.mode == "aws"
    
    def test_llm_backend_switching(self, sample_sop_content):
        """Test that LLM backends switch correctly between modes."""
        # Test local mode with HuggingFace
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('transformers.pipeline') as mock_pipeline:
            
            mock_pipe = MagicMock()
            mock_pipe.return_value = [{"generated_text": "Test response"}]
            mock_pipeline.return_value = mock_pipe
            
            from sop_qa_tool.services.ontology_extractor import OntologyExtractor
            extractor = OntologyExtractor()
            
            # Test local mode extraction
            result = extractor.extract_sop_structure("test content")
            assert result is not None
        
        # Test AWS mode with Bedrock
        mock_settings_aws = Settings()
        mock_settings_aws.mode = "aws"
        mock_settings_aws.bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws), \
             patch('boto3.client') as mock_boto:
            
            mock_bedrock = MagicMock()
            mock_boto.return_value = mock_bedrock
            
            mock_bedrock.invoke_model.return_value = {
                'body': MagicMock(read=lambda: json.dumps({
                    "content": [{"text": json.dumps({"title": "Test SOP"})}]
                }).encode())
            }
            
            from sop_qa_tool.services.ontology_extractor import OntologyExtractor
            extractor = OntologyExtractor()
            
            # Test AWS mode extraction
            result = extractor.extract_sop_structure("test content")
            assert result is not None
    
    def test_performance_differences_between_modes(self, client, sample_sop_content):
        """Test that performance characteristics differ appropriately between modes."""
        # This test validates that we acknowledge different performance expectations
        # Local mode: slower but works offline
        # AWS mode: faster but requires internet
        
        # Test local mode performance expectations
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Local mode should have higher response time tolerance
            max_local_response_time = 6.0  # seconds
            
            # Create test document
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_sop_content)
                temp_file = f.name
            
            try:
                # Ingest and query
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("perf_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                
                time.sleep(2)
                
                start_time = time.time()
                response = client.post(
                    "/ask",
                    json={"question": "What is this procedure about?", "filters": {}}
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                print(f"Local mode response time: {response_time:.3f}s")
                
                # Local mode should be within its tolerance
                assert response_time < max_local_response_time
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_error_handling_consistency(self, client):
        """Test that error handling is consistent across modes."""
        # Test local mode error handling
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test invalid file upload
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("test.exe", b"invalid content", "application/exe")}
            )
            
            # Should handle error gracefully
            assert response.status_code in [400, 422]  # Bad request or validation error
        
        # Test AWS mode error handling
        mock_settings_aws = Settings()
        mock_settings_aws.mode = "aws"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws):
            # Test invalid file upload
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("test.exe", b"invalid content", "application/exe")}
            )
            
            # Should handle error gracefully (same as local mode)
            assert response.status_code in [400, 422]
    
    def test_health_check_mode_awareness(self, client):
        """Test that health checks report mode-specific component status."""
        # Test local mode health check
        mock_settings_local = Settings()
        mock_settings_local.mode = "local"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            response = client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert "mode" in health_data
            assert health_data["mode"] == "local"
        
        # Test AWS mode health check
        mock_settings_aws = Settings()
        mock_settings_aws.mode = "aws"
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws):
            response = client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert "mode" in health_data
            assert health_data["mode"] == "aws"
    
    def teardown_method(self, method):
        """Clean up test data after each test."""
        # Clean up test directories
        for path in ["./test_data_local", "./test_data_aws", "./test_faiss"]:
            test_path = Path(path)
            if test_path.exists():
                import shutil
                shutil.rmtree(test_path, ignore_errors=True)
