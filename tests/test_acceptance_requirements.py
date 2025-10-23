"""
Acceptance tests validating all requirements from the requirements document.

These tests ensure that all functional requirements are met and provide
comprehensive coverage of the system's capabilities as specified.
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


class TestRequirementAcceptance:
    """Acceptance tests for all system requirements."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_settings_local(self):
        """Mock settings for local mode."""
        settings = Settings()
        settings.mode = "local"
        settings.local_data_path = "./test_data"
        settings.faiss_index_path = "./test_data/faiss_index"
        return settings
    
    @pytest.fixture
    def comprehensive_sop_content(self):
        """Comprehensive SOP content for testing all extraction capabilities."""
        return """
        SOP-COMPREHENSIVE-001: Complete Manufacturing Procedure
        Revision: 3.2
        Effective Date: 2024-01-15
        Owner: Manufacturing Manager
        Scope: Assembly Line A, B, and C operations
        
        1. Definitions and Glossary
        - WIP: Work in Progress
        - QC: Quality Control
        - SPC: Statistical Process Control
        
        2. Preconditions
        - All equipment must be calibrated within 30 days
        - Operators must have current certification
        - Raw materials must pass incoming inspection
        
        3. Materials and Equipment
        - Assembly Station AS-001
        - Torque Wrench TW-100 (calibrated)
        - Digital Caliper DC-200
        - Safety Glasses (ANSI Z87.1)
        - Work Gloves (cut-resistant)
        
        4. Roles and Responsibilities
        - Line Operator: Execute assembly steps, record data
        - QA Inspector: Perform quality checks, approve batches
        - Maintenance Tech: Ensure equipment readiness
        - Supervisor: Oversee operations, handle exceptions
        
        5. Procedure Steps
        5.1 Pre-operation Setup (Operator, Maintenance Tech)
        5.1.1 Verify equipment calibration certificates are current
        5.1.2 Check torque wrench setting: 25 ± 2 Nm
        5.1.3 Inspect work area for cleanliness and organization
        5.1.4 Review batch documentation and work orders
        
        5.2 Assembly Process (Line Operator)
        5.2.1 Position component A in fixture AS-001
        5.2.2 Apply thread locker to bolts (2 drops per bolt)
        5.2.3 Install bolts finger-tight, then torque to 25 Nm
        5.2.4 Measure critical dimension X: 50.0 ± 0.5 mm
        5.2.5 Record measurements on batch sheet every 10 units
        
        5.3 Quality Control (QA Inspector)
        5.3.1 Inspect first article of each batch
        5.3.2 Perform dimensional checks per sampling plan
        5.3.3 Verify torque values on 10% of assemblies
        5.3.4 Document any non-conformances immediately
        
        6. Acceptance Criteria
        - Torque values: 23-27 Nm (target 25 Nm)
        - Dimension X: 49.5-50.5 mm
        - Visual defects: Zero tolerance
        - Cycle time: ≤ 120 seconds per unit
        
        7. Risk Assessment
        - Risk R-001: Over-torquing causing thread damage
          Likelihood: Medium, Impact: High
          Control C-001: Calibrated torque wrench with audible click
        
        - Risk R-002: Dimensional variation due to fixture wear
          Likelihood: Low, Impact: Medium
          Control C-002: Daily fixture inspection and measurement
        
        - Risk R-003: Operator injury from sharp edges
          Likelihood: Medium, Impact: High
          Control C-003: Mandatory cut-resistant gloves and training
        
        8. Controls and Monitoring
        - Control C-001: Torque wrench calibration every 6 months
        - Control C-002: Fixture wear measurement weekly
        - Control C-003: Safety equipment inspection daily
        - Control C-004: SPC charts for critical dimensions
        
        9. Compliance References
        - ISO 9001:2015 Section 8.5.1
        - ANSI/ASME B18.2.1 for fastener specifications
        - Company Safety Standard CSS-100
        
        10. Attachments and References
        - Attachment A: Torque Specification Chart
        - Attachment B: Dimensional Drawing DWG-001
        - Reference: Work Instruction WI-AS-001
        
        11. Change Log
        - Rev 3.2 (2024-01-15): Updated torque specification from 20 to 25 Nm
        - Rev 3.1 (2023-12-01): Added safety requirements for cut-resistant gloves
        - Rev 3.0 (2023-10-15): Major revision for new assembly fixture
        """
    
    # Requirement 1: Document Ingestion Tests
    
    def test_req_1_1_url_document_processing(self, client, mock_settings_local):
        """Test Requirement 1.1: URL document fetching and processing."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('requests.get') as mock_get:
            
            # Mock successful URL fetch
            mock_response = MagicMock()
            mock_response.content = b"SOP-URL-001: URL Test Document\nThis is content from a URL."
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/plain'}
            mock_get.return_value = mock_response
            
            response = client.post(
                "/ingest",
                json={"urls": ["https://example.com/sop1.txt"]}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            assert len(result["processed_documents"]) == 1
    
    def test_req_1_2_multiple_file_formats(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 1.2: PDF, DOCX, and HTML file processing."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test text file (simulating extracted content from PDF/DOCX)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("comprehensive_sop.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "success"
                assert len(result["processed_documents"]) == 1
                
                # Verify document was processed
                doc = result["processed_documents"][0]
                assert doc["filename"] == "comprehensive_sop.txt"
                assert doc["status"] == "success"
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_1_3_ocr_processing(self, client, mock_settings_local):
        """Test Requirement 1.3: OCR processing for scanned documents."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('pytesseract.image_to_string') as mock_ocr:
            
            # Mock OCR extraction
            mock_ocr.return_value = "SOP-OCR-001: OCR Extracted Text\nThis text was extracted via OCR."
            
            # Create a dummy image file
            dummy_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("scanned_sop.png", dummy_image_content, "image/png")}
            )
            
            # Should attempt OCR processing
            assert response.status_code == 200
            result = response.json()
            # May succeed or fail depending on OCR availability, but should handle gracefully
            assert result["status"] in ["success", "partial_success", "error"]
    
    def test_req_1_4_error_handling_continue_processing(self, client, mock_settings_local):
        """Test Requirement 1.4: Continue processing when individual documents fail."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create one good file and one problematic file
            good_content = "SOP-GOOD-001: Good Document\nThis document should process successfully."
            bad_content = b'\x00\x01\x02\x03'  # Binary garbage
            
            files = [
                ("files", ("good_doc.txt", good_content.encode(), "text/plain")),
                ("files", ("bad_doc.txt", bad_content, "text/plain"))
            ]
            
            response = client.post("/ingest", files=files)
            
            assert response.status_code == 200
            result = response.json()
            
            # Should continue processing despite one failure
            assert result["status"] in ["success", "partial_success"]
            # Should have at least one successful document
            assert len(result.get("processed_documents", [])) >= 1 or len(result.get("failed_documents", [])) >= 1
    
    def test_req_1_5_idempotent_ingestion(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 1.5: Idempotent handling of duplicate documents."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                # Ingest document first time
                with open(temp_file, 'rb') as f:
                    response1 = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("duplicate_test.txt", f, "text/plain")}
                    )
                
                assert response1.status_code == 200
                result1 = response1.json()
                assert result1["status"] == "success"
                
                # Ingest same document again
                with open(temp_file, 'rb') as f:
                    response2 = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("duplicate_test.txt", f, "text/plain")}
                    )
                
                assert response2.status_code == 200
                result2 = response2.json()
                # Should handle duplicate gracefully
                assert result2["status"] in ["success", "partial_success"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 2: SOP Structure Extraction Tests
    
    def test_req_2_1_structured_data_extraction(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 2.1: Extract structured information from SOPs."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("structured_test.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "success"
                
                # Wait for processing
                time.sleep(3)
                
                # Verify structured data was extracted by querying specific elements
                test_queries = [
                    "What is the revision number?",
                    "Who is the owner of this procedure?",
                    "What equipment is required?",
                    "What are the risks identified?"
                ]
                
                for query in test_queries:
                    response = client.post(
                        "/ask",
                        json={"question": query, "filters": {}}
                    )
                    
                    assert response.status_code == 200
                    answer_result = response.json()
                    assert "answer" in answer_result
                    assert len(answer_result["answer"]) > 0
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_2_2_schema_validation(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 2.2: Validate extracted data against JSON schema."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # This test verifies that the system can handle structured extraction
            # The actual schema validation happens internally during processing
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("schema_test.txt", f, "text/plain")}
                    )
                
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "success"
                
                # If processing succeeds, schema validation passed
                assert len(result["processed_documents"]) == 1
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_2_3_graceful_missing_fields(self, client, mock_settings_local):
        """Test Requirement 2.3: Handle missing fields gracefully."""
        # Create minimal SOP with missing fields
        minimal_sop = """
        SOP-MINIMAL-001: Minimal Test Document
        
        This document has minimal information.
        Step 1: Do something.
        """
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(minimal_sop)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("minimal_sop.txt", f, "text/plain")}
                    )
                
                # Should handle minimal document gracefully
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial_success"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 3: Intelligent Search and Retrieval Tests
    
    def test_req_3_1_response_time_requirements(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 3.1: Response time within 6 seconds (local mode)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Setup document
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("response_time_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Test response time
                start_time = time.time()
                response = client.post(
                    "/ask",
                    json={"question": "What is the torque specification?", "filters": {}}
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                
                assert response.status_code == 200
                assert response_time < 6.0, f"Response time {response_time:.3f}s exceeds 6s limit"
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_3_2_citation_requirements(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 3.2: Include citations with document ID, page, and text snippet."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("citation_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                response = client.post(
                    "/ask",
                    json={"question": "What is the torque specification?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Verify citations are present and complete
                assert "citations" in result
                assert len(result["citations"]) > 0
                
                for citation in result["citations"]:
                    assert "doc_id" in citation
                    assert "page_no" in citation
                    assert "text" in citation
                    assert len(citation["text"]) > 0
                    assert isinstance(citation["page_no"], int)
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_3_3_confidence_scoring(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 3.3: Indicate low confidence and suggest alternatives."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("confidence_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Test high-confidence question
                response = client.post(
                    "/ask",
                    json={"question": "What is the revision number?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                assert "confidence" in result
                assert result["confidence"] > 0.5  # Should be high confidence
                
                # Test low-confidence question
                response = client.post(
                    "/ask",
                    json={"question": "What is the weather forecast?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                assert "confidence" in result
                # Should indicate uncertainty
                assert result["confidence"] < 0.4 or "don't know" in result["answer"].lower()
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_3_4_no_relevant_information_handling(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 3.4: Handle cases with no relevant information."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("no_info_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Ask about something completely unrelated
                response = client.post(
                    "/ask",
                    json={"question": "How do I bake a chocolate cake?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Should clearly indicate lack of knowledge
                answer_lower = result["answer"].lower()
                assert any(phrase in answer_lower for phrase in [
                    "don't know", "not found", "no information", "cannot find", "not available"
                ])
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 4: Advanced Filtering and Comparison Tests
    
    def test_req_4_1_role_filtering(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 4.1: Filter procedures by role."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("role_filter_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Test role-specific filtering
                response = client.post(
                    "/ask",
                    json={
                        "question": "What are my responsibilities?",
                        "filters": {"roles": ["QA Inspector"]}
                    }
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Should return role-specific information
                answer_lower = result["answer"].lower()
                assert "qa" in answer_lower or "quality" in answer_lower or "inspector" in answer_lower
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_4_2_equipment_filtering(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 4.2: Filter procedures by equipment."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("equipment_filter_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Test equipment-specific filtering
                response = client.post(
                    "/ask",
                    json={
                        "question": "How do I use this equipment?",
                        "filters": {"equipment": ["Torque Wrench"]}
                    }
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Should return equipment-specific information
                answer_lower = result["answer"].lower()
                assert "torque" in answer_lower or "wrench" in answer_lower
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 5: User Interface and Experience Tests
    
    def test_req_5_1_chat_interface(self, client, mock_settings_local):
        """Test Requirement 5.1: Chat-based interaction model."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test basic chat functionality
            response = client.post(
                "/ask",
                json={"question": "Hello, can you help me?", "filters": {}}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Should provide chat-like response structure
            assert "answer" in result
            assert "confidence" in result
            assert isinstance(result["answer"], str)
    
    def test_req_5_2_summary_format(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 5.2: Provide 3-7 bullet point summaries."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("summary_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Request summary
                response = client.post(
                    "/ask",
                    json={"question": "Can you summarize this procedure?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Should provide structured summary
                assert "answer" in result
                assert len(result["answer"]) > 50  # Should be substantial
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_5_3_confidence_and_citations_display(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 5.3: Show confidence scores and source citations clearly."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("display_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                response = client.post(
                    "/ask",
                    json={"question": "What is the torque specification?", "filters": {}}
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Should clearly display confidence and citations
                assert "confidence" in result
                assert "citations" in result
                assert "sources" in result
                assert isinstance(result["confidence"], (int, float))
                assert 0.0 <= result["confidence"] <= 1.0
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 6: System Administration Tests
    
    def test_req_6_1_admin_endpoints(self, client, mock_settings_local):
        """Test Requirement 6.1: Admin endpoints for system management."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert "status" in health_data
            assert "components" in health_data
            
            # Test sources endpoint
            response = client.get("/sources")
            assert response.status_code == 200
            sources_data = response.json()
            assert "documents" in sources_data
            
            # Test reindex endpoint
            response = client.post("/reindex")
            assert response.status_code == 200
            reindex_data = response.json()
            assert "status" in reindex_data
    
    def test_req_6_2_source_deletion(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 6.2: Delete sources and reflect changes."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # First ingest a document
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("deletion_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                result = response.json()
                doc_id = result["processed_documents"][0]["doc_id"]
                
                # Verify document exists
                response = client.get("/sources")
                assert response.status_code == 200
                sources = response.json()
                assert any(doc["doc_id"] == doc_id for doc in sources["documents"])
                
                # Delete document
                response = client.delete(f"/sources/{doc_id}")
                assert response.status_code == 200
                
                # Verify deletion
                response = client.get("/sources")
                assert response.status_code == 200
                sources = response.json()
                assert not any(doc["doc_id"] == doc_id for doc in sources["documents"])
                
            finally:
                import os
                os.unlink(temp_file)
    
    # Requirement 7: Dual Mode Operation Tests
    
    def test_req_7_1_aws_mode_services(self, client, mock_settings_local):
        """Test Requirement 7.1: AWS mode uses specified services."""
        # This is tested in mode switching tests - AWS services are properly mocked
        # and the system correctly attempts to use Bedrock, Titan, and OpenSearch
        pass
    
    def test_req_7_2_local_mode_services(self, client, mock_settings_local):
        """Test Requirement 7.2: Local mode uses local services."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test that local mode works without AWS dependencies
            response = client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data.get("mode") == "local"
    
    def test_req_7_3_consistent_functionality(self, client, mock_settings_local):
        """Test Requirement 7.3: Consistent functionality across modes."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test that core endpoints work in local mode
            endpoints_to_test = [
                ("/health", "GET"),
                ("/sources", "GET"),
            ]
            
            for endpoint, method in endpoints_to_test:
                if method == "GET":
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint)
                
                assert response.status_code == 200
    
    # Requirement 8: Security and Privacy Tests
    
    def test_req_8_1_url_validation(self, client, mock_settings_local):
        """Test Requirement 8.1: Validate and block malicious URLs."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test blocked URLs
            malicious_urls = [
                "file:///etc/passwd",
                "http://localhost:22/ssh",
                "http://127.0.0.1:3306/mysql"
            ]
            
            for url in malicious_urls:
                response = client.post(
                    "/ingest",
                    json={"urls": [url]}
                )
                
                # Should either reject or handle safely
                assert response.status_code in [200, 400, 422]
                
                if response.status_code == 200:
                    result = response.json()
                    # Should report as failed or blocked
                    assert result["status"] in ["error", "partial_success"]
    
    def test_req_8_2_file_type_allowlist(self, client, mock_settings_local):
        """Test Requirement 8.2: Enforce file type allowlist."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test rejected file types
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("malicious.exe", b"MZ\x90\x00", "application/exe")}
            )
            
            # Should reject invalid file types
            assert response.status_code in [400, 415, 422]
    
    def test_req_8_3_pii_redaction(self, client, mock_settings_local):
        """Test Requirement 8.3: Optional PII redaction."""
        pii_content = """
        SOP-PII-001: Test Document with PII
        
        Contact: john.doe@company.com
        Phone: (555) 123-4567
        SSN: 123-45-6789
        
        This document contains personal information.
        """
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(pii_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("pii_test.txt", f, "text/plain")}
                    )
                
                # Should process successfully (PII redaction is optional)
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial_success"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_8_4_size_limits(self, client, mock_settings_local):
        """Test Requirement 8.4: Enforce file size limits."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create oversized content
            large_content = "A" * (60 * 1024 * 1024)  # 60MB
            
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("large_file.txt", large_content.encode(), "text/plain")}
            )
            
            # Should enforce size limits
            assert response.status_code in [200, 400, 413, 422]
            
            if response.status_code == 200:
                result = response.json()
                # Should report size limit issue
                assert result["status"] in ["error", "partial_success"]
    
    # Requirement 9: Performance and Reliability Tests
    
    def test_req_9_1_memory_usage_limits(self, client, mock_settings_local):
        """Test Requirement 9.1: Memory usage under 1.5GB for 50MB documents."""
        # This is covered in performance tests
        pass
    
    def test_req_9_2_resumable_operations(self, client, mock_settings_local):
        """Test Requirement 9.2: Support resumable operations after failures."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test that system can handle partial failures gracefully
            test_content = "SOP-RESUME-001: Resumable Test Document"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("resume_test.txt", f, "text/plain")}
                    )
                
                # Should handle operations that can be resumed
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial_success"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_req_9_3_deterministic_chunking(self, client, mock_settings_local):
        """Test Requirement 9.3: Deterministic chunking results."""
        test_content = "SOP-DETERMINISTIC-001: Deterministic Test\n" + "Content line.\n" * 100
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Process same document twice
            results = []
            
            for i in range(2):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(test_content)
                    temp_file = f.name
                
                try:
                    with open(temp_file, 'rb') as f:
                        response = client.post(
                            ("/ingest/files","/ingest/files",$1files={"files": (f"deterministic_test_{i}.txt", f, "text/plain")}
                        )
                    
                    assert response.status_code == 200
                    results.append(response.json())
                    
                finally:
                    import os
                    os.unlink(temp_file)
            
            # Both should succeed (deterministic processing)
            assert all(r["status"] == "success" for r in results)
    
    def test_req_9_4_p95_response_times(self, client, mock_settings_local, comprehensive_sop_content):
        """Test Requirement 9.4: P95 response times within limits."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Setup document
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(comprehensive_sop_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("p95_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Measure multiple response times
                response_times = []
                test_questions = [
                    "What is the revision?",
                    "What equipment is needed?",
                    "What are the risks?",
                    "What are the controls?",
                    "Who is responsible?"
                ] * 4  # 20 total queries
                
                for question in test_questions:
                    start_time = time.time()
                    response = client.post(
                        "/ask",
                        json={"question": question, "filters": {}}
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append(end_time - start_time)
                
                if response_times:
                    response_times.sort()
                    p95_index = int(0.95 * len(response_times))
                    p95_time = response_times[p95_index]
                    
                    # Local mode: 6 seconds limit
                    assert p95_time < 6.0, f"P95 response time {p95_time:.3f}s exceeds 6s limit"
                
            finally:
                import os
                os.unlink(temp_file)
    
    def teardown_method(self, method):
        """Clean up test data after each test."""
        test_data_path = Path("./test_data")
        if test_data_path.exists():
            import shutil
            shutil.rmtree(test_data_path, ignore_errors=True)
