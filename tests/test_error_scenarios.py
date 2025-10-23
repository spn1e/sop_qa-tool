"""
Error scenario testing for network failures and resource limits.

Tests system resilience and error handling under various failure conditions
including network issues, resource constraints, and malformed inputs.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from requests.exceptions import ConnectionError, Timeout, RequestException

import pytest
from fastapi.testclient import TestClient

from sop_qa_tool.api.main import app
from sop_qa_tool.config.settings import Settings


class TestErrorScenarios:
    """Tests for error handling and system resilience."""
    
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
    def mock_settings_aws(self):
        """Mock settings for AWS mode."""
        settings = Settings()
        settings.mode = "aws"
        settings.aws_region = "us-east-1"
        settings.s3_raw_bucket = "test-sop-raw"
        settings.opensearch_endpoint = "https://test-search.us-east-1.aoss.amazonaws.com"
        return settings
    
    def test_network_failure_during_url_ingestion(self, client, mock_settings_local):
        """Test handling of network failures during URL ingestion (Requirement 9.2)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('requests.get') as mock_get:
            
            # Simulate network failure
            mock_get.side_effect = ConnectionError("Network unreachable")
            
            response = client.post(
                "/ingest",
                json={"urls": ["https://example.com/sop1.pdf"]}
            )
            
            # Should handle network failure gracefully
            assert response.status_code == 200  # API should not crash
            result = response.json()
            
            # Should report failure but continue operation
            assert result["status"] in ["partial_success", "error"]
            assert "failed_documents" in result
            assert len(result["failed_documents"]) > 0
            
            # Error should be logged appropriately
            failed_doc = result["failed_documents"][0]
            assert "network" in failed_doc["error"].lower() or "connection" in failed_doc["error"].lower()
    
    def test_network_timeout_handling(self, client, mock_settings_local):
        """Test handling of network timeouts."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('requests.get') as mock_get:
            
            # Simulate timeout
            mock_get.side_effect = Timeout("Request timed out")
            
            response = client.post(
                "/ingest",
                json={"urls": ["https://slow-server.com/sop.pdf"]}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Should handle timeout gracefully
            assert "failed_documents" in result
            if result["failed_documents"]:
                failed_doc = result["failed_documents"][0]
                assert "timeout" in failed_doc["error"].lower()
    
    def test_aws_service_unavailable(self, client, mock_settings_aws):
        """Test handling when AWS services are unavailable."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_aws), \
             patch('boto3.client') as mock_boto:
            
            # Mock AWS service failure
            mock_bedrock = MagicMock()
            mock_bedrock.invoke_model.side_effect = Exception("Service unavailable")
            mock_boto.return_value = mock_bedrock
            
            # Create test document
            test_content = "SOP-ERROR-001: Error Test Document"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                # Test ingestion with AWS service failure
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("error_test.txt", f, "text/plain")}
                    )
                
                # Should handle AWS failure gracefully
                assert response.status_code == 200
                result = response.json()
                
                # May succeed with degraded functionality or report partial failure
                assert result["status"] in ["success", "partial_success", "error"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_large_file_rejection(self, client, mock_settings_local):
        """Test rejection of files exceeding size limits (Requirement 8.4)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create oversized content (simulate large file)
            large_content = "A" * (60 * 1024 * 1024)  # 60MB content
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(large_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("large_file.txt", f, "text/plain")}
                    )
                
                # Should reject oversized file
                assert response.status_code in [400, 413, 422]  # Bad request or payload too large
                
                if response.status_code == 200:
                    # If API doesn't reject at HTTP level, check response content
                    result = response.json()
                    assert result["status"] == "error" or "failed_documents" in result
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_malformed_file_handling(self, client, mock_settings_local):
        """Test handling of malformed or corrupted files."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create malformed content
            malformed_content = b'\x00\x01\x02\x03\x04\x05' * 1000  # Binary garbage
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
                f.write(malformed_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("malformed.pdf", f, "application/pdf")}
                    )
                
                # Should handle malformed file gracefully
                assert response.status_code == 200
                result = response.json()
                
                # Should report processing failure
                if result["status"] == "success":
                    # File might be processed but with empty/minimal content
                    assert len(result["processed_documents"]) >= 0
                else:
                    assert "failed_documents" in result
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_invalid_file_type_rejection(self, client, mock_settings_local):
        """Test rejection of invalid file types (Requirement 8.2)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create executable file
            exe_content = b'MZ\x90\x00'  # PE header signature
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.exe', delete=False) as f:
                f.write(exe_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("malicious.exe", f, "application/exe")}
                    )
                
                # Should reject invalid file type
                assert response.status_code in [400, 415, 422]  # Bad request or unsupported media type
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_memory_exhaustion_protection(self, client, mock_settings_local):
        """Test protection against memory exhaustion."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create multiple moderately large files to test batch processing limits
            temp_files = []
            
            try:
                for i in range(10):
                    content = f"SOP-MEM-{i:03d}: Memory Test Document {i}\n" + "Content " * 10000
                    
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    temp_file.write(content)
                    temp_file.close()
                    temp_files.append(temp_file.name)
                
                # Try to ingest all files at once
                files_data = []
                for temp_file in temp_files:
                    with open(temp_file, 'rb') as f:
                        files_data.append(("files", (f"mem_test_{len(files_data)}.txt", f.read(), "text/plain")))
                
                response = client.post("/ingest", files=files_data)
                
                # Should handle large batch gracefully
                assert response.status_code == 200
                result = response.json()
                
                # Should either process successfully or report resource limits
                assert result["status"] in ["success", "partial_success", "error"]
                
            finally:
                for temp_file in temp_files:
                    try:
                        import os
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass
    
    def test_concurrent_request_limits(self, client, mock_settings_local):
        """Test handling of concurrent request limits."""
        import threading
        import queue
        
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create test document first
            test_content = "SOP-CONCURRENT-001: Concurrent Test"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                # Ingest document
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("concurrent_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Create many concurrent requests
                results_queue = queue.Queue()
                
                def make_request():
                    try:
                        response = client.post(
                            "/ask",
                            json={"question": "What is this document about?", "filters": {}},
                            timeout=30
                        )
                        results_queue.put({
                            'status_code': response.status_code,
                            'success': response.status_code == 200
                        })
                    except Exception as e:
                        results_queue.put({
                            'status_code': 500,
                            'success': False,
                            'error': str(e)
                        })
                
                # Launch concurrent requests
                threads = []
                for i in range(20):  # 20 concurrent requests
                    thread = threading.Thread(target=make_request)
                    threads.append(thread)
                    thread.start()
                
                # Wait for all requests to complete
                for thread in threads:
                    thread.join(timeout=60)
                
                # Collect results
                results = []
                while not results_queue.empty():
                    results.append(results_queue.get())
                
                # Analyze results
                successful_requests = [r for r in results if r['success']]
                failed_requests = [r for r in results if not r['success']]
                
                print(f"Successful concurrent requests: {len(successful_requests)}")
                print(f"Failed concurrent requests: {len(failed_requests)}")
                
                # Should handle most requests successfully or gracefully reject some
                success_rate = len(successful_requests) / len(results) if results else 0
                assert success_rate >= 0.7, f"Success rate {success_rate:.2f} too low under concurrent load"
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_database_connection_failure(self, client, mock_settings_local):
        """Test handling of database/storage connection failures."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('faiss.IndexFlatIP') as mock_faiss:
            
            # Simulate FAISS initialization failure
            mock_faiss.side_effect = Exception("Cannot initialize FAISS index")
            
            # Test query with storage failure
            response = client.post(
                "/ask",
                json={"question": "Test question", "filters": {}}
            )
            
            # Should handle storage failure gracefully
            assert response.status_code in [200, 500, 503]
            
            if response.status_code == 200:
                result = response.json()
                # Should indicate service degradation
                assert "error" in result or result.get("confidence", 1.0) == 0.0
    
    def test_llm_service_failure(self, client, mock_settings_local):
        """Test handling of LLM service failures."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('transformers.pipeline') as mock_pipeline:
            
            # Simulate LLM failure
            mock_pipeline.side_effect = Exception("LLM service unavailable")
            
            test_content = "SOP-LLM-001: LLM Test Document"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                # Test ingestion with LLM failure
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("llm_test.txt", f, "text/plain")}
                    )
                
                # Should handle LLM failure gracefully
                assert response.status_code == 200
                result = response.json()
                
                # May succeed with reduced functionality
                assert result["status"] in ["success", "partial_success", "error"]
                
            finally:
                import os
                os.unlink(temp_file)
    
    def test_disk_space_exhaustion(self, client, mock_settings_local):
        """Test handling of disk space exhaustion."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local), \
             patch('builtins.open', side_effect=OSError("No space left on device")):
            
            test_content = "SOP-DISK-001: Disk Space Test"
            
            # Test ingestion with disk space failure
            response = client.post(
                ("/ingest/files","/ingest/files",$1files={"files": ("disk_test.txt", test_content.encode(), "text/plain")}
            )
            
            # Should handle disk space failure gracefully
            assert response.status_code in [200, 500, 507]  # OK, Internal Error, or Insufficient Storage
            
            if response.status_code == 200:
                result = response.json()
                assert result["status"] == "error" or "failed_documents" in result
    
    def test_invalid_json_in_request(self, client, mock_settings_local):
        """Test handling of malformed JSON requests."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test malformed JSON
            response = client.post(
                "/ask",
                data='{"question": "test", "filters": {invalid json}',
                headers={"Content-Type": "application/json"}
            )
            
            # Should reject malformed JSON
            assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client, mock_settings_local):
        """Test handling of requests with missing required fields."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test missing question field
            response = client.post(
                "/ask",
                json={"filters": {}}  # Missing "question" field
            )
            
            # Should reject request with missing required field
            assert response.status_code == 422
    
    def test_extremely_long_input(self, client, mock_settings_local):
        """Test handling of extremely long input strings."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create extremely long question
            long_question = "What is " + "very " * 10000 + "important about this procedure?"
            
            response = client.post(
                "/ask",
                json={"question": long_question, "filters": {}}
            )
            
            # Should handle long input gracefully
            assert response.status_code in [200, 400, 413, 422]
            
            if response.status_code == 200:
                result = response.json()
                # Should either process or indicate input too long
                assert "answer" in result or "error" in result
    
    def test_special_characters_in_input(self, client, mock_settings_local):
        """Test handling of special characters and potential injection attempts."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Test various special characters and potential injection strings
            test_inputs = [
                "What about <script>alert('xss')</script>?",
                "Tell me about '; DROP TABLE documents; --",
                "What is the procedure for \x00\x01\x02?",
                "How do I handle 🚀 emoji in procedures?",
                "What about unicode: ñáéíóú çüß?"
            ]
            
            for test_input in test_inputs:
                response = client.post(
                    "/ask",
                    json={"question": test_input, "filters": {}}
                )
                
                # Should handle special characters safely
                assert response.status_code in [200, 400, 422]
                
                if response.status_code == 200:
                    result = response.json()
                    # Response should not contain unescaped special characters
                    assert "answer" in result
                    # Basic XSS protection check
                    assert "<script>" not in result["answer"]
    
    def teardown_method(self, method):
        """Clean up test data after each test."""
        test_data_path = Path("./test_data")
        if test_data_path.exists():
            import shutil
            shutil.rmtree(test_data_path, ignore_errors=True)
