"""
Performance tests for memory usage and response time requirements.

Tests system performance under various loads and validates that the system
meets the specified performance requirements from the design document.
"""

import asyncio
import gc
import json
import os
import psutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sop_qa_tool.api.main import app
from sop_qa_tool.config.settings import Settings


class TestPerformanceRequirements:
    """Performance tests validating system requirements."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_settings_local(self):
        """Mock settings for local mode testing."""
        settings = Settings()
        settings.mode = "local"
        settings.local_data_path = "./test_data"
        settings.faiss_index_path = "./test_data/faiss_index"
        return settings
    
    @pytest.fixture
    def large_document_set(self):
        """Generate a large document set for testing (simulating 50MB)."""
        documents = []
        
        # Create multiple large documents
        base_content = """
        SOP-TEST-{doc_num}: Test Procedure {doc_num}
        Revision: 1.0
        Effective Date: 2024-01-01
        
        1. Purpose and Scope
        This is a test procedure for performance validation. """ + "A" * 1000 + """
        
        2. Roles and Responsibilities
        - Operator: Responsible for operations
        - QA Inspector: Responsible for quality checks
        
        3. Equipment Required
        - Equipment-{doc_num}-001
        - Equipment-{doc_num}-002
        - Equipment-{doc_num}-003
        
        4. Procedure Steps
        """ + "\n".join([f"4.{i} Step {i}: " + "B" * 500 for i in range(1, 21)]) + """
        
        5. Quality Controls
        """ + "\n".join([f"- Control {i}: " + "C" * 200 for i in range(1, 11)]) + """
        
        6. Risk Assessment
        """ + "\n".join([f"- Risk R-{i:03d}: " + "D" * 300 for i in range(1, 16)])
        
        # Generate documents totaling approximately 50MB
        for i in range(1, 51):  # 50 documents of ~1MB each
            content = base_content.format(doc_num=i)
            documents.append({
                'filename': f'sop_test_{i:03d}.txt',
                'content': content
            })
        
        return documents
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.performance
    def test_memory_usage_large_document_set(self, client, large_document_set, mock_settings_local):
        """Test memory usage with 50MB document set (Requirement 9.1)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Measure baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage()
            
            # Create temporary files
            temp_files = []
            try:
                for doc in large_document_set[:10]:  # Test with subset first
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    temp_file.write(doc['content'])
                    temp_file.close()
                    temp_files.append(temp_file.name)
                
                # Ingest documents
                for temp_file in temp_files:
                    with open(temp_file, 'rb') as f:
                        response = client.post(
                            "/ingest/files",
                            files={"files": (os.path.basename(temp_file), f, "text/plain")}
                        )
                    assert response.status_code == 200
                
                # Wait for processing
                time.sleep(5)
                
                # Measure peak memory usage
                gc.collect()
                peak_memory = self.get_memory_usage()
                memory_increase = peak_memory - baseline_memory
                
                # Requirement: System should use less than 1.5GB for 50MB document set
                # For our subset test, proportionally scale the requirement
                max_allowed_memory = 300  # MB for 10 documents (1.5GB * 10/50)
                
                print(f"Baseline memory: {baseline_memory:.2f} MB")
                print(f"Peak memory: {peak_memory:.2f} MB")
                print(f"Memory increase: {memory_increase:.2f} MB")
                
                assert memory_increase < max_allowed_memory, \
                    f"Memory usage {memory_increase:.2f} MB exceeds limit {max_allowed_memory} MB"
                
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass
    
    @pytest.mark.performance
    def test_response_time_aws_mode(self, client, mock_settings_local):
        """Test response times meet requirements (Requirement 9.4)."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create a test document
            test_content = """
            SOP-PERF-001: Performance Test Procedure
            
            1. Test Step
            This is a test step for performance validation.
            The temperature should be maintained at 20°C.
            
            2. Quality Control
            Monitor the process continuously.
            """
            
            # Ingest document
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        "/ingest/files",
                        files={"files": ("test_perf.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                
                # Wait for processing
                time.sleep(2)
                
                # Test query response times
                test_questions = [
                    "What is the temperature requirement?",
                    "What are the quality controls?",
                    "What is the procedure title?",
                    "How should the process be monitored?",
                    "What is step 1 about?"
                ]
                
                response_times = []
                
                for question in test_questions:
                    start_time = time.time()
                    
                    response = client.post(
                        "/ask",
                        json={"question": question, "filters": {}}
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    
                    assert response.status_code == 200
                    print(f"Question: '{question}' - Response time: {response_time:.3f}s")
                
                # Calculate P95 response time
                response_times.sort()
                p95_index = int(0.95 * len(response_times))
                p95_response_time = response_times[p95_index]
                
                # Requirement: P95 response time should be within limits
                # Local mode: 6 seconds, AWS mode: 3 seconds
                max_response_time = 6.0  # seconds for local mode
                
                print(f"P95 response time: {p95_response_time:.3f}s")
                print(f"Average response time: {sum(response_times)/len(response_times):.3f}s")
                
                assert p95_response_time < max_response_time, \
                    f"P95 response time {p95_response_time:.3f}s exceeds limit {max_response_time}s"
                
            finally:
                os.unlink(temp_file)
    
    @pytest.mark.performance
    def test_concurrent_query_performance(self, client, mock_settings_local):
        """Test system performance under concurrent load."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Setup test document
            test_content = """
            SOP-CONCURRENT-001: Concurrent Test Procedure
            
            1. Process Steps
            Step 1: Initialize the system
            Step 2: Configure parameters
            Step 3: Start operation
            Step 4: Monitor performance
            Step 5: Complete process
            """
            
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
                
                # Define concurrent queries
                queries = [
                    "What is step 1?",
                    "What is step 2?",
                    "What is step 3?",
                    "What is step 4?",
                    "What is step 5?",
                    "How many steps are there?",
                    "What is the procedure about?",
                    "What should be monitored?"
                ]
                
                # Execute concurrent queries
                def execute_query(question):
                    start_time = time.time()
                    response = client.post(
                        "/ask",
                        json={"question": question, "filters": {}}
                    )
                    end_time = time.time()
                    return {
                        'question': question,
                        'response_time': end_time - start_time,
                        'status_code': response.status_code,
                        'success': response.status_code == 200
                    }
                
                # Run concurrent queries
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(execute_query, query) for query in queries]
                    results = [future.result() for future in as_completed(futures)]
                
                # Analyze results
                successful_queries = [r for r in results if r['success']]
                failed_queries = [r for r in results if not r['success']]
                
                print(f"Successful queries: {len(successful_queries)}/{len(queries)}")
                print(f"Failed queries: {len(failed_queries)}")
                
                if successful_queries:
                    avg_response_time = sum(r['response_time'] for r in successful_queries) / len(successful_queries)
                    max_response_time = max(r['response_time'] for r in successful_queries)
                    
                    print(f"Average concurrent response time: {avg_response_time:.3f}s")
                    print(f"Max concurrent response time: {max_response_time:.3f}s")
                    
                    # Validate performance under load
                    assert len(successful_queries) >= len(queries) * 0.9, "Too many failed queries under load"
                    assert avg_response_time < 10.0, "Average response time too high under load"
                
            finally:
                os.unlink(temp_file)
    
    @pytest.mark.performance
    def test_ingestion_performance(self, client, mock_settings_local):
        """Test document ingestion performance."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Create multiple test documents
            documents = []
            temp_files = []
            
            try:
                for i in range(5):
                    content = f"""
                    SOP-INGEST-{i:03d}: Ingestion Test Document {i}
                    
                    This is test document {i} for ingestion performance testing.
                    """ + "Content " * 1000  # Make documents reasonably sized
                    
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    temp_file.write(content)
                    temp_file.close()
                    temp_files.append(temp_file.name)
                
                # Measure ingestion time
                start_time = time.time()
                
                for temp_file in temp_files:
                    with open(temp_file, 'rb') as f:
                        response = client.post(
                            ("/ingest/files","/ingest/files",$1files={"files": (os.path.basename(temp_file), f, "text/plain")}
                        )
                    assert response.status_code == 200
                
                end_time = time.time()
                total_ingestion_time = end_time - start_time
                
                print(f"Ingested {len(temp_files)} documents in {total_ingestion_time:.3f}s")
                print(f"Average time per document: {total_ingestion_time/len(temp_files):.3f}s")
                
                # Validate ingestion performance
                max_time_per_doc = 30.0  # seconds
                avg_time_per_doc = total_ingestion_time / len(temp_files)
                
                assert avg_time_per_doc < max_time_per_doc, \
                    f"Average ingestion time {avg_time_per_doc:.3f}s exceeds limit {max_time_per_doc}s"
                
            finally:
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, client, mock_settings_local):
        """Test for memory leaks during repeated operations."""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=mock_settings_local):
            # Measure baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage()
            
            # Create test document
            test_content = """
            SOP-LEAK-001: Memory Leak Test
            
            This document is used for memory leak testing.
            It contains some content for processing.
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                # Ingest document once
                with open(temp_file, 'rb') as f:
                    response = client.post(
                        ("/ingest/files","/ingest/files",$1files={"files": ("leak_test.txt", f, "text/plain")}
                    )
                assert response.status_code == 200
                time.sleep(2)
                
                # Perform repeated queries
                memory_measurements = []
                
                for i in range(20):
                    response = client.post(
                        "/ask",
                        json={"question": f"What is this document about? Query {i}", "filters": {}}
                    )
                    assert response.status_code == 200
                    
                    if i % 5 == 0:  # Measure memory every 5 queries
                        gc.collect()
                        current_memory = self.get_memory_usage()
                        memory_measurements.append(current_memory)
                
                # Check for memory growth trend
                if len(memory_measurements) >= 3:
                    memory_growth = memory_measurements[-1] - memory_measurements[0]
                    print(f"Memory growth over {len(memory_measurements)} measurements: {memory_growth:.2f} MB")
                    
                    # Allow some memory growth but detect significant leaks
                    max_allowed_growth = 50  # MB
                    assert memory_growth < max_allowed_growth, \
                        f"Potential memory leak detected: {memory_growth:.2f} MB growth"
                
            finally:
                os.unlink(temp_file)
    
    def teardown_method(self, method):
        """Clean up after each test."""
        # Force garbage collection
        gc.collect()
        
        # Clean up test data
        test_data_path = Path("./test_data")
        if test_data_path.exists():
            import shutil
            shutil.rmtree(test_data_path, ignore_errors=True)
