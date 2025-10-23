"""
Pytest configuration and shared fixtures for integration tests.

Provides common test setup, fixtures, and configuration for all integration
and acceptance tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "acceptance: marks tests as acceptance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    # Create test data directory
    test_data_dir = Path("./test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    if test_data_dir.exists():
        import shutil
        shutil.rmtree(test_data_dir, ignore_errors=True)


@pytest.fixture
def temp_file_factory():
    """Factory for creating temporary files in tests."""
    temp_files = []
    
    def create_temp_file(content: str, suffix: str = ".txt", mode: str = "w"):
        temp_file = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False)
        if mode == "w":
            temp_file.write(content)
        else:
            temp_file.write(content.encode() if isinstance(content, str) else content)
        temp_file.close()
        temp_files.append(temp_file.name)
        return temp_file.name
    
    yield create_temp_file
    
    # Cleanup temporary files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except FileNotFoundError:
            pass


@pytest.fixture
def mock_aws_services():
    """Mock AWS services for testing."""
    with patch('boto3.client') as mock_boto_client, \
         patch('opensearchpy.OpenSearch') as mock_opensearch:
        
        # Mock Bedrock
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = {
            'body': MagicMock(read=lambda: b'{"content": [{"text": "Mocked response"}]}')
        }
        
        # Mock S3
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {'ETag': '"mock-etag"'}
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'Mock file content')
        }
        
        # Mock Textract
        mock_textract = MagicMock()
        mock_textract.detect_document_text.return_value = {
            'Blocks': [
                {
                    'BlockType': 'LINE',
                    'Text': 'Mock OCR extracted text'
                }
            ]
        }
        
        # Configure boto3 client mock
        def mock_client_factory(service_name, **kwargs):
            service_mocks = {
                'bedrock-runtime': mock_bedrock,
                's3': mock_s3,
                'textract': mock_textract
            }
            return service_mocks.get(service_name, MagicMock())
        
        mock_boto_client.side_effect = mock_client_factory
        
        # Mock OpenSearch
        mock_os_instance = MagicMock()
        mock_os_instance.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_source': {
                            'chunk_text': 'Mock search result',
                            'doc_id': 'mock_doc_001',
                            'metadata': {'page_no': 1}
                        },
                        '_score': 0.95
                    }
                ]
            }
        }
        mock_opensearch.return_value = mock_os_instance
        
        yield {
            'bedrock': mock_bedrock,
            's3': mock_s3,
            'textract': mock_textract,
            'opensearch': mock_os_instance
        }


@pytest.fixture
def sample_sop_documents():
    """Provide sample SOP documents for testing."""
    return {
        'simple': """
        SOP-SIMPLE-001: Simple Test Procedure
        Revision: 1.0
        
        1. Purpose
        This is a simple test procedure.
        
        2. Steps
        2.1 Step one: Do something
        2.2 Step two: Do something else
        """,
        
        'complex': """
        SOP-COMPLEX-001: Complex Manufacturing Procedure
        Revision: 2.1
        Effective Date: 2024-01-15
        Owner: Manufacturing Manager
        
        1. Purpose and Scope
        This procedure covers complex manufacturing operations.
        
        2. Roles and Responsibilities
        - Operator: Execute manufacturing steps
        - QA Inspector: Perform quality checks
        - Supervisor: Oversee operations
        
        3. Equipment Required
        - Manufacturing Station MS-001
        - Measurement Device MD-100
        - Safety Equipment (PPE)
        
        4. Procedure Steps
        4.1 Pre-operation Setup
        4.1.1 Verify equipment calibration
        4.1.2 Check safety equipment
        4.1.3 Review work orders
        
        4.2 Manufacturing Process
        4.2.1 Load materials into station
        4.2.2 Execute manufacturing sequence
        4.2.3 Monitor process parameters
        4.2.4 Record measurements every 30 minutes
        
        5. Quality Controls
        - Dimensional tolerance: ±0.1mm
        - Surface finish: Ra 1.6 μm max
        - Visual inspection: 100% of parts
        
        6. Risk Assessment
        - Risk R-001: Equipment malfunction
          Control C-001: Regular maintenance schedule
        - Risk R-002: Operator injury
          Control C-002: Mandatory PPE and training
        """,
        
        'minimal': """
        SOP-MINIMAL-001: Minimal Document
        
        This document has minimal structure.
        Step 1: Do something.
        """,
        
        'with_pii': """
        SOP-PII-001: Document with Personal Information
        
        Contact Information:
        - Manager: john.doe@company.com
        - Phone: (555) 123-4567
        - Emergency: 911
        
        Employee ID: EMP-12345
        Social Security: 123-45-6789
        
        This document contains PII for testing redaction.
        """
    }


@pytest.fixture
def performance_test_data():
    """Generate data for performance testing."""
    def generate_large_document(size_kb: int = 100) -> str:
        """Generate a document of approximately the specified size in KB."""
        base_content = """
        SOP-PERF-{doc_id}: Performance Test Document {doc_id}
        Revision: 1.0
        
        1. Purpose
        This document is generated for performance testing.
        
        2. Content Section
        """
        
        # Add content to reach desired size
        content_line = "This is a line of content for performance testing. " * 10
        lines_needed = (size_kb * 1024) // len(content_line)
        
        full_content = base_content + "\n".join([
            f"2.{i} {content_line}" for i in range(1, lines_needed + 1)
        ])
        
        return full_content
    
    return {
        'generate_large_document': generate_large_document,
        'small_doc': generate_large_document(10),   # 10KB
        'medium_doc': generate_large_document(100), # 100KB
        'large_doc': generate_large_document(1000)  # 1MB
    }


@pytest.fixture
def error_test_scenarios():
    """Provide test scenarios for error handling."""
    return {
        'network_errors': [
            'ConnectionError',
            'Timeout',
            'HTTPError',
            'RequestException'
        ],
        'malicious_urls': [
            'file:///etc/passwd',
            'http://localhost:22/',
            'http://127.0.0.1:3306/',
            'ftp://internal.server/',
            'javascript:alert("xss")'
        ],
        'invalid_files': [
            ('malicious.exe', b'MZ\x90\x00', 'application/exe'),
            ('script.js', b'alert("xss")', 'application/javascript'),
            ('binary.bin', b'\x00\x01\x02\x03', 'application/octet-stream')
        ],
        'malformed_content': [
            b'\x00\x01\x02\x03' * 1000,  # Binary garbage
            b'\xff\xfe' + 'Invalid UTF-16'.encode('utf-8'),  # Encoding issues
            b'<html><script>alert("xss")</script></html>'  # Potential XSS
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Clean up any test files or directories
    cleanup_paths = [
        "./test_data",
        "./test_faiss",
        "./test_data_local",
        "./test_data_aws"
    ]
    
    for path in cleanup_paths:
        test_path = Path(path)
        if test_path.exists():
            import shutil
            shutil.rmtree(test_path, ignore_errors=True)


# Custom pytest markers for test categorization
pytestmark = [
    pytest.mark.integration,
]
