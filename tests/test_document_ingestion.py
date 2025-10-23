"""
Unit tests for Document Ingestion Service

Tests document ingestion functionality including URL fetching, file uploads,
text extraction, security validation, and error handling.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi import UploadFile
import io

from sop_qa_tool.services.document_ingestion import (
    DocumentIngestionService,
    DocumentSource,
    DocumentText,
    IngestResult,
    BatchIngestResult
)
from sop_qa_tool.services.security import SecurityValidator
from sop_qa_tool.services.text_extraction import TextExtractor, TextExtractionError


class MockUploadFile:
    """Mock UploadFile for testing"""
    def __init__(self, filename: str, file: io.BytesIO, content_type: str = None):
        self.filename = filename
        self.file = file
        self.content_type = content_type
    
    async def read(self) -> bytes:
        return self.file.read()
    
    def seek(self, position: int):
        return self.file.seek(position)


class TestDocumentIngestionService:
    """Test cases for DocumentIngestionService"""
    
    @pytest.fixture
    def service(self):
        """Create DocumentIngestionService instance for testing"""
        return DocumentIngestionService()
    
    @pytest.fixture
    def mock_security_validator(self):
        """Mock security validator"""
        validator = Mock(spec=SecurityValidator)
        validator.validate_url.return_value = True
        validator.validate_file_upload.return_value = True
        return validator
    
    @pytest.fixture
    def mock_text_extractor(self):
        """Mock text extractor"""
        extractor = Mock(spec=TextExtractor)
        extractor.extract_text.return_value = ("Sample extracted text", {
            'extraction_method': 'test',
            'page_count': 1,
            'word_count': 3
        })
        return extractor
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        # Minimal PDF header
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF'
    
    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing"""
        return b'<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Test Document</h1><p>Sample content</p></body></html>'
    
    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing"""
        return b'This is a sample text document with some content for testing.'
    
    def test_generate_doc_id(self, service):
        """Test document ID generation"""
        import time
        doc_id1 = service._generate_doc_id("test_source")
        time.sleep(0.01)  # Small delay to ensure different timestamp
        doc_id2 = service._generate_doc_id("test_source")
        
        # IDs should be different due to timestamp
        assert doc_id1 != doc_id2
        assert doc_id1.startswith("doc_")
        assert doc_id2.startswith("doc_")
    
    def test_extract_title_from_url(self, service):
        """Test title extraction from URLs"""
        # Test with filename
        title1 = service._extract_title_from_url("https://example.com/sop-filling-procedure.pdf")
        assert title1 == "Sop Filling Procedure"
        
        # Test with domain only
        title2 = service._extract_title_from_url("https://www.example.com/")
        assert title2 == "Example.com"
        
        # Test with underscores and hyphens
        title3 = service._extract_title_from_url("https://example.com/test_file-name.docx")
        assert title3 == "Test File Name"
    
    def test_extract_title_from_filename(self, service):
        """Test title extraction from filenames"""
        title1 = service._extract_title_from_filename("sop-filling-procedure.pdf")
        assert title1 == "Sop Filling Procedure"
        
        title2 = service._extract_title_from_filename("test_file_name.docx")
        assert title2 == "Test File Name"
    
    @pytest.mark.asyncio
    async def test_fetch_url_with_retry_success(self, service):
        """Test successful URL fetching"""
        mock_response = Mock()
        mock_response.content = b"test content"
        mock_response.status_code = 200
        mock_response.headers = {
            'content-type': 'text/html',
            'content-length': '12'
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service.http_client, 'get', return_value=mock_response):
            content, metadata = await service._fetch_url_with_retry("https://example.com/test.html")
            
            assert content == b"test content"
            assert metadata['status_code'] == 200
            assert metadata['content_type'] == 'text/html'
    
    @pytest.mark.asyncio
    async def test_fetch_url_with_retry_failure(self, service):
        """Test URL fetching with retries on failure"""
        import httpx
        
        with patch.object(service.http_client, 'get', side_effect=httpx.RequestError("Connection failed")):
            with pytest.raises(httpx.RequestError):
                await service._fetch_url_with_retry("https://example.com/test.html", max_retries=2)
    
    @pytest.mark.asyncio
    async def test_ingest_url_success(self, service, sample_html_content):
        """Test successful URL ingestion"""
        url = "https://example.com/test.html"
        
        # Mock dependencies
        service.security_validator = Mock()
        service.security_validator.validate_url.return_value = True
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(return_value=(
            "Extracted text content",
            {'extraction_method': 'html', 'page_count': 1}
        ))
        
        # Mock HTTP client
        mock_response = Mock()
        mock_response.content = sample_html_content
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service.http_client, 'get', return_value=mock_response):
            result = await service.ingest_url(url)
        
        assert result.success is True
        assert result.document is not None
        assert result.document.title == "Test"  # From test.html filename
        assert result.document.content == "Extracted text content"
        assert result.document.source.source_type == "url"
        assert result.document.source.source_value == url
    
    @pytest.mark.asyncio
    async def test_ingest_url_security_failure(self, service):
        """Test URL ingestion with security validation failure"""
        url = "file:///etc/passwd"
        
        service.security_validator = Mock()
        service.security_validator.validate_url.return_value = False
        
        result = await service.ingest_url(url)
        
        assert result.success is False
        assert "security validation" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_ingest_url_size_limit_exceeded(self, service):
        """Test URL ingestion with file size limit exceeded"""
        url = "https://example.com/large-file.pdf"
        
        service.security_validator = Mock()
        service.security_validator.validate_url.return_value = True
        
        # Mock large content
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        mock_response = Mock()
        mock_response.content = large_content
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service.http_client, 'get', return_value=mock_response):
            result = await service.ingest_url(url)
        
        assert result.success is False
        assert "exceeds limit" in result.error_message
    
    @pytest.mark.asyncio
    async def test_ingest_file_success(self, service, sample_pdf_content):
        """Test successful file ingestion"""
        # Create mock UploadFile
        file_content = io.BytesIO(sample_pdf_content)
        upload_file = MockUploadFile(
            filename="test.pdf",
            file=file_content,
            content_type="application/pdf"
        )
        
        # Mock dependencies
        service.security_validator = Mock()
        service.security_validator.validate_file_upload.return_value = True
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(return_value=(
            "Extracted PDF content",
            {'extraction_method': 'pdf', 'page_count': 1}
        ))
        
        result = await service.ingest_file(upload_file)
        
        assert result.success is True
        assert result.document is not None
        assert result.document.title == "Test"
        assert result.document.content == "Extracted PDF content"
        assert result.document.source.source_type == "file"
        assert result.document.source.original_filename == "test.pdf"
    
    @pytest.mark.asyncio
    async def test_ingest_file_security_failure(self, service):
        """Test file ingestion with security validation failure"""
        file_content = io.BytesIO(b"malicious content")
        upload_file = MockUploadFile(
            filename="malicious.exe",
            file=file_content,
            content_type="application/octet-stream"
        )
        
        service.security_validator = Mock()
        service.security_validator.validate_file_upload.return_value = False
        
        result = await service.ingest_file(upload_file)
        
        assert result.success is False
        assert "security validation" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_ingest_file_extraction_error(self, service, sample_pdf_content):
        """Test file ingestion with text extraction error"""
        file_content = io.BytesIO(sample_pdf_content)
        upload_file = MockUploadFile(
            filename="test.pdf",
            file=file_content,
            content_type="application/pdf"
        )
        
        service.security_validator = Mock()
        service.security_validator.validate_file_upload.return_value = True
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(
            side_effect=TextExtractionError("Extraction failed")
        )
        
        result = await service.ingest_file(upload_file)
        
        assert result.success is False
        assert "Extraction failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_ingest_urls_batch_success(self, service, sample_html_content):
        """Test successful batch URL ingestion"""
        urls = [
            "https://example.com/doc1.html",
            "https://example.com/doc2.html"
        ]
        
        # Mock dependencies
        service.security_validator = Mock()
        service.security_validator.validate_url.return_value = True
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(return_value=(
            "Extracted content",
            {'extraction_method': 'html', 'page_count': 1}
        ))
        
        # Mock HTTP client
        mock_response = Mock()
        mock_response.content = sample_html_content
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service.http_client, 'get', return_value=mock_response):
            result = await service.ingest_urls(urls)
        
        assert result.total_documents == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 2
        assert all(r.success for r in result.results)
    
    @pytest.mark.asyncio
    async def test_ingest_urls_batch_partial_failure(self, service, sample_html_content):
        """Test batch URL ingestion with partial failures"""
        urls = [
            "https://example.com/doc1.html",
            "file:///etc/passwd",  # This should fail security validation
            "https://example.com/doc3.html"
        ]
        
        # Mock dependencies
        service.security_validator = Mock()
        def validate_url_side_effect(url):
            return not url.startswith("file://")
        service.security_validator.validate_url.side_effect = validate_url_side_effect
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(return_value=(
            "Extracted content",
            {'extraction_method': 'html', 'page_count': 1}
        ))
        
        # Mock HTTP client
        mock_response = Mock()
        mock_response.content = sample_html_content
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(service.http_client, 'get', return_value=mock_response):
            result = await service.ingest_urls(urls)
        
        assert result.total_documents == 3
        assert result.successful == 2
        assert result.failed == 1
        assert len(result.results) == 3
        
        # Check that the file:// URL failed
        failed_result = next(r for r in result.results if not r.success)
        assert "security validation" in failed_result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_ingest_files_batch_success(self, service, sample_pdf_content, sample_text_content):
        """Test successful batch file ingestion"""
        files = [
            MockUploadFile(
                filename="test1.pdf",
                file=io.BytesIO(sample_pdf_content),
                content_type="application/pdf"
            ),
            MockUploadFile(
                filename="test2.txt",
                file=io.BytesIO(sample_text_content),
                content_type="text/plain"
            )
        ]
        
        # Mock dependencies
        service.security_validator = Mock()
        service.security_validator.validate_file_upload.return_value = True
        
        service.text_extractor = Mock()
        service.text_extractor.extract_text = AsyncMock(return_value=(
            "Extracted content",
            {'extraction_method': 'test', 'page_count': 1}
        ))
        
        result = await service.ingest_files(files)
        
        assert result.total_documents == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 2
        assert all(r.success for r in result.results)


class TestDocumentSource:
    """Test cases for DocumentSource model"""
    
    def test_document_source_creation(self):
        """Test DocumentSource model creation"""
        source = DocumentSource(
            source_type="url",
            source_value="https://example.com/test.pdf",
            original_filename="test.pdf",
            content_type="application/pdf",
            size_bytes=1024
        )
        
        assert source.source_type == "url"
        assert source.source_value == "https://example.com/test.pdf"
        assert source.original_filename == "test.pdf"
        assert source.content_type == "application/pdf"
        assert source.size_bytes == 1024
    
    def test_document_source_minimal(self):
        """Test DocumentSource with minimal required fields"""
        source = DocumentSource(
            source_type="file",
            source_value="test.txt"
        )
        
        assert source.source_type == "file"
        assert source.source_value == "test.txt"
        assert source.original_filename is None
        assert source.content_type is None
        assert source.size_bytes is None


class TestDocumentText:
    """Test cases for DocumentText model"""
    
    def test_document_text_creation(self):
        """Test DocumentText model creation"""
        source = DocumentSource(
            source_type="url",
            source_value="https://example.com/test.pdf"
        )
        
        doc = DocumentText(
            doc_id="test_doc_123",
            title="Test Document",
            content="This is test content",
            metadata={"test": "value"},
            source=source,
            extraction_method="pdf",
            page_count=5,
            language="en",
            processing_time_seconds=1.5
        )
        
        assert doc.doc_id == "test_doc_123"
        assert doc.title == "Test Document"
        assert doc.content == "This is test content"
        assert doc.metadata == {"test": "value"}
        assert doc.source == source
        assert doc.extraction_method == "pdf"
        assert doc.page_count == 5
        assert doc.language == "en"
        assert doc.processing_time_seconds == 1.5


class TestIngestResult:
    """Test cases for IngestResult model"""
    
    def test_ingest_result_success(self):
        """Test successful IngestResult creation"""
        source = DocumentSource(
            source_type="file",
            source_value="test.txt"
        )
        
        doc = DocumentText(
            doc_id="test_doc",
            title="Test",
            content="Content",
            source=source,
            extraction_method="text",
            processing_time_seconds=1.0
        )
        
        result = IngestResult(
            success=True,
            doc_id="test_doc",
            document=doc,
            processing_time_seconds=1.0
        )
        
        assert result.success is True
        assert result.doc_id == "test_doc"
        assert result.document == doc
        assert result.error_message is None
        assert result.warnings == []
    
    def test_ingest_result_failure(self):
        """Test failed IngestResult creation"""
        result = IngestResult(
            success=False,
            doc_id="failed_doc",
            error_message="Processing failed",
            warnings=["Warning message"],
            processing_time_seconds=0.5
        )
        
        assert result.success is False
        assert result.doc_id == "failed_doc"
        assert result.document is None
        assert result.error_message == "Processing failed"
        assert result.warnings == ["Warning message"]


class TestBatchIngestResult:
    """Test cases for BatchIngestResult model"""
    
    def test_batch_ingest_result_creation(self):
        """Test BatchIngestResult creation"""
        results = [
            IngestResult(success=True, processing_time_seconds=1.0),
            IngestResult(success=False, error_message="Failed", processing_time_seconds=0.5)
        ]
        
        batch_result = BatchIngestResult(
            total_documents=2,
            successful=1,
            failed=1,
            results=results,
            total_processing_time_seconds=1.5
        )
        
        assert batch_result.total_documents == 2
        assert batch_result.successful == 1
        assert batch_result.failed == 1
        assert len(batch_result.results) == 2
        assert batch_result.total_processing_time_seconds == 1.5


# Integration tests
class TestDocumentIngestionIntegration:
    """Integration tests for document ingestion"""
    
    @pytest.mark.asyncio
    async def test_real_text_file_ingestion(self):
        """Test ingestion of a real text file"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
            temp_path = Path(f.name)
        
        try:
            # Create UploadFile from the temporary file
            with open(temp_path, 'rb') as f:
                file_content = io.BytesIO(f.read())
            
            upload_file = MockUploadFile(
                filename="test.txt",
                file=file_content,
                content_type="text/plain"
            )
            
            # Test ingestion
            async with DocumentIngestionService() as service:
                result = await service.ingest_file(upload_file)
            
            assert result.success is True
            assert result.document is not None
            assert "test document" in result.document.content.lower()
            assert result.document.source.source_type == "file"
            
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_real_html_content_ingestion(self):
        """Test ingestion of HTML content"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SOP Test Document</title>
        </head>
        <body>
            <h1>Standard Operating Procedure</h1>
            <h2>Safety Requirements</h2>
            <p>Always wear protective equipment.</p>
            <h2>Procedure Steps</h2>
            <ol>
                <li>Step 1: Preparation</li>
                <li>Step 2: Execution</li>
                <li>Step 3: Cleanup</li>
            </ol>
        </body>
        </html>
        """
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = Path(f.name)
        
        try:
            # Create UploadFile
            with open(temp_path, 'rb') as f:
                file_content = io.BytesIO(f.read())
            
            upload_file = MockUploadFile(
                filename="sop_test.html",
                file=file_content,
                content_type="text/html"
            )
            
            # Test ingestion
            async with DocumentIngestionService() as service:
                result = await service.ingest_file(upload_file)
            
            assert result.success is True
            assert result.document is not None
            assert "standard operating procedure" in result.document.content.lower()
            assert "safety requirements" in result.document.content.lower()
            assert "step 1" in result.document.content.lower()
            
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
