#!/usr/bin/env python3
"""
Security Service Demonstration

This script demonstrates the security validation features implemented in the SOP Q&A Tool.
It shows URL validation, file validation, input sanitization, and PII redaction capabilities.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock
from fastapi import UploadFile

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sop_qa_tool.services.security import SecurityValidator
from sop_qa_tool.config.settings import get_settings


def demo_url_validation():
    """Demonstrate URL validation with SSRF protection"""
    print("=" * 60)
    print("URL VALIDATION DEMO")
    print("=" * 60)
    
    validator = SecurityValidator()
    
    # Test URLs
    test_urls = [
        # Valid URLs
        ("https://example.com/document.pdf", "Valid HTTPS URL"),
        ("http://docs.company.com/sop.html", "Valid HTTP URL"),
        
        # Invalid schemes
        ("file:///etc/passwd", "Blocked file:// scheme"),
        ("javascript:alert('xss')", "Blocked JavaScript scheme"),
        ("ftp://malicious.com/backdoor", "Blocked FTP scheme"),
        
        # Localhost blocking
        ("http://localhost/internal", "Blocked localhost"),
        ("https://127.0.0.1/metadata", "Blocked loopback IP"),
        ("http://192.168.1.1/config", "Blocked private IP"),
        
        # Malformed URLs
        ("not-a-url", "Malformed URL"),
        ("http://", "Incomplete URL"),
        ("", "Empty URL"),
    ]
    
    for url, description in test_urls:
        result = validator.validate_url(url)
        status = "✅ ALLOWED" if result else "❌ BLOCKED"
        print(f"{status} | {description:<25} | {url}")
    
    print()


def demo_file_validation():
    """Demonstrate file upload validation"""
    print("=" * 60)
    print("FILE VALIDATION DEMO")
    print("=" * 60)
    
    validator = SecurityValidator()
    validator.settings.allowed_file_types = ["pdf", "docx", "html", "txt"]
    
    # Test files
    test_files = [
        # Valid files
        ("document.pdf", "application/pdf", "Valid PDF file"),
        ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Valid DOCX file"),
        ("page.html", "text/html", "Valid HTML file"),
        ("data.txt", "text/plain", "Valid text file"),
        
        # Invalid extensions
        ("malware.exe", "application/octet-stream", "Dangerous executable"),
        ("script.bat", "application/x-msdos-program", "Dangerous batch file"),
        ("virus.js", "application/javascript", "Dangerous JavaScript"),
        
        # Path traversal attempts
        ("../../../etc/passwd", "text/plain", "Path traversal attempt"),
        ("..\\..\\windows\\system32\\config\\sam", "text/plain", "Windows path traversal"),
        
        # Windows reserved names
        ("con.txt", "text/plain", "Windows reserved name"),
        ("aux.pdf", "application/pdf", "Windows reserved name"),
        ("nul.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Windows reserved name"),
        
        # Control characters
        ("file\x00.txt", "text/plain", "Null byte injection"),
        ("file\r\n.txt", "text/plain", "CRLF injection"),
    ]
    
    for filename, content_type, description in test_files:
        # Create mock upload file
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = filename
        upload_file.content_type = content_type
        upload_file.size = 1024
        
        result = validator.validate_file_upload(upload_file)
        status = "✅ ALLOWED" if result else "❌ BLOCKED"
        print(f"{status} | {description:<25} | {filename}")
    
    print()


def demo_query_validation():
    """Demonstrate query validation and sanitization"""
    print("=" * 60)
    print("QUERY VALIDATION DEMO")
    print("=" * 60)
    
    validator = SecurityValidator()
    
    # Test queries
    test_queries = [
        # Valid queries
        ("What are the safety requirements?", "Normal question"),
        ("How do I operate the filling machine?", "Normal operational question"),
        ("Show me the maintenance schedule", "Normal request"),
        
        # SQL injection attempts
        ("'; DROP TABLE documents; --", "SQL injection with DROP"),
        ("What is UNION SELECT password FROM users", "SQL injection with UNION"),
        ("Show me admin'--", "SQL comment injection"),
        ("Requirements' OR '1'='1", "SQL boolean injection"),
        
        # XSS attempts
        ("Requirements <script>alert('xss')</script>", "XSS with script tag"),
        ("What is <img src=x onerror=alert('xss')>", "XSS with img tag"),
        ("Process javascript:alert('xss')", "XSS with JavaScript URL"),
        
        # Oversized input
        ("What are the requirements? " * 100, "Oversized query"),
        
        # Empty/invalid input
        ("", "Empty query"),
        ("   ", "Whitespace only"),
    ]
    
    for query, description in test_queries:
        result = validator.validate_query(query)
        status = "✅ ALLOWED" if result else "❌ BLOCKED"
        
        # Show sanitized version for valid queries
        if result:
            sanitized = validator.sanitize_input(query)
            if sanitized != query:
                print(f"{status} | {description:<25} | Original: {query[:50]}...")
                print(f"     | {'Sanitized:':<25} | {sanitized[:50]}...")
            else:
                print(f"{status} | {description:<25} | {query[:50]}...")
        else:
            print(f"{status} | {description:<25} | {query[:50]}...")
    
    print()


def demo_pii_redaction():
    """Demonstrate PII redaction capabilities"""
    print("=" * 60)
    print("PII REDACTION DEMO")
    print("=" * 60)
    
    validator = SecurityValidator()
    validator.settings.enable_pii_redaction = True
    
    # Test text with PII
    test_texts = [
        "Contact John Doe at john.doe@example.com or call 555-123-4567",
        "Employee SSN: 123-45-6789, Credit Card: 4532 1234 5678 9012",
        "Server IP: 192.168.1.100, Email: admin@company.com",
        "Phone: (555) 987-6543, Alt phone: 555.111.2222",
        "No PII in this text about safety procedures",
    ]
    
    for text in test_texts:
        redacted = validator.redact_pii(text)
        if redacted != text:
            print(f"Original:  {text}")
            print(f"Redacted:  {redacted}")
            print()
        else:
            print(f"No PII:    {text}")
            print()


def demo_batch_validation():
    """Demonstrate batch validation capabilities"""
    print("=" * 60)
    print("BATCH VALIDATION DEMO")
    print("=" * 60)
    
    validator = SecurityValidator()
    
    # URL batch validation
    print("URL Batch Validation:")
    urls = [
        "https://example.com/doc1.pdf",
        "https://example.org/doc2.html",
        "file:///etc/passwd",  # Should be blocked
        "http://localhost/internal",  # Should be blocked
        "https://example.net/doc3.txt",
    ]
    
    url_result = validator.validate_url_batch(urls)
    print(f"Valid URLs: {len(url_result['valid_urls'])}")
    print(f"Invalid URLs: {len(url_result['invalid_urls'])}")
    if url_result['invalid_urls']:
        print(f"Blocked: {', '.join(url_result['invalid_urls'])}")
    print()
    
    # File batch validation
    print("File Batch Validation:")
    files = []
    test_filenames = [
        ("document.pdf", "application/pdf"),
        ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("malware.exe", "application/octet-stream"),  # Should be blocked
        ("script.bat", "application/x-msdos-program"),  # Should be blocked
        ("data.txt", "text/plain"),
    ]
    
    for filename, content_type in test_filenames:
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = filename
        upload_file.content_type = content_type
        upload_file.size = 1024
        files.append(upload_file)
    
    validator.settings.allowed_file_types = ["pdf", "docx", "txt"]
    file_result = validator.validate_batch_upload(files)
    print(f"Valid files: {len(file_result['valid_files'])}")
    print(f"Invalid files: {len(file_result['invalid_files'])}")
    if file_result['invalid_files']:
        print(f"Blocked: {', '.join(file_result['invalid_files'])}")
    print()


def demo_security_configuration():
    """Demonstrate security configuration"""
    print("=" * 60)
    print("SECURITY CONFIGURATION")
    print("=" * 60)
    
    validator = SecurityValidator()
    config = validator.get_security_summary()
    
    print("Current Security Settings:")
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(map(str, value))}")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    print("Security Headers:")
    headers = validator.create_security_headers()
    for header, value in headers.items():
        print(f"  {header}: {value}")
    
    print()


def main():
    """Run all security demonstrations"""
    print("SOP Q&A Tool - Security Features Demonstration")
    print("=" * 60)
    print()
    
    try:
        demo_url_validation()
        demo_file_validation()
        demo_query_validation()
        demo_pii_redaction()
        demo_batch_validation()
        demo_security_configuration()
        
        print("=" * 60)
        print("SECURITY DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("Key Security Features Demonstrated:")
        print("✅ URL validation with SSRF protection")
        print("✅ File upload validation and type checking")
        print("✅ SQL injection and XSS detection")
        print("✅ Input sanitization and size limits")
        print("✅ PII redaction capabilities")
        print("✅ Batch validation for efficiency")
        print("✅ Security headers and configuration")
        print()
        print("All security validations are integrated into the API endpoints")
        print("and provide comprehensive protection against malicious inputs.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())