"""
Security Service

Provides input validation, content filtering, and protection against malicious inputs.
Includes SSRF protection, file type validation, and PII redaction capabilities.
"""

import ipaddress
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from urllib.parse import urlparse

from fastapi import UploadFile

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation service for inputs and content"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Blocked URL schemes
        self.blocked_schemes = {'file', 'ftp', 'gopher', 'ldap', 'dict'}
        
        # Private IP ranges (RFC 1918, RFC 4193, etc.)
        self.private_networks = [
            ipaddress.IPv4Network('10.0.0.0/8'),
            ipaddress.IPv4Network('172.16.0.0/12'),
            ipaddress.IPv4Network('192.168.0.0/16'),
            ipaddress.IPv4Network('127.0.0.0/8'),  # Loopback
            ipaddress.IPv4Network('169.254.0.0/16'),  # Link-local
            ipaddress.IPv6Network('::1/128'),  # IPv6 loopback
            ipaddress.IPv6Network('fc00::/7'),  # IPv6 unique local
            ipaddress.IPv6Network('fe80::/10'),  # IPv6 link-local
        ]
        
        # Dangerous file extensions
        self.dangerous_extensions = {
            'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar',
            'app', 'deb', 'pkg', 'rpm', 'dmg', 'iso', 'msi', 'run', 'bin'
        }
        
        # Windows reserved names (case-insensitive)
        self.windows_reserved_names = {
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'com5',
            'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3', 'lpt4',
            'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
        }
        
        # MIME type mappings for allowed file types
        self.allowed_mime_types = {
            'pdf': ['application/pdf'],
            'docx': [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-word.document.macroEnabled.12'
            ],
            'html': ['text/html', 'application/xhtml+xml'],
            'txt': ['text/plain', 'text/csv'],
            'md': ['text/markdown', 'text/x-markdown', 'text/plain']
        }
        
        # PII patterns for redaction
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL for security issues including SSRF protection
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is safe, False otherwise
        """
        try:
            if not url or not url.strip():
                logger.warning("Empty URL provided")
                return False
            
            parsed = urlparse(url.strip())
            
            # Check for basic URL structure
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Malformed URL: {url}")
                return False
            
            # Check scheme
            if parsed.scheme.lower() in self.blocked_schemes:
                logger.warning(f"Blocked URL scheme: {parsed.scheme} in {url}")
                return False
            
            # Only allow HTTP/HTTPS
            if parsed.scheme.lower() not in ['http', 'https']:
                logger.warning(f"Invalid URL scheme: {parsed.scheme} in {url}")
                return False
            
            # Check for localhost/private IPs if blocking is enabled
            if self.settings.block_localhost_urls:
                hostname = parsed.hostname
                if not hostname:
                    logger.warning(f"No hostname in URL: {url}")
                    return False
                
                # Check for localhost variants
                localhost_variants = {'localhost', '127.0.0.1', '::1', '0.0.0.0'}
                if hostname.lower() in localhost_variants:
                    logger.warning(f"Blocked localhost URL: {url}")
                    return False
                
                # Check for private IP addresses
                try:
                    ip = ipaddress.ip_address(hostname)
                    for network in self.private_networks:
                        if ip in network:
                            logger.warning(f"Blocked private IP URL: {url}")
                            return False
                except ValueError:
                    # Not an IP address, continue with domain validation
                    pass
                
                # Check for suspicious domains
                suspicious_patterns = [
                    r'.*\.local$',
                    r'.*\.internal$',
                    r'.*\.corp$',
                    r'.*\.lan$'
                ]
                
                for pattern in suspicious_patterns:
                    if re.match(pattern, hostname, re.IGNORECASE):
                        logger.warning(f"Blocked suspicious domain: {hostname} in {url}")
                        return False
            
            # Check URL length
            if len(url) > 2048:
                logger.warning(f"URL too long: {len(url)} characters")
                return False
            
            logger.debug(f"URL validation passed: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return False
    
    def validate_file_upload(self, file: UploadFile) -> bool:
        """
        Validate uploaded file for security issues
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            True if file is safe, False otherwise
        """
        try:
            filename = file.filename or ""
            content_type = file.content_type or ""
            
            # Check filename
            if not filename:
                logger.warning("File upload has no filename")
                return False
            
            # Check for path traversal
            if '..' in filename or '/' in filename or '\\' in filename:
                logger.warning(f"Potential path traversal in filename: {filename}")
                return False
            
            # Check for null bytes and control characters
            if '\x00' in filename or any(ord(c) < 32 for c in filename):
                logger.warning(f"Null bytes or control characters in filename: {filename}")
                return False
            
            # Check for Windows reserved names
            base_name = Path(filename).stem.lower()
            if base_name in self.windows_reserved_names:
                logger.warning(f"Windows reserved filename: {filename}")
                return False
            
            # Get file extension
            file_ext = Path(filename).suffix.lower().lstrip('.')
            
            # Check if extension is allowed
            if file_ext not in self.settings.allowed_file_types:
                logger.warning(f"File extension not allowed: {file_ext} (filename: {filename})")
                return False
            
            # Check for dangerous extensions
            if file_ext in self.dangerous_extensions:
                logger.warning(f"Dangerous file extension: {file_ext} (filename: {filename})")
                return False
            
            # Validate MIME type if provided
            if content_type and file_ext in self.allowed_mime_types:
                allowed_mimes = self.allowed_mime_types[file_ext]
                # Extract base MIME type (ignore charset, etc.)
                base_mime = content_type.split(';')[0].strip().lower()
                if base_mime not in allowed_mimes:
                    logger.warning(f"MIME type mismatch: {content_type} for extension {file_ext}")
                    # Don't reject, just warn - browsers can be inconsistent with MIME types
            
            # Check filename length
            if len(filename) > 255:
                logger.warning(f"Filename too long: {len(filename)} characters")
                return False
            
            logger.debug(f"File validation passed: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating file upload: {e}")
            return False
    
    def validate_file_type_by_content(self, file_path: Path) -> Optional[str]:
        """
        Validate file type by examining file content (magic bytes)
        
        Args:
            file_path: Path to file to examine
            
        Returns:
            Detected file type or None if not recognized/allowed
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # PDF signature
            if header.startswith(b'%PDF'):
                return 'pdf'
            
            # DOCX signature (ZIP with specific structure)
            if header.startswith(b'PK\x03\x04'):
                # Could be DOCX, need to check further
                try:
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        if '[Content_Types].xml' in zip_file.namelist():
                            return 'docx'
                except:
                    pass
                return None
            
            # HTML signature
            html_patterns = [b'<!DOCTYPE html', b'<html', b'<HTML', b'<!doctype html']
            for pattern in html_patterns:
                if pattern in header.lower():
                    return 'html'
            
            # Text files (UTF-8 or ASCII)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(100)  # Try to read some text
                return 'txt'
            except UnicodeDecodeError:
                pass
            
            logger.warning(f"Unrecognized file type for: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error validating file content: {e}")
            return None
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input text
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        max_length = 10000  # Reasonable limit for queries
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Input text truncated to {max_length} characters")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def redact_pii(self, text: str) -> str:
        """
        Redact personally identifiable information from text
        
        Args:
            text: Text to redact PII from
            
        Returns:
            Text with PII redacted
        """
        if not self.settings.enable_pii_redaction or not text:
            return text
        
        redacted_text = text
        
        # Apply PII patterns
        for pii_type, pattern in self.pii_patterns.items():
            replacement = f"[REDACTED_{pii_type.upper()}]"
            redacted_text = pattern.sub(replacement, redacted_text)
        
        return redacted_text
    
    def validate_query(self, query: str) -> bool:
        """
        Validate user query for security issues
        
        Args:
            query: User query to validate
            
        Returns:
            True if query is safe, False otherwise
        """
        if not query or not query.strip():
            return False
        
        # Check length
        if len(query) > 1000:
            logger.warning(f"Query too long: {len(query)} characters")
            return False
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r'exec\s*\(',
            r'xp_cmdshell',
            r'sp_executesql',
            r';\s*--',
            r"'\s*--",
            r"'\s*or\s*'",
            r"'\s*and\s*'",
            r'1\s*=\s*1',
            r'1\s*or\s*1',
        ]
        
        # Check for XSS patterns
        xss_patterns = [
            r'<script[^>]*>',
            r'</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<img[^>]*onerror',
            r'<svg[^>]*onload',
            r'<body[^>]*onload',
            r'<input[^>]*onfocus',
        ]
        
        query_lower = query.lower()
        
        # Check SQL injection patterns
        for pattern in sql_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Potential SQL injection attempt in query: {pattern}")
                return False
        
        # Check XSS patterns
        for pattern in xss_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Potential XSS attempt in query: {pattern}")
                return False
        
        return True
    
    def get_allowed_file_extensions(self) -> Set[str]:
        """Get set of allowed file extensions"""
        return set(self.settings.allowed_file_types)
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum allowed file size in bytes"""
        return self.settings.get_max_file_size_bytes()
    
    def validate_file_size(self, file: UploadFile) -> bool:
        """
        Validate file size against configured limits
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            True if file size is acceptable, False otherwise
        """
        if not file.size:
            # If size is not available, we'll check during processing
            return True
        
        max_size = self.get_max_file_size_bytes()
        if file.size > max_size:
            logger.warning(f"File {file.filename} size {file.size} exceeds limit {max_size}")
            return False
        
        return True
    
    def validate_batch_upload(self, files: List[UploadFile]) -> Dict[str, List[str]]:
        """
        Validate a batch of file uploads and return validation results
        
        Args:
            files: List of FastAPI UploadFile objects
            
        Returns:
            Dictionary with 'valid' and 'invalid' file lists and error messages
        """
        results = {
            'valid_files': [],
            'invalid_files': [],
            'errors': []
        }
        
        if not files:
            results['errors'].append("No files provided")
            return results
        
        # Check total batch size
        total_size = sum(file.size or 0 for file in files)
        max_batch_size = self.get_max_file_size_bytes() * 5  # Allow 5x single file limit for batch
        
        if total_size > max_batch_size:
            results['errors'].append(f"Total batch size {total_size} exceeds limit {max_batch_size}")
            return results
        
        for file in files:
            filename = file.filename or "unknown"
            
            # Validate individual file
            if not self.validate_file_upload(file):
                results['invalid_files'].append(filename)
                results['errors'].append(f"File validation failed: {filename}")
                continue
            
            if not self.validate_file_size(file):
                results['invalid_files'].append(filename)
                results['errors'].append(f"File size validation failed: {filename}")
                continue
            
            results['valid_files'].append(filename)
        
        return results
    
    def validate_url_batch(self, urls: List[str]) -> Dict[str, List[str]]:
        """
        Validate a batch of URLs and return validation results
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            Dictionary with 'valid' and 'invalid' URL lists and error messages
        """
        results = {
            'valid_urls': [],
            'invalid_urls': [],
            'errors': []
        }
        
        if not urls:
            results['errors'].append("No URLs provided")
            return results
        
        # Check batch size limit
        if len(urls) > 100:  # Reasonable batch limit
            results['errors'].append(f"Too many URLs in batch: {len(urls)} (max: 100)")
            return results
        
        for url in urls:
            if not self.validate_url(url):
                results['invalid_urls'].append(url)
                results['errors'].append(f"URL validation failed: {url}")
            else:
                results['valid_urls'].append(url)
        
        return results
    
    def create_security_headers(self) -> Dict[str, str]:
        """
        Create security headers for HTTP responses
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "WARNING"):
        """
        Log security-related events for monitoring
        
        Args:
            event_type: Type of security event (e.g., "blocked_url", "file_rejected")
            details: Event details dictionary
            severity: Log severity level
        """
        log_entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            "details": details,
            "severity": severity
        }
        
        if severity == "ERROR":
            logger.error(f"Security event: {event_type} - {details}")
        elif severity == "WARNING":
            logger.warning(f"Security event: {event_type} - {details}")
        else:
            logger.info(f"Security event: {event_type} - {details}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current security configuration
        
        Returns:
            Dictionary with security settings summary
        """
        return {
            "allowed_file_types": list(self.get_allowed_file_extensions()),
            "max_file_size_mb": self.settings.max_file_size_mb,
            "pii_redaction_enabled": self.settings.enable_pii_redaction,
            "localhost_blocking_enabled": self.settings.block_localhost_urls,
            "blocked_schemes": list(self.blocked_schemes),
            "dangerous_extensions": list(self.dangerous_extensions)
        }