"""
Document Ingestion Service

Handles document intake from URLs and file uploads with robust error handling,
security validation, and text extraction capabilities.
"""

import asyncio
import hashlib
import logging
import mimetypes
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse
import uuid

import httpx
import requests
from fastapi import UploadFile
from pydantic import BaseModel, Field

from ..config.settings import get_settings
from .text_extraction import TextExtractor
from .security import SecurityValidator


logger = logging.getLogger(__name__)


class DocumentSource(BaseModel):
    """Document source information"""
    source_type: str = Field(..., description="Type of source: 'url' or 'file'")
    source_value: str = Field(..., description="URL or filename")
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class DocumentText(BaseModel):
    """Extracted document text with metadata"""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Extracted text content")
    metadata: Dict = Field(default_factory=dict, description="Document metadata")
    source: DocumentSource = Field(..., description="Source information")
    extraction_method: str = Field(..., description="Method used for text extraction")
    page_count: Optional[int] = None
    language: Optional[str] = None
    processing_time_seconds: float = Field(..., description="Time taken to process")


class IngestResult(BaseModel):
    """Result of document ingestion operation"""
    success: bool = Field(..., description="Whether ingestion was successful")
    doc_id: Optional[str] = None
    document: Optional[DocumentText] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: float = Field(..., description="Total processing time")


class BatchIngestResult(BaseModel):
    """Result of batch document ingestion"""
    total_documents: int = Field(..., description="Total number of documents processed")
    successful: int = Field(..., description="Number of successfully processed documents")
    failed: int = Field(..., description="Number of failed documents")
    results: List[IngestResult] = Field(..., description="Individual results")
    total_processing_time_seconds: float = Field(..., description="Total batch processing time")


class DocumentIngestionService:
    """Service for ingesting documents from various sources"""
    
    def __init__(self):
        self.settings = get_settings()
        self.security_validator = SecurityValidator()
        self.text_extractor = TextExtractor()
        
        # Configure HTTP client with security settings
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.request_timeout_seconds),
            limits=httpx.Limits(max_connections=self.settings.max_concurrent_requests),
            follow_redirects=True,
            max_redirects=3
        )
        
        # Sync client for fallback
        self.sync_http_client = requests.Session()
        self.sync_http_client.timeout = self.settings.request_timeout_seconds
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    def _generate_doc_id(self, source: str) -> str:
        """Generate unique document ID based on source"""
        # Create deterministic ID based on source
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        # Use more precise timestamp with microseconds
        timestamp = int(time.time() * 1000000)  # microseconds
        return f"doc_{source_hash}_{timestamp}"
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from URL"""
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if filename and filename != '/':
            # Remove extension and clean up
            title = Path(filename).stem
            # Replace underscores and hyphens with spaces
            title = re.sub(r'[-_]+', ' ', title)
            # Capitalize words
            title = ' '.join(word.capitalize() for word in title.split())
            return title
        else:
            # Use domain name
            return parsed.netloc.replace('www.', '').capitalize()
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract title from filename"""
        title = Path(filename).stem
        # Replace underscores and hyphens with spaces
        title = re.sub(r'[-_]+', ' ', title)
        # Capitalize words
        title = ' '.join(word.capitalize() for word in title.split())
        return title
    
    async def _fetch_url_with_retry(self, url: str, max_retries: int = 3) -> Tuple[bytes, Dict]:
        """Fetch URL content with retry logic and exponential backoff"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching URL (attempt {attempt + 1}/{max_retries}): {url}")
                
                response = await self.http_client.get(url)
                response.raise_for_status()
                
                # Extract metadata
                metadata = {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'content_length': len(response.content),
                    'last_modified': response.headers.get('last-modified'),
                    'etag': response.headers.get('etag'),
                    'server': response.headers.get('server'),
                }
                
                logger.info(f"Successfully fetched URL: {url} ({len(response.content)} bytes)")
                return response.content, metadata
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code in [404, 403, 401]:
                    # Don't retry for client errors
                    raise e
                logger.warning(f"HTTP error fetching {url} (attempt {attempt + 1}): {e}")
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(f"Request error fetching {url} (attempt {attempt + 1}): {e}")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_exception or Exception(f"Failed to fetch URL after {max_retries} attempts")
    
    async def ingest_url(self, url: str) -> IngestResult:
        """Ingest a single document from URL"""
        start_time = time.time()
        doc_id = self._generate_doc_id(url)
        
        try:
            logger.info(f"Starting URL ingestion: {url}")
            
            # Security validation
            if not self.security_validator.validate_url(url):
                return IngestResult(
                    success=False,
                    doc_id=doc_id,
                    error_message=f"URL failed security validation: {url}",
                    processing_time_seconds=time.time() - start_time
                )
            
            # Fetch content
            content_bytes, fetch_metadata = await self._fetch_url_with_retry(url)
            
            # Validate file size
            if len(content_bytes) > self.settings.get_max_file_size_bytes():
                return IngestResult(
                    success=False,
                    doc_id=doc_id,
                    error_message=f"File size ({len(content_bytes)} bytes) exceeds limit ({self.settings.get_max_file_size_bytes()} bytes)",
                    processing_time_seconds=time.time() - start_time
                )
            
            # Determine content type
            content_type = fetch_metadata.get('content_type', '')
            if not content_type:
                content_type, _ = mimetypes.guess_type(url)
                content_type = content_type or 'application/octet-stream'
            
            # Create document source
            source = DocumentSource(
                source_type="url",
                source_value=url,
                original_filename=Path(urlparse(url).path).name,
                content_type=content_type,
                size_bytes=len(content_bytes)
            )
            
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(url).suffix) as temp_file:
                temp_file.write(content_bytes)
                temp_path = Path(temp_file.name)
            
            try:
                # Extract text
                extracted_text, extraction_metadata = await self.text_extractor.extract_text(
                    temp_path, content_type
                )
                
                # Create document
                document = DocumentText(
                    doc_id=doc_id,
                    title=self._extract_title_from_url(url),
                    content=extracted_text,
                    metadata={
                        **fetch_metadata,
                        **extraction_metadata,
                        'ingestion_timestamp': time.time(),
                        'ingestion_method': 'url_fetch'
                    },
                    source=source,
                    extraction_method=extraction_metadata.get('extraction_method', 'unknown'),
                    page_count=extraction_metadata.get('page_count'),
                    language=extraction_metadata.get('language'),
                    processing_time_seconds=time.time() - start_time
                )
                
                logger.info(f"Successfully ingested URL: {url} -> {doc_id}")
                return IngestResult(
                    success=True,
                    doc_id=doc_id,
                    document=document,
                    processing_time_seconds=time.time() - start_time
                )
                
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {e}", exc_info=True)
            return IngestResult(
                success=False,
                doc_id=doc_id,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time
            )
    
    async def ingest_file(self, file: UploadFile) -> IngestResult:
        """Ingest a single uploaded file"""
        start_time = time.time()
        doc_id = self._generate_doc_id(file.filename or "unknown")
        
        try:
            logger.info(f"Starting file ingestion: {file.filename}")
            
            # Security validation
            if not self.security_validator.validate_file_upload(file):
                return IngestResult(
                    success=False,
                    doc_id=doc_id,
                    error_message=f"File failed security validation: {file.filename}",
                    processing_time_seconds=time.time() - start_time
                )
            
            # Read file content
            content_bytes = await file.read()
            
            # Validate file size
            if len(content_bytes) > self.settings.get_max_file_size_bytes():
                return IngestResult(
                    success=False,
                    doc_id=doc_id,
                    error_message=f"File size ({len(content_bytes)} bytes) exceeds limit ({self.settings.get_max_file_size_bytes()} bytes)",
                    processing_time_seconds=time.time() - start_time
                )
            
            # Create document source
            source = DocumentSource(
                source_type="file",
                source_value=file.filename or "unknown",
                original_filename=file.filename,
                content_type=file.content_type,
                size_bytes=len(content_bytes)
            )
            
            # Save to temporary file for processing
            suffix = Path(file.filename or "").suffix if file.filename else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content_bytes)
                temp_path = Path(temp_file.name)
            
            try:
                # Extract text
                extracted_text, extraction_metadata = await self.text_extractor.extract_text(
                    temp_path, file.content_type or ""
                )
                
                # Create document
                document = DocumentText(
                    doc_id=doc_id,
                    title=self._extract_title_from_filename(file.filename or "unknown"),
                    content=extracted_text,
                    metadata={
                        **extraction_metadata,
                        'ingestion_timestamp': time.time(),
                        'ingestion_method': 'file_upload',
                        'original_filename': file.filename
                    },
                    source=source,
                    extraction_method=extraction_metadata.get('extraction_method', 'unknown'),
                    page_count=extraction_metadata.get('page_count'),
                    language=extraction_metadata.get('language'),
                    processing_time_seconds=time.time() - start_time
                )
                
                logger.info(f"Successfully ingested file: {file.filename} -> {doc_id}")
                return IngestResult(
                    success=True,
                    doc_id=doc_id,
                    document=document,
                    processing_time_seconds=time.time() - start_time
                )
                
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
            return IngestResult(
                success=False,
                doc_id=doc_id,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time
            )
    
    async def ingest_urls(self, urls: List[str]) -> BatchIngestResult:
        """Ingest multiple documents from URLs"""
        start_time = time.time()
        
        logger.info(f"Starting batch URL ingestion: {len(urls)} URLs")
        
        # Process URLs concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        
        async def process_url(url: str) -> IngestResult:
            async with semaphore:
                return await self.ingest_url(url)
        
        # Execute all ingestions
        results = await asyncio.gather(
            *[process_url(url) for url in urls],
            return_exceptions=True
        )
        
        # Process results
        processed_results = []
        successful = 0
        failed = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions that weren't caught
                processed_results.append(IngestResult(
                    success=False,
                    doc_id=self._generate_doc_id(urls[i]),
                    error_message=str(result),
                    processing_time_seconds=0
                ))
                failed += 1
            else:
                processed_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch URL ingestion complete: {successful} successful, {failed} failed, {total_time:.2f}s total")
        
        return BatchIngestResult(
            total_documents=len(urls),
            successful=successful,
            failed=failed,
            results=processed_results,
            total_processing_time_seconds=total_time
        )
    
    async def ingest_files(self, files: List[UploadFile]) -> BatchIngestResult:
        """Ingest multiple uploaded files"""
        start_time = time.time()
        
        logger.info(f"Starting batch file ingestion: {len(files)} files")
        
        # Process files sequentially to avoid memory issues with large files
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            result = await self.ingest_file(file)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch file ingestion complete: {successful} successful, {failed} failed, {total_time:.2f}s total")
        
        return BatchIngestResult(
            total_documents=len(files),
            successful=successful,
            failed=failed,
            results=results,
            total_processing_time_seconds=total_time
        )


# Convenience functions for direct usage
async def ingest_url(url: str) -> IngestResult:
    """Convenience function to ingest a single URL"""
    async with DocumentIngestionService() as service:
        return await service.ingest_url(url)


async def ingest_urls(urls: List[str]) -> BatchIngestResult:
    """Convenience function to ingest multiple URLs"""
    async with DocumentIngestionService() as service:
        return await service.ingest_urls(urls)


async def ingest_file(file: UploadFile) -> IngestResult:
    """Convenience function to ingest a single file"""
    async with DocumentIngestionService() as service:
        return await service.ingest_file(file)


async def ingest_files(files: List[UploadFile]) -> BatchIngestResult:
    """Convenience function to ingest multiple files"""
    async with DocumentIngestionService() as service:
        return await service.ingest_files(files)