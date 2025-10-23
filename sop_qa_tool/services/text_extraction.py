"""
Text Extraction Service

Handles text extraction from various document formats including PDF, DOCX, HTML,
and provides OCR capabilities using AWS Textract (primary) and local OCR (fallback).
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import mimetypes

# Document processing
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

# OCR libraries
import pytesseract
from PIL import Image

# AWS SDK (optional)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class TextExtractionError(Exception):
    """Custom exception for text extraction errors"""
    pass


class OCRService:
    """OCR service with AWS Textract primary and local OCR fallback"""
    
    def __init__(self):
        self.settings = get_settings()
        self.textract_client = None
        
        # Initialize AWS Textract client if in AWS mode
        if self.settings.is_aws_mode() and AWS_AVAILABLE:
            try:
                self.textract_client = boto3.client(
                    'textract',
                    region_name=self.settings.aws_region
                )
                logger.info("AWS Textract client initialized")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Failed to initialize Textract client: {e}")
                self.textract_client = None
    
    async def extract_text_from_image(self, image_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        start_time = time.time()
        
        try:
            # Try AWS Textract first if available
            if self.textract_client and self.settings.is_aws_mode():
                try:
                    text, metadata = await self._extract_with_textract(image_path)
                    metadata['ocr_method'] = 'aws_textract'
                    metadata['processing_time'] = time.time() - start_time
                    return text, metadata
                except Exception as e:
                    logger.warning(f"Textract failed, falling back to local OCR: {e}")
            
            # Fallback to local OCR
            text, metadata = await self._extract_with_local_ocr(image_path)
            metadata['ocr_method'] = 'local_tesseract'
            metadata['processing_time'] = time.time() - start_time
            return text, metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            raise TextExtractionError(f"OCR failed: {e}")
    
    async def _extract_with_textract(self, image_path: Path) -> Tuple[str, Dict]:
        """Extract text using AWS Textract"""
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
            
            # Call Textract
            response = self.textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            
            # Extract text from response
            text_blocks = []
            confidence_scores = []
            
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    text_blocks.append(block['Text'])
                    confidence_scores.append(block.get('Confidence', 0))
            
            extracted_text = '\n'.join(text_blocks)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            metadata = {
                'textract_confidence': avg_confidence,
                'textract_blocks': len(response.get('Blocks', [])),
                'textract_lines': len(text_blocks)
            }
            
            logger.info(f"Textract extraction successful: {len(extracted_text)} chars, {avg_confidence:.2f}% confidence")
            return extracted_text, metadata
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'InvalidImageFormatException':
                raise TextExtractionError(f"Invalid image format for Textract: {image_path}")
            elif error_code == 'DocumentTooLargeException':
                raise TextExtractionError(f"Image too large for Textract: {image_path}")
            else:
                raise TextExtractionError(f"Textract error: {error_code}")
    
    async def _extract_with_local_ocr(self, image_path: Path) -> Tuple[str, Dict]:
        """Extract text using local Tesseract OCR"""
        try:
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def run_ocr():
                # Open and process image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Extract text with confidence data
                    ocr_data = pytesseract.image_to_data(
                        img, 
                        output_type=pytesseract.Output.DICT,
                        config='--psm 6'  # Uniform block of text
                    )
                    
                    # Filter out low-confidence text
                    min_confidence = 30
                    text_parts = []
                    confidences = []
                    
                    for i, conf in enumerate(ocr_data['conf']):
                        if int(conf) > min_confidence:
                            text = ocr_data['text'][i].strip()
                            if text:
                                text_parts.append(text)
                                confidences.append(int(conf))
                    
                    extracted_text = ' '.join(text_parts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    return extracted_text, {
                        'tesseract_confidence': avg_confidence,
                        'tesseract_words': len(text_parts),
                        'min_confidence_threshold': min_confidence
                    }
            
            extracted_text, metadata = await loop.run_in_executor(None, run_ocr)
            
            logger.info(f"Local OCR extraction successful: {len(extracted_text)} chars, {metadata['tesseract_confidence']:.2f}% confidence")
            return extracted_text, metadata
            
        except Exception as e:
            raise TextExtractionError(f"Local OCR failed: {e}")


class TextExtractor:
    """Main text extraction service supporting multiple document formats"""
    
    def __init__(self):
        self.settings = get_settings()
        self.ocr_service = OCRService()
    
    async def extract_text(self, file_path: Path, content_type: str = "") -> Tuple[str, Dict]:
        """
        Extract text from document file
        
        Args:
            file_path: Path to document file
            content_type: MIME type of the document
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        start_time = time.time()
        
        try:
            # Determine file type
            file_ext = file_path.suffix.lower().lstrip('.')
            if not content_type:
                content_type, _ = mimetypes.guess_type(str(file_path))
                content_type = content_type or ""
            
            logger.info(f"Extracting text from {file_path} (type: {file_ext}, mime: {content_type})")
            
            # Route to appropriate extraction method
            if file_ext == 'pdf' or 'pdf' in content_type.lower():
                text, metadata = await self._extract_from_pdf(file_path)
            elif file_ext == 'docx' or 'wordprocessingml' in content_type.lower():
                text, metadata = await self._extract_from_docx(file_path)
            elif file_ext in ['html', 'htm'] or 'html' in content_type.lower():
                text, metadata = await self._extract_from_html(file_path)
            elif file_ext in ['txt', 'md'] or 'text/plain' in content_type.lower() or 'markdown' in content_type.lower():
                text, metadata = await self._extract_from_text(file_path)
            else:
                # Try auto-detection
                text, metadata = await self._extract_auto(file_path)
            
            # Post-process text
            text = self._clean_extracted_text(text)
            
            # Add common metadata
            metadata.update({
                'file_path': str(file_path),
                'file_size_bytes': file_path.stat().st_size,
                'extraction_time': time.time() - start_time,
                'content_type': content_type,
                'file_extension': file_ext,
                'character_count': len(text),
                'word_count': len(text.split()) if text else 0
            })
            
            logger.info(f"Text extraction successful: {len(text)} chars, {metadata.get('word_count', 0)} words")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise TextExtractionError(f"Failed to extract text from {file_path}: {e}")
    
    async def _extract_from_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from PDF file"""
        try:
            # Use unstructured to partition PDF
            elements = partition_pdf(
                filename=str(file_path),
                strategy="auto",  # Will use OCR if needed
                infer_table_structure=True,
                extract_images_in_pdf=True,
                include_page_breaks=True
            )
            
            # Extract text and metadata
            text_parts = []
            page_count = 0
            table_count = 0
            image_count = 0
            
            for element in elements:
                text_parts.append(str(element))
                
                # Count different element types
                if hasattr(element, 'metadata') and element.metadata:
                    if element.metadata.get('page_number'):
                        page_count = max(page_count, element.metadata.get('page_number', 0))
                    if 'table' in str(type(element)).lower():
                        table_count += 1
                    if 'image' in str(type(element)).lower():
                        image_count += 1
            
            extracted_text = '\n'.join(text_parts)
            
            metadata = {
                'extraction_method': 'unstructured_pdf',
                'page_count': page_count,
                'table_count': table_count,
                'image_count': image_count,
                'element_count': len(elements)
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.warning(f"PDF extraction with unstructured failed: {e}")
            # Fallback to OCR if PDF extraction fails
            return await self._extract_with_ocr_fallback(file_path)
    
    async def _extract_from_docx(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from DOCX file"""
        try:
            elements = partition_docx(
                filename=str(file_path),
                infer_table_structure=True
            )
            
            text_parts = []
            table_count = 0
            
            for element in elements:
                text_parts.append(str(element))
                if 'table' in str(type(element)).lower():
                    table_count += 1
            
            extracted_text = '\n'.join(text_parts)
            
            metadata = {
                'extraction_method': 'unstructured_docx',
                'table_count': table_count,
                'element_count': len(elements)
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            raise TextExtractionError(f"DOCX extraction failed: {e}")
    
    async def _extract_from_html(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from HTML file"""
        try:
            elements = partition_html(
                filename=str(file_path),
                include_page_breaks=False
            )
            
            text_parts = []
            link_count = 0
            
            for element in elements:
                text_parts.append(str(element))
                if hasattr(element, 'metadata') and element.metadata:
                    # Check if metadata has link_urls attribute
                    if hasattr(element.metadata, 'link_urls') and element.metadata.link_urls:
                        link_count += len(element.metadata.link_urls)
            
            extracted_text = '\n'.join(text_parts)
            
            metadata = {
                'extraction_method': 'unstructured_html',
                'link_count': link_count,
                'element_count': len(elements)
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            raise TextExtractionError(f"HTML extraction failed: {e}")
    
    async def _extract_from_text(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from plain text file"""
        try:
            elements = partition_text(filename=str(file_path))
            
            text_parts = []
            for element in elements:
                text_parts.append(str(element))
            
            extracted_text = '\n'.join(text_parts)
            
            metadata = {
                'extraction_method': 'unstructured_text',
                'element_count': len(elements)
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            # Fallback to simple file reading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                
                metadata = {
                    'extraction_method': 'simple_text_read',
                    'encoding': 'utf-8'
                }
                
                return extracted_text, metadata
                
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    extracted_text = f.read()
                
                metadata = {
                    'extraction_method': 'simple_text_read',
                    'encoding': 'latin-1'
                }
                
                return extracted_text, metadata
    
    async def _extract_auto(self, file_path: Path) -> Tuple[str, Dict]:
        """Auto-detect and extract text from file"""
        try:
            elements = partition(filename=str(file_path))
            
            text_parts = []
            for element in elements:
                text_parts.append(str(element))
            
            extracted_text = '\n'.join(text_parts)
            
            metadata = {
                'extraction_method': 'unstructured_auto',
                'element_count': len(elements)
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            raise TextExtractionError(f"Auto extraction failed: {e}")
    
    async def _extract_with_ocr_fallback(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text using OCR as fallback method"""
        try:
            # Convert PDF to images and OCR each page
            # This is a simplified approach - in production, you might want to use pdf2image
            logger.info(f"Attempting OCR fallback for {file_path}")
            
            # For now, try to treat as image directly
            text, metadata = await self.ocr_service.extract_text_from_image(file_path)
            metadata['extraction_method'] = 'ocr_fallback'
            
            return text, metadata
            
        except Exception as e:
            raise TextExtractionError(f"OCR fallback failed: {e}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = text.strip()
        
        return text
    
    def supports_file_type(self, file_extension: str) -> bool:
        """Check if file type is supported for text extraction"""
        supported_types = {'pdf', 'docx', 'html', 'htm', 'txt', 'md'}
        return file_extension.lower().lstrip('.') in supported_types