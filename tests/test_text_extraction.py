"""
Unit tests for Text Extraction Service

Tests text extraction functionality including PDF, DOCX, HTML processing,
and OCR capabilities with AWS Textract and local OCR fallback.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import io

from sop_qa_tool.services.text_extraction import (
    TextExtractor,
    OCRService,
    TextExtractionError
)


class TestOCRService:
    """Test cases for OCRService"""
    
    @pytest.fixture
    def ocr_service(self):
        """Create OCRService instance for testing"""
        return OCRService()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple white image with black text
        img = Image.new('RGB', (200, 100), color='white')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def test_ocr_service_initialization_aws_mode(self, ocr_service):
        """Test OCR service initialization in AWS mode"""
        with patch('sop_qa_tool.services.text_extraction.boto3') as mock_boto3:
            ocr_service.settings.mode = "aws"
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            
            # Reinitialize
            ocr_service.__init__()
            
            assert ocr_service.textract_client == mock_client
    
    def test_ocr_service_initialization_local_mode(self, ocr_service):
        """Test OCR service initialization in local mode"""
        ocr_service.settings.mode = "local"
        ocr_service.__init__()
        
        assert ocr_service.textract_client is None
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image_textract_success(self, ocr_service, sample_image):
        """Test successful text extraction using AWS Textract"""
        # Mock Textract client
        mock_client = Mock()
        mock_response = {
            'Blocks': [
                {
                    'BlockType': 'LINE',
                    'Text': 'Sample text line 1',
                    'Confidence': 95.5
                },
                {
                    'BlockType': 'LINE',
                    'Text': 'Sample text line 2',
                    'Confidence': 92.3
                },
                {
                    'BlockType': 'WORD',
                    'Text': 'Sample',
                    'Confidence': 98.1
                }
            ]
        }
        mock_client.detect_document_text.return_value = mock_response
        
        ocr_service.textract_client = mock_client
        ocr_service.settings.mode = "aws"
        
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(sample_image)
            temp_path = Path(f.name)
        
        try:
            text, metadata = await ocr_service.extract_text_from_image(temp_path)
            
            assert text == "Sample text line 1\nSample text line 2"
            assert metadata['ocr_method'] == 'aws_textract'
            assert metadata['textract_confidence'] == (95.5 + 92.3) / 2
            assert metadata['textract_lines'] == 2
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image_textract_fallback(self, ocr_service, sample_image):
        """Test fallback to local OCR when Textract fails"""
        # Mock Textract failure
        mock_client = Mock()
        mock_client.detect_document_text.side_effect = Exception("Textract failed")
        
        ocr_service.textract_client = mock_client
        ocr_service.settings.mode = "aws"
        
        # Mock local OCR success
        with patch('sop_qa_tool.services.text_extraction.pytesseract') as mock_tesseract:
            mock_tesseract.image_to_data.return_value = {
                'text': ['Sample', 'text', 'from', 'local', 'OCR'],
                'conf': [85, 90, 88, 92, 87]
            }
            mock_tesseract.Output.DICT = 'dict'
            
            # Create temporary image file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(sample_image)
                temp_path = Path(f.name)
            
            try:
                text, metadata = await ocr_service.extract_text_from_image(temp_path)
                
                assert "Sample text from local OCR" in text
                assert metadata['ocr_method'] == 'local_tesseract'
                assert 'tesseract_confidence' in metadata
                
            finally:
                temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image_local_ocr_only(self, ocr_service, sample_image):
        """Test local OCR when Textract is not available"""
        ocr_service.textract_client = None
        ocr_service.settings.mode = "local"
        
        with patch('sop_qa_tool.services.text_extraction.pytesseract') as mock_tesseract:
            mock_tesseract.image_to_data.return_value = {
                'text': ['Local', 'OCR', 'text', 'extraction'],
                'conf': [75, 80, 85, 90]
            }
            mock_tesseract.Output.DICT = 'dict'
            
            # Create temporary image file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(sample_image)
                temp_path = Path(f.name)
            
            try:
                text, metadata = await ocr_service.extract_text_from_image(temp_path)
                
                assert "Local OCR text extraction" in text
                assert metadata['ocr_method'] == 'local_tesseract'
                assert metadata['tesseract_confidence'] == (75 + 80 + 85 + 90) / 4
                
            finally:
                temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image_ocr_failure(self, ocr_service, sample_image):
        """Test handling of OCR failures"""
        ocr_service.textract_client = None
        
        with patch('sop_qa_tool.services.text_extraction.pytesseract') as mock_tesseract:
            mock_tesseract.image_to_data.side_effect = Exception("OCR failed")
            
            # Create temporary image file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(sample_image)
                temp_path = Path(f.name)
            
            try:
                with pytest.raises(TextExtractionError):
                    await ocr_service.extract_text_from_image(temp_path)
                    
            finally:
                temp_path.unlink(missing_ok=True)


class TestTextExtractor:
    """Test cases for TextExtractor"""
    
    @pytest.fixture
    def text_extractor(self):
        """Create TextExtractor instance for testing"""
        return TextExtractor()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF'
    
    @pytest.fixture
    def sample_html_content(self):
        """Sample HTML content for testing"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>SOP Document</title>
</head>
<body>
    <h1>Standard Operating Procedure</h1>
    <h2>Safety Requirements</h2>
    <p>Always wear protective equipment when operating machinery.</p>
    <h2>Procedure Steps</h2>
    <ol>
        <li>Preparation phase</li>
        <li>Execution phase</li>
        <li>Cleanup phase</li>
    </ol>
</body>
</html>"""
    
    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing"""
        return """Standard Operating Procedure - Filling Process

1. Safety Requirements
   - Wear safety goggles
   - Use protective gloves
   - Ensure proper ventilation

2. Equipment Setup
   - Check filler temperature
   - Verify pressure settings
   - Test emergency stops

3. Operating Procedure
   - Start filling sequence
   - Monitor fill levels
   - Complete quality checks"""
    
    def test_text_extractor_initialization(self, text_extractor):
        """Test TextExtractor initialization"""
        assert text_extractor.settings is not None
        assert text_extractor.ocr_service is not None
    
    @pytest.mark.asyncio
    async def test_extract_text_from_html(self, text_extractor, sample_html_content):
        """Test text extraction from HTML file"""
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(sample_html_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_html') as mock_partition:
                # Mock unstructured elements
                mock_elements = [
                    Mock(__str__=lambda: "Standard Operating Procedure"),
                    Mock(__str__=lambda: "Safety Requirements"),
                    Mock(__str__=lambda: "Always wear protective equipment when operating machinery."),
                    Mock(__str__=lambda: "Procedure Steps"),
                    Mock(__str__=lambda: "1. Preparation phase"),
                    Mock(__str__=lambda: "2. Execution phase"),
                    Mock(__str__=lambda: "3. Cleanup phase")
                ]
                
                # Add metadata to some elements
                for i, element in enumerate(mock_elements):
                    element.metadata = {'link_urls': [] if i % 2 == 0 else ['http://example.com']}
                
                mock_partition.return_value = mock_elements
                
                text, metadata = await text_extractor.extract_text(temp_path, "text/html")
                
                assert "Standard Operating Procedure" in text
                assert "Safety Requirements" in text
                assert "protective equipment" in text
                assert metadata['extraction_method'] == 'unstructured_html'
                assert metadata['element_count'] == len(mock_elements)
                assert metadata['file_extension'] == 'html'
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_text_file(self, text_extractor, sample_text_content):
        """Test text extraction from plain text file"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_text_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_text') as mock_partition:
                mock_elements = [
                    Mock(__str__=lambda: "Standard Operating Procedure - Filling Process"),
                    Mock(__str__=lambda: "1. Safety Requirements"),
                    Mock(__str__=lambda: "- Wear safety goggles"),
                    Mock(__str__=lambda: "2. Equipment Setup"),
                    Mock(__str__=lambda: "- Check filler temperature")
                ]
                mock_partition.return_value = mock_elements
                
                text, metadata = await text_extractor.extract_text(temp_path, "text/plain")
                
                assert "Standard Operating Procedure" in text
                assert "Safety Requirements" in text
                assert metadata['extraction_method'] == 'unstructured_text'
                assert metadata['element_count'] == len(mock_elements)
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_success(self, text_extractor, sample_pdf_content):
        """Test successful PDF text extraction"""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(sample_pdf_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_pdf') as mock_partition:
                mock_elements = [
                    Mock(__str__=lambda: "PDF Document Title"),
                    Mock(__str__=lambda: "This is content from page 1"),
                    Mock(__str__=lambda: "This is content from page 2")
                ]
                
                # Add metadata
                mock_elements[0].metadata = {'page_number': 1}
                mock_elements[1].metadata = {'page_number': 1}
                mock_elements[2].metadata = {'page_number': 2}
                
                mock_partition.return_value = mock_elements
                
                text, metadata = await text_extractor.extract_text(temp_path, "application/pdf")
                
                assert "PDF Document Title" in text
                assert "content from page 1" in text
                assert "content from page 2" in text
                assert metadata['extraction_method'] == 'unstructured_pdf'
                assert metadata['page_count'] == 2
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_ocr_fallback(self, text_extractor, sample_pdf_content):
        """Test PDF extraction with OCR fallback"""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(sample_pdf_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_pdf') as mock_partition:
                mock_partition.side_effect = Exception("PDF parsing failed")
                
                # Mock OCR service
                text_extractor.ocr_service = Mock()
                text_extractor.ocr_service.extract_text_from_image = AsyncMock(return_value=(
                    "OCR extracted text from PDF",
                    {'ocr_method': 'local_tesseract', 'tesseract_confidence': 85.0}
                ))
                
                text, metadata = await text_extractor.extract_text(temp_path, "application/pdf")
                
                assert text == "OCR extracted text from PDF"
                assert metadata['extraction_method'] == 'ocr_fallback'
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_docx(self, text_extractor):
        """Test DOCX text extraction"""
        # Create a minimal DOCX-like file (just for testing structure)
        docx_content = b'PK\x03\x04' + b'fake docx content'
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            f.write(docx_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_docx') as mock_partition:
                mock_elements = [
                    Mock(__str__=lambda: "DOCX Document Title"),
                    Mock(__str__=lambda: "Paragraph 1 content"),
                    Mock(__str__=lambda: "Table content")
                ]
                mock_partition.return_value = mock_elements
                
                text, metadata = await text_extractor.extract_text(temp_path, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                
                assert "DOCX Document Title" in text
                assert "Paragraph 1 content" in text
                assert metadata['extraction_method'] == 'unstructured_docx'
                assert metadata['element_count'] == len(mock_elements)
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_auto_detection(self, text_extractor):
        """Test auto-detection of file type"""
        # Create a file with unknown extension
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b'Some content for auto-detection')
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition') as mock_partition:
                mock_elements = [Mock(__str__=lambda: "Auto-detected content")]
                mock_partition.return_value = mock_elements
                
                text, metadata = await text_extractor.extract_text(temp_path, "")
                
                assert text == "Auto-detected content"
                assert metadata['extraction_method'] == 'unstructured_auto'
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_failure(self, text_extractor):
        """Test handling of text extraction failures"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'Test content')
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_text') as mock_partition:
                mock_partition.side_effect = Exception("Extraction failed")
                
                with pytest.raises(TextExtractionError):
                    await text_extractor.extract_text(temp_path, "text/plain")
                    
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_text_encoding_fallback(self, text_extractor):
        """Test text extraction with encoding fallback"""
        # Create file with non-UTF-8 content
        latin1_content = "Café résumé naïve".encode('latin-1')
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(latin1_content)
            temp_path = Path(f.name)
        
        try:
            with patch('sop_qa_tool.services.text_extraction.partition_text') as mock_partition:
                mock_partition.side_effect = Exception("Unstructured failed")
                
                text, metadata = await text_extractor.extract_text(temp_path, "text/plain")
                
                assert "Café résumé naïve" in text
                assert metadata['extraction_method'] == 'simple_text_read'
                assert metadata['encoding'] in ['utf-8', 'latin-1']
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_clean_extracted_text(self, text_extractor):
        """Test text cleaning functionality"""
        messy_text = """
        
        This   is    messy     text
        
        
        With   excessive    whitespace
        
        
        And multiple   paragraph   breaks
        
        """
        
        cleaned = text_extractor._clean_extracted_text(messy_text)
        
        # Should normalize whitespace
        assert "This is messy text" in cleaned
        assert "With excessive whitespace" in cleaned
        assert "And multiple paragraph breaks" in cleaned
        
        # Should not have excessive newlines
        assert "\n\n\n" not in cleaned
        assert "   " not in cleaned
    
    def test_clean_extracted_text_empty(self, text_extractor):
        """Test cleaning of empty text"""
        assert text_extractor._clean_extracted_text("") == ""
        assert text_extractor._clean_extracted_text(None) == ""
        assert text_extractor._clean_extracted_text("   ") == ""
    
    def test_supports_file_type(self, text_extractor):
        """Test file type support checking"""
        supported_types = ['pdf', 'docx', 'html', 'htm', 'txt']
        unsupported_types = ['jpg', 'png', 'mp4', 'exe', 'zip']
        
        for file_type in supported_types:
            assert text_extractor.supports_file_type(file_type) is True
            assert text_extractor.supports_file_type(f".{file_type}") is True
            assert text_extractor.supports_file_type(file_type.upper()) is True
        
        for file_type in unsupported_types:
            assert text_extractor.supports_file_type(file_type) is False


class TestTextExtractionIntegration:
    """Integration tests for text extraction"""
    
    @pytest.mark.asyncio
    async def test_real_text_file_extraction(self):
        """Test extraction from a real text file"""
        content = """Standard Operating Procedure
Safety First Manufacturing

1. Personal Protective Equipment
   - Safety glasses required at all times
   - Steel-toed boots in production areas
   - Hard hats in designated zones

2. Emergency Procedures
   - Know location of emergency exits
   - Fire extinguisher locations marked in red
   - Emergency contact: 911

3. Quality Control
   - Inspect all materials before use
   - Document any defects immediately
   - Report to supervisor within 1 hour"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            extractor = TextExtractor()
            text, metadata = await extractor.extract_text(temp_path, "text/plain")
            
            assert "Standard Operating Procedure" in text
            assert "Personal Protective Equipment" in text
            assert "Emergency Procedures" in text
            assert "Quality Control" in text
            assert metadata['character_count'] > 0
            assert metadata['word_count'] > 0
            assert metadata['file_extension'] == 'txt'
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_real_html_file_extraction(self):
        """Test extraction from a real HTML file"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Manufacturing Safety Protocol</title>
</head>
<body>
    <header>
        <h1>Manufacturing Safety Protocol</h1>
        <p>Revision 2.1 - Effective Date: 2024-01-15</p>
    </header>
    
    <main>
        <section id="ppe">
            <h2>Personal Protective Equipment</h2>
            <ul>
                <li>Safety goggles must be worn in all production areas</li>
                <li>Cut-resistant gloves required for handling materials</li>
                <li>Non-slip footwear mandatory on factory floor</li>
            </ul>
        </section>
        
        <section id="procedures">
            <h2>Standard Procedures</h2>
            <ol>
                <li>Pre-shift safety inspection</li>
                <li>Equipment calibration check</li>
                <li>Material quality verification</li>
                <li>Production line startup sequence</li>
            </ol>
        </section>
        
        <section id="emergency">
            <h2>Emergency Response</h2>
            <p>In case of emergency, immediately:</p>
            <ol>
                <li>Stop all machinery using emergency stops</li>
                <li>Evacuate personnel to designated safe areas</li>
                <li>Contact emergency services: <strong>911</strong></li>
                <li>Notify plant manager: <strong>ext. 2500</strong></li>
            </ol>
        </section>
    </main>
    
    <footer>
        <p>Document ID: SOP-SAFETY-001</p>
        <p>Next Review Date: 2024-07-15</p>
    </footer>
</body>
</html>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = Path(f.name)
        
        try:
            extractor = TextExtractor()
            text, metadata = await extractor.extract_text(temp_path, "text/html")
            
            # Check that key content is extracted
            assert "Manufacturing Safety Protocol" in text
            assert "Personal Protective Equipment" in text
            assert "Safety goggles must be worn" in text
            assert "Emergency Response" in text
            assert "SOP-SAFETY-001" in text
            
            # Check metadata
            assert metadata['file_extension'] == 'html'
            assert metadata['character_count'] > 0
            assert metadata['word_count'] > 50  # Should have many words
            
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
