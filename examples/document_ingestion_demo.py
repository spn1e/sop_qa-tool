"""
Document Ingestion Service Demo

Demonstrates the document ingestion functionality including:
- Text file processing
- HTML file processing  
- Security validation
- Text extraction
"""

import asyncio
import tempfile
from pathlib import Path
import io
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sop_qa_tool.services.document_ingestion import DocumentIngestionService


class MockUploadFile:
    """Mock UploadFile for demo"""
    def __init__(self, filename: str, file: io.BytesIO, content_type: str = None):
        self.filename = filename
        self.file = file
        self.content_type = content_type
    
    async def read(self) -> bytes:
        return self.file.read()


async def demo_text_file_ingestion():
    """Demo text file ingestion"""
    print("=== Text File Ingestion Demo ===")
    
    # Create sample SOP content
    sop_content = """Standard Operating Procedure - Equipment Maintenance

1. Safety Requirements
   - Wear safety goggles at all times
   - Use lockout/tagout procedures
   - Ensure proper ventilation

2. Daily Inspection Checklist
   - Check fluid levels
   - Inspect belts and hoses
   - Test emergency stops
   - Record readings in logbook

3. Weekly Maintenance
   - Lubricate moving parts
   - Clean filters
   - Check calibration
   - Update maintenance records

4. Emergency Procedures
   - Immediate shutdown: Press red emergency button
   - Contact supervisor: ext. 2500
   - Evacuate area if necessary
   - Call emergency services: 911

Document ID: SOP-MAINT-001
Revision: 2.1
Effective Date: 2024-01-15
Next Review: 2024-07-15"""

    # Create mock upload file
    file_content = io.BytesIO(sop_content.encode('utf-8'))
    upload_file = MockUploadFile(
        filename="sop_maintenance.txt",
        file=file_content,
        content_type="text/plain"
    )
    
    # Process the file
    async with DocumentIngestionService() as service:
        result = await service.ingest_file(upload_file)
    
    if result.success:
        print(f"✅ Successfully ingested: {result.document.title}")
        print(f"📄 Document ID: {result.doc_id}")
        print(f"📊 Content length: {len(result.document.content)} characters")
        print(f"⏱️  Processing time: {result.processing_time_seconds:.2f} seconds")
        print(f"🔧 Extraction method: {result.document.extraction_method}")
        
        # Show first 200 characters of extracted content
        print(f"\n📝 Content preview:")
        print(result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content)
        
        # Show metadata
        print(f"\n📋 Metadata:")
        for key, value in result.document.metadata.items():
            if key not in ['ingestion_timestamp']:  # Skip timestamp for cleaner output
                print(f"   {key}: {value}")
    else:
        print(f"❌ Ingestion failed: {result.error_message}")


async def demo_html_file_ingestion():
    """Demo HTML file ingestion"""
    print("\n=== HTML File Ingestion Demo ===")
    
    # Create sample HTML SOP content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chemical Handling Safety Protocol</title>
</head>
<body>
    <header>
        <h1>Chemical Handling Safety Protocol</h1>
        <p><strong>Document ID:</strong> SOP-CHEM-003</p>
        <p><strong>Revision:</strong> 1.4 | <strong>Effective Date:</strong> 2024-02-01</p>
    </header>
    
    <main>
        <section id="ppe">
            <h2>Personal Protective Equipment</h2>
            <ul>
                <li>Chemical-resistant gloves (nitrile or neoprene)</li>
                <li>Safety goggles with side shields</li>
                <li>Lab coat or chemical-resistant apron</li>
                <li>Closed-toe shoes with chemical-resistant soles</li>
            </ul>
        </section>
        
        <section id="handling">
            <h2>Handling Procedures</h2>
            <ol>
                <li><strong>Pre-handling inspection:</strong> Check container integrity</li>
                <li><strong>Work area preparation:</strong> Ensure adequate ventilation</li>
                <li><strong>Transfer procedures:</strong> Use appropriate funnels and containers</li>
                <li><strong>Spill prevention:</strong> Work over spill trays</li>
            </ol>
        </section>
        
        <section id="emergency">
            <h2>Emergency Response</h2>
            <div class="warning">
                <h3>In case of chemical spill:</h3>
                <ol>
                    <li>Evacuate immediate area</li>
                    <li>Alert others in the vicinity</li>
                    <li>Contact emergency response team: <strong>ext. 3333</strong></li>
                    <li>Refer to chemical-specific SDS for cleanup procedures</li>
                </ol>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Next Review Date: 2024-08-01</p>
        <p>Approved by: Safety Manager</p>
    </footer>
</body>
</html>"""

    # Create mock upload file
    file_content = io.BytesIO(html_content.encode('utf-8'))
    upload_file = MockUploadFile(
        filename="chemical_safety.html",
        file=file_content,
        content_type="text/html"
    )
    
    # Process the file
    async with DocumentIngestionService() as service:
        result = await service.ingest_file(upload_file)
    
    if result.success:
        print(f"✅ Successfully ingested: {result.document.title}")
        print(f"📄 Document ID: {result.doc_id}")
        print(f"📊 Content length: {len(result.document.content)} characters")
        print(f"⏱️  Processing time: {result.processing_time_seconds:.2f} seconds")
        print(f"🔧 Extraction method: {result.document.extraction_method}")
        
        # Show first 300 characters of extracted content
        print(f"\n📝 Content preview:")
        print(result.document.content[:300] + "..." if len(result.document.content) > 300 else result.document.content)
        
        # Show source info
        print(f"\n📋 Source Information:")
        print(f"   Type: {result.document.source.source_type}")
        print(f"   Filename: {result.document.source.original_filename}")
        print(f"   Content Type: {result.document.source.content_type}")
        print(f"   Size: {result.document.source.size_bytes} bytes")
    else:
        print(f"❌ Ingestion failed: {result.error_message}")


async def demo_security_validation():
    """Demo security validation"""
    print("\n=== Security Validation Demo ===")
    
    # Test various security scenarios
    test_cases = [
        ("Valid HTTPS URL", "https://example.com/sop.pdf"),
        ("Blocked file:// URL", "file:///etc/passwd"),
        ("Blocked localhost URL", "http://localhost/internal.html"),
        ("Valid HTTP URL", "http://docs.company.com/procedures.html"),
        ("Invalid scheme", "javascript:alert('xss')"),
    ]
    
    async with DocumentIngestionService() as service:
        for description, url in test_cases:
            is_valid = service.security_validator.validate_url(url)
            status = "✅ ALLOWED" if is_valid else "🚫 BLOCKED"
            print(f"{status} {description}: {url}")


async def demo_batch_processing():
    """Demo batch file processing"""
    print("\n=== Batch Processing Demo ===")
    
    # Create multiple sample files
    files = []
    
    # File 1: Safety checklist
    content1 = """Daily Safety Checklist
1. PPE inspection complete
2. Emergency exits clear
3. Fire extinguishers accessible
4. First aid kit stocked"""
    
    files.append(MockUploadFile(
        filename="safety_checklist.txt",
        file=io.BytesIO(content1.encode('utf-8')),
        content_type="text/plain"
    ))
    
    # File 2: Equipment log
    content2 = """Equipment Maintenance Log
Date: 2024-01-15
Equipment: Conveyor Belt #3
Status: Operational
Last Service: 2024-01-10
Next Service: 2024-02-10"""
    
    files.append(MockUploadFile(
        filename="equipment_log.txt",
        file=io.BytesIO(content2.encode('utf-8')),
        content_type="text/plain"
    ))
    
    # Process batch
    async with DocumentIngestionService() as service:
        batch_result = await service.ingest_files(files)
    
    print(f"📊 Batch Processing Results:")
    print(f"   Total documents: {batch_result.total_documents}")
    print(f"   Successful: {batch_result.successful}")
    print(f"   Failed: {batch_result.failed}")
    print(f"   Total processing time: {batch_result.total_processing_time_seconds:.2f} seconds")
    
    print(f"\n📋 Individual Results:")
    for i, result in enumerate(batch_result.results, 1):
        status = "✅" if result.success else "❌"
        if result.success:
            print(f"   {status} File {i}: {result.document.title} ({result.doc_id})")
        else:
            print(f"   {status} File {i}: {result.error_message}")


async def main():
    """Run all demos"""
    print("🚀 Document Ingestion Service Demo")
    print("=" * 50)
    
    try:
        await demo_text_file_ingestion()
        await demo_html_file_ingestion()
        await demo_security_validation()
        await demo_batch_processing()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())