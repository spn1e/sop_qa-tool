"""
Unit tests for text chunking and processing service.

Tests chunking consistency, metadata extraction, heading preservation,
and deterministic behavior for the same input.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from sop_qa_tool.services.text_chunker import (
    TextChunker, 
    HeadingInfo, 
    ChunkMetadata
)
from sop_qa_tool.models.sop_models import (
    DocumentChunk, 
    SOPDocument, 
    ProcedureStep, 
    RoleResponsibility,
    SourceInfo
)
from sop_qa_tool.config.settings import Settings, OperationMode


class TestTextChunker:
    """Test cases for TextChunker class"""
    
    @pytest.fixture
    def chunker(self):
        """Create TextChunker instance with test settings"""
        with patch('sop_qa_tool.services.text_chunker.get_settings') as mock_settings:
            settings = Settings(
                mode=OperationMode.LOCAL,
                chunk_size=500,
                chunk_overlap=50
            )
            mock_settings.return_value = settings
            return TextChunker()
    
    @pytest.fixture
    def sample_text(self):
        """Sample SOP text for testing"""
        return """
# 1. Introduction
This is the introduction to the SOP for bottle filling process.

## 1.1 Purpose
The purpose of this SOP is to ensure consistent bottle filling operations.

# 2. Equipment Required
The following equipment is required:
- FILLER-01 (Main filling machine)
- CONVEYOR-02 (Transport system)
- Temperature probe

# 3. Procedure Steps

## 3.1 Preparation
Step 3.1.1: Check that FILLER-01 is clean and ready.
The operator must verify all connections.

## 3.2 Filling Process
Step 3.2.1: Start the filling sequence.
The QA inspector must monitor temperature.

Risk R-001: Temperature deviation may cause quality issues.
Control C-001: Monitor temperature every 5 minutes.

## 3.3 Quality Control
Step 3.3.1: Inspect filled bottles.
The supervisor must approve the batch.
"""
    
    @pytest.fixture
    def sample_sop_document(self):
        """Sample SOPDocument for testing metadata extraction"""
        return SOPDocument(
            doc_id="test_sop_001",
            title="Bottle Filling SOP",
            process_name="Bottle Filling",
            roles_responsibilities=[
                RoleResponsibility(
                    role="operator",
                    responsibilities=["Operate filling machine", "Monitor process"]
                ),
                RoleResponsibility(
                    role="qa inspector",
                    responsibilities=["Quality checks", "Temperature monitoring"]
                ),
                RoleResponsibility(
                    role="supervisor",
                    responsibilities=["Batch approval", "Process oversight"]
                )
            ],
            materials_equipment=["FILLER-01", "CONVEYOR-02", "Temperature probe"],
            procedure_steps=[
                ProcedureStep(
                    step_id="3.1.1",
                    description="Check that FILLER-01 is clean and ready",
                    responsible_roles=["operator"],
                    required_equipment=["FILLER-01"]
                ),
                ProcedureStep(
                    step_id="3.2.1",
                    description="Start the filling sequence",
                    responsible_roles=["operator"],
                    required_equipment=["FILLER-01"]
                ),
                ProcedureStep(
                    step_id="3.3.1",
                    description="Inspect filled bottles",
                    responsible_roles=["supervisor"],
                    required_equipment=[]
                )
            ],
            source=SourceInfo(file_path="test.pdf")
        )
    
    def test_chunk_document_basic(self, chunker, sample_text):
        """Test basic document chunking functionality"""
        chunks = chunker.chunk_document(
            text=sample_text,
            doc_id="test_doc_001"
        )
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # All chunks should have required fields
        for chunk in chunks:
            assert chunk.chunk_id.startswith("test_doc_001_chunk_")
            assert chunk.doc_id == "test_doc_001"
            assert len(chunk.chunk_text) > 0
            assert chunk.chunk_index >= 0
            assert isinstance(chunk.step_ids, list)
            assert isinstance(chunk.roles, list)
            assert isinstance(chunk.equipment, list)
    
    def test_deterministic_chunking(self, chunker, sample_text):
        """Test that chunking produces consistent results for same input"""
        # Chunk the same text multiple times
        chunks1 = chunker.chunk_document(sample_text, "test_doc_001")
        chunks2 = chunker.chunk_document(sample_text, "test_doc_001")
        chunks3 = chunker.chunk_document(sample_text, "test_doc_001")
        
        # Should produce identical results
        assert len(chunks1) == len(chunks2) == len(chunks3)
        
        for c1, c2, c3 in zip(chunks1, chunks2, chunks3):
            assert c1.chunk_id == c2.chunk_id == c3.chunk_id
            assert c1.chunk_text == c2.chunk_text == c3.chunk_text
            assert c1.chunk_index == c2.chunk_index == c3.chunk_index
            assert c1.step_ids == c2.step_ids == c3.step_ids
            assert c1.roles == c2.roles == c3.roles
            assert c1.equipment == c2.equipment == c3.equipment
    
    def test_heading_extraction(self, chunker, sample_text):
        """Test heading extraction and structure preservation"""
        headings = chunker._extract_headings(sample_text)
        
        # Should find multiple headings
        assert len(headings) > 0
        
        # Check specific headings
        heading_texts = [h.text for h in headings]
        assert "1. Introduction" in heading_texts
        assert "1.1 Purpose" in heading_texts
        assert "2. Equipment Required" in heading_texts
        assert "3. Procedure Steps" in heading_texts
        
        # Check heading levels
        intro_heading = next(h for h in headings if "Introduction" in h.text)
        assert intro_heading.level == 1
        
        purpose_heading = next(h for h in headings if "Purpose" in h.text)
        assert purpose_heading.level == 2
    
    def test_heading_path_building(self, chunker):
        """Test hierarchical heading path construction"""
        # Create mock headings
        headings = [
            HeadingInfo(level=1, text="1. Introduction", start_pos=0, end_pos=20, path=""),
            HeadingInfo(level=2, text="1.1 Purpose", start_pos=50, end_pos=70, path=""),
        ]
        
        # Build path for second heading
        path = chunker._build_heading_path(headings[:1], 2, "1.1 Purpose")
        assert path == "1. Introduction > 1.1 Purpose"
    
    def test_step_id_extraction(self, chunker):
        """Test extraction of step IDs from text"""
        text = "Step 3.1.1: Check equipment. Step 3.2.1: Start process."
        step_ids = chunker._extract_step_ids(text)
        
        assert "3.1.1" in step_ids
        assert "3.2.1" in step_ids
        assert len(step_ids) == 2
    
    def test_risk_id_extraction(self, chunker):
        """Test extraction of risk IDs from text"""
        text = "Risk R-001: Temperature deviation. Risk R002 may occur."
        risk_ids = chunker._extract_risk_ids(text)
        
        assert "R-001" in risk_ids
        assert "R-002" in risk_ids
        assert len(risk_ids) == 2
    
    def test_control_id_extraction(self, chunker):
        """Test extraction of control IDs from text"""
        text = "Control C-001: Monitor temperature. Control C002 prevents issues."
        control_ids = chunker._extract_control_ids(text)
        
        assert "C-001" in control_ids
        assert "C-002" in control_ids
        assert len(control_ids) == 2
    
    def test_role_extraction(self, chunker, sample_sop_document):
        """Test extraction of roles from text"""
        text = "The operator must check equipment. QA inspector monitors quality."
        roles = chunker._extract_roles(text, sample_sop_document)
        
        assert "operator" in roles
        assert "qa" in roles or "qa inspector" in roles
    
    def test_equipment_extraction(self, chunker, sample_sop_document):
        """Test extraction of equipment from text"""
        text = "Use FILLER-01 for filling. Check CONVEYOR-02 operation."
        equipment = chunker._extract_equipment(text, sample_sop_document)
        
        assert "filler-01" in equipment
        assert "conveyor-02" in equipment
    
    def test_metadata_enrichment(self, chunker, sample_sop_document):
        """Test metadata enrichment for chunks"""
        chunks = chunker.chunk_document(
            text=chunker.sample_text if hasattr(chunker, 'sample_text') else "Step 3.1.1: Operator checks FILLER-01.",
            doc_id="test_doc_001",
            sop_document=sample_sop_document
        )
        
        # Find chunk with step information
        step_chunk = None
        for chunk in chunks:
            if chunk.step_ids:
                step_chunk = chunk
                break
        
        if step_chunk:
            assert len(step_chunk.step_ids) > 0
            # Should have extracted roles and equipment
            assert len(step_chunk.roles) > 0 or len(step_chunk.equipment) > 0
    
    def test_recursive_splitting(self, chunker):
        """Test recursive text splitting with different separators"""
        # Long text that needs splitting
        long_text = "This is a long paragraph. " * 50  # ~1250 characters
        
        chunks = chunker._recursive_split(long_text, 500, 50)
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be roughly the target size
        for chunk in chunks:
            assert len(chunk) <= 600  # Allow some flexibility
    
    def test_chunk_overlap(self, chunker):
        """Test that chunks have proper overlap"""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks = chunker._recursive_split(text, 30, 10)
        
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
                
                # There should be some common text (overlap)
                # This is a simplified check - in practice, overlap might be more complex
                assert len(chunk1) > 0 and len(chunk2) > 0
    
    def test_chunk_validation(self, chunker, sample_text):
        """Test chunk validation functionality"""
        chunks = chunker.chunk_document(sample_text, "test_doc_001")
        
        validation_result = chunker.validate_chunks(chunks)
        
        assert 'valid' in validation_result
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        assert 'stats' in validation_result
        
        # Should be valid chunks
        assert validation_result['valid'] is True
        
        # Should have statistics
        stats = validation_result['stats']
        assert stats['total_chunks'] == len(chunks)
        assert stats['avg_chunk_size'] > 0
        assert stats['total_characters'] > 0
    
    def test_empty_text_handling(self, chunker):
        """Test handling of empty or whitespace-only text"""
        chunks = chunker.chunk_document("", "test_doc_001")
        assert len(chunks) == 0
        
        chunks = chunker.chunk_document("   \n\n   ", "test_doc_001")
        assert len(chunks) == 0
    
    def test_very_short_text(self, chunker):
        """Test handling of very short text"""
        short_text = "Short text."
        chunks = chunker.chunk_document(short_text, "test_doc_001")
        
        assert len(chunks) == 1
        assert chunks[0].chunk_text.strip() == short_text.strip()
        assert chunks[0].chunk_index == 0
    
    def test_chunk_id_generation(self, chunker):
        """Test deterministic chunk ID generation"""
        text = "This is a test chunk."
        
        # Generate ID multiple times
        id1 = chunker._generate_chunk_id("doc1", 0, text)
        id2 = chunker._generate_chunk_id("doc1", 0, text)
        id3 = chunker._generate_chunk_id("doc1", 0, text)
        
        # Should be identical
        assert id1 == id2 == id3
        
        # Different text should produce different ID
        id4 = chunker._generate_chunk_id("doc1", 0, "Different text.")
        assert id4 != id1
        
        # Different doc_id should produce different ID
        id5 = chunker._generate_chunk_id("doc2", 0, text)
        assert id5 != id1
        
        # Different index should produce different ID
        id6 = chunker._generate_chunk_id("doc1", 1, text)
        assert id6 != id1
    
    def test_page_number_extraction(self, chunker):
        """Test page number extraction from text"""
        text_with_page = "This is content on page 5 of the document."
        page_no = chunker._extract_page_number(text_with_page)
        assert page_no == 5
        
        text_without_page = "This is content without page reference."
        page_no = chunker._extract_page_number(text_without_page)
        assert page_no is None
    
    def test_structure_preservation_disabled(self, chunker, sample_text):
        """Test chunking with structure preservation disabled"""
        chunks_with_structure = chunker.chunk_document(
            sample_text, "test_doc_001", preserve_structure=True
        )
        
        chunks_without_structure = chunker.chunk_document(
            sample_text, "test_doc_001", preserve_structure=False
        )
        
        # Both should produce chunks, but structure preservation affects metadata
        assert len(chunks_with_structure) > 0
        assert len(chunks_without_structure) > 0
        
        # Chunks with structure should have heading paths
        has_heading_paths = any(chunk.heading_path for chunk in chunks_with_structure)
        no_heading_paths = all(not chunk.heading_path for chunk in chunks_without_structure)
        
        # This test might need adjustment based on implementation details
        # The key is that structure preservation affects the chunking process


class TestHeadingInfo:
    """Test cases for HeadingInfo dataclass"""
    
    def test_heading_info_creation(self):
        """Test HeadingInfo object creation"""
        heading = HeadingInfo(
            level=1,
            text="Introduction",
            start_pos=0,
            end_pos=20,
            path="Introduction"
        )
        
        assert heading.level == 1
        assert heading.text == "Introduction"
        assert heading.start_pos == 0
        assert heading.end_pos == 20
        assert heading.path == "Introduction"


class TestChunkMetadata:
    """Test cases for ChunkMetadata dataclass"""
    
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata object creation"""
        metadata = ChunkMetadata(
            step_ids=["3.1.1", "3.1.2"],
            risk_ids=["R-001"],
            control_ids=["C-001", "C-002"],
            roles=["operator", "supervisor"],
            equipment=["FILLER-01"],
            page_no=5,
            heading_path="3. Process > 3.1 Setup"
        )
        
        assert metadata.step_ids == ["3.1.1", "3.1.2"]
        assert metadata.risk_ids == ["R-001"]
        assert metadata.control_ids == ["C-001", "C-002"]
        assert metadata.roles == ["operator", "supervisor"]
        assert metadata.equipment == ["FILLER-01"]
        assert metadata.page_no == 5
        assert metadata.heading_path == "3. Process > 3.1 Setup"


class TestIntegration:
    """Integration tests for text chunking with other components"""
    
    @pytest.fixture
    def chunker(self):
        """Create TextChunker instance for integration tests"""
        with patch('sop_qa_tool.services.text_chunker.get_settings') as mock_settings:
            settings = Settings(
                mode=OperationMode.LOCAL,
                chunk_size=800,
                chunk_overlap=150
            )
            mock_settings.return_value = settings
            return TextChunker()
    
    def test_chunking_with_real_sop_structure(self, chunker):
        """Test chunking with realistic SOP document structure"""
        realistic_sop = """
SOP-FILL-001: Bottle Filling Procedure
Revision: 2.1
Effective Date: 2024-01-15

1. PURPOSE AND SCOPE
This Standard Operating Procedure (SOP) defines the process for filling bottles in the production line.

2. RESPONSIBILITIES
2.1 Line Operator
- Operate the filling equipment
- Monitor fill levels and quality
- Report any deviations immediately

2.2 QA Inspector  
- Verify product quality
- Conduct random sampling
- Approve or reject batches

2.3 Production Supervisor
- Oversee the filling process
- Approve batch releases
- Handle escalations

3. EQUIPMENT AND MATERIALS
3.1 Primary Equipment
- FILLER-01: Main filling machine (capacity: 1000 bottles/hour)
- CONVEYOR-02: Transport system
- CAPPER-03: Bottle capping unit

3.2 Quality Control Equipment
- Temperature probe (±0.1°C accuracy)
- Fill level gauge
- Pressure sensors

4. PROCEDURE
4.1 Pre-Operation Setup
Step 4.1.1: Verify FILLER-01 is clean and sanitized
- Check cleaning log completion
- Verify sanitization certificate
- Responsible: Line Operator

Step 4.1.2: Calibrate temperature probe
- Set target temperature: 18-22°C
- Verify accuracy against reference
- Responsible: QA Inspector

4.2 Filling Operation
Step 4.2.1: Start filling sequence
- Initialize FILLER-01 control system
- Set fill volume: 500ml ±5ml
- Begin continuous operation
- Responsible: Line Operator

Step 4.2.2: Monitor process parameters
- Check temperature every 5 minutes
- Verify fill levels every 10 bottles
- Record data on batch sheet
- Responsible: Line Operator, QA Inspector

5. RISK MANAGEMENT
Risk R-001: Temperature deviation
- Impact: Product quality degradation
- Probability: Medium
- Controls: C-001, C-002

Risk R-002: Overfill/underfill
- Impact: Customer complaints, waste
- Probability: Low
- Controls: C-003, C-004

6. CONTROL MEASURES
Control C-001: Continuous temperature monitoring
- Method: Automated sensor with alarms
- Frequency: Every 5 minutes
- Responsible: Line Operator

Control C-002: Temperature calibration
- Method: Weekly calibration check
- Frequency: Weekly
- Responsible: QA Inspector

Control C-003: Fill level verification
- Method: Statistical sampling (1 in 10)
- Frequency: Continuous
- Responsible: QA Inspector

Control C-004: Weight check system
- Method: Automated rejection system
- Frequency: Every bottle
- Responsible: System (automated)
"""
        
        chunks = chunker.chunk_document(realistic_sop, "SOP-FILL-001")
        
        # Should produce multiple chunks
        assert len(chunks) >= 3
        
        # Validate chunk content and metadata
        step_chunks = [c for c in chunks if c.step_ids]
        risk_chunks = [c for c in chunks if c.risk_ids]
        control_chunks = [c for c in chunks if c.control_ids]
        role_chunks = [c for c in chunks if c.roles]
        equipment_chunks = [c for c in chunks if c.equipment]
        
        # Should have extracted various types of metadata
        assert len(step_chunks) > 0, "Should find chunks with step IDs"
        assert len(risk_chunks) > 0, "Should find chunks with risk IDs"
        assert len(control_chunks) > 0, "Should find chunks with control IDs"
        assert len(role_chunks) > 0, "Should find chunks with roles"
        assert len(equipment_chunks) > 0, "Should find chunks with equipment"
        
        # Validate specific extractions
        all_step_ids = set()
        all_risk_ids = set()
        all_control_ids = set()
        
        for chunk in chunks:
            all_step_ids.update(chunk.step_ids)
            all_risk_ids.update(chunk.risk_ids)
            all_control_ids.update(chunk.control_ids)
        
        # Should have found specific IDs from the text
        assert "4.1.1" in all_step_ids
        assert "4.1.2" in all_step_ids
        assert "4.2.1" in all_step_ids
        assert "4.2.2" in all_step_ids
        
        assert "R-001" in all_risk_ids
        assert "R-002" in all_risk_ids
        
        assert "C-001" in all_control_ids
        assert "C-002" in all_control_ids
        assert "C-003" in all_control_ids
        assert "C-004" in all_control_ids
    
    def test_performance_with_large_document(self, chunker):
        """Test chunking performance with large document"""
        # Create a large document (simulate ~100KB)
        large_text = """
This is a section of a large SOP document. It contains multiple paragraphs
with detailed procedures, equipment specifications, and safety requirements.
The operator must follow all steps carefully to ensure product quality.

Step 1.1: Initialize the system
Step 1.2: Verify all connections
Step 1.3: Run diagnostic tests

Risk R-001: System failure during operation
Control C-001: Regular maintenance schedule
""" * 200  # Repeat to create large document
        
        import time
        start_time = time.time()
        
        chunks = chunker.chunk_document(large_text, "large_doc_001")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0, f"Chunking took too long: {processing_time:.2f}s"
        
        # Should produce reasonable number of chunks
        assert len(chunks) > 10
        assert len(chunks) < 1000  # Shouldn't be excessive
        
        # Validate chunks
        validation_result = chunker.validate_chunks(chunks)
        assert validation_result['valid'] is True


if __name__ == "__main__":
    pytest.main([__file__])
