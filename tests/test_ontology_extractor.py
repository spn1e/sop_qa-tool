"""
Unit tests for the Ontology Extraction Service.

Tests cover LLM-powered extraction, fallback mechanisms, validation,
merging logic, and error handling scenarios.

Requirements: 2.1, 2.2, 2.3, 7.2
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from sop_qa_tool.services.ontology_extractor import OntologyExtractor
from sop_qa_tool.models.sop_models import (
    SOPDocument, DocumentChunk, ExtractionResult, ValidationResult,
    SourceInfo, ProcedureStep, Risk, Control, RoleResponsibility,
    RiskCategory, ControlType, PriorityLevel
)
from sop_qa_tool.config.settings import Settings, OperationMode


class TestOntologyExtractor:
    """Test suite for OntologyExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create OntologyExtractor instance for testing."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(mode=OperationMode.LOCAL)
            return OntologyExtractor()
    
    @pytest.fixture
    def sample_source_info(self):
        """Create sample source information."""
        return SourceInfo(
            url="https://example.com/sop.pdf",
            page_range=[1, 5],
            last_modified=datetime.utcnow(),
            file_size=1024000
        )
    
    @pytest.fixture
    def sample_sop_text(self):
        """Sample SOP text for testing extraction."""
        return """
        SOP-FILL-001: Bottle Filling Procedure
        Revision: 2.1
        Effective Date: 2024-01-15
        
        1. Purpose and Scope
        This procedure covers the standard bottle filling process for liquid products.
        
        2. Roles and Responsibilities
        2.1 Operator: Responsible for equipment setup and monitoring
        2.2 QA Inspector: Responsible for quality verification
        
        3. Equipment and Materials
        - Filling Machine FM-001
        - Temperature Probe TP-200
        - Safety Glasses
        - Product bottles
        
        4. Procedure Steps
        4.1 Pre-operation Setup
        4.1.1 Verify filling machine is clean and calibrated
        4.1.2 Check temperature probe calibration (±0.5°C)
        
        4.2 Filling Process
        4.2.1 Start filling machine and monitor temperature (18-22°C)
        4.2.2 Inspect first 10 bottles for proper fill level
        
        5. Risks and Controls
        Risk R-001: Temperature deviation may affect product quality
        Control C-001: Continuous temperature monitoring with alarms
        
        6. Quality Checkpoints
        - Visual inspection of fill levels
        - Temperature log verification
        """
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="doc1_chunk_0",
                doc_id="doc1",
                chunk_text="SOP-FILL-001: Bottle Filling Procedure\nRevision: 2.1\nThis procedure covers bottle filling.",
                chunk_index=0,
                page_no=1,
                heading_path="1. Purpose and Scope",
                step_ids=[],
                roles=["Operator"],
                equipment=["Filling Machine"]
            ),
            DocumentChunk(
                chunk_id="doc1_chunk_1",
                doc_id="doc1",
                chunk_text="4.1.1 Verify filling machine is clean\n4.1.2 Check temperature probe",
                chunk_index=1,
                page_no=2,
                heading_path="4. Procedure Steps > 4.1 Pre-operation",
                step_ids=["4.1.1", "4.1.2"],
                roles=["Operator"],
                equipment=["Filling Machine", "Temperature Probe"]
            )
        ]
    
    def test_initialization_local_mode(self):
        """Test extractor initialization in local mode."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(mode=OperationMode.LOCAL)
            extractor = OntologyExtractor()
            
            assert extractor.settings.mode == OperationMode.LOCAL
            assert extractor._bedrock_client is None
            assert extractor._local_model is None
    
    def test_initialization_aws_mode(self):
        """Test extractor initialization in AWS mode."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                mode=OperationMode.AWS,
                opensearch_endpoint="https://test.aoss.amazonaws.com",
                s3_raw_bucket="test-raw",
                s3_chunks_bucket="test-chunks"
            )
            extractor = OntologyExtractor()
            
            assert extractor.settings.mode == OperationMode.AWS
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    @patch('boto3.Session')
    def test_get_bedrock_client_success(self, mock_session):
        """Test successful Bedrock client initialization."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                mode=OperationMode.AWS,
                aws_profile="test-profile",
                aws_region="us-east-1",
                opensearch_endpoint="https://test.aoss.amazonaws.com",
                s3_raw_bucket="test-raw",
                s3_chunks_bucket="test-chunks"
            )
            
            mock_bedrock = Mock()
            mock_session.return_value.client.return_value = mock_bedrock
            
            extractor = OntologyExtractor()
            client = extractor._get_bedrock_client()
            
            assert client == mock_bedrock
            mock_session.assert_called_once_with(
                profile_name="test-profile",
                region_name="us-east-1"
            )
    
    def test_rule_based_extraction(self, extractor, sample_sop_text):
        """Test rule-based extraction fallback."""
        result = extractor._rule_based_extraction(sample_sop_text)
        
        assert isinstance(result, dict)
        assert "title" in result
        assert "procedure_steps" in result
        assert len(result["procedure_steps"]) > 0
        
        # Check that steps were extracted
        step_ids = [step["step_id"] for step in result["procedure_steps"]]
        assert "4.1.1" in step_ids
        assert "4.1.2" in step_ids
        
        # Check equipment extraction
        assert "materials_equipment" in result
        assert len(result["materials_equipment"]) > 0
    
    def test_extract_from_text_local_mode(self, extractor, sample_sop_text, sample_source_info):
        """Test text extraction in local mode."""
        with patch.object(extractor, '_extract_with_local_model') as mock_extract:
            mock_sop = SOPDocument(
                doc_id="test_doc",
                title="Test SOP",
                process_name="Test Process",
                source=sample_source_info
            )
            mock_extract.return_value = ExtractionResult(
                success=True,
                sop_document=mock_sop
            )
            
            result = extractor.extract_from_text(sample_sop_text, "test_doc", sample_source_info)
            
            assert result.success
            assert result.sop_document is not None
            assert result.sop_document.doc_id == "test_doc"
            assert result.processing_time_seconds is not None
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    @patch('boto3.Session')
    def test_extract_with_bedrock_success(self, mock_session):
        """Test successful extraction using Bedrock."""
        # Setup AWS mode
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                mode=OperationMode.AWS,
                bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                opensearch_endpoint="https://test.aoss.amazonaws.com",
                s3_raw_bucket="test-raw",
                s3_chunks_bucket="test-chunks"
            )
            
            # Mock Bedrock response
            mock_bedrock = Mock()
            mock_response = {
                'body': Mock()
            }
            mock_response['body'].read.return_value = json.dumps({
                'content': [{'text': '{"title": "Test SOP", "process_name": "Test Process", "procedure_steps": []}'}]
            }).encode()
            mock_bedrock.invoke_model.return_value = mock_response
            mock_session.return_value.client.return_value = mock_bedrock
            
            extractor = OntologyExtractor()
            source_info = SourceInfo(url="https://test.com/sop.pdf")
            
            result = extractor._extract_with_bedrock("Test SOP content", "test_doc", source_info)
            
            assert result.success
            assert result.sop_document is not None
            assert result.sop_document.title == "Test SOP"
    
    def test_create_sop_document_valid_data(self, extractor, sample_source_info):
        """Test SOP document creation with valid data."""
        extracted_data = {
            "title": "Test SOP",
            "process_name": "Test Process",
            "revision": "1.0",
            "procedure_steps": [
                {
                    "step_id": "1",
                    "description": "Test step",
                    "responsible_roles": ["Operator"],
                    "required_equipment": ["Machine-1"]
                }
            ],
            "risks": [
                {
                    "risk_id": "R-001",
                    "description": "Test risk",
                    "category": "safety"
                }
            ]
        }
        
        sop_doc = extractor._create_sop_document(extracted_data, "test_doc", sample_source_info)
        
        assert isinstance(sop_doc, SOPDocument)
        assert sop_doc.doc_id == "test_doc"
        assert sop_doc.title == "Test SOP"
        assert len(sop_doc.procedure_steps) == 1
        assert len(sop_doc.risks) == 1
    
    def test_create_sop_document_invalid_data(self, extractor, sample_source_info):
        """Test SOP document creation with invalid data falls back to minimal document."""
        extracted_data = {
            "title": "Test SOP",
            "process_name": "Test Process",
            "procedure_steps": [
                {
                    "step_id": "invalid_step_id_format!!!",  # Invalid step ID
                    "description": "Test step"
                }
            ]
        }
        
        sop_doc = extractor._create_sop_document(extracted_data, "test_doc", sample_source_info)
        
        assert isinstance(sop_doc, SOPDocument)
        assert sop_doc.doc_id == "test_doc"
        assert sop_doc.title == "Test SOP"
        # Should fall back to minimal document due to validation error
        assert len(sop_doc.procedure_steps) == 0
    
    def test_merge_extractions_single_extraction(self, extractor, sample_source_info):
        """Test merging with single extraction returns the same document."""
        sop_doc = SOPDocument(
            doc_id="test_doc",
            title="Test SOP",
            process_name="Test Process",
            source=sample_source_info
        )
        
        result = extractor._merge_extractions([sop_doc], "merged_doc", sample_source_info)
        
        assert result.doc_id == "merged_doc"
        assert result.title == "Test SOP"
    
    def test_merge_extractions_multiple_documents(self, extractor, sample_source_info):
        """Test merging multiple partial extractions."""
        # Create two partial extractions
        sop1 = SOPDocument(
            doc_id="doc1",
            title="Test SOP",
            process_name="Test Process",
            source=sample_source_info,
            procedure_steps=[
                ProcedureStep(step_id="1", description="Step 1"),
                ProcedureStep(step_id="2", description="Step 2")
            ],
            risks=[
                Risk(risk_id="R-001", description="Risk 1", category=RiskCategory.SAFETY)
            ]
        )
        
        sop2 = SOPDocument(
            doc_id="doc2",
            title="Test SOP",
            process_name="Test Process",
            source=sample_source_info,
            procedure_steps=[
                ProcedureStep(step_id="3", description="Step 3"),
                ProcedureStep(step_id="1", description="Step 1 duplicate")  # Duplicate
            ],
            risks=[
                Risk(risk_id="R-002", description="Risk 2", category=RiskCategory.QUALITY)
            ],
            controls=[
                Control(control_id="C-001", description="Control 1", control_type=ControlType.PREVENTIVE)
            ]
        )
        
        merged = extractor._merge_extractions([sop1, sop2], "merged_doc", sample_source_info)
        
        assert merged.doc_id == "merged_doc"
        assert len(merged.procedure_steps) == 3  # 1, 2, 3 (duplicate removed)
        assert len(merged.risks) == 2  # R-001, R-002
        assert len(merged.controls) == 1  # C-001
        
        # Check step ordering
        step_ids = [step.step_id for step in merged.procedure_steps]
        assert step_ids == ["1", "2", "3"]
    
    def test_parse_step_id_valid(self, extractor):
        """Test step ID parsing for sorting."""
        assert extractor._parse_step_id("1") == (1,)
        assert extractor._parse_step_id("1.2") == (1, 2)
        assert extractor._parse_step_id("1.2.3") == (1, 2, 3)
    
    def test_parse_step_id_invalid(self, extractor):
        """Test step ID parsing with invalid format."""
        assert extractor._parse_step_id("invalid") == (999,)
        assert extractor._parse_step_id("1.a.3") == (999,)
    
    def test_validate_extraction_valid_document(self, extractor, sample_source_info):
        """Test validation of a valid SOP document."""
        sop_doc = SOPDocument(
            doc_id="test_doc",
            title="Complete SOP",
            process_name="Test Process",
            source=sample_source_info,
            revision="1.0",
            procedure_steps=[
                ProcedureStep(step_id="1", description="Step 1"),
                ProcedureStep(step_id="2", description="Step 2")
            ],
            risks=[
                Risk(risk_id="R-001", description="Risk 1", category=RiskCategory.SAFETY, affected_steps=["1"])
            ],
            controls=[
                Control(control_id="C-001", description="Control 1", control_type=ControlType.PREVENTIVE, applicable_steps=["1"])
            ],
            roles_responsibilities=[
                RoleResponsibility(role="Operator", responsibilities=["Operate equipment"])
            ]
        )
        
        result = extractor._validate_extraction(sop_doc)
        
        assert result.is_valid
        assert result.completeness_score > 0.5
        assert len(result.errors) == 0
    
    def test_validate_extraction_missing_steps(self, extractor, sample_source_info):
        """Test validation with missing procedure steps."""
        sop_doc = SOPDocument(
            doc_id="test_doc",
            title="Incomplete SOP",
            process_name="Test Process",
            source=sample_source_info
            # No procedure steps
        )
        
        result = extractor._validate_extraction(sop_doc)
        
        assert not result.is_valid
        assert "No procedure steps found" in result.errors
        assert result.completeness_score < 0.5
    
    def test_validate_extraction_referential_integrity(self, extractor, sample_source_info):
        """Test validation of referential integrity between risks/controls and steps."""
        sop_doc = SOPDocument(
            doc_id="test_doc",
            title="Test SOP",
            process_name="Test Process",
            source=sample_source_info,
            procedure_steps=[
                ProcedureStep(step_id="1", description="Step 1")
            ],
            risks=[
                Risk(risk_id="R-001", description="Risk 1", category=RiskCategory.SAFETY, affected_steps=["999"])  # Non-existent step
            ]
        )
        
        result = extractor._validate_extraction(sop_doc)
        
        assert result.is_valid  # Still valid, but has warnings
        assert any("references non-existent step" in warning for warning in result.warnings)
    
    def test_extract_from_chunks_success(self, extractor, sample_chunks, sample_source_info):
        """Test successful extraction from multiple chunks."""
        with patch.object(extractor, 'extract_from_text') as mock_extract:
            # Mock successful extractions from each chunk
            mock_sop1 = SOPDocument(
                doc_id="doc1_chunk_0",
                title="Test SOP",
                process_name="Test Process",
                source=sample_source_info,
                procedure_steps=[ProcedureStep(step_id="1", description="Step 1")]
            )
            mock_sop2 = SOPDocument(
                doc_id="doc1_chunk_1", 
                title="Test SOP",
                process_name="Test Process",
                source=sample_source_info,
                procedure_steps=[ProcedureStep(step_id="2", description="Step 2")]
            )
            
            mock_extract.side_effect = [
                ExtractionResult(success=True, sop_document=mock_sop1),
                ExtractionResult(success=True, sop_document=mock_sop2)
            ]
            
            result = extractor.extract_from_chunks(sample_chunks, "doc1", sample_source_info)
            
            assert result.success
            assert result.sop_document is not None
            assert len(result.sop_document.procedure_steps) == 2
            assert result.processing_time_seconds is not None
    
    def test_extract_from_chunks_partial_failure(self, extractor, sample_chunks, sample_source_info):
        """Test extraction from chunks with some failures."""
        with patch.object(extractor, 'extract_from_text') as mock_extract:
            # First chunk succeeds, second fails
            mock_sop = SOPDocument(
                doc_id="doc1_chunk_0",
                title="Test SOP", 
                process_name="Test Process",
                source=sample_source_info,
                procedure_steps=[ProcedureStep(step_id="1", description="Step 1")]
            )
            
            mock_extract.side_effect = [
                ExtractionResult(success=True, sop_document=mock_sop),
                ExtractionResult(success=False, errors=["Extraction failed"])
            ]
            
            result = extractor.extract_from_chunks(sample_chunks, "doc1", sample_source_info)
            
            assert result.success  # Should still succeed with partial results
            assert result.sop_document is not None
            assert len(result.warnings) > 0
    
    def test_extract_from_chunks_complete_failure(self, extractor, sample_chunks, sample_source_info):
        """Test extraction from chunks with complete failure."""
        with patch.object(extractor, 'extract_from_text') as mock_extract:
            # All chunks fail
            mock_extract.return_value = ExtractionResult(success=False, errors=["Extraction failed"])
            
            result = extractor.extract_from_chunks(sample_chunks, "doc1", sample_source_info)
            
            assert not result.success
            assert "No successful extractions from any chunks" in result.errors[0]
    
    def test_get_sop_schema(self, extractor):
        """Test SOP schema generation."""
        schema = extractor._get_sop_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "procedure_steps" in schema["properties"]
        assert "risks" in schema["properties"]
        assert "controls" in schema["properties"]
    
    def test_error_handling_bedrock_failure(self, extractor, sample_sop_text, sample_source_info):
        """Test error handling when Bedrock fails."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                mode=OperationMode.AWS,
                opensearch_endpoint="https://test.aoss.amazonaws.com",
                s3_raw_bucket="test-raw", 
                s3_chunks_bucket="test-chunks"
            )
            
            # Mock both Bedrock failure and disable local fallback
            with patch.object(extractor, '_get_bedrock_client') as mock_client, \
                 patch.object(extractor, '_extract_with_local_model') as mock_local:
                mock_client.side_effect = Exception("Bedrock unavailable")
                mock_local.side_effect = Exception("Local model unavailable")
                
                result = extractor.extract_from_text(sample_sop_text, "test_doc", sample_source_info)
                
                assert not result.success
                assert "Extraction failed" in result.errors[0]
    
    def test_extraction_with_empty_text(self, extractor, sample_source_info):
        """Test extraction with empty or minimal text."""
        result = extractor.extract_from_text("", "test_doc", sample_source_info)
        
        # Should handle gracefully and return minimal document
        assert result.success or len(result.errors) > 0
    
    def test_extraction_with_non_sop_text(self, extractor, sample_source_info):
        """Test extraction with non-SOP text content."""
        non_sop_text = "This is just a regular document with no SOP structure. It talks about various topics but has no procedure steps or manufacturing content."
        
        result = extractor.extract_from_text(non_sop_text, "test_doc", sample_source_info)
        
        # Should still create a document but with minimal content
        if result.success:
            assert result.sop_document is not None
            assert result.sop_document.doc_id == "test_doc"


@pytest.fixture
def sample_sop_document():
    """Create a sample SOP document for testing."""
    return SOPDocument(
        doc_id="SOP-001",
        title="Bottle Filling Procedure",
        process_name="Bottle Filling",
        revision="2.1",
        source=SourceInfo(url="https://example.com/sop.pdf"),
        procedure_steps=[
            ProcedureStep(
                step_id="1",
                description="Setup equipment",
                responsible_roles=["Operator"],
                required_equipment=["Filling Machine"]
            ),
            ProcedureStep(
                step_id="2", 
                description="Start filling process",
                responsible_roles=["Operator"],
                safety_notes=["Wear safety glasses"]
            )
        ],
        risks=[
            Risk(
                risk_id="R-001",
                description="Equipment malfunction",
                category=RiskCategory.OPERATIONAL,
                overall_rating=PriorityLevel.HIGH,
                affected_steps=["1", "2"]
            )
        ],
        controls=[
            Control(
                control_id="C-001",
                description="Regular maintenance",
                control_type=ControlType.PREVENTIVE,
                applicable_risks=["R-001"],
                applicable_steps=["1"]
            )
        ],
        roles_responsibilities=[
            RoleResponsibility(
                role="Operator",
                responsibilities=["Equipment operation", "Safety compliance"],
                qualifications=["Basic training", "Safety certification"]
            )
        ]
    )


class TestIntegrationScenarios:
    """Integration tests for complex extraction scenarios."""
    
    def test_complete_sop_extraction_workflow(self, sample_sop_document):
        """Test complete workflow from text to validated SOP document."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(mode=OperationMode.LOCAL)
            
            extractor = OntologyExtractor()
            
            # Mock the rule-based extraction to return structured data
            with patch.object(extractor, '_rule_based_extraction') as mock_extract:
                mock_extract.return_value = {
                    "title": "Bottle Filling Procedure",
                    "process_name": "Bottle Filling",
                    "revision": "2.1",
                    "procedure_steps": [
                        {"step_id": "1", "description": "Setup equipment"},
                        {"step_id": "2", "description": "Start filling"}
                    ],
                    "risks": [
                        {"risk_id": "R-001", "description": "Equipment failure", "category": "operational"}
                    ]
                }
                
                source_info = SourceInfo(url="https://example.com/test.pdf")
                result = extractor.extract_from_text("Sample SOP text", "SOP-001", source_info)
                
                assert result.success
                assert result.sop_document.title == "Bottle Filling Procedure"
                assert len(result.sop_document.procedure_steps) == 2
                assert result.processing_time_seconds is not None
    
    def test_multi_chunk_complex_sop(self):
        """Test extraction from a complex multi-chunk SOP document."""
        with patch('sop_qa_tool.services.ontology_extractor.get_settings') as mock_settings:
            mock_settings.return_value = Settings(mode=OperationMode.LOCAL)
            
            extractor = OntologyExtractor()
            
            # Create complex chunks representing different sections
            chunks = [
                DocumentChunk(
                    chunk_id="sop_chunk_0",
                    doc_id="complex_sop",
                    chunk_text="SOP-COMPLEX-001: Multi-Phase Manufacturing Process\nRevision 3.2\nOwner: Production Manager",
                    chunk_index=0,
                    heading_path="1. Document Information"
                ),
                DocumentChunk(
                    chunk_id="sop_chunk_1", 
                    doc_id="complex_sop",
                    chunk_text="Phase 1: Preparation\n1.1 Gather materials\n1.2 Setup workstation\n1.3 Calibrate equipment",
                    chunk_index=1,
                    heading_path="2. Phase 1 - Preparation",
                    step_ids=["1.1", "1.2", "1.3"]
                ),
                DocumentChunk(
                    chunk_id="sop_chunk_2",
                    doc_id="complex_sop", 
                    chunk_text="Phase 2: Production\n2.1 Start production line\n2.2 Monitor quality parameters\nRisk: Line stoppage due to equipment failure",
                    chunk_index=2,
                    heading_path="3. Phase 2 - Production",
                    step_ids=["2.1", "2.2"],
                    risk_ids=["R-001"]
                )
            ]
            
            source_info = SourceInfo(url="https://example.com/complex_sop.pdf")
            
            with patch.object(extractor, 'extract_from_text') as mock_extract:
                # Mock different extractions for each chunk
                mock_extractions = [
                    ExtractionResult(
                        success=True,
                        sop_document=SOPDocument(
                            doc_id="sop_chunk_0",
                            title="Multi-Phase Manufacturing Process",
                            process_name="Multi-Phase Manufacturing",
                            revision="3.2",
                            source=source_info
                        )
                    ),
                    ExtractionResult(
                        success=True,
                        sop_document=SOPDocument(
                            doc_id="sop_chunk_1",
                            title="Multi-Phase Manufacturing Process",
                            process_name="Multi-Phase Manufacturing",
                            source=source_info,
                            procedure_steps=[
                                ProcedureStep(step_id="1.1", description="Gather materials"),
                                ProcedureStep(step_id="1.2", description="Setup workstation"),
                                ProcedureStep(step_id="1.3", description="Calibrate equipment")
                            ]
                        )
                    ),
                    ExtractionResult(
                        success=True,
                        sop_document=SOPDocument(
                            doc_id="sop_chunk_2",
                            title="Multi-Phase Manufacturing Process", 
                            process_name="Multi-Phase Manufacturing",
                            source=source_info,
                            procedure_steps=[
                                ProcedureStep(step_id="2.1", description="Start production line"),
                                ProcedureStep(step_id="2.2", description="Monitor quality parameters")
                            ],
                            risks=[
                                Risk(risk_id="R-001", description="Line stoppage", category=RiskCategory.OPERATIONAL)
                            ]
                        )
                    )
                ]
                mock_extract.side_effect = mock_extractions
                
                result = extractor.extract_from_chunks(chunks, "complex_sop", source_info)
                
                assert result.success
                assert result.sop_document.title == "Multi-Phase Manufacturing Process"
                assert result.sop_document.revision == "3.2"
                assert len(result.sop_document.procedure_steps) == 5  # All steps merged
                assert len(result.sop_document.risks) == 1
                
                # Verify step ordering
                step_ids = [step.step_id for step in result.sop_document.procedure_steps]
                assert step_ids == ["1.1", "1.2", "1.3", "2.1", "2.2"]
