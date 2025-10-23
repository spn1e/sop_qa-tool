"""
Unit tests for SOP data models and validation functions.

Tests cover model validation, serialization, and schema compliance checking
as specified in requirements 2.2 and 2.3.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from pydantic import ValidationError

from sop_qa_tool.models import (
    SOPDocument,
    ProcedureStep,
    Risk,
    Control,
    RoleResponsibility,
    Definition,
    ChangeLogEntry,
    SourceInfo,
    DocumentChunk,
    ExtractionResult,
    ValidationResult,
    PriorityLevel,
    StepType,
    RiskCategory,
    ControlType,
    SOPValidator,
    validate_sop_schema_compliance,
    validate_required_extraction_fields
)


class TestSOPModels:
    """Test cases for SOP Pydantic models."""
    
    def test_source_info_creation(self):
        """Test SourceInfo model creation and validation."""
        source = SourceInfo(
            url="https://example.com/sop.pdf",
            page_range=[1, 5],
            last_modified=datetime.utcnow(),
            file_size=1024
        )
        
        assert source.url == "https://example.com/sop.pdf"
        assert source.page_range == [1, 5]
        assert source.file_size == 1024
    
    def test_source_info_invalid_page_range(self):
        """Test SourceInfo validation with invalid page range."""
        with pytest.raises(ValidationError) as exc_info:
            SourceInfo(
                url="https://example.com/sop.pdf",
                page_range=[5, 1]  # Invalid: start > end
            )
        
        assert "Start page must be less than or equal to end page" in str(exc_info.value)
    
    def test_procedure_step_creation(self):
        """Test ProcedureStep model creation and validation."""
        step = ProcedureStep(
            step_id="1.2.1",
            title="Temperature Check",
            description="Check that the temperature is within acceptable range",
            step_type=StepType.VERIFICATION,
            responsible_roles=["Operator", "QA Inspector"],
            required_equipment=["Thermometer", "Data Logger"],
            duration_minutes=5,
            acceptance_criteria=["Temperature between 20-25°C"]
        )
        
        assert step.step_id == "1.2.1"
        assert step.step_type == StepType.VERIFICATION
        assert len(step.responsible_roles) == 2
        assert step.duration_minutes == 5
    
    def test_procedure_step_invalid_step_id(self):
        """Test ProcedureStep validation with invalid step ID format."""
        with pytest.raises(ValidationError) as exc_info:
            ProcedureStep(
                step_id="1.2.a",  # Invalid: contains letter
                description="Test step"
            )
        
        assert "Step ID must follow format" in str(exc_info.value)
    
    def test_procedure_step_invalid_duration(self):
        """Test ProcedureStep validation with invalid duration."""
        with pytest.raises(ValidationError) as exc_info:
            ProcedureStep(
                step_id="1.1",
                description="Test step",
                duration_minutes=-5  # Invalid: negative duration
            )
        
        assert "Duration must be positive" in str(exc_info.value)
    
    def test_risk_creation(self):
        """Test Risk model creation."""
        risk = Risk(
            risk_id="R-001",
            description="Equipment malfunction during operation",
            category=RiskCategory.OPERATIONAL,
            probability=PriorityLevel.MEDIUM,
            severity=PriorityLevel.HIGH,
            overall_rating=PriorityLevel.HIGH,
            affected_steps=["2.1", "2.2"],
            potential_consequences=["Production delay", "Quality issues"]
        )
        
        assert risk.risk_id == "R-001"
        assert risk.category == RiskCategory.OPERATIONAL
        assert risk.overall_rating == PriorityLevel.HIGH
        assert len(risk.affected_steps) == 2
    
    def test_control_creation(self):
        """Test Control model creation."""
        control = Control(
            control_id="C-001",
            description="Regular equipment maintenance schedule",
            control_type=ControlType.PREVENTIVE,
            effectiveness=PriorityLevel.HIGH,
            applicable_risks=["R-001"],
            responsible_roles=["Maintenance Technician"],
            verification_method="Maintenance log review",
            frequency="Weekly"
        )
        
        assert control.control_id == "C-001"
        assert control.control_type == ControlType.PREVENTIVE
        assert "R-001" in control.applicable_risks
    
    def test_sop_document_creation(self):
        """Test complete SOPDocument creation."""
        now = datetime.utcnow()
        
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Bottle Filling Standard Operating Procedure",
            process_name="Bottle Filling",
            revision="1.2",
            effective_date=now,
            owner_role="Production Manager",
            scope="Applies to all bottle filling operations",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(
                    step_id="1.1",
                    description="Prepare filling station"
                )
            ]
        )
        
        assert sop.doc_id == "SOP-001"
        assert sop.title == "Bottle Filling Standard Operating Procedure"
        assert len(sop.procedure_steps) == 1
        assert sop.extraction_timestamp is not None
    
    def test_sop_document_invalid_dates(self):
        """Test SOPDocument validation with invalid date range."""
        now = datetime.utcnow()
        
        with pytest.raises(ValidationError) as exc_info:
            SOPDocument(
                doc_id="SOP-001",
                title="Test SOP",
                process_name="Test Process",
                effective_date=now,
                expiry_date=now - timedelta(days=1),  # Invalid: expiry before effective
                source=SourceInfo(url="https://example.com/sop.pdf")
            )
        
        assert "Effective date must be before expiry date" in str(exc_info.value)
    
    def test_sop_document_confidence_validation(self):
        """Test SOPDocument confidence score validation."""
        with pytest.raises(ValidationError) as exc_info:
            SOPDocument(
                doc_id="SOP-001",
                title="Test SOP",
                process_name="Test Process",
                source=SourceInfo(url="https://example.com/sop.pdf"),
                extraction_confidence=1.5  # Invalid: > 1.0
            )
        
        assert "Confidence score must be between 0 and 1" in str(exc_info.value)
    
    def test_sop_document_helper_methods(self):
        """Test SOPDocument helper methods."""
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Test SOP",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(
                    step_id="1.1",
                    description="Step 1",
                    responsible_roles=["Operator"]
                ),
                ProcedureStep(
                    step_id="1.2",
                    description="Step 2",
                    responsible_roles=["QA Inspector"]
                )
            ],
            risks=[
                Risk(
                    risk_id="R-001",
                    description="Safety risk",
                    category=RiskCategory.SAFETY,
                    overall_rating=PriorityLevel.HIGH
                ),
                Risk(
                    risk_id="R-002",
                    description="Quality risk",
                    category=RiskCategory.QUALITY,
                    overall_rating=PriorityLevel.MEDIUM
                )
            ],
            controls=[
                Control(
                    control_id="C-001",
                    description="Safety control",
                    control_type=ControlType.PREVENTIVE,
                    applicable_risks=["R-001"]
                )
            ]
        )
        
        # Test get_steps_by_role
        operator_steps = sop.get_steps_by_role("Operator")
        assert len(operator_steps) == 1
        assert operator_steps[0].step_id == "1.1"
        
        # Test get_risks_by_category
        safety_risks = sop.get_risks_by_category(RiskCategory.SAFETY)
        assert len(safety_risks) == 1
        assert safety_risks[0].risk_id == "R-001"
        
        # Test get_high_priority_risks
        high_risks = sop.get_high_priority_risks()
        assert len(high_risks) == 1
        assert high_risks[0].risk_id == "R-001"
        
        # Test get_controls_for_risk
        controls_for_r001 = sop.get_controls_for_risk("R-001")
        assert len(controls_for_r001) == 1
        assert controls_for_r001[0].control_id == "C-001"
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk model creation."""
        chunk = DocumentChunk(
            chunk_id="doc1_chunk_001",
            doc_id="doc1",
            chunk_text="This is a sample chunk of text from the document.",
            chunk_index=0,
            page_no=1,
            heading_path="1. Introduction > 1.1 Overview",
            step_ids=["1.1"],
            roles=["Operator"],
            equipment=["Machine-A"],
            embedding=[0.1, -0.2, 0.3, 0.4],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert chunk.chunk_id == "doc1_chunk_001"
        assert chunk.chunk_index == 0
        assert len(chunk.embedding) == 4
        assert "Operator" in chunk.roles
    
    def test_document_chunk_invalid_index(self):
        """Test DocumentChunk validation with invalid chunk index."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                chunk_id="doc1_chunk_001",
                doc_id="doc1",
                chunk_text="Sample text",
                chunk_index=-1  # Invalid: negative index
            )
        
        assert "Chunk index must be non-negative" in str(exc_info.value)
    
    def test_extraction_result_creation(self):
        """Test ExtractionResult model creation."""
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Test SOP",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf")
        )
        
        result = ExtractionResult(
            success=True,
            sop_document=sop,
            chunks=[
                DocumentChunk(
                    chunk_id="chunk1",
                    doc_id="SOP-001",
                    chunk_text="Sample chunk",
                    chunk_index=0
                )
            ],
            processing_time_seconds=2.5
        )
        
        assert result.success is True
        assert result.sop_document.doc_id == "SOP-001"
        assert len(result.chunks) == 1
        assert result.processing_time_seconds == 2.5
    
    def test_validation_result_creation(self):
        """Test ValidationResult model creation."""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing required field: title"],
            warnings=["Low completeness score"],
            completeness_score=0.6
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.completeness_score == 0.6


class TestSOPValidator:
    """Test cases for SOPValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SOPValidator()
        
        # Create a sample valid SOP for testing
        self.valid_sop = SOPDocument(
            doc_id="SOP-001",
            title="Bottle Filling Standard Operating Procedure",
            process_name="Bottle Filling",
            revision="1.0",
            effective_date=datetime.utcnow(),
            owner_role="Production Manager",
            scope="All bottle filling operations",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(
                    step_id="1.1",
                    description="Prepare the filling station and verify all equipment is operational",
                    responsible_roles=["Operator"],
                    required_equipment=["Filling Machine", "Conveyor Belt"]
                ),
                ProcedureStep(
                    step_id="1.2",
                    description="Perform quality check on bottles before filling",
                    responsible_roles=["QA Inspector"]
                )
            ],
            roles_responsibilities=[
                RoleResponsibility(
                    role="Operator",
                    responsibilities=["Operate filling machine", "Monitor process"]
                ),
                RoleResponsibility(
                    role="QA Inspector",
                    responsibilities=["Quality checks", "Documentation"]
                )
            ],
            risks=[
                Risk(
                    risk_id="R-001",
                    description="Equipment malfunction during filling",
                    category=RiskCategory.OPERATIONAL,
                    overall_rating=PriorityLevel.MEDIUM,
                    affected_steps=["1.1"]
                )
            ],
            controls=[
                Control(
                    control_id="C-001",
                    description="Regular equipment maintenance",
                    control_type=ControlType.PREVENTIVE,
                    applicable_risks=["R-001"],
                    applicable_steps=["1.1"]
                )
            ]
        )
    
    def test_validate_valid_sop(self):
        """Test validation of a valid SOP document."""
        result = self.validator.validate_sop_document(self.valid_sop)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.completeness_score > 0.7
    
    def test_validate_sop_with_short_title(self):
        """Test validation with short title."""
        sop = SOPDocument(
            doc_id="SOP-001",
            title="SOP",  # Too short
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf")
        )
        
        result = self.validator.validate_sop_document(sop)
        
        assert result.is_valid is False
        assert any("Title too short" in error for error in result.errors)
    
    def test_validate_sop_with_duplicate_step_ids(self):
        """Test validation with duplicate step IDs."""
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Test SOP Document",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(step_id="1.1", description="First step"),
                ProcedureStep(step_id="1.1", description="Duplicate step")  # Duplicate ID
            ]
        )
        
        result = self.validator.validate_sop_document(sop)
        
        assert result.is_valid is False
        assert any("Duplicate step ID" in error for error in result.errors)
    
    def test_validate_sop_with_invalid_cross_references(self):
        """Test validation with invalid cross-references."""
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Test SOP Document",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(step_id="1.1", description="Test step")
            ],
            risks=[
                Risk(
                    risk_id="R-001",
                    description="Test risk",
                    category=RiskCategory.SAFETY,
                    affected_steps=["2.1"]  # Non-existent step
                )
            ]
        )
        
        result = self.validator.validate_sop_document(sop)
        
        assert result.is_valid is False
        assert any("references non-existent step" in error for error in result.errors)
    
    def test_validate_extraction_result_success(self):
        """Test validation of successful extraction result."""
        result = ExtractionResult(
            success=True,
            sop_document=self.valid_sop,
            chunks=[
                DocumentChunk(
                    chunk_id="chunk1",
                    doc_id="SOP-001",
                    chunk_text="Sample chunk text content",
                    chunk_index=0
                )
            ],
            processing_time_seconds=1.5
        )
        
        validation = self.validator.validate_extraction_result(result)
        
        assert validation.is_valid is True
        assert len(validation.errors) == 0
    
    def test_validate_extraction_result_failure(self):
        """Test validation of failed extraction result."""
        result = ExtractionResult(
            success=False,
            sop_document=None,
            errors=["Failed to parse document"],
            processing_time_seconds=0.5
        )
        
        validation = self.validator.validate_extraction_result(result)
        
        # Should be valid since failure is properly documented
        assert validation.is_valid is True
    
    def test_validate_chunks_with_duplicates(self):
        """Test chunk validation with duplicate IDs."""
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                doc_id="doc1",
                chunk_text="First chunk",
                chunk_index=0
            ),
            DocumentChunk(
                chunk_id="chunk1",  # Duplicate ID
                doc_id="doc1",
                chunk_text="Second chunk",
                chunk_index=1
            )
        ]
        
        result = ExtractionResult(
            success=True,
            chunks=chunks
        )
        
        validation = self.validator.validate_extraction_result(result)
        
        assert validation.is_valid is False
        assert any("Duplicate chunk ID" in error for error in validation.errors)
    
    def test_completeness_score_calculation(self):
        """Test completeness score calculation."""
        # Create SOP with minimal fields
        minimal_sop = SOPDocument(
            doc_id="SOP-001",
            title="Minimal SOP",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(step_id="1.1", description="Single step")
            ]
        )
        
        result = self.validator.validate_sop_document(minimal_sop)
        
        # Should have lower completeness score due to missing optional fields
        assert result.completeness_score < 0.5
        
        # Full SOP should have higher score
        full_result = self.validator.validate_sop_document(self.valid_sop)
        assert full_result.completeness_score > result.completeness_score


class TestValidationFunctions:
    """Test cases for standalone validation functions."""
    
    def test_validate_sop_schema_compliance_valid(self):
        """Test schema compliance validation with valid data."""
        valid_data = {
            "doc_id": "SOP-001",
            "title": "Test SOP",
            "process_name": "Test Process",
            "source": {
                "url": "https://example.com/sop.pdf"
            }
        }
        
        is_valid, errors = validate_sop_schema_compliance(valid_data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_sop_schema_compliance_invalid(self):
        """Test schema compliance validation with invalid data."""
        invalid_data = {
            "doc_id": "SOP-001",
            "title": "Test SOP",
            # Missing required 'process_name' and 'source'
        }
        
        is_valid, errors = validate_sop_schema_compliance(invalid_data)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("process_name" in error for error in errors)
    
    def test_validate_required_extraction_fields(self):
        """Test required field extraction validation."""
        # Create SOP with good field coverage
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Complete SOP",
            process_name="Test Process",
            revision="1.0",
            effective_date=datetime.utcnow(),
            source=SourceInfo(url="https://example.com/sop.pdf"),
            procedure_steps=[
                ProcedureStep(step_id="1.1", description="Test step")
            ],
            risks=[
                Risk(
                    risk_id="R-001",
                    description="Test risk",
                    category=RiskCategory.SAFETY
                )
            ],
            controls=[
                Control(
                    control_id="C-001",
                    description="Test control",
                    control_type=ControlType.PREVENTIVE
                )
            ],
            roles_responsibilities=[
                RoleResponsibility(
                    role="Operator",
                    responsibilities=["Test responsibility"]
                )
            ]
        )
        
        result = validate_required_extraction_fields(sop, min_coverage=0.8)
        
        assert result.is_valid is True
        assert result.completeness_score >= 0.8
    
    def test_validate_required_extraction_fields_insufficient(self):
        """Test required field validation with insufficient coverage."""
        # Create minimal SOP
        sop = SOPDocument(
            doc_id="SOP-001",
            title="Minimal SOP",
            process_name="Test Process",
            source=SourceInfo(url="https://example.com/sop.pdf")
            # Missing most required fields
        )
        
        result = validate_required_extraction_fields(sop, min_coverage=0.8)
        
        assert result.is_valid is False
        assert result.completeness_score < 0.8
        assert any("below minimum" in error for error in result.errors)


if __name__ == "__main__":
    pytest.main([__file__])
