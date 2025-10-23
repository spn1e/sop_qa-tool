"""
Validation functions for SOP data models and schema compliance checking.

This module provides comprehensive validation for extracted SOP data to ensure
schema compliance and data quality as specified in requirements 2.2 and 2.3.
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError
import re
from datetime import datetime

from .sop_models import (
    SOPDocument, ProcedureStep, Risk, Control, RoleResponsibility,
    ValidationResult, ExtractionResult, DocumentChunk
)


class SOPValidator:
    """Comprehensive validator for SOP documents and their components."""
    
    # Required fields for completeness scoring
    CRITICAL_FIELDS = [
        'title', 'process_name', 'procedure_steps'
    ]
    
    IMPORTANT_FIELDS = [
        'revision', 'effective_date', 'owner_role', 'scope',
        'roles_responsibilities', 'risks', 'controls'
    ]
    
    OPTIONAL_FIELDS = [
        'definitions_glossary', 'preconditions', 'materials_equipment',
        'acceptance_criteria', 'compliance_refs', 'attachments_refs', 'change_log'
    ]
    
    def __init__(self):
        self.validation_rules = {
            'min_procedure_steps': 1,
            'max_step_id_depth': 4,  # e.g., 1.2.3.4
            'min_title_length': 5,
            'max_title_length': 200,
            'required_step_fields': ['step_id', 'description'],
            'required_risk_fields': ['risk_id', 'description', 'category'],
            'required_control_fields': ['control_id', 'description', 'control_type']
        }
    
    def validate_sop_document(self, sop: SOPDocument) -> ValidationResult:
        """
        Comprehensive validation of an SOP document.
        
        Args:
            sop: SOPDocument instance to validate
            
        Returns:
            ValidationResult with validation status, errors, warnings, and completeness score
        """
        errors = []
        warnings = []
        
        try:
            # Basic Pydantic validation is already done at model creation
            # Additional business logic validation
            
            # Validate title
            title_validation = self._validate_title(sop.title)
            errors.extend(title_validation['errors'])
            warnings.extend(title_validation['warnings'])
            
            # Validate procedure steps
            steps_validation = self._validate_procedure_steps(sop.procedure_steps)
            errors.extend(steps_validation['errors'])
            warnings.extend(steps_validation['warnings'])
            
            # Validate risks
            risks_validation = self._validate_risks(sop.risks)
            errors.extend(risks_validation['errors'])
            warnings.extend(risks_validation['warnings'])
            
            # Validate controls
            controls_validation = self._validate_controls(sop.controls)
            errors.extend(controls_validation['errors'])
            warnings.extend(controls_validation['warnings'])
            
            # Validate cross-references
            cross_ref_validation = self._validate_cross_references(sop)
            errors.extend(cross_ref_validation['errors'])
            warnings.extend(cross_ref_validation['warnings'])
            
            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(sop)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                completeness_score=completeness_score
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed with exception: {str(e)}"],
                warnings=[],
                completeness_score=0.0
            )
    
    def _validate_title(self, title: str) -> Dict[str, List[str]]:
        """Validate document title."""
        errors = []
        warnings = []
        
        if len(title) < self.validation_rules['min_title_length']:
            errors.append(f"Title too short (minimum {self.validation_rules['min_title_length']} characters)")
        
        if len(title) > self.validation_rules['max_title_length']:
            warnings.append(f"Title very long (over {self.validation_rules['max_title_length']} characters)")
        
        # Check for common SOP title patterns
        sop_patterns = [r'sop', r'standard operating procedure', r'work instruction', r'procedure']
        if not any(re.search(pattern, title.lower()) for pattern in sop_patterns):
            warnings.append("Title doesn't contain typical SOP keywords")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_procedure_steps(self, steps: List[ProcedureStep]) -> Dict[str, List[str]]:
        """Validate procedure steps."""
        errors = []
        warnings = []
        
        if len(steps) < self.validation_rules['min_procedure_steps']:
            errors.append(f"Too few procedure steps (minimum {self.validation_rules['min_procedure_steps']})")
        
        step_ids = set()
        for i, step in enumerate(steps):
            # Check for duplicate step IDs
            if step.step_id in step_ids:
                errors.append(f"Duplicate step ID: {step.step_id}")
            step_ids.add(step.step_id)
            
            # Validate step ID depth
            depth = len(step.step_id.split('.'))
            if depth > self.validation_rules['max_step_id_depth']:
                warnings.append(f"Step {step.step_id} has deep nesting (depth {depth})")
            
            # Check required fields
            for field in self.validation_rules['required_step_fields']:
                if not getattr(step, field, None):
                    errors.append(f"Step {step.step_id} missing required field: {field}")
            
            # Validate step description length
            if len(step.description) < 10:
                warnings.append(f"Step {step.step_id} has very short description")
            
            # Check for safety-related steps without safety notes
            safety_keywords = ['safety', 'hazard', 'danger', 'caution', 'warning']
            if any(keyword in step.description.lower() for keyword in safety_keywords):
                if not step.safety_notes:
                    warnings.append(f"Step {step.step_id} mentions safety but has no safety notes")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_risks(self, risks: List[Risk]) -> Dict[str, List[str]]:
        """Validate risks."""
        errors = []
        warnings = []
        
        risk_ids = set()
        for risk in risks:
            # Check for duplicate risk IDs
            if risk.risk_id in risk_ids:
                errors.append(f"Duplicate risk ID: {risk.risk_id}")
            risk_ids.add(risk.risk_id)
            
            # Check required fields
            for field in self.validation_rules['required_risk_fields']:
                if not getattr(risk, field, None):
                    errors.append(f"Risk {risk.risk_id} missing required field: {field}")
            
            # Validate risk rating consistency
            if risk.probability and risk.severity and not risk.overall_rating:
                warnings.append(f"Risk {risk.risk_id} has probability and severity but no overall rating")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_controls(self, controls: List[Control]) -> Dict[str, List[str]]:
        """Validate controls."""
        errors = []
        warnings = []
        
        control_ids = set()
        for control in controls:
            # Check for duplicate control IDs
            if control.control_id in control_ids:
                errors.append(f"Duplicate control ID: {control.control_id}")
            control_ids.add(control.control_id)
            
            # Check required fields
            for field in self.validation_rules['required_control_fields']:
                if not getattr(control, field, None):
                    errors.append(f"Control {control.control_id} missing required field: {field}")
            
            # Check if control addresses any risks
            if not control.applicable_risks:
                warnings.append(f"Control {control.control_id} doesn't address any specific risks")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_cross_references(self, sop: SOPDocument) -> Dict[str, List[str]]:
        """Validate cross-references between different sections."""
        errors = []
        warnings = []
        
        # Collect all valid IDs
        step_ids = {step.step_id for step in sop.procedure_steps}
        risk_ids = {risk.risk_id for risk in sop.risks}
        control_ids = {control.control_id for control in sop.controls}
        role_names = {role.role for role in sop.roles_responsibilities}
        
        # Validate risk references in steps
        for step in sop.procedure_steps:
            # Check if referenced roles exist
            for role in step.responsible_roles:
                if role not in role_names and role_names:  # Only warn if we have roles defined
                    warnings.append(f"Step {step.step_id} references undefined role: {role}")
        
        # Validate step references in risks
        for risk in sop.risks:
            for step_id in risk.affected_steps:
                if step_id not in step_ids:
                    errors.append(f"Risk {risk.risk_id} references non-existent step: {step_id}")
        
        # Validate risk references in controls
        for control in sop.controls:
            for risk_id in control.applicable_risks:
                if risk_id not in risk_ids:
                    errors.append(f"Control {control.control_id} references non-existent risk: {risk_id}")
            
            for step_id in control.applicable_steps:
                if step_id not in step_ids:
                    errors.append(f"Control {control.control_id} references non-existent step: {step_id}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _calculate_completeness_score(self, sop: SOPDocument) -> float:
        """Calculate completeness score based on field population."""
        total_score = 0.0
        max_score = 0.0
        
        # Critical fields (weight: 3)
        for field in self.CRITICAL_FIELDS:
            max_score += 3.0
            value = getattr(sop, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    total_score += 3.0
                elif isinstance(value, str) and value.strip():
                    total_score += 3.0
                elif value is not None:
                    total_score += 3.0
        
        # Important fields (weight: 2)
        for field in self.IMPORTANT_FIELDS:
            max_score += 2.0
            value = getattr(sop, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    total_score += 2.0
                elif isinstance(value, str) and value.strip():
                    total_score += 2.0
                elif value is not None:
                    total_score += 2.0
        
        # Optional fields (weight: 1)
        for field in self.OPTIONAL_FIELDS:
            max_score += 1.0
            value = getattr(sop, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    total_score += 1.0
                elif isinstance(value, str) and value.strip():
                    total_score += 1.0
                elif value is not None:
                    total_score += 1.0
        
        return total_score / max_score if max_score > 0 else 0.0
    
    def validate_extraction_result(self, result: ExtractionResult) -> ValidationResult:
        """Validate an extraction result."""
        errors = []
        warnings = []
        
        if not result.success:
            if not result.errors:
                errors.append("Extraction marked as failed but no errors provided")
        
        if result.sop_document:
            sop_validation = self.validate_sop_document(result.sop_document)
            errors.extend(sop_validation.errors)
            warnings.extend(sop_validation.warnings)
        elif result.success:
            errors.append("Extraction marked as successful but no SOP document provided")
        
        # Validate chunks
        chunk_validation = self._validate_chunks(result.chunks)
        errors.extend(chunk_validation['errors'])
        warnings.extend(chunk_validation['warnings'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            completeness_score=sop_validation.completeness_score if result.sop_document else 0.0
        )
    
    def _validate_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, List[str]]:
        """Validate document chunks."""
        errors = []
        warnings = []
        
        chunk_ids = set()
        for chunk in chunks:
            # Check for duplicate chunk IDs
            if chunk.chunk_id in chunk_ids:
                errors.append(f"Duplicate chunk ID: {chunk.chunk_id}")
            chunk_ids.add(chunk.chunk_id)
            
            # Validate chunk text length
            if len(chunk.chunk_text.strip()) < 10:
                warnings.append(f"Chunk {chunk.chunk_id} has very short text content")
            
            # Validate embedding dimensions if present
            if chunk.embedding:
                if not isinstance(chunk.embedding, list):
                    errors.append(f"Chunk {chunk.chunk_id} embedding must be a list")
                elif len(chunk.embedding) not in [384, 768, 1536]:  # Common embedding dimensions
                    warnings.append(f"Chunk {chunk.chunk_id} has unusual embedding dimension: {len(chunk.embedding)}")
        
        return {'errors': errors, 'warnings': warnings}


def validate_sop_schema_compliance(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate raw data against SOP schema before creating Pydantic models.
    
    Args:
        data: Raw dictionary data to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Try to create SOPDocument from data
        sop = SOPDocument(**data)
        return True, []
    except ValidationError as e:
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            errors.append(f"Field '{field_path}': {error['msg']}")
        return False, errors
    except Exception as e:
        return False, [f"Unexpected validation error: {str(e)}"]


def validate_required_extraction_fields(sop: SOPDocument, min_coverage: float = 0.8) -> ValidationResult:
    """
    Validate that extracted SOP meets minimum field coverage requirements.
    
    As per requirement 2.1: "extract at least 80% of the following fields when present"
    
    Args:
        sop: SOPDocument to validate
        min_coverage: Minimum coverage ratio (default 0.8 for 80%)
        
    Returns:
        ValidationResult indicating if minimum coverage is met
    """
    required_fields = [
        'title', 'revision', 'effective_date', 'procedure_steps',
        'risks', 'controls', 'roles_responsibilities'
    ]
    
    present_fields = 0
    total_fields = len(required_fields)
    
    for field in required_fields:
        value = getattr(sop, field, None)
        if value:
            if isinstance(value, list) and len(value) > 0:
                present_fields += 1
            elif isinstance(value, str) and value.strip():
                present_fields += 1
            elif value is not None:
                present_fields += 1
    
    coverage_ratio = present_fields / total_fields
    is_valid = coverage_ratio >= min_coverage
    
    errors = []
    warnings = []
    
    if not is_valid:
        errors.append(f"Field coverage {coverage_ratio:.1%} below minimum {min_coverage:.1%}")
    
    if coverage_ratio < 0.6:
        warnings.append("Very low field coverage - extraction quality may be poor")
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        completeness_score=coverage_ratio
    )