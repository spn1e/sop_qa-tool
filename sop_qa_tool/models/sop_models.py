"""
Pydantic models for SOP (Standard Operating Procedure) ontology.

This module defines the data structures for representing SOPs and their components
including procedure steps, risks, controls, roles, and other manufacturing-specific
elements as specified in requirements 2.2 and 2.3.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class PriorityLevel(str, Enum):
    """Priority levels for risks and controls."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StepType(str, Enum):
    """Types of procedure steps."""
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    CLEANUP = "cleanup"
    SAFETY_CHECK = "safety_check"


class RiskCategory(str, Enum):
    """Categories of risks in manufacturing processes."""
    SAFETY = "safety"
    QUALITY = "quality"
    ENVIRONMENTAL = "environmental"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"


class ControlType(str, Enum):
    """Types of controls for risk mitigation."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    ADMINISTRATIVE = "administrative"


class SourceInfo(BaseModel):
    """Information about the source document."""
    url: Optional[str] = Field(None, description="Source URL if document was fetched from web")
    file_path: Optional[str] = Field(None, description="Local file path")
    page_range: Optional[List[int]] = Field(None, description="Page range [start, end] where content was found")
    last_modified: Optional[datetime] = Field(None, description="Last modification timestamp")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    @field_validator('page_range')
    @classmethod
    def validate_page_range(cls, v):
        if v is not None and len(v) == 2 and v[0] > v[1]:
            raise ValueError("Start page must be less than or equal to end page")
        return v


class Definition(BaseModel):
    """Glossary definition for terms used in the SOP."""
    term: str = Field(..., description="The term being defined")
    definition: str = Field(..., description="Definition of the term")
    category: Optional[str] = Field(None, description="Category of the term (e.g., 'equipment', 'process')")


class RoleResponsibility(BaseModel):
    """Role and associated responsibilities in the SOP."""
    role: str = Field(..., description="Job title or role name")
    responsibilities: List[str] = Field(..., description="List of specific responsibilities")
    qualifications: Optional[List[str]] = Field(None, description="Required qualifications or certifications")
    authority_level: Optional[str] = Field(None, description="Level of authority (e.g., 'operator', 'supervisor', 'manager')")


class ProcedureStep(BaseModel):
    """Individual step in a procedure."""
    step_id: str = Field(..., description="Unique identifier for the step (e.g., '3.2.1')")
    title: Optional[str] = Field(None, description="Brief title of the step")
    description: str = Field(..., description="Detailed description of what to do")
    step_type: Optional[StepType] = Field(None, description="Type of step")
    responsible_roles: List[str] = Field(default_factory=list, description="Roles responsible for this step")
    required_equipment: List[str] = Field(default_factory=list, description="Equipment needed for this step")
    materials: List[str] = Field(default_factory=list, description="Materials or consumables needed")
    duration_minutes: Optional[int] = Field(None, description="Expected duration in minutes")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites before performing this step")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Criteria to verify step completion")
    safety_notes: List[str] = Field(default_factory=list, description="Safety considerations for this step")
    quality_checkpoints: List[str] = Field(default_factory=list, description="Quality verification points")
    
    @field_validator('step_id')
    @classmethod
    def validate_step_id(cls, v):
        # Validate step ID format (e.g., "1", "1.1", "1.1.1")
        if not re.match(r'^\d+(\.\d+)*$', v):
            raise ValueError("Step ID must follow format like '1', '1.1', or '1.1.1'")
        return v
    
    @field_validator('duration_minutes')
    @classmethod
    def validate_duration(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v


class Risk(BaseModel):
    """Risk associated with the procedure."""
    risk_id: str = Field(..., description="Unique identifier for the risk")
    description: str = Field(..., description="Description of the risk")
    category: RiskCategory = Field(..., description="Category of risk")
    probability: Optional[PriorityLevel] = Field(None, description="Likelihood of occurrence")
    severity: Optional[PriorityLevel] = Field(None, description="Severity of impact")
    overall_rating: Optional[PriorityLevel] = Field(None, description="Overall risk rating")
    affected_steps: List[str] = Field(default_factory=list, description="Step IDs where this risk applies")
    potential_consequences: List[str] = Field(default_factory=list, description="Potential consequences if risk occurs")
    triggers: List[str] = Field(default_factory=list, description="Conditions that could trigger this risk")


class Control(BaseModel):
    """Control measure to mitigate risks."""
    control_id: str = Field(..., description="Unique identifier for the control")
    description: str = Field(..., description="Description of the control measure")
    control_type: ControlType = Field(..., description="Type of control")
    effectiveness: Optional[PriorityLevel] = Field(None, description="Effectiveness rating")
    applicable_risks: List[str] = Field(default_factory=list, description="Risk IDs this control addresses")
    applicable_steps: List[str] = Field(default_factory=list, description="Step IDs where this control applies")
    responsible_roles: List[str] = Field(default_factory=list, description="Roles responsible for implementing control")
    verification_method: Optional[str] = Field(None, description="How to verify control is working")
    frequency: Optional[str] = Field(None, description="How often control should be applied/checked")


class ChangeLogEntry(BaseModel):
    """Entry in the document change log."""
    version: str = Field(..., description="Version number")
    date: datetime = Field(..., description="Date of change")
    author: str = Field(..., description="Person who made the change")
    description: str = Field(..., description="Description of what changed")
    approval_status: Optional[str] = Field(None, description="Approval status of the change")


class SOPDocument(BaseModel):
    """Complete SOP document with all structured information."""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    process_name: str = Field(..., description="Name of the process this SOP covers")
    revision: Optional[str] = Field(None, description="Document revision number")
    effective_date: Optional[datetime] = Field(None, description="Date when this version becomes effective")
    expiry_date: Optional[datetime] = Field(None, description="Date when this version expires")
    owner_role: Optional[str] = Field(None, description="Role responsible for maintaining this SOP")
    scope: Optional[str] = Field(None, description="Scope and applicability of the SOP")
    
    # Content sections
    definitions_glossary: List[Definition] = Field(default_factory=list, description="Definitions and glossary terms")
    preconditions: List[str] = Field(default_factory=list, description="Conditions that must be met before starting")
    materials_equipment: List[str] = Field(default_factory=list, description="Required materials and equipment")
    roles_responsibilities: List[RoleResponsibility] = Field(default_factory=list, description="Roles and their responsibilities")
    procedure_steps: List[ProcedureStep] = Field(default_factory=list, description="Detailed procedure steps")
    risks: List[Risk] = Field(default_factory=list, description="Identified risks")
    controls: List[Control] = Field(default_factory=list, description="Control measures")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Overall acceptance criteria")
    compliance_refs: List[str] = Field(default_factory=list, description="Compliance references (standards, regulations)")
    attachments_refs: List[str] = Field(default_factory=list, description="References to attachments or related documents")
    change_log: List[ChangeLogEntry] = Field(default_factory=list, description="Document change history")
    
    # Metadata
    source: SourceInfo = Field(..., description="Information about the source document")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this data was extracted")
    extraction_confidence: Optional[float] = Field(None, description="Confidence score for extraction quality (0-1)")
    
    @field_validator('extraction_confidence')
    @classmethod
    def validate_confidence(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("Confidence score must be between 0 and 1")
        return v
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.effective_date and self.expiry_date and self.effective_date >= self.expiry_date:
            raise ValueError("Effective date must be before expiry date")
        return self
    
    def get_steps_by_role(self, role: str) -> List[ProcedureStep]:
        """Get all procedure steps assigned to a specific role."""
        return [step for step in self.procedure_steps if role in step.responsible_roles]
    
    def get_risks_by_category(self, category: RiskCategory) -> List[Risk]:
        """Get all risks of a specific category."""
        return [risk for risk in self.risks if risk.category == category]
    
    def get_high_priority_risks(self) -> List[Risk]:
        """Get all high or critical priority risks."""
        return [risk for risk in self.risks 
                if risk.overall_rating in [PriorityLevel.HIGH, PriorityLevel.CRITICAL]]
    
    def get_controls_for_risk(self, risk_id: str) -> List[Control]:
        """Get all controls that address a specific risk."""
        return [control for control in self.controls if risk_id in control.applicable_risks]


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document for vector storage."""
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    doc_id: str = Field(..., description="ID of the parent document")
    chunk_text: str = Field(..., description="The actual text content")
    chunk_index: int = Field(..., description="Sequential index of this chunk in the document")
    
    # Metadata for enhanced retrieval
    page_no: Optional[int] = Field(None, description="Page number where this chunk appears")
    heading_path: Optional[str] = Field(None, description="Hierarchical path of headings (e.g., '3. Process > 3.2 Setup')")
    step_ids: List[str] = Field(default_factory=list, description="Step IDs referenced in this chunk")
    risk_ids: List[str] = Field(default_factory=list, description="Risk IDs referenced in this chunk")
    control_ids: List[str] = Field(default_factory=list, description="Control IDs referenced in this chunk")
    roles: List[str] = Field(default_factory=list, description="Roles mentioned in this chunk")
    equipment: List[str] = Field(default_factory=list, description="Equipment mentioned in this chunk")
    
    # Vector storage metadata
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for this chunk")
    embedding_model: Optional[str] = Field(None, description="Model used to generate embedding")
    
    @field_validator('chunk_index')
    @classmethod
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v


class ExtractionResult(BaseModel):
    """Result of SOP extraction process."""
    success: bool = Field(..., description="Whether extraction was successful")
    sop_document: Optional[SOPDocument] = Field(None, description="Extracted SOP document")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Text chunks for vector storage")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during extraction")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for extraction")
    
    @field_validator('processing_time_seconds')
    @classmethod
    def validate_processing_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


class ValidationResult(BaseModel):
    """Result of schema validation."""
    is_valid: bool = Field(..., description="Whether the data passes validation")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    completeness_score: Optional[float] = Field(None, description="Completeness score (0-1)")
    
    @field_validator('completeness_score')
    @classmethod
    def validate_completeness(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("Completeness score must be between 0 and 1")
        return v

class GoldenDatasetItem(BaseModel):
    """Item in the golden dataset for evaluation."""
    question: str = Field(..., description="Test question")
    expected_answer: str = Field(..., description="Expected/ground truth answer")
    category: str = Field(..., description="Question category (e.g., 'safety', 'procedure', 'equipment')")
    difficulty: str = Field(default="medium", description="Question difficulty level")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply during retrieval")
    source_documents: List[str] = Field(default_factory=list, description="Expected source document IDs")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the question")


class EvaluationMetric(BaseModel):
    """Individual evaluation metric result."""
    score: float = Field(..., description="Metric score (0.0 to 1.0)")
    threshold: float = Field(..., description="Target threshold for this metric")
    passed: bool = Field(..., description="Whether the metric passed the threshold")


class EvaluationResult(BaseModel):
    """Complete evaluation result with RAGAS metrics."""
    timestamp: datetime = Field(..., description="When the evaluation was run")
    dataset_size: int = Field(..., description="Number of items in the evaluation dataset")
    evaluation_time_seconds: float = Field(..., description="Time taken to run evaluation")
    metrics: Dict[str, EvaluationMetric] = Field(..., description="Individual metric results")
    overall_pass_rate: float = Field(..., description="Percentage of metrics that passed thresholds")
    raw_results: Dict[str, Any] = Field(..., description="Raw RAGAS evaluation results")


class BenchmarkResult(BaseModel):
    """Performance benchmark results."""
    timestamp: datetime = Field(..., description="When the benchmark was run")
    test_queries_count: int = Field(..., description="Number of test queries used")
    concurrent_users_tested: List[int] = Field(..., description="List of concurrent user counts tested")
    iterations_per_test: int = Field(..., description="Number of iterations per test configuration")
    results: Dict[str, Any] = Field(..., description="Detailed benchmark results")


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report combining metrics and benchmarks."""
    evaluation_result: EvaluationResult = Field(..., description="RAGAS evaluation results")
    benchmark_result: BenchmarkResult = Field(..., description="Performance benchmark results")
    summary: Dict[str, Any] = Field(..., description="Executive summary of results")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")