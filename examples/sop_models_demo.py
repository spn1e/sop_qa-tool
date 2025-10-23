#!/usr/bin/env python3
"""
Demonstration script for SOP data models and validation.

This script shows how to create, validate, and work with SOP documents
using the Pydantic models defined in the sop_qa_tool.models module.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the project root to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sop_qa_tool.models import (
    SOPDocument,
    ProcedureStep,
    Risk,
    Control,
    RoleResponsibility,
    Definition,
    SourceInfo,
    DocumentChunk,
    ExtractionResult,
    PriorityLevel,
    StepType,
    RiskCategory,
    ControlType,
    SOPValidator,
    validate_sop_schema_compliance,
    validate_required_extraction_fields
)


def create_sample_sop() -> SOPDocument:
    """Create a comprehensive sample SOP document."""
    
    # Create source information
    source = SourceInfo(
        url="https://factory.example.com/sops/bottle-filling-v1.2.pdf",
        page_range=[1, 15],
        last_modified=datetime.utcnow() - timedelta(days=30),
        file_size=2048576  # 2MB
    )
    
    # Define roles and responsibilities
    roles = [
        RoleResponsibility(
            role="Production Operator",
            responsibilities=[
                "Operate bottle filling equipment",
                "Monitor filling process parameters",
                "Perform routine quality checks",
                "Document production data"
            ],
            qualifications=["Basic machine operation training", "Safety certification"],
            authority_level="operator"
        ),
        RoleResponsibility(
            role="QA Inspector",
            responsibilities=[
                "Verify product quality standards",
                "Conduct sampling and testing",
                "Review production documentation",
                "Approve batch release"
            ],
            qualifications=["Quality assurance certification", "Laboratory training"],
            authority_level="supervisor"
        ),
        RoleResponsibility(
            role="Line Supervisor",
            responsibilities=[
                "Oversee production operations",
                "Coordinate with maintenance",
                "Handle non-conformances",
                "Ensure safety compliance"
            ],
            qualifications=["Supervisory training", "Process knowledge certification"],
            authority_level="supervisor"
        )
    ]
    
    # Define procedure steps
    steps = [
        ProcedureStep(
            step_id="1.1",
            title="Pre-operation Setup",
            description="Verify all equipment is clean, calibrated, and ready for operation. Check that all safety systems are functional.",
            step_type=StepType.PREPARATION,
            responsible_roles=["Production Operator"],
            required_equipment=["Filling Machine FM-001", "Conveyor System", "Safety Interlocks"],
            materials=["Cleaning Solution", "Calibration Standards"],
            duration_minutes=15,
            prerequisites=["Equipment maintenance completed", "Cleaning verification passed"],
            acceptance_criteria=[
                "All equipment status lights are green",
                "Calibration certificates are current",
                "Safety interlocks test successfully"
            ],
            safety_notes=[
                "Ensure LOTO procedures are followed",
                "Verify emergency stop functionality"
            ]
        ),
        ProcedureStep(
            step_id="1.2",
            title="Product Setup",
            description="Configure filling parameters for the specific product being processed. Load product specifications and verify settings.",
            step_type=StepType.PREPARATION,
            responsible_roles=["Production Operator", "QA Inspector"],
            required_equipment=["HMI Terminal", "Recipe Database"],
            duration_minutes=10,
            acceptance_criteria=[
                "Correct recipe loaded and verified",
                "Fill volume within specification",
                "Product changeover documented"
            ]
        ),
        ProcedureStep(
            step_id="2.1",
            title="Start Production",
            description="Begin filling operation following established startup sequence. Monitor initial production for quality compliance.",
            step_type=StepType.EXECUTION,
            responsible_roles=["Production Operator"],
            required_equipment=["Filling Machine FM-001"],
            duration_minutes=5,
            acceptance_criteria=[
                "Smooth startup achieved",
                "Initial samples pass quality check",
                "Production rate within target range"
            ],
            quality_checkpoints=[
                "First 10 bottles inspected for fill level",
                "Cap torque verification"
            ]
        ),
        ProcedureStep(
            step_id="2.2",
            title="Continuous Monitoring",
            description="Monitor production parameters continuously. Perform periodic quality checks and document results.",
            step_type=StepType.EXECUTION,
            responsible_roles=["Production Operator", "QA Inspector"],
            required_equipment=["Process Monitoring System", "Quality Testing Equipment"],
            acceptance_criteria=[
                "All parameters within control limits",
                "Quality checks pass acceptance criteria",
                "Documentation complete and accurate"
            ],
            quality_checkpoints=[
                "Hourly fill weight verification",
                "Visual inspection every 30 minutes",
                "Cap integrity testing"
            ]
        ),
        ProcedureStep(
            step_id="3.1",
            title="End of Run Procedures",
            description="Complete production run following proper shutdown sequence. Secure equipment and complete documentation.",
            step_type=StepType.CLEANUP,
            responsible_roles=["Production Operator", "Line Supervisor"],
            required_equipment=["Cleaning System"],
            duration_minutes=20,
            acceptance_criteria=[
                "Equipment properly shut down",
                "Cleaning cycle completed",
                "All documentation submitted"
            ]
        )
    ]
    
    # Define risks
    risks = [
        Risk(
            risk_id="R-001",
            description="Contamination of product during filling process",
            category=RiskCategory.QUALITY,
            probability=PriorityLevel.MEDIUM,
            severity=PriorityLevel.HIGH,
            overall_rating=PriorityLevel.HIGH,
            affected_steps=["2.1", "2.2"],
            potential_consequences=[
                "Product recall",
                "Customer complaints",
                "Regulatory action",
                "Brand damage"
            ],
            triggers=[
                "Equipment malfunction",
                "Inadequate cleaning",
                "Environmental contamination"
            ]
        ),
        Risk(
            risk_id="R-002",
            description="Operator injury from moving equipment",
            category=RiskCategory.SAFETY,
            probability=PriorityLevel.LOW,
            severity=PriorityLevel.CRITICAL,
            overall_rating=PriorityLevel.HIGH,
            affected_steps=["1.1", "2.1", "3.1"],
            potential_consequences=[
                "Personal injury",
                "Production shutdown",
                "Regulatory investigation",
                "Workers compensation claims"
            ],
            triggers=[
                "Bypassed safety systems",
                "Inadequate training",
                "Equipment malfunction"
            ]
        ),
        Risk(
            risk_id="R-003",
            description="Incorrect fill volume leading to customer complaints",
            category=RiskCategory.QUALITY,
            probability=PriorityLevel.MEDIUM,
            severity=PriorityLevel.MEDIUM,
            overall_rating=PriorityLevel.MEDIUM,
            affected_steps=["1.2", "2.1", "2.2"],
            potential_consequences=[
                "Customer dissatisfaction",
                "Regulatory non-compliance",
                "Product rework"
            ]
        )
    ]
    
    # Define controls
    controls = [
        Control(
            control_id="C-001",
            description="Preventive maintenance program for filling equipment",
            control_type=ControlType.PREVENTIVE,
            effectiveness=PriorityLevel.HIGH,
            applicable_risks=["R-001", "R-002"],
            applicable_steps=["1.1"],
            responsible_roles=["Line Supervisor"],
            verification_method="Maintenance records review",
            frequency="Weekly"
        ),
        Control(
            control_id="C-002",
            description="Safety interlock system prevents operation with guards open",
            control_type=ControlType.PREVENTIVE,
            effectiveness=PriorityLevel.HIGH,
            applicable_risks=["R-002"],
            applicable_steps=["1.1", "2.1"],
            responsible_roles=["Production Operator"],
            verification_method="Functional testing",
            frequency="Daily"
        ),
        Control(
            control_id="C-003",
            description="Statistical process control for fill weight monitoring",
            control_type=ControlType.DETECTIVE,
            effectiveness=PriorityLevel.MEDIUM,
            applicable_risks=["R-003"],
            applicable_steps=["2.2"],
            responsible_roles=["QA Inspector"],
            verification_method="Control chart review",
            frequency="Hourly"
        )
    ]
    
    # Create the complete SOP document
    sop = SOPDocument(
        doc_id="SOP-FILL-001",
        title="Bottle Filling Standard Operating Procedure",
        process_name="Automated Bottle Filling",
        revision="1.2",
        effective_date=datetime.utcnow() - timedelta(days=30),
        expiry_date=datetime.utcnow() + timedelta(days=335),  # 1 year from effective
        owner_role="Production Manager",
        scope="Applies to all automated bottle filling operations on Line 1 and Line 2",
        
        definitions_glossary=[
            Definition(
                term="LOTO",
                definition="Lockout/Tagout - Safety procedure to ensure equipment is properly shut off",
                category="safety"
            ),
            Definition(
                term="HMI",
                definition="Human Machine Interface - Control panel for equipment operation",
                category="equipment"
            )
        ],
        
        preconditions=[
            "Equipment maintenance is current",
            "Operators are trained and certified",
            "Product specifications are available",
            "Quality testing equipment is calibrated"
        ],
        
        materials_equipment=[
            "Filling Machine FM-001",
            "Conveyor System CS-001",
            "HMI Terminal",
            "Quality Testing Equipment",
            "Cleaning Solution Type A",
            "Calibration Standards"
        ],
        
        roles_responsibilities=roles,
        procedure_steps=steps,
        risks=risks,
        controls=controls,
        
        acceptance_criteria=[
            "All filled bottles meet volume specifications (±2%)",
            "No contamination detected in quality testing",
            "Production rate achieves target efficiency (>95%)",
            "All safety systems function properly",
            "Documentation is complete and accurate"
        ],
        
        compliance_refs=[
            "FDA 21 CFR Part 110 - Good Manufacturing Practices",
            "ISO 22000:2018 - Food Safety Management",
            "OSHA 29 CFR 1910 - Occupational Safety Standards"
        ],
        
        attachments_refs=[
            "Attachment A: Equipment Specifications",
            "Attachment B: Quality Testing Procedures",
            "Attachment C: Emergency Response Procedures"
        ],
        
        source=source,
        extraction_confidence=0.95
    )
    
    return sop


def create_sample_chunks(sop: SOPDocument) -> list[DocumentChunk]:
    """Create sample document chunks for the SOP."""
    
    chunks = [
        DocumentChunk(
            chunk_id=f"{sop.doc_id}_chunk_001",
            doc_id=sop.doc_id,
            chunk_text="This Standard Operating Procedure covers the automated bottle filling process for production lines 1 and 2. The procedure ensures consistent product quality while maintaining safety standards.",
            chunk_index=0,
            page_no=1,
            heading_path="1. Introduction > 1.1 Purpose",
            roles=["Production Operator", "QA Inspector"],
            equipment=["Filling Machine FM-001"]
        ),
        DocumentChunk(
            chunk_id=f"{sop.doc_id}_chunk_002",
            doc_id=sop.doc_id,
            chunk_text="Step 1.1: Pre-operation Setup - Verify all equipment is clean, calibrated, and ready for operation. Check that all safety systems are functional. Duration: 15 minutes.",
            chunk_index=1,
            page_no=3,
            heading_path="2. Procedure > 2.1 Setup Phase",
            step_ids=["1.1"],
            roles=["Production Operator"],
            equipment=["Filling Machine FM-001", "Conveyor System", "Safety Interlocks"]
        ),
        DocumentChunk(
            chunk_id=f"{sop.doc_id}_chunk_003",
            doc_id=sop.doc_id,
            chunk_text="Risk R-001: Contamination of product during filling process. This is a high-priority risk that could lead to product recall and customer complaints. Controlled by preventive maintenance program C-001.",
            chunk_index=2,
            page_no=8,
            heading_path="4. Risk Assessment > 4.1 Quality Risks",
            risk_ids=["R-001"],
            control_ids=["C-001"]
        )
    ]
    
    return chunks


def demonstrate_validation():
    """Demonstrate the validation capabilities."""
    
    print("=== SOP Models Validation Demo ===\n")
    
    # Create sample SOP
    print("1. Creating sample SOP document...")
    sop = create_sample_sop()
    print(f"   Created SOP: {sop.title}")
    print(f"   Document ID: {sop.doc_id}")
    print(f"   Procedure steps: {len(sop.procedure_steps)}")
    print(f"   Risks identified: {len(sop.risks)}")
    print(f"   Controls defined: {len(sop.controls)}")
    
    # Validate the SOP
    print("\n2. Validating SOP document...")
    validator = SOPValidator()
    validation_result = validator.validate_sop_document(sop)
    
    print(f"   Validation result: {'PASS' if validation_result.is_valid else 'FAIL'}")
    print(f"   Completeness score: {validation_result.completeness_score:.2%}")
    print(f"   Errors: {len(validation_result.errors)}")
    print(f"   Warnings: {len(validation_result.warnings)}")
    
    if validation_result.warnings:
        print("   Warnings:")
        for warning in validation_result.warnings:
            print(f"     - {warning}")
    
    # Test helper methods
    print("\n3. Testing SOP helper methods...")
    operator_steps = sop.get_steps_by_role("Production Operator")
    print(f"   Steps for Production Operator: {len(operator_steps)}")
    
    safety_risks = sop.get_risks_by_category(RiskCategory.SAFETY)
    print(f"   Safety risks: {len(safety_risks)}")
    
    high_risks = sop.get_high_priority_risks()
    print(f"   High priority risks: {len(high_risks)}")
    
    # Create and validate chunks
    print("\n4. Creating and validating document chunks...")
    chunks = create_sample_chunks(sop)
    
    extraction_result = ExtractionResult(
        success=True,
        sop_document=sop,
        chunks=chunks,
        processing_time_seconds=2.5
    )
    
    extraction_validation = validator.validate_extraction_result(extraction_result)
    print(f"   Extraction validation: {'PASS' if extraction_validation.is_valid else 'FAIL'}")
    print(f"   Chunks created: {len(chunks)}")
    
    # Test schema compliance
    print("\n5. Testing schema compliance...")
    sop_dict = sop.model_dump()
    is_compliant, errors = validate_sop_schema_compliance(sop_dict)
    print(f"   Schema compliance: {'PASS' if is_compliant else 'FAIL'}")
    
    # Test required field coverage
    print("\n6. Testing required field coverage...")
    coverage_result = validate_required_extraction_fields(sop, min_coverage=0.8)
    print(f"   Field coverage: {'PASS' if coverage_result.is_valid else 'FAIL'}")
    print(f"   Coverage score: {coverage_result.completeness_score:.2%}")
    
    # Demonstrate JSON serialization
    print("\n7. JSON serialization test...")
    json_str = sop.model_dump_json(indent=2)
    print(f"   JSON size: {len(json_str):,} characters")
    
    # Test deserialization
    sop_from_json = SOPDocument.model_validate_json(json_str)
    print(f"   Deserialization: {'SUCCESS' if sop_from_json.doc_id == sop.doc_id else 'FAILED'}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_validation()