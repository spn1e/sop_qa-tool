"""
Ontology Extraction Demo

This script demonstrates the ontology extraction capabilities of the SOP Q&A tool,
showing how structured information is extracted from SOP documents using both
AWS Bedrock and local fallback methods.

Requirements: 2.1, 2.2, 2.3, 7.2
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from sop_qa_tool.services.ontology_extractor import OntologyExtractor
from sop_qa_tool.services.summarizer import SOPSummarizer
from sop_qa_tool.models.sop_models import SourceInfo, DocumentChunk
from sop_qa_tool.config.settings import get_settings


def create_sample_sop_text():
    """Create a comprehensive sample SOP text for demonstration."""
    return """
    SOP-WELD-001: Arc Welding Safety Procedure
    Revision: 3.1
    Effective Date: 2024-01-15
    Owner: Safety Manager
    Scope: This procedure applies to all arc welding operations in the manufacturing facility.
    
    1. PURPOSE AND SCOPE
    This Standard Operating Procedure (SOP) establishes safety requirements and procedures 
    for arc welding operations to protect personnel from electrical shock, burns, fumes, 
    and fire hazards.
    
    2. DEFINITIONS
    Arc Welding: A welding process that uses an electric arc to create heat to melt and join metals.
    PPE: Personal Protective Equipment required for safe welding operations.
    Hot Work Permit: Authorization required for welding operations in designated areas.
    
    3. ROLES AND RESPONSIBILITIES
    3.1 Certified Welder
    - Perform welding operations according to specifications
    - Conduct pre-welding safety inspections
    - Maintain welding equipment in safe condition
    - Report safety hazards immediately
    
    3.2 Safety Inspector
    - Issue hot work permits
    - Verify PPE compliance
    - Monitor welding operations for safety compliance
    - Investigate welding-related incidents
    
    3.3 Maintenance Technician
    - Perform routine maintenance on welding equipment
    - Calibrate welding machines quarterly
    - Replace worn or damaged components
    
    4. REQUIRED EQUIPMENT AND MATERIALS
    - Arc welding machine (Model WM-500 or equivalent)
    - Welding electrodes (E7018 classification)
    - Welding helmet with auto-darkening filter (Shade 10-13)
    - Leather welding gloves
    - Fire-resistant welding jacket
    - Safety boots with metatarsal guards
    - Fire extinguisher (Class C minimum)
    - Ventilation system or portable fume extractor
    
    5. PROCEDURE STEPS
    5.1 Pre-Welding Preparation
    5.1.1 Obtain hot work permit from Safety Inspector
    5.1.2 Inspect welding area for fire hazards within 35-foot radius
    5.1.3 Position fire extinguisher within 10 feet of work area
    5.1.4 Verify adequate ventilation (minimum 2000 CFM for enclosed spaces)
    5.1.5 Test welding machine for proper grounding (resistance < 1 ohm)
    
    5.2 Personal Protective Equipment
    5.2.1 Don fire-resistant welding jacket and pants
    5.2.2 Put on leather welding gloves (minimum 14-inch cuff length)
    5.2.3 Wear safety boots with metatarsal protection
    5.2.4 Use welding helmet with appropriate shade filter (10-13)
    5.2.5 Ensure no exposed skin areas
    
    5.3 Welding Operations
    5.3.1 Set welding parameters according to material specifications
    5.3.2 Maintain arc length of 1/8 to 1/4 inch
    5.3.3 Monitor weld pool temperature (1500-1800°F optimal)
    5.3.4 Inspect each weld pass for defects before proceeding
    5.3.5 Allow cooling time between passes (minimum 2 minutes)
    
    5.4 Post-Welding Activities
    5.4.1 Allow workpiece to cool completely (below 100°F)
    5.4.2 Clean welding area of slag and spatter
    5.4.3 Conduct final visual inspection of completed welds
    5.4.4 Document welding parameters and inspection results
    5.4.5 Return hot work permit to Safety Inspector
    
    6. SAFETY RISKS AND HAZARDS
    Risk R-001: Electrical shock from faulty equipment or improper grounding
    - Probability: Medium
    - Severity: Critical
    - Affected Steps: 5.1.5, 5.3.1-5.3.5
    - Consequences: Serious injury or death, equipment damage
    
    Risk R-002: Burns from hot metal, sparks, or UV radiation
    - Probability: High
    - Severity: High  
    - Affected Steps: 5.2.1-5.2.5, 5.3.1-5.3.5, 5.4.1
    - Consequences: First to third-degree burns, eye damage
    
    Risk R-003: Fire or explosion from combustible materials
    - Probability: Medium
    - Severity: Critical
    - Affected Steps: 5.1.2, 5.1.3, 5.3.1-5.3.5
    - Consequences: Property damage, injuries, fatalities
    
    Risk R-004: Inhalation of toxic welding fumes
    - Probability: High
    - Severity: Medium
    - Affected Steps: 5.1.4, 5.3.1-5.3.5
    - Consequences: Respiratory illness, long-term health effects
    
    7. CONTROL MEASURES
    Control C-001: Equipment grounding and electrical safety inspection
    - Type: Preventive
    - Applicable Risks: R-001
    - Responsible Role: Maintenance Technician
    - Frequency: Before each use and monthly
    - Verification: Resistance measurement < 1 ohm
    
    Control C-002: Personal protective equipment requirements
    - Type: Preventive
    - Applicable Risks: R-002
    - Responsible Role: Certified Welder, Safety Inspector
    - Frequency: Every welding operation
    - Verification: Visual inspection and compliance check
    
    Control C-003: Fire prevention and suppression measures
    - Type: Preventive/Detective
    - Applicable Risks: R-003
    - Responsible Role: Certified Welder, Safety Inspector
    - Frequency: Before and during welding operations
    - Verification: Fire watch and extinguisher placement
    
    Control C-004: Ventilation and fume extraction
    - Type: Preventive
    - Applicable Risks: R-004
    - Responsible Role: Maintenance Technician
    - Frequency: Continuous during welding
    - Verification: Airflow measurement ≥ 2000 CFM
    
    8. QUALITY CHECKPOINTS
    - Visual inspection of weld appearance and penetration
    - Dimensional verification of weld size and profile
    - Non-destructive testing (dye penetrant or magnetic particle)
    - Documentation of welding parameters and consumables used
    
    9. COMPLIANCE REFERENCES
    - OSHA 29 CFR 1910.252 - General requirements for welding
    - AWS D1.1 - Structural Welding Code - Steel
    - ANSI Z49.1 - Safety in Welding, Cutting, and Allied Processes
    - Company Safety Manual Section 4.3 - Hot Work Operations
    
    10. CHANGE LOG
    Version 3.1 - 2024-01-15 - Safety Manager - Added fume extraction requirements
    Version 3.0 - 2023-08-20 - Safety Manager - Updated PPE specifications
    Version 2.5 - 2023-03-10 - Production Manager - Revised welding parameters
    """


def create_sample_chunks():
    """Create sample document chunks for multi-chunk extraction demo."""
    full_text = create_sample_sop_text()
    sections = full_text.split('\n\n')
    
    chunks = []
    for i, section in enumerate(sections[:5]):  # Use first 5 sections
        if section.strip():
            chunk = DocumentChunk(
                chunk_id=f"weld_sop_chunk_{i}",
                doc_id="SOP-WELD-001",
                chunk_text=section.strip(),
                chunk_index=i,
                page_no=i // 2 + 1,  # Simulate page numbers
                heading_path=f"Section {i+1}",
                step_ids=[],
                roles=[],
                equipment=[]
            )
            chunks.append(chunk)
    
    return chunks


def demo_single_text_extraction():
    """Demonstrate extraction from a single text document."""
    print("=" * 80)
    print("DEMO 1: Single Text Extraction")
    print("=" * 80)
    
    # Initialize extractor
    extractor = OntologyExtractor()
    
    # Create sample data
    sop_text = create_sample_sop_text()
    source_info = SourceInfo(
        url="https://company.com/sops/SOP-WELD-001.pdf",
        page_range=[1, 8],
        last_modified=datetime.utcnow(),
        file_size=45000
    )
    
    print("Extracting structured information from SOP text...")
    print(f"Text length: {len(sop_text)} characters")
    print(f"Source: {source_info.url}")
    print()
    
    # Perform extraction
    result = extractor.extract_from_text(sop_text, "SOP-WELD-001", source_info)
    
    # Display results
    if result.success:
        sop_doc = result.sop_document
        print("✅ Extraction successful!")
        print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
        print()
        
        print("📋 Document Information:")
        print(f"  Title: {sop_doc.title}")
        print(f"  Process: {sop_doc.process_name}")
        print(f"  Revision: {sop_doc.revision}")
        print(f"  Owner: {sop_doc.owner_role}")
        print(f"  Scope: {sop_doc.scope[:100]}..." if sop_doc.scope else "  Scope: Not specified")
        print()
        
        print("👥 Roles and Responsibilities:")
        for role in sop_doc.roles_responsibilities:
            print(f"  • {role.role}: {len(role.responsibilities)} responsibilities")
            if role.qualifications:
                print(f"    Qualifications: {', '.join(role.qualifications)}")
        print()
        
        print("🔧 Equipment and Materials:")
        for equipment in sop_doc.materials_equipment[:10]:  # Show first 10
            print(f"  • {equipment}")
        if len(sop_doc.materials_equipment) > 10:
            print(f"  ... and {len(sop_doc.materials_equipment) - 10} more items")
        print()
        
        print("📝 Procedure Steps:")
        for step in sop_doc.procedure_steps[:8]:  # Show first 8 steps
            print(f"  {step.step_id}: {step.description[:60]}...")
            if step.responsible_roles:
                print(f"    👤 Responsible: {', '.join(step.responsible_roles)}")
            if step.safety_notes:
                print(f"    ⚠️  Safety: {len(step.safety_notes)} notes")
        if len(sop_doc.procedure_steps) > 8:
            print(f"  ... and {len(sop_doc.procedure_steps) - 8} more steps")
        print()
        
        print("⚠️  Risks Identified:")
        for risk in sop_doc.risks:
            rating = risk.overall_rating.value if risk.overall_rating else "Unknown"
            print(f"  {risk.risk_id}: {risk.description[:50]}...")
            print(f"    Category: {risk.category.value.title()}, Rating: {rating.title()}")
        print()
        
        print("🛡️  Control Measures:")
        for control in sop_doc.controls:
            print(f"  {control.control_id}: {control.description[:50]}...")
            print(f"    Type: {control.control_type.value.title()}")
            if control.applicable_risks:
                print(f"    Addresses: {', '.join(control.applicable_risks)}")
        print()
        
    else:
        print("❌ Extraction failed!")
        for error in result.errors:
            print(f"  Error: {error}")
        print()
    
    if result.warnings:
        print("⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")
        print()


def demo_multi_chunk_extraction():
    """Demonstrate extraction from multiple document chunks."""
    print("=" * 80)
    print("DEMO 2: Multi-Chunk Extraction and Merging")
    print("=" * 80)
    
    # Initialize extractor
    extractor = OntologyExtractor()
    
    # Create sample chunks
    chunks = create_sample_chunks()
    source_info = SourceInfo(
        url="https://company.com/sops/SOP-WELD-001.pdf",
        page_range=[1, 8],
        last_modified=datetime.utcnow(),
        file_size=45000
    )
    
    print(f"Processing {len(chunks)} document chunks...")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.chunk_text)} chars - {chunk.heading_path}")
    print()
    
    # Perform extraction
    result = extractor.extract_from_chunks(chunks, "SOP-WELD-001", source_info)
    
    # Display results
    if result.success:
        sop_doc = result.sop_document
        print("✅ Multi-chunk extraction successful!")
        print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
        print()
        
        print("📊 Merged Document Statistics:")
        print(f"  Total procedure steps: {len(sop_doc.procedure_steps)}")
        print(f"  Total risks identified: {len(sop_doc.risks)}")
        print(f"  Total controls: {len(sop_doc.controls)}")
        print(f"  Total roles: {len(sop_doc.roles_responsibilities)}")
        print(f"  Equipment items: {len(sop_doc.materials_equipment)}")
        print()
        
        # Show step sequence to demonstrate proper merging
        if sop_doc.procedure_steps:
            print("🔄 Step Sequence (showing proper merge order):")
            for step in sop_doc.procedure_steps[:10]:
                print(f"  {step.step_id}: {step.description[:50]}...")
            print()
        
    else:
        print("❌ Multi-chunk extraction failed!")
        for error in result.errors:
            print(f"  Error: {error}")
    
    if result.warnings:
        print("⚠️  Processing Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")
        print()


def demo_validation_and_quality_assessment():
    """Demonstrate validation and quality assessment of extracted data."""
    print("=" * 80)
    print("DEMO 3: Validation and Quality Assessment")
    print("=" * 80)
    
    # Initialize extractor
    extractor = OntologyExtractor()
    
    # Extract document first
    sop_text = create_sample_sop_text()
    source_info = SourceInfo(url="https://company.com/sops/SOP-WELD-001.pdf")
    
    result = extractor.extract_from_text(sop_text, "SOP-WELD-001", source_info)
    
    if result.success and result.sop_document:
        sop_doc = result.sop_document
        
        # Perform validation
        validation_result = extractor._validate_extraction(sop_doc)
        
        print("🔍 Document Validation Results:")
        print(f"  Overall Status: {'✅ Valid' if validation_result.is_valid else '❌ Invalid'}")
        print(f"  Completeness Score: {validation_result.completeness_score:.2%}")
        print()
        
        if validation_result.errors:
            print("❌ Validation Errors:")
            for error in validation_result.errors:
                print(f"  • {error}")
            print()
        
        if validation_result.warnings:
            print("⚠️  Validation Warnings:")
            for warning in validation_result.warnings:
                print(f"  • {warning}")
            print()
        
        # Quality metrics
        print("📈 Quality Metrics:")
        print(f"  Document has title: {'✅' if sop_doc.title and sop_doc.title != 'Untitled SOP' else '❌'}")
        print(f"  Has procedure steps: {'✅' if sop_doc.procedure_steps else '❌'}")
        print(f"  Has risk analysis: {'✅' if sop_doc.risks else '❌'}")
        print(f"  Has control measures: {'✅' if sop_doc.controls else '❌'}")
        print(f"  Has role definitions: {'✅' if sop_doc.roles_responsibilities else '❌'}")
        print(f"  Has revision info: {'✅' if sop_doc.revision else '❌'}")
        print()
        
        # Referential integrity check
        step_ids = {step.step_id for step in sop_doc.procedure_steps}
        risk_step_refs = set()
        control_step_refs = set()
        
        for risk in sop_doc.risks:
            risk_step_refs.update(risk.affected_steps)
        
        for control in sop_doc.controls:
            control_step_refs.update(control.applicable_steps)
        
        invalid_risk_refs = risk_step_refs - step_ids
        invalid_control_refs = control_step_refs - step_ids
        
        print("🔗 Referential Integrity:")
        print(f"  Valid risk-step references: {len(risk_step_refs - invalid_risk_refs)}/{len(risk_step_refs)}")
        print(f"  Valid control-step references: {len(control_step_refs - invalid_control_refs)}/{len(control_step_refs)}")
        
        if invalid_risk_refs:
            print(f"  ⚠️  Invalid risk references: {', '.join(invalid_risk_refs)}")
        if invalid_control_refs:
            print(f"  ⚠️  Invalid control references: {', '.join(invalid_control_refs)}")
        print()


def demo_summarization():
    """Demonstrate document summarization capabilities."""
    print("=" * 80)
    print("DEMO 4: Document Summarization")
    print("=" * 80)
    
    # Initialize services
    extractor = OntologyExtractor()
    summarizer = SOPSummarizer()
    
    # Extract document
    sop_text = create_sample_sop_text()
    source_info = SourceInfo(url="https://company.com/sops/SOP-WELD-001.pdf")
    
    result = extractor.extract_from_text(sop_text, "SOP-WELD-001", source_info)
    
    if result.success and result.sop_document:
        sop_doc = result.sop_document
        
        # Create summary
        summary = summarizer.create_document_summary(sop_doc)
        
        print("📄 Document Summary:")
        print(f"  Title: {summary['title']}")
        print(f"  Process: {summary['process_name']}")
        print(f"  Revision: {summary['revision']}")
        print()
        
        print("📊 Key Metrics:")
        metrics = summary['metrics']
        print(f"  Total Steps: {metrics['total_steps']}")
        print(f"  Total Risks: {metrics['total_risks']} (High Priority: {metrics['high_priority_risks']})")
        print(f"  Total Controls: {metrics['total_controls']}")
        print(f"  Total Roles: {metrics['total_roles']}")
        print(f"  Equipment Items: {metrics['total_equipment']}")
        print()
        
        print("🎯 Overview:")
        overview = summary['overview']
        print(f"  Complexity Level: {overview['complexity_level'].title()}")
        print(f"  Safety Focus: {'Yes' if overview['safety_focus'] else 'No'}")
        print(f"  Key Roles: {', '.join(overview['key_roles'])}")
        print()
        
        print("⚠️  Critical Information:")
        critical = summary['critical_info']
        if critical['critical_risks']:
            print("  High-Priority Risks:")
            for risk in critical['critical_risks'][:3]:
                print(f"    • {risk['risk_id']}: {risk['description'][:50]}... ({risk['rating'].title()})")
        
        if critical['safety_critical_steps']:
            print("  Safety-Critical Steps:")
            for step in critical['safety_critical_steps'][:3]:
                print(f"    • {step['step_id']}: {step['title']}")
        print()
        
        print("🔍 Quick Reference:")
        quick_ref = summary['quick_reference']
        if quick_ref['step_sequence']:
            print("  First Few Steps:")
            for step in quick_ref['step_sequence'][:5]:
                duration = f" ({step['duration']} min)" if step['duration'] else ""
                print(f"    {step['step_id']}: {step['title']}{duration}")
        print()


def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms."""
    print("=" * 80)
    print("DEMO 5: Error Handling and Fallback")
    print("=" * 80)
    
    extractor = OntologyExtractor()
    source_info = SourceInfo(url="https://company.com/test.pdf")
    
    # Test with empty text
    print("Testing with empty text...")
    result = extractor.extract_from_text("", "empty_doc", source_info)
    print(f"Result: {'Success' if result.success else 'Failed'}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print()
    
    # Test with non-SOP text
    print("Testing with non-SOP text...")
    non_sop_text = "This is just a regular document about weather patterns and has nothing to do with manufacturing or procedures."
    result = extractor.extract_from_text(non_sop_text, "weather_doc", source_info)
    print(f"Result: {'Success' if result.success else 'Failed'}")
    if result.success and result.sop_document:
        print(f"Created minimal document: {result.sop_document.title}")
        print(f"Steps extracted: {len(result.sop_document.procedure_steps)}")
    print()
    
    # Test with malformed chunks
    print("Testing with malformed chunks...")
    bad_chunks = [
        DocumentChunk(
            chunk_id="bad_chunk",
            doc_id="bad_doc",
            chunk_text="",  # Empty chunk
            chunk_index=0
        )
    ]
    
    result = extractor.extract_from_chunks(bad_chunks, "bad_doc", source_info)
    print(f"Result: {'Success' if result.success else 'Failed'}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    print()


def main():
    """Run all demonstration scenarios."""
    print("🔬 SOP Ontology Extraction Service Demo")
    print("This demo showcases the structured information extraction capabilities")
    print("of the SOP Q&A tool using both AWS Bedrock and local fallback methods.")
    print()
    
    # Check settings
    settings = get_settings()
    print(f"Current mode: {settings.mode.value.upper()}")
    print(f"Using {'AWS Bedrock' if settings.is_aws_mode() else 'Local extraction'} for processing")
    print()
    
    try:
        # Run all demos
        demo_single_text_extraction()
        demo_multi_chunk_extraction()
        demo_validation_and_quality_assessment()
        demo_summarization()
        demo_error_handling()
        
        print("=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()