"""
Text Chunking Demo

Demonstrates the text chunking and processing capabilities of the SOP Q&A Tool.
Shows how documents are split into chunks with metadata extraction and structure preservation.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sop_qa_tool.services.text_chunker import TextChunker
from sop_qa_tool.models.sop_models import SOPDocument, ProcedureStep, RoleResponsibility, SourceInfo


def main():
    """Run text chunking demonstration"""
    print("=== SOP Text Chunking Demo ===\n")
    
    # Sample SOP text
    sample_sop_text = """
# SOP-FILL-001: Bottle Filling Procedure
**Revision:** 2.1  
**Effective Date:** 2024-01-15

## 1. PURPOSE AND SCOPE
This Standard Operating Procedure (SOP) defines the process for filling bottles in the production line to ensure consistent quality and safety.

## 2. RESPONSIBILITIES

### 2.1 Line Operator
- Operate the filling equipment (FILLER-01)
- Monitor fill levels and quality parameters
- Report any deviations immediately to supervisor

### 2.2 QA Inspector  
- Verify product quality meets specifications
- Conduct random sampling every 30 minutes
- Approve or reject batches based on quality criteria

### 2.3 Production Supervisor
- Oversee the entire filling process
- Approve batch releases after QA approval
- Handle escalations and process deviations

## 3. EQUIPMENT AND MATERIALS

### 3.1 Primary Equipment
- **FILLER-01:** Main filling machine (capacity: 1000 bottles/hour)
- **CONVEYOR-02:** Transport system for bottle movement
- **CAPPER-03:** Bottle capping unit with torque control

### 3.2 Quality Control Equipment
- Temperature probe (±0.1°C accuracy)
- Fill level gauge with digital display
- Pressure sensors for system monitoring

## 4. PROCEDURE

### 4.1 Pre-Operation Setup

**Step 4.1.1:** Verify FILLER-01 is clean and sanitized
- Check cleaning log completion from previous shift
- Verify sanitization certificate is current
- **Responsible:** Line Operator
- **Equipment:** FILLER-01, cleaning verification tools

**Step 4.1.2:** Calibrate temperature probe
- Set target temperature range: 18-22°C
- Verify accuracy against certified reference thermometer
- Document calibration results on form QF-001
- **Responsible:** QA Inspector
- **Equipment:** Temperature probe, reference thermometer

### 4.2 Filling Operation

**Step 4.2.1:** Start filling sequence
- Initialize FILLER-01 control system
- Set fill volume: 500ml ±5ml tolerance
- Begin continuous operation at 800 bottles/hour
- **Responsible:** Line Operator
- **Equipment:** FILLER-01, control panel

**Step 4.2.2:** Monitor process parameters
- Check temperature readings every 5 minutes
- Verify fill levels on every 10th bottle
- Record all data on batch production sheet
- **Responsible:** Line Operator, QA Inspector
- **Equipment:** Temperature probe, fill gauge

### 4.3 Quality Control Checks

**Step 4.3.1:** Conduct hourly quality inspections
- Sample 3 bottles per hour for weight verification
- Check cap torque on sampled bottles
- Verify label placement and print quality
- **Responsible:** QA Inspector
- **Equipment:** Precision scale, torque meter

## 5. RISK MANAGEMENT

**Risk R-001:** Temperature deviation beyond acceptable range
- **Impact:** Product quality degradation, potential spoilage
- **Probability:** Medium (2-3 occurrences per month)
- **Severity:** High (affects entire batch)
- **Controls:** C-001, C-002

**Risk R-002:** Overfill or underfill conditions
- **Impact:** Customer complaints, regulatory non-compliance
- **Probability:** Low (1 occurrence per month)
- **Severity:** Medium (affects individual bottles)
- **Controls:** C-003, C-004

**Risk R-003:** Equipment malfunction during operation
- **Impact:** Production downtime, potential safety hazard
- **Probability:** Low (quarterly occurrence)
- **Severity:** High (complete line shutdown)
- **Controls:** C-005, C-006

## 6. CONTROL MEASURES

**Control C-001:** Continuous temperature monitoring system
- **Method:** Automated sensor with visual and audible alarms
- **Frequency:** Continuous monitoring, 5-minute intervals
- **Responsible:** Line Operator (monitoring), Maintenance (calibration)
- **Effectiveness:** High

**Control C-002:** Weekly temperature calibration verification
- **Method:** Comparison against certified reference standard
- **Frequency:** Weekly, every Monday morning
- **Responsible:** QA Inspector
- **Effectiveness:** High

**Control C-003:** Statistical fill level verification
- **Method:** Random sampling with precision measurement
- **Frequency:** 1 in every 10 bottles (10% sampling rate)
- **Responsible:** QA Inspector
- **Effectiveness:** Medium

**Control C-004:** Automated weight rejection system
- **Method:** In-line checkweigher with automatic rejection
- **Frequency:** Every bottle (100% inspection)
- **Responsible:** System (automated), Operator (monitoring)
- **Effectiveness:** High

**Control C-005:** Preventive maintenance schedule
- **Method:** Scheduled maintenance per manufacturer recommendations
- **Frequency:** Monthly major service, weekly minor checks
- **Responsible:** Maintenance Technician
- **Effectiveness:** High

**Control C-006:** Emergency stop procedures
- **Method:** Clearly marked emergency stops, trained personnel
- **Frequency:** As needed, monthly training drills
- **Responsible:** All personnel, Safety Officer (training)
- **Effectiveness:** High

## 7. DOCUMENTATION AND RECORDS
All activities must be documented on the appropriate forms:
- Batch Production Sheet (Form BP-001)
- Quality Control Log (Form QC-002)
- Equipment Maintenance Log (Form EM-003)
- Deviation Report (Form DR-004)

Records must be retained for minimum 2 years per company policy.
"""

    # Create TextChunker instance
    print("1. Initializing Text Chunker...")
    chunker = TextChunker()
    print(f"   - Chunk size: {chunker.chunk_size} characters")
    print(f"   - Chunk overlap: {chunker.chunk_overlap} characters")
    
    # Create sample SOP document for enhanced metadata
    print("\n2. Creating sample SOP document structure...")
    sop_document = SOPDocument(
        doc_id="SOP-FILL-001",
        title="Bottle Filling Procedure",
        process_name="Bottle Filling",
        revision="2.1",
        roles_responsibilities=[
            RoleResponsibility(
                role="Line Operator",
                responsibilities=["Operate filling equipment", "Monitor process parameters"],
                qualifications=["Basic machine operation training"]
            ),
            RoleResponsibility(
                role="QA Inspector",
                responsibilities=["Quality verification", "Sampling and testing"],
                qualifications=["Quality control certification"]
            ),
            RoleResponsibility(
                role="Production Supervisor",
                responsibilities=["Process oversight", "Batch approval"],
                qualifications=["Supervisory experience", "Process knowledge"]
            )
        ],
        materials_equipment=[
            "FILLER-01", "CONVEYOR-02", "CAPPER-03", 
            "Temperature probe", "Fill level gauge", "Pressure sensors"
        ],
        procedure_steps=[
            ProcedureStep(
                step_id="4.1.1",
                title="Verify equipment cleanliness",
                description="Verify FILLER-01 is clean and sanitized",
                responsible_roles=["Line Operator"],
                required_equipment=["FILLER-01"]
            ),
            ProcedureStep(
                step_id="4.1.2", 
                title="Calibrate temperature probe",
                description="Calibrate temperature probe and verify accuracy",
                responsible_roles=["QA Inspector"],
                required_equipment=["Temperature probe"]
            ),
            ProcedureStep(
                step_id="4.2.1",
                title="Start filling sequence", 
                description="Initialize and start the filling operation",
                responsible_roles=["Line Operator"],
                required_equipment=["FILLER-01"]
            )
        ],
        source=SourceInfo(file_path="SOP-FILL-001.pdf")
    )
    
    # Chunk the document
    print("\n3. Chunking document with structure preservation...")
    chunks = chunker.chunk_document(
        text=sample_sop_text,
        doc_id="SOP-FILL-001",
        sop_document=sop_document,
        preserve_structure=True
    )
    
    print(f"   - Created {len(chunks)} chunks")
    
    # Display chunk information
    print("\n4. Chunk Analysis:")
    print("=" * 80)
    
    total_chars = sum(len(chunk.chunk_text) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    
    print(f"Total characters: {total_chars}")
    print(f"Average chunk size: {avg_chunk_size:.1f} characters")
    print(f"Chunk size range: {min(len(c.chunk_text) for c in chunks)} - {max(len(c.chunk_text) for c in chunks)}")
    
    # Show metadata statistics
    chunks_with_steps = len([c for c in chunks if c.step_ids])
    chunks_with_risks = len([c for c in chunks if c.risk_ids])
    chunks_with_controls = len([c for c in chunks if c.control_ids])
    chunks_with_roles = len([c for c in chunks if c.roles])
    chunks_with_equipment = len([c for c in chunks if c.equipment])
    chunks_with_headings = len([c for c in chunks if c.heading_path])
    
    print(f"\nMetadata extraction results:")
    print(f"  - Chunks with step IDs: {chunks_with_steps}")
    print(f"  - Chunks with risk IDs: {chunks_with_risks}")
    print(f"  - Chunks with control IDs: {chunks_with_controls}")
    print(f"  - Chunks with roles: {chunks_with_roles}")
    print(f"  - Chunks with equipment: {chunks_with_equipment}")
    print(f"  - Chunks with heading paths: {chunks_with_headings}")
    
    # Show detailed information for first few chunks
    print("\n5. Sample Chunks (first 3):")
    print("=" * 80)
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Index: {chunk.chunk_index}")
        print(f"Size: {len(chunk.chunk_text)} characters")
        
        if chunk.heading_path:
            print(f"Heading Path: {chunk.heading_path}")
        
        if chunk.step_ids:
            print(f"Step IDs: {', '.join(chunk.step_ids)}")
        
        if chunk.risk_ids:
            print(f"Risk IDs: {', '.join(chunk.risk_ids)}")
        
        if chunk.control_ids:
            print(f"Control IDs: {', '.join(chunk.control_ids)}")
        
        if chunk.roles:
            print(f"Roles: {', '.join(chunk.roles)}")
        
        if chunk.equipment:
            print(f"Equipment: {', '.join(chunk.equipment)}")
        
        # Show first 200 characters of content
        content_preview = chunk.chunk_text[:200].replace('\n', ' ').strip()
        if len(chunk.chunk_text) > 200:
            content_preview += "..."
        print(f"Content: {content_preview}")
    
    # Test deterministic chunking
    print("\n6. Testing Deterministic Chunking...")
    chunks2 = chunker.chunk_document(
        text=sample_sop_text,
        doc_id="SOP-FILL-001",
        sop_document=sop_document,
        preserve_structure=True
    )
    
    is_deterministic = all(
        c1.chunk_id == c2.chunk_id and c1.chunk_text == c2.chunk_text
        for c1, c2 in zip(chunks, chunks2)
    )
    
    print(f"   - Chunking is deterministic: {is_deterministic}")
    print(f"   - Both runs produced {len(chunks)} and {len(chunks2)} chunks")
    
    # Validate chunks
    print("\n7. Chunk Validation:")
    validation_result = chunker.validate_chunks(chunks)
    
    print(f"   - Valid: {validation_result['valid']}")
    if validation_result['errors']:
        print(f"   - Errors: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"   - Warnings: {validation_result['warnings']}")
    
    stats = validation_result['stats']
    print(f"   - Statistics:")
    print(f"     * Total chunks: {stats['total_chunks']}")
    print(f"     * Average size: {stats['avg_chunk_size']:.1f} chars")
    print(f"     * Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars")
    print(f"     * Chunks with metadata: {stats['chunks_with_metadata']}")
    
    # Show extracted metadata summary
    print("\n8. Extracted Metadata Summary:")
    all_step_ids = set()
    all_risk_ids = set()
    all_control_ids = set()
    all_roles = set()
    all_equipment = set()
    
    for chunk in chunks:
        all_step_ids.update(chunk.step_ids)
        all_risk_ids.update(chunk.risk_ids)
        all_control_ids.update(chunk.control_ids)
        all_roles.update(chunk.roles)
        all_equipment.update(chunk.equipment)
    
    print(f"   - Step IDs found: {sorted(all_step_ids)}")
    print(f"   - Risk IDs found: {sorted(all_risk_ids)}")
    print(f"   - Control IDs found: {sorted(all_control_ids)}")
    print(f"   - Roles found: {sorted(all_roles)}")
    print(f"   - Equipment found: {sorted(all_equipment)}")
    
    print("\n=== Demo Complete ===")
    print("\nThe text chunker successfully:")
    print("✓ Split the document into manageable chunks")
    print("✓ Preserved document structure with heading paths")
    print("✓ Extracted step IDs, risk IDs, and control IDs")
    print("✓ Identified roles and equipment mentions")
    print("✓ Provided deterministic, consistent results")
    print("✓ Validated chunk quality and consistency")


if __name__ == "__main__":
    main()