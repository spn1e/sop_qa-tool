#!/usr/bin/env python3
"""
Vector Storage Service Demo

This script demonstrates the vector storage functionality including:
- Indexing document chunks with embeddings
- Searching with metadata filters
- Document deletion and index management
- Both FAISS (local) and OpenSearch (AWS) modes
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from sop_qa_tool.services.vectorstore import VectorStoreService
from sop_qa_tool.models.sop_models import DocumentChunk
from sop_qa_tool.config.settings import get_settings


def create_sample_chunks():
    """Create sample document chunks for demonstration"""
    chunks = [
        DocumentChunk(
            chunk_id="sop_001_chunk_001",
            doc_id="sop_001",
            chunk_text="Step 1: Prepare the filling equipment by checking temperature settings to 75°C ± 2°C. Ensure all sensors are calibrated and functioning properly.",
            chunk_index=0,
            page_no=1,
            heading_path="1. Equipment Preparation",
            step_ids=["1.1", "1.2"],
            roles=["Operator", "Maintenance Technician"],
            equipment=["Filling Machine", "Temperature Controller", "Sensors"]
        ),
        DocumentChunk(
            chunk_id="sop_001_chunk_002",
            doc_id="sop_001", 
            chunk_text="Step 2: Verify safety protocols are in place. Check emergency stop buttons, safety guards, and ensure all personnel are wearing appropriate PPE.",
            chunk_index=1,
            page_no=1,
            heading_path="2. Safety Verification",
            step_ids=["2.1", "2.2"],
            roles=["Safety Officer", "Operator"],
            equipment=["Emergency Stop", "Safety Guards", "PPE"]
        ),
        DocumentChunk(
            chunk_id="sop_001_chunk_003",
            doc_id="sop_001",
            chunk_text="Step 3: Begin the filling process by starting the conveyor belt at 50 units/minute. Monitor fill levels and adjust as needed.",
            chunk_index=2,
            page_no=2,
            heading_path="3. Filling Process",
            step_ids=["3.1"],
            roles=["Operator"],
            equipment=["Conveyor Belt", "Fill Level Monitor"]
        ),
        DocumentChunk(
            chunk_id="sop_002_chunk_001",
            doc_id="sop_002",
            chunk_text="Quality inspection procedure: Inspect each product for defects using visual inspection and measurement tools. Record findings in QC log.",
            chunk_index=0,
            page_no=1,
            heading_path="1. Quality Control",
            step_ids=["1.1"],
            roles=["QA Inspector"],
            equipment=["Measurement Tools", "QC Log"]
        ),
        DocumentChunk(
            chunk_id="sop_002_chunk_002",
            doc_id="sop_002",
            chunk_text="If defects are found, quarantine the affected products and notify the production supervisor immediately. Document the issue in the defect log.",
            chunk_index=1,
            page_no=1,
            heading_path="2. Defect Handling",
            step_ids=["2.1", "2.2"],
            roles=["QA Inspector", "Production Supervisor"],
            equipment=["Quarantine Area", "Defect Log"]
        )
    ]
    return chunks


def create_sample_embeddings(num_chunks: int, dimensions: int = 384):
    """Create sample embeddings for demonstration"""
    # Create random embeddings and normalize them
    embeddings = np.random.rand(num_chunks, dimensions).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


async def demonstrate_vector_storage():
    """Demonstrate vector storage functionality"""
    print("🔧 Vector Storage Service Demo")
    print("=" * 50)
    
    # Get settings and show current mode
    settings = get_settings()
    print(f"Operating Mode: {settings.mode.value.upper()}")
    print(f"Data Path: {settings.local_data_path}")
    print()
    
    # Initialize vector store service
    print("📦 Initializing Vector Store Service...")
    try:
        service = VectorStoreService()
        store_type = service.get_store_type()
        print(f"✅ Initialized {store_type.upper()} vector store")
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        return
    
    print()
    
    # Create sample data
    print("📝 Creating sample document chunks...")
    chunks = create_sample_chunks()
    embeddings = create_sample_embeddings(len(chunks))
    print(f"✅ Created {len(chunks)} chunks with {embeddings.shape[1]}D embeddings")
    print()
    
    # Clear any existing data
    print("🧹 Clearing existing index...")
    await service.clear_index()
    print("✅ Index cleared")
    print()
    
    # Index the chunks
    print("📊 Indexing document chunks...")
    success = await service.index_chunks(chunks, embeddings)
    if success:
        print("✅ Successfully indexed all chunks")
    else:
        print("❌ Failed to index chunks")
        return
    print()
    
    # Get index statistics
    print("📈 Index Statistics:")
    stats = await service.get_stats()
    print(f"  • Total Documents: {stats.total_documents}")
    print(f"  • Total Chunks: {stats.total_chunks}")
    print(f"  • Index Size: {stats.index_size_mb:.2f} MB")
    print(f"  • Dimensions: {stats.dimensions}")
    print(f"  • Model: {stats.model_name}")
    print()
    
    # Demonstrate basic search
    print("🔍 Basic Vector Search:")
    query_embedding = embeddings[0]  # Use first chunk as query
    results = await service.search(query_embedding, top_k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.chunk_id}] Score: {result.score:.4f}")
        print(f"     Text: {result.chunk_text[:80]}...")
        print(f"     Roles: {', '.join(result.metadata.get('roles', []))}")
        print()
    
    # Demonstrate filtered search
    print("🎯 Filtered Search (QA Inspector role):")
    filtered_results = await service.search(
        query_embedding,
        filters={"roles": ["QA Inspector"]},
        top_k=5
    )
    
    print(f"Found {len(filtered_results)} results:")
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. [{result.chunk_id}] Score: {result.score:.4f}")
        print(f"     Text: {result.chunk_text[:80]}...")
        print(f"     Roles: {', '.join(result.metadata.get('roles', []))}")
        print()
    
    # Demonstrate equipment filter
    print("⚙️ Equipment Filter (Filling Machine):")
    equipment_results = await service.search(
        query_embedding,
        filters={"equipment": ["Filling Machine"]},
        top_k=5
    )
    
    print(f"Found {len(equipment_results)} results:")
    for i, result in enumerate(equipment_results, 1):
        print(f"  {i}. [{result.chunk_id}] Score: {result.score:.4f}")
        print(f"     Equipment: {', '.join(result.metadata.get('equipment', []))}")
        print()
    
    # Demonstrate document deletion
    print("🗑️ Document Deletion:")
    print("Deleting document 'sop_001'...")
    delete_success = await service.delete_document("sop_001")
    if delete_success:
        print("✅ Document deleted successfully")
    else:
        print("❌ Failed to delete document")
    
    # Check stats after deletion
    stats_after_delete = await service.get_stats()
    print(f"Documents after deletion: {stats_after_delete.total_documents}")
    print(f"Chunks after deletion: {stats_after_delete.total_chunks}")
    print()
    
    # Verify deletion with search
    print("🔍 Search after deletion (should only find sop_002):")
    post_delete_results = await service.search(query_embedding, top_k=5)
    print(f"Found {len(post_delete_results)} results:")
    for result in post_delete_results:
        print(f"  • [{result.chunk_id}] from {result.doc_id}")
    print()
    
    # Final cleanup
    print("🧹 Final cleanup...")
    await service.clear_index()
    final_stats = await service.get_stats()
    print(f"✅ Index cleared - Documents: {final_stats.total_documents}, Chunks: {final_stats.total_chunks}")
    
    print("\n🎉 Demo completed successfully!")


async def main():
    """Main demo function"""
    try:
        await demonstrate_vector_storage()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())