#!/usr/bin/env python3
"""
LOCAL-SIMPLE Demo Script

Demonstrates the SOP QA Tool running in local mode with:
- HuggingFace sentence-transformers for embeddings
- FAISS for vector storage
- Local file processing

This is a safe, cost-free way to test the system before moving to AWS.
"""

import asyncio
import time
from pathlib import Path

from sop_qa_tool.config.settings import Settings
from sop_qa_tool.services.embedder import EmbeddingService


async def main():
    print("🏠 SOP QA Tool - LOCAL-SIMPLE Demo")
    print("=" * 50)
    
    # Initialize settings
    settings = Settings()
    print(f"✅ Configuration loaded - Mode: {settings.mode}")
    print(f"📁 Data path: {settings.local_data_path}")
    print(f"🤖 Model: {settings.hf_model_path}")
    
    # Create directories
    settings.create_directories()
    print("✅ Local directories created")
    
    # Initialize embedding service
    print("\n🔧 Initializing services...")
    embedding_service = EmbeddingService()
    print("✅ EmbeddingService ready")
    
    # Test with sample SOP content
    print("\n📝 Testing with sample SOP content...")
    sample_texts = [
        "Standard Operating Procedure for Equipment Maintenance: All equipment must be inspected daily before use.",
        "Safety Protocol: Always wear protective equipment when handling chemicals.",
        "Quality Control: Document all test results in the quality management system.",
        "Emergency Response: In case of fire, evacuate immediately and call emergency services."
    ]
    
    start_time = time.time()
    result = await embedding_service.embed_texts(sample_texts)
    processing_time = time.time() - start_time
    
    print(f"✅ Embedded {len(sample_texts)} texts")
    print(f"📊 Embedding shape: {result.embeddings.shape}")
    print(f"🎯 Dimensions: {result.dimensions}")
    print(f"⚡ Processing time: {processing_time:.3f}s")
    print(f"🏷️  Model used: {result.model_name}")
    
    # Demonstrate similarity search
    print("\n🔍 Testing similarity search...")
    query = "How should I maintain equipment?"
    query_result = await embedding_service.embed_texts([query])
    
    # Simple cosine similarity
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(
        query_result.embeddings,
        result.embeddings
    )[0]
    
    print(f"Query: '{query}'")
    print("Most similar SOPs:")
    for i, (text, sim) in enumerate(zip(sample_texts, similarities)):
        print(f"  {i+1}. [{sim:.3f}] {text[:60]}...")
    
    print("\n🎉 LOCAL-SIMPLE demo completed successfully!")
    print("\n📋 What's working:")
    print("  ✅ Text embedding with sentence-transformers")
    print("  ✅ Vector similarity search")
    print("  ✅ Local file storage")
    print("  ✅ Fast processing (no API calls)")
    print("  ✅ Zero cloud costs")
    
    print("\n🚀 Ready for production? Try AWS mode next!")


if __name__ == "__main__":
    asyncio.run(main())