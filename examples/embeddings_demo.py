"""
Embeddings Service Demo

Demonstrates the usage of the EmbeddingService for both AWS and local modes,
including batch processing, caching, and dimension validation.
"""

import asyncio
import logging
import numpy as np
import time
from pathlib import Path

from sop_qa_tool.services.embedder import EmbeddingService
from sop_qa_tool.config.settings import Settings, OperationMode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_local_mode():
    """Demonstrate embeddings service in local mode"""
    print("\n" + "="*60)
    print("EMBEDDINGS SERVICE DEMO - LOCAL MODE")
    print("="*60)
    
    # Configure for local mode
    settings = Settings(
        mode=OperationMode.LOCAL,
        local_data_path=Path("./data"),
        hf_model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_batch_size=3
    )
    
    # Override global settings for demo
    import sop_qa_tool.config.settings
    sop_qa_tool.config.settings.settings = settings
    
    try:
        # Initialize service
        print("Initializing EmbeddingService in local mode...")
        service = EmbeddingService()
        print(f"✓ Service initialized with model: {service._get_model_name()}")
        print(f"✓ Expected dimensions: {service._get_expected_dimensions()}")
        
        # Sample texts for embedding
        sample_texts = [
            "Step 3.1: Check the temperature of the filling machine",
            "Ensure all safety equipment is properly worn",
            "Quality control inspection must be performed every hour",
            "The conveyor belt speed should be set to 2.5 m/s",
            "Record all measurements in the quality log",
            "If temperature exceeds 85°C, stop the process immediately"
        ]
        
        print(f"\nEmbedding {len(sample_texts)} sample texts...")
        
        # Generate embeddings
        start_time = time.time()
        result = await service.embed_texts(sample_texts, batch_size=3)
        processing_time = time.time() - start_time
        
        print(f"✓ Embeddings generated in {processing_time:.2f}s")
        print(f"✓ Shape: {result.embeddings.shape}")
        print(f"✓ Dimensions: {result.dimensions}")
        print(f"✓ Model: {result.model_name}")
        print(f"✓ Cached count: {result.cached_count}")
        print(f"✓ Error count: {result.error_count}")
        
        # Validate embeddings
        validation = service.validate_embeddings(result.embeddings)
        print(f"\nValidation results:")
        print(f"✓ Valid: {validation['valid']}")
        if validation['errors']:
            print(f"✗ Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"⚠ Warnings: {validation['warnings']}")
        print(f"✓ Stats: {validation['stats']}")
        
        # Test caching
        print(f"\nTesting caching with same texts...")
        start_time = time.time()
        cached_result = await service.embed_texts(sample_texts[:3])
        cache_time = time.time() - start_time
        
        print(f"✓ Cached embeddings retrieved in {cache_time:.4f}s")
        print(f"✓ Cached count: {cached_result.cached_count}")
        
        # Verify cached embeddings are identical
        np.testing.assert_array_equal(
            result.embeddings[:3], 
            cached_result.embeddings
        )
        print(f"✓ Cached embeddings are identical to original")
        
        # Test single query embedding
        print(f"\nTesting single query embedding...")
        query = "What is the maximum temperature allowed?"
        query_embedding = await service.embed_query(query)
        print(f"✓ Query embedding shape: {query_embedding.shape}")
        
        # Calculate similarity with sample texts
        print(f"\nCalculating similarities with sample texts:")
        similarities = np.dot(result.embeddings, query_embedding)
        
        for i, (text, similarity) in enumerate(zip(sample_texts, similarities)):
            print(f"  {i+1}. Similarity: {similarity:.3f} - {text[:50]}...")
        
        # Show most similar text
        most_similar_idx = np.argmax(similarities)
        print(f"\nMost similar text: {sample_texts[most_similar_idx]}")
        print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
        
        # Cache statistics
        cache_stats = service.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"✓ Total entries: {cache_stats['total_entries']}")
        print(f"✓ Total size: {cache_stats['total_size_mb']:.2f} MB")
        print(f"✓ Model counts: {cache_stats['model_counts']}")
        
        print(f"\n✓ Local mode demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in local mode demo: {e}")
        logger.exception("Local mode demo failed")


async def demo_aws_mode():
    """Demonstrate embeddings service in AWS mode (if configured)"""
    print("\n" + "="*60)
    print("EMBEDDINGS SERVICE DEMO - AWS MODE")
    print("="*60)
    
    # Check if AWS credentials are available
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            print("⚠ AWS credentials not found, skipping AWS mode demo")
            return
    except ImportError:
        print("⚠ boto3 not installed, skipping AWS mode demo")
        return
    
    # Configure for AWS mode
    settings = Settings(
        mode=OperationMode.AWS,
        aws_region="us-east-1",
        titan_embeddings_id="amazon.titan-embed-text-v2:0",
        embedding_batch_size=2
    )
    
    # Override global settings for demo
    import sop_qa_tool.config.settings
    sop_qa_tool.config.settings.settings = settings
    
    try:
        # Initialize service
        print("Initializing EmbeddingService in AWS mode...")
        service = EmbeddingService()
        print(f"✓ Service initialized with model: {service._get_model_name()}")
        print(f"✓ Expected dimensions: {service._get_expected_dimensions()}")
        
        # Sample texts (fewer for AWS demo to avoid costs)
        sample_texts = [
            "Step 1: Verify equipment calibration",
            "Step 2: Check safety protocols",
            "Step 3: Begin production process"
        ]
        
        print(f"\nEmbedding {len(sample_texts)} sample texts...")
        
        # Generate embeddings
        start_time = time.time()
        result = await service.embed_texts(sample_texts)
        processing_time = time.time() - start_time
        
        print(f"✓ Embeddings generated in {processing_time:.2f}s")
        print(f"✓ Shape: {result.embeddings.shape}")
        print(f"✓ Dimensions: {result.dimensions}")
        print(f"✓ Model: {result.model_name}")
        
        # Validate embeddings
        validation = service.validate_embeddings(result.embeddings)
        print(f"\nValidation results:")
        print(f"✓ Valid: {validation['valid']}")
        if validation['warnings']:
            print(f"⚠ Warnings: {validation['warnings']}")
        print(f"✓ Stats: {validation['stats']}")
        
        # Test normalization (Titan embeddings should be normalized)
        norms = np.linalg.norm(result.embeddings, axis=1)
        print(f"\nEmbedding norms (should be ~1.0 for Titan):")
        for i, norm in enumerate(norms):
            print(f"  Text {i+1}: {norm:.4f}")
        
        print(f"\n✓ AWS mode demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Error in AWS mode demo: {e}")
        logger.exception("AWS mode demo failed")


async def demo_performance_comparison():
    """Compare performance between cached and uncached embeddings"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON DEMO")
    print("="*60)
    
    # Use local mode for consistent testing
    settings = Settings(
        mode=OperationMode.LOCAL,
        local_data_path=Path("./data"),
        hf_model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_batch_size=5
    )
    
    import sop_qa_tool.config.settings
    sop_qa_tool.config.settings.settings = settings
    
    try:
        service = EmbeddingService()
        
        # Generate test texts
        test_texts = [
            f"This is test text number {i} for performance comparison."
            for i in range(20)
        ]
        
        print(f"Testing with {len(test_texts)} texts...")
        
        # Clear cache to start fresh
        service.clear_cache()
        
        # First run - no cache
        print("\nFirst run (no cache):")
        start_time = time.time()
        result1 = await service.embed_texts(test_texts)
        time1 = time.time() - start_time
        print(f"✓ Time: {time1:.3f}s")
        print(f"✓ Cached count: {result1.cached_count}")
        
        # Second run - with cache
        print("\nSecond run (with cache):")
        start_time = time.time()
        result2 = await service.embed_texts(test_texts)
        time2 = time.time() - start_time
        print(f"✓ Time: {time2:.3f}s")
        print(f"✓ Cached count: {result2.cached_count}")
        
        # Performance improvement
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n✓ Speedup from caching: {speedup:.1f}x")
        
        # Verify results are identical
        np.testing.assert_array_equal(result1.embeddings, result2.embeddings)
        print(f"✓ Results are identical")
        
        # Test partial caching
        print("\nTesting partial caching (half new texts):")
        mixed_texts = test_texts[:10] + [f"New text {i}" for i in range(10)]
        
        start_time = time.time()
        result3 = await service.embed_texts(mixed_texts)
        time3 = time.time() - start_time
        
        print(f"✓ Time: {time3:.3f}s")
        print(f"✓ Cached count: {result3.cached_count}")
        print(f"✓ New embeddings: {len(mixed_texts) - result3.cached_count}")
        
        print(f"\n✓ Performance comparison completed!")
        
    except Exception as e:
        print(f"✗ Error in performance demo: {e}")
        logger.exception("Performance demo failed")


async def main():
    """Run all demos"""
    print("EMBEDDINGS SERVICE DEMONSTRATION")
    print("This demo shows the capabilities of the EmbeddingService")
    print("including local and AWS modes, caching, and validation.")
    
    # Run local mode demo
    await demo_local_mode()
    
    # Run AWS mode demo (if available)
    await demo_aws_mode()
    
    # Run performance comparison
    await demo_performance_comparison()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())