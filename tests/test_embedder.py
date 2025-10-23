"""
Unit tests for the Embeddings Service.

Tests embedding generation, dimension consistency, caching, retry logic,
and rate limiting for both AWS and local modes.
"""

import asyncio
import json
import numpy as np
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from sop_qa_tool.services.embedder import EmbeddingService, RateLimiter, EmbeddingResult
from sop_qa_tool.config.settings import Settings, OperationMode


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_calls_within_limit(self):
        """Test that rate limiter allows calls within the limit"""
        limiter = RateLimiter(max_calls=5, time_window=1.0)
        
        start_time = time.time()
        
        # Make 5 calls - should all be immediate
        for _ in range(5):
            await limiter.acquire()
        
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be very fast
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_calls(self):
        """Test that rate limiter blocks calls exceeding the limit"""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # Make 2 calls quickly
        await limiter.acquire()
        await limiter.acquire()
        
        # Third call should be delayed
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time
        
        assert elapsed >= 0.9  # Should wait almost the full window


class TestEmbeddingService:
    """Test embedding service functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def local_settings(self, temp_dir):
        """Settings for local mode testing"""
        return Settings(
            mode=OperationMode.LOCAL,
            local_data_path=temp_dir / "data",
            hf_model_path="sentence-transformers/all-MiniLM-L6-v2",
            embedding_batch_size=4
        )
    
    @pytest.fixture
    def aws_settings(self, temp_dir):
        """Settings for AWS mode testing"""
        return Settings(
            mode=OperationMode.AWS,
            aws_region="us-east-1",
            titan_embeddings_id="amazon.titan-embed-text-v2:0",
            embedding_batch_size=4,
            local_data_path=temp_dir / "data",  # For cache
            opensearch_endpoint="https://test-endpoint.us-east-1.aoss.amazonaws.com",
            s3_raw_bucket="test-raw-bucket",
            s3_chunks_bucket="test-chunks-bucket"
        )
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock sentence transformer model"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        return mock_model
    
    @pytest.fixture
    def mock_aws_client(self):
        """Mock AWS Bedrock client"""
        mock_client = Mock()
        
        # Mock successful response
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'embedding': np.random.rand(768).tolist()
        }).encode()
        
        mock_client.invoke_model.return_value = mock_response
        mock_client.list_foundation_models.return_value = {}
        
        return mock_client
    
    def test_local_mode_initialization(self, local_settings, mock_sentence_transformer):
        """Test initialization in local mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                assert service.settings.is_local_mode()
                assert service._local_model is not None
                assert service._aws_client is None
    
    def test_aws_mode_initialization(self, aws_settings, mock_aws_client):
        """Test initialization in AWS mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('boto3.Session') as mock_session:
                mock_session.return_value.client.return_value = mock_aws_client
                
                service = EmbeddingService()
                
                assert service.settings.is_aws_mode()
                assert service._aws_client is not None
                assert service._local_model is None
                assert service._rate_limiter is not None
    
    def test_aws_mode_requires_boto3(self, aws_settings):
        """Test that AWS mode fails gracefully without boto3"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('sop_qa_tool.services.embedder.AWS_AVAILABLE', False):
                with pytest.raises(ImportError, match="boto3 is required"):
                    EmbeddingService()
    
    def test_text_hashing(self, local_settings, mock_sentence_transformer):
        """Test text hashing for cache keys"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # Same text should produce same hash
                hash1 = service._get_text_hash("test text")
                hash2 = service._get_text_hash("test text")
                assert hash1 == hash2
                
                # Different text should produce different hash
                hash3 = service._get_text_hash("different text")
                assert hash1 != hash3
                
                # Hash should be consistent length
                assert len(hash1) == 16
    
    def test_cache_key_generation(self, local_settings, mock_sentence_transformer):
        """Test cache key generation"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                key1 = service._get_cache_key("test", "model1")
                key2 = service._get_cache_key("test", "model2")
                key3 = service._get_cache_key("different", "model1")
                
                # Same text, different model should be different
                assert key1 != key2
                
                # Different text, same model should be different
                assert key1 != key3
                
                # Keys should contain model name
                assert "model1" in key1
                assert "model2" in key2
    
    @pytest.mark.asyncio
    async def test_local_embedding_generation(self, local_settings, mock_sentence_transformer):
        """Test embedding generation in local mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                texts = ["Hello world", "Test text"]
                result = await service.embed_texts(texts)
                
                assert isinstance(result, EmbeddingResult)
                assert result.embeddings.shape[0] == 2
                assert result.embeddings.shape[1] == 384  # Local model dimensions
                assert result.dimensions == 384
                assert result.processing_time > 0
                assert "sentence-transformers" in result.model_name
    
    @pytest.mark.asyncio
    async def test_aws_embedding_generation(self, aws_settings, mock_aws_client):
        """Test embedding generation in AWS mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('boto3.Session') as mock_session:
                mock_session.return_value.client.return_value = mock_aws_client
                
                service = EmbeddingService()
                
                texts = ["Hello world", "Test text"]
                result = await service.embed_texts(texts)
                
                assert isinstance(result, EmbeddingResult)
                assert result.embeddings.shape[0] == 2
                assert result.embeddings.shape[1] == 768  # Titan dimensions
                assert result.dimensions == 768
                assert result.processing_time > 0
                assert "titan" in result.model_name.lower()
    
    @pytest.mark.asyncio
    async def test_single_query_embedding(self, local_settings, mock_sentence_transformer):
        """Test single query embedding"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                embedding = await service.embed_query("test query")
                
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, local_settings, mock_sentence_transformer):
        """Test batch processing with multiple batches"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                # Mock to return different embeddings for each call
                mock_sentence_transformer.encode.side_effect = [
                    np.random.rand(2, 384).astype(np.float32),  # First batch
                    np.random.rand(2, 384).astype(np.float32),  # Second batch
                    np.random.rand(1, 384).astype(np.float32)   # Third batch
                ]
                
                service = EmbeddingService()
                
                # 5 texts with batch size 2 should create 3 batches
                texts = [f"text {i}" for i in range(5)]
                result = await service.embed_texts(texts, batch_size=2)
                
                assert result.embeddings.shape[0] == 5
                assert mock_sentence_transformer.encode.call_count == 3
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, local_settings, mock_sentence_transformer, temp_dir):
        """Test embedding caching"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # First call should compute embeddings
                texts = ["cached text"]
                result1 = await service.embed_texts(texts)
                assert result1.cached_count == 0
                
                # Second call should use cache
                result2 = await service.embed_texts(texts)
                assert result2.cached_count == 1
                
                # Verify embeddings are identical
                np.testing.assert_array_equal(result1.embeddings, result2.embeddings)
    
    @pytest.mark.asyncio
    async def test_cache_persistence(self, local_settings, mock_sentence_transformer, temp_dir):
        """Test that cache persists across service instances"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                # First service instance
                service1 = EmbeddingService()
                texts = ["persistent text"]
                result1 = await service1.embed_texts(texts)
                service1._save_cache()  # Explicitly save
                
                # Second service instance should load cache
                service2 = EmbeddingService()
                result2 = await service2.embed_texts(texts)
                
                assert result2.cached_count == 1
                np.testing.assert_array_equal(result1.embeddings, result2.embeddings)
    
    @pytest.mark.asyncio
    async def test_aws_retry_logic(self, aws_settings, mock_aws_client):
        """Test retry logic for AWS API failures"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('boto3.Session') as mock_session:
                # Mock client that fails twice then succeeds
                from botocore.exceptions import ClientError
                
                mock_aws_client.invoke_model.side_effect = [
                    ClientError({'Error': {'Code': 'ThrottlingException'}}, 'InvokeModel'),
                    ClientError({'Error': {'Code': 'ThrottlingException'}}, 'InvokeModel'),
                    {
                        'body': Mock()
                    }
                ]
                
                # Mock successful response for third attempt
                mock_response_body = json.dumps({'embedding': np.random.rand(768).tolist()})
                mock_aws_client.invoke_model.return_value['body'].read.return_value = mock_response_body.encode()
                
                mock_session.return_value.client.return_value = mock_aws_client
                
                service = EmbeddingService()
                
                # Should succeed after retries
                result = await service.embed_texts(["test text"])
                assert result.embeddings.shape[0] == 1
                assert mock_aws_client.invoke_model.call_count == 3
    
    @pytest.mark.asyncio
    async def test_aws_retry_exhaustion(self, aws_settings, mock_aws_client):
        """Test that AWS retries are exhausted and error is raised"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('boto3.Session') as mock_session:
                from botocore.exceptions import ClientError
                
                # Mock client that always fails
                mock_aws_client.invoke_model.side_effect = ClientError(
                    {'Error': {'Code': 'ThrottlingException'}}, 'InvokeModel'
                )
                mock_session.return_value.client.return_value = mock_aws_client
                
                service = EmbeddingService()
                
                # Should raise error after max retries
                with pytest.raises(ClientError):
                    await service.embed_texts(["test text"])
    
    def test_dimension_validation_local(self, local_settings, mock_sentence_transformer):
        """Test dimension validation for local mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # Valid embeddings
                valid_embeddings = np.random.rand(5, 384).astype(np.float32)
                result = service.validate_embeddings(valid_embeddings)
                assert result['valid'] is True
                assert result['stats']['shape'] == (5, 384)
                
                # Invalid dimensions
                invalid_embeddings = np.random.rand(5, 768).astype(np.float32)
                result = service.validate_embeddings(invalid_embeddings)
                assert result['valid'] is False
                assert any("384 dimensions" in error for error in result['errors'])
    
    def test_dimension_validation_aws(self, aws_settings, mock_aws_client):
        """Test dimension validation for AWS mode"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=aws_settings):
            with patch('boto3.Session') as mock_session:
                mock_session.return_value.client.return_value = mock_aws_client
                
                service = EmbeddingService()
                
                # Valid embeddings
                valid_embeddings = np.random.rand(5, 768).astype(np.float32)
                result = service.validate_embeddings(valid_embeddings)
                assert result['valid'] is True
                assert result['stats']['shape'] == (5, 768)
                
                # Invalid dimensions
                invalid_embeddings = np.random.rand(5, 384).astype(np.float32)
                result = service.validate_embeddings(invalid_embeddings)
                assert result['valid'] is False
                assert any("768 dimensions" in error for error in result['errors'])
    
    def test_validation_edge_cases(self, local_settings, mock_sentence_transformer):
        """Test validation with edge cases"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # Empty array
                result = service.validate_embeddings(np.array([]))
                assert result['valid'] is False
                assert any("Empty embeddings" in error for error in result['errors'])
                
                # NaN values
                nan_embeddings = np.full((2, 384), np.nan, dtype=np.float32)
                result = service.validate_embeddings(nan_embeddings)
                assert result['valid'] is False
                assert any("NaN values" in error for error in result['errors'])
                
                # Infinite values
                inf_embeddings = np.full((2, 384), np.inf, dtype=np.float32)
                result = service.validate_embeddings(inf_embeddings)
                assert result['valid'] is False
                assert any("infinite values" in error for error in result['errors'])
                
                # Zero vectors
                zero_embeddings = np.zeros((2, 384), dtype=np.float32)
                result = service.validate_embeddings(zero_embeddings)
                assert result['valid'] is True  # Zero vectors are valid, just warned
                assert any("zero vectors" in warning for warning in result['warnings'])
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, local_settings, mock_sentence_transformer):
        """Test handling of empty input"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                result = await service.embed_texts([])
                assert result.embeddings.size == 0
                assert result.dimensions == 0
                assert result.processing_time >= 0
    
    def test_cache_stats(self, local_settings, mock_sentence_transformer):
        """Test cache statistics"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # Clear cache first
                service.clear_cache()
                
                # Empty cache
                stats = service.get_cache_stats()
                assert stats['total_entries'] == 0
                
                # Add some cache entries
                embedding = np.random.rand(384).astype(np.float32)
                service._store_in_cache("text1", "model1", embedding)
                service._store_in_cache("text2", "model1", embedding)
                service._store_in_cache("text3", "model2", embedding)
                
                stats = service.get_cache_stats()
                assert stats['total_entries'] == 3
                assert stats['model_counts']['model1'] == 2
                assert stats['model_counts']['model2'] == 1
                assert stats['total_size_mb'] > 0
    
    def test_cache_clearing(self, local_settings, mock_sentence_transformer):
        """Test cache clearing functionality"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                
                # Clear cache first
                service.clear_cache()
                
                # Add cache entry
                embedding = np.random.rand(384).astype(np.float32)
                service._store_in_cache("text", "model", embedding)
                assert len(service._cache) == 1
                
                # Clear cache
                service.clear_cache()
                assert len(service._cache) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_batch(self, local_settings, mock_sentence_transformer):
        """Test error handling during batch processing"""
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                # Mock to fail on second batch
                mock_sentence_transformer.encode.side_effect = [
                    np.random.rand(2, 384).astype(np.float32),  # First batch succeeds
                    Exception("Model error"),  # Second batch fails
                    np.random.rand(1, 384).astype(np.float32)   # Third batch succeeds
                ]
                
                service = EmbeddingService()
                
                # Clear cache to ensure fresh processing
                service.clear_cache()
                
                # Use unique texts to avoid cache hits
                texts = [f"unique_error_test_text_{i}_{time.time()}" for i in range(5)]
                result = await service.embed_texts(texts, batch_size=2)
                
                # Should have some results despite errors
                assert result.embeddings.shape[0] == 5
                assert result.error_count == 2  # Two texts in failed batch
    
    def test_model_name_retrieval(self, local_settings, mock_sentence_transformer):
        """Test model name retrieval for local mode"""
        # Local mode
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                assert service._get_model_name() == local_settings.hf_model_path
    
    def test_expected_dimensions(self, local_settings, mock_sentence_transformer):
        """Test expected dimensions for local mode"""
        # Local mode
        with patch('sop_qa_tool.config.settings.get_settings', return_value=local_settings):
            with patch('sop_qa_tool.services.embedder.SentenceTransformer', return_value=mock_sentence_transformer):
                service = EmbeddingService()
                assert service._get_expected_dimensions() == 384


if __name__ == "__main__":
    pytest.main([__file__])
