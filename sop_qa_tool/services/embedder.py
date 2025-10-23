"""
Embeddings Service Implementation

Provides vector embeddings for text using Amazon Titan Text Embeddings v2 (AWS mode)
or sentence-transformers (local mode). Includes batch processing, caching, dimension
validation, retry logic, and rate limiting.
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    embeddings: np.ndarray
    dimensions: int
    model_name: str
    processing_time: float
    cached_count: int = 0
    error_count: int = 0


@dataclass
class EmbeddingCache:
    """Cache entry for embeddings"""
    embedding: np.ndarray
    timestamp: float
    model_name: str
    text_hash: str


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this call
        self.calls.append(now)


class EmbeddingService:
    """
    Embeddings service supporting both AWS Titan and local sentence-transformers.
    
    Features:
    - Dual mode operation (AWS/local)
    - Batch processing for efficiency
    - Persistent caching
    - Dimension validation
    - Retry logic with exponential backoff
    - Rate limiting for API calls
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._local_model = None
        self._aws_client = None
        self._cache = {}
        self._cache_file = None
        self._rate_limiter = None
        
        # Initialize based on mode
        if self.settings.is_aws_mode():
            self._init_aws_mode()
        else:
            self._init_local_mode()
        
        # Load cache
        self._load_cache()
    
    def _init_aws_mode(self):
        """Initialize AWS Bedrock client and rate limiter"""
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS mode. Install with: pip install boto3")
        
        try:
            # Initialize Bedrock client
            session = boto3.Session(profile_name=self.settings.aws_profile)
            self._aws_client = session.client(
                'bedrock-runtime',
                region_name=self.settings.aws_region
            )
            
            # Test connection
            self._aws_client.list_foundation_models()
            logger.info("AWS Bedrock client initialized successfully")
            
            # Initialize rate limiter (Titan has generous limits, but we'll be conservative)
            self._rate_limiter = RateLimiter(max_calls=100, time_window=60.0)
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {e}")
            raise
    
    def _init_local_mode(self):
        """Initialize local sentence-transformers model"""
        try:
            logger.info(f"Loading local model: {self.settings.hf_model_path}")
            self._local_model = SentenceTransformer(self.settings.hf_model_path)
            logger.info("Local sentence-transformers model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        if self.settings.is_local_mode():
            self._cache_file = self.settings.local_data_path / "embeddings_cache.pkl"
        else:
            # For AWS mode, we could use S3 or local cache
            self._cache_file = Path("./data/embeddings_cache.pkl")
        
        if self._cache_file and self._cache_file.exists():
            try:
                with open(self._cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._cache = {}
        else:
            self._cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        if not self._cache_file:
            return
        
        try:
            # Ensure directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination"""
        text_hash = self._get_text_hash(text)
        return f"{model_name}:{text_hash}"
    
    def _get_from_cache(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available"""
        cache_key = self._get_cache_key(text, model_name)
        
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            # Check if cache entry is still valid (not too old)
            age_hours = (time.time() - cache_entry.timestamp) / 3600
            if age_hours < 24 * 7:  # Cache valid for 1 week
                return cache_entry.embedding
        
        return None
    
    def _store_in_cache(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model_name)
        text_hash = self._get_text_hash(text)
        
        cache_entry = EmbeddingCache(
            embedding=embedding,
            timestamp=time.time(),
            model_name=model_name,
            text_hash=text_hash
        )
        
        self._cache[cache_key] = cache_entry
        
        # Periodically save cache
        if len(self._cache) % 100 == 0:
            self._save_cache()
    
    async def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts with batch processing.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                dimensions=0,
                model_name="none",
                processing_time=0.0
            )
        
        start_time = time.time()
        batch_size = batch_size or self.settings.embedding_batch_size
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        cached_count = 0
        
        model_name = self._get_model_name()
        
        for i, text in enumerate(texts):
            cached_embedding = self._get_from_cache(text, model_name)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                cached_count += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        logger.info(f"Found {cached_count} cached embeddings, computing {len(uncached_texts)} new ones")
        
        # Process uncached texts in batches
        error_count = 0
        
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]
                
                try:
                    if self.settings.is_aws_mode():
                        batch_embeddings = await self._embed_batch_aws(batch_texts)
                    else:
                        batch_embeddings = await self._embed_batch_local(batch_texts)
                    
                    # Store results and cache
                    for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                        original_index = batch_indices[i]
                        embeddings[original_index] = embedding
                        self._store_in_cache(text, model_name, embedding)
                    
                except Exception as e:
                    logger.error(f"Failed to embed batch {batch_start}-{batch_end}: {e}")
                    error_count += len(batch_texts)
                    
                    # Fill with zero embeddings for failed texts
                    zero_embedding = np.zeros(self._get_expected_dimensions())
                    for i in range(len(batch_texts)):
                        original_index = batch_indices[i]
                        embeddings[original_index] = zero_embedding
        
        # Convert to numpy array and validate
        embeddings_array = np.array([emb for emb in embeddings if emb is not None])
        
        if len(embeddings_array) == 0:
            raise ValueError("No embeddings could be generated")
        
        # Validate dimensions
        expected_dims = self._get_expected_dimensions()
        actual_dims = embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 0
        
        if actual_dims != expected_dims:
            raise ValueError(f"Embedding dimension mismatch: expected {expected_dims}, got {actual_dims}")
        
        processing_time = time.time() - start_time
        
        # Save cache
        self._save_cache()
        
        return EmbeddingResult(
            embeddings=embeddings_array,
            dimensions=actual_dims,
            model_name=model_name,
            processing_time=processing_time,
            cached_count=cached_count,
            error_count=error_count
        )
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        result = await self.embed_texts([query])
        return result.embeddings[0]
    
    async def _embed_batch_aws(self, texts: List[str]) -> List[np.ndarray]:
        """Embed batch of texts using AWS Titan"""
        if not self._aws_client:
            raise RuntimeError("AWS client not initialized")
        
        embeddings = []
        
        for text in texts:
            # Rate limiting
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            
            # Retry logic with exponential backoff
            max_retries = 3
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Prepare request
                    request_body = {
                        "inputText": text[:8000],  # Titan has input length limits
                        "dimensions": 768,  # Titan v2 supports 768 dimensions
                        "normalize": True
                    }
                    
                    # Make API call
                    response = self._aws_client.invoke_model(
                        modelId=self.settings.titan_embeddings_id,
                        body=json.dumps(request_body),
                        contentType='application/json',
                        accept='application/json'
                    )
                    
                    # Parse response
                    response_body = json.loads(response['body'].read())
                    embedding = np.array(response_body['embedding'], dtype=np.float32)
                    embeddings.append(embedding)
                    break
                    
                except (ClientError, BotoCoreError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"AWS embedding failed after {max_retries} attempts: {e}")
                        raise
                    
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"AWS embedding attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        return embeddings
    
    async def _embed_batch_local(self, texts: List[str]) -> List[np.ndarray]:
        """Embed batch of texts using local sentence-transformers"""
        if not self._local_model:
            raise RuntimeError("Local model not initialized")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                embeddings = await loop.run_in_executor(
                    executor, 
                    self._local_model.encode, 
                    texts
                )
            
            # Convert to list of numpy arrays
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise
    
    def _get_model_name(self) -> str:
        """Get the current model name for caching"""
        if self.settings.is_aws_mode():
            return self.settings.titan_embeddings_id
        else:
            return self.settings.hf_model_path
    
    def _get_expected_dimensions(self) -> int:
        """Get expected embedding dimensions for current mode"""
        if self.settings.is_aws_mode():
            return 768  # Titan Text Embeddings v2
        else:
            return 384  # all-MiniLM-L6-v2
    
    def validate_embeddings(self, embeddings: np.ndarray) -> Dict[str, any]:
        """
        Validate embedding array for consistency and correctness.
        
        Args:
            embeddings: Numpy array of embeddings to validate
            
        Returns:
            Dictionary with validation results
        """
        if embeddings.size == 0:
            return {
                'valid': False,
                'errors': ['Empty embeddings array'],
                'warnings': [],
                'stats': {}
            }
        
        errors = []
        warnings = []
        
        # Check dimensions
        expected_dims = self._get_expected_dimensions()
        if len(embeddings.shape) != 2:
            errors.append(f"Expected 2D array, got {len(embeddings.shape)}D")
        elif embeddings.shape[1] != expected_dims:
            errors.append(f"Expected {expected_dims} dimensions, got {embeddings.shape[1]}")
        
        # Check for NaN or infinite values
        if np.isnan(embeddings).any():
            errors.append("Embeddings contain NaN values")
        
        if np.isinf(embeddings).any():
            errors.append("Embeddings contain infinite values")
        
        # Check if embeddings are normalized (for AWS mode)
        if self.settings.is_aws_mode():
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                warnings.append("Embeddings may not be properly normalized")
        
        # Check for zero vectors
        zero_vectors = np.all(embeddings == 0, axis=1).sum()
        if zero_vectors > 0:
            warnings.append(f"Found {zero_vectors} zero vectors")
        
        # Calculate statistics
        stats = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            'zero_vectors': int(zero_vectors)
        }
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get statistics about the embedding cache"""
        if not self._cache:
            return {'total_entries': 0, 'cache_file_exists': False}
        
        # Group by model
        model_counts = {}
        total_size = 0
        oldest_timestamp = float('inf')
        newest_timestamp = 0
        
        for cache_entry in self._cache.values():
            model = cache_entry.model_name
            model_counts[model] = model_counts.get(model, 0) + 1
            total_size += cache_entry.embedding.nbytes
            oldest_timestamp = min(oldest_timestamp, cache_entry.timestamp)
            newest_timestamp = max(newest_timestamp, cache_entry.timestamp)
        
        return {
            'total_entries': len(self._cache),
            'model_counts': model_counts,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_file_exists': self._cache_file.exists() if self._cache_file else False,
            'oldest_entry_age_hours': (time.time() - oldest_timestamp) / 3600 if oldest_timestamp != float('inf') else 0,
            'newest_entry_age_hours': (time.time() - newest_timestamp) / 3600 if newest_timestamp > 0 else 0
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        if self._cache_file and self._cache_file.exists():
            self._cache_file.unlink()
        logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup: save cache on destruction"""
        try:
            self._save_cache()
        except:
            pass  # Ignore errors during cleanup