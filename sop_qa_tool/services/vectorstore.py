"""
Vector Storage Service

Provides vector storage and retrieval capabilities using OpenSearch Serverless (AWS mode)
or FAISS (local mode). Supports metadata filtering, document deletion, and index management
as specified in requirements 7.1, 7.2, 4.1, and 6.3.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle

import numpy as np
import faiss

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from opensearchpy.exceptions import OpenSearchException
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from ..config.settings import get_settings
from ..models.sop_models import DocumentChunk, SourceInfo


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search operation"""
    chunk_id: str
    doc_id: str
    chunk_text: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[SourceInfo] = None


@dataclass
class IndexStats:
    """Statistics about the vector index"""
    total_documents: int
    total_chunks: int
    index_size_mb: float
    last_updated: Optional[float] = None
    model_name: Optional[str] = None
    dimensions: Optional[int] = None


class VectorStore(ABC):
    """Abstract base class for vector storage implementations"""
    
    @abstractmethod
    async def index_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: np.ndarray
    ) -> bool:
        """Index document chunks with their embeddings"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: np.ndarray, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """Get index statistics"""
        pass
    
    @abstractmethod
    async def clear_index(self) -> bool:
        """Clear the entire index"""
        pass


class OpenSearchVectorStore(VectorStore):
    """
    OpenSearch Serverless vector store implementation for AWS mode.
    
    Features:
    - HNSW algorithm with cosine similarity
    - Metadata filtering with hybrid search
    - Automatic index creation and management
    - Batch indexing for efficiency
    """
    
    def __init__(self):
        if not AWS_AVAILABLE:
            raise ImportError("opensearch-py and boto3 are required for AWS mode")
        
        self.settings = get_settings()
        self._client = None
        self._index_name = "sop-chunks"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenSearch client with AWS authentication"""
        try:
            # Get AWS credentials
            session = boto3.Session(profile_name=self.settings.aws_profile)
            credentials = session.get_credentials()
            
            # Create OpenSearch client with SigV4 auth
            self._client = OpenSearch(
                hosts=[{
                    'host': self.settings.opensearch_endpoint.replace('https://', ''),
                    'port': 443
                }],
                http_auth=('', ''),  # Empty auth, will use SigV4
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                http_compress=True,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            info = self._client.info()
            logger.info(f"Connected to OpenSearch: {info['version']['number']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise
    
    async def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            if not self._client.indices.exists(index=self._index_name):
                # Define index mapping for vector search
                mapping = {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100,
                            "knn.algo_param.ef_construction": 128,
                            "knn.space_type": "cosinesimil"
                        }
                    },
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "chunk_text": {
                                "type": "text", 
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 768,  # Titan embeddings dimension
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 128,
                                        "m": 24
                                    }
                                }
                            },
                            "metadata": {
                                "properties": {
                                    "page_no": {"type": "integer"},
                                    "heading_path": {"type": "text"},
                                    "roles": {"type": "keyword"},
                                    "equipment": {"type": "keyword"},
                                    "step_ids": {"type": "keyword"},
                                    "risk_ids": {"type": "keyword"},
                                    "control_ids": {"type": "keyword"}
                                }
                            },
                            "source": {
                                "properties": {
                                    "url": {"type": "keyword"},
                                    "page_range": {"type": "integer"}
                                }
                            },
                            "timestamp": {"type": "date"}
                        }
                    }
                }
                
                self._client.indices.create(index=self._index_name, body=mapping)
                logger.info(f"Created OpenSearch index: {self._index_name}")
            
        except Exception as e:
            logger.error(f"Failed to create OpenSearch index: {e}")
            raise
    
    async def index_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: np.ndarray
    ) -> bool:
        """Index document chunks with their embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Ensure index exists before indexing
        await self._ensure_index_exists()
        
        try:
            # Prepare bulk indexing data
            bulk_data = []
            timestamp = time.time()
            
            for chunk, embedding in zip(chunks, embeddings):
                # Index action
                bulk_data.append({
                    "index": {
                        "_index": self._index_name,
                        "_id": chunk.chunk_id
                    }
                })
                
                # Document data
                doc_data = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_text": chunk.chunk_text,
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "page_no": chunk.page_no,
                        "heading_path": chunk.heading_path,
                        "roles": chunk.roles,
                        "equipment": chunk.equipment,
                        "step_ids": chunk.step_ids,
                        "risk_ids": chunk.risk_ids,
                        "control_ids": chunk.control_ids
                    },
                    "timestamp": timestamp
                }
                
                # Add source info if available
                if hasattr(chunk, 'source') and chunk.source:
                    doc_data["source"] = {
                        "url": chunk.source.url,
                        "page_range": chunk.source.page_range
                    }
                
                bulk_data.append(doc_data)
            
            # Execute bulk indexing
            response = self._client.bulk(body=bulk_data, refresh=True)
            
            # Check for errors
            if response.get("errors"):
                error_items = [item for item in response["items"] if "error" in item.get("index", {})]
                logger.error(f"Bulk indexing errors: {error_items}")
                return False
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks with optional metadata filtering"""
        try:
            # Build query
            query = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding.tolist(),
                            "k": top_k
                        }
                    }
                },
                "_source": {
                    "excludes": ["embedding"]  # Don't return embeddings in results
                }
            }
            
            # Add filters if provided
            if filters:
                bool_query = {"bool": {"must": [query["query"]]}}
                
                # Add metadata filters
                if "roles" in filters:
                    bool_query["bool"]["must"].append({
                        "terms": {"metadata.roles": filters["roles"]}
                    })
                
                if "equipment" in filters:
                    bool_query["bool"]["must"].append({
                        "terms": {"metadata.equipment": filters["equipment"]}
                    })
                
                if "doc_id" in filters:
                    bool_query["bool"]["must"].append({
                        "term": {"doc_id": filters["doc_id"]}
                    })
                
                if "step_ids" in filters:
                    bool_query["bool"]["must"].append({
                        "terms": {"metadata.step_ids": filters["step_ids"]}
                    })
                
                query["query"] = bool_query
            
            # Execute search
            response = self._client.search(index=self._index_name, body=query)
            
            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                # Create source info if available
                source_info = None
                if "source" in source:
                    source_info = SourceInfo(
                        url=source["source"].get("url"),
                        page_range=source["source"].get("page_range")
                    )
                
                result = SearchResult(
                    chunk_id=source["chunk_id"],
                    doc_id=source["doc_id"],
                    chunk_text=source["chunk_text"],
                    score=hit["_score"],
                    metadata=source.get("metadata", {}),
                    source=source_info
                )
                results.append(result)
            
            logger.debug(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Delete by query
            delete_query = {
                "query": {
                    "term": {"doc_id": doc_id}
                }
            }
            
            response = self._client.delete_by_query(
                index=self._index_name,
                body=delete_query,
                refresh=True
            )
            
            deleted_count = response.get("deleted", 0)
            logger.info(f"Deleted {deleted_count} chunks for document {doc_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def get_stats(self) -> IndexStats:
        """Get index statistics"""
        try:
            # Get index stats
            stats_response = self._client.indices.stats(index=self._index_name)
            index_stats = stats_response["indices"][self._index_name]
            
            # Get document count
            count_response = self._client.count(index=self._index_name)
            total_chunks = count_response["count"]
            
            # Get unique document count
            agg_query = {
                "size": 0,
                "aggs": {
                    "unique_docs": {
                        "cardinality": {
                            "field": "doc_id"
                        }
                    }
                }
            }
            agg_response = self._client.search(index=self._index_name, body=agg_query)
            total_documents = agg_response["aggregations"]["unique_docs"]["value"]
            
            # Calculate index size
            index_size_bytes = index_stats["total"]["store"]["size_in_bytes"]
            index_size_mb = index_size_bytes / (1024 * 1024)
            
            return IndexStats(
                total_documents=total_documents,
                total_chunks=total_chunks,
                index_size_mb=index_size_mb,
                last_updated=time.time(),
                model_name="amazon.titan-embed-text-v2:0",
                dimensions=768
            )
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                index_size_mb=0.0
            )
    
    async def clear_index(self) -> bool:
        """Clear the entire index"""
        try:
            # Delete all documents
            delete_query = {"query": {"match_all": {}}}
            response = self._client.delete_by_query(
                index=self._index_name,
                body=delete_query,
                refresh=True
            )
            
            deleted_count = response.get("deleted", 0)
            logger.info(f"Cleared index: deleted {deleted_count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation for local mode.
    
    Features:
    - IndexFlatIP for cosine similarity
    - Persistent storage with incremental updates
    - Metadata stored in separate JSON files
    - Memory-efficient batch operations
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._index = None
        self._metadata = {}
        self._doc_to_chunks = {}
        self._chunk_counter = 0
        
        # File paths
        self._index_file = self.settings.faiss_index_path / "vector.index"
        self._metadata_file = self.settings.faiss_index_path / "metadata.json"
        self._doc_mapping_file = self.settings.faiss_index_path / "doc_mapping.json"
        
        # Ensure directory exists
        self.settings.faiss_index_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self._index_file.exists():
                # Load FAISS index
                self._index = faiss.read_index(str(self._index_file))
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
                
                # Load metadata
                if self._metadata_file.exists():
                    with open(self._metadata_file, 'r') as f:
                        self._metadata = json.load(f)
                
                # Load document mapping
                if self._doc_mapping_file.exists():
                    with open(self._doc_mapping_file, 'r') as f:
                        self._doc_to_chunks = json.load(f)
                
                # Set chunk counter
                self._chunk_counter = len(self._metadata)
                
            else:
                # Create new index (will be initialized when first chunks are added)
                self._index = None
                self._metadata = {}
                self._doc_to_chunks = {}
                self._chunk_counter = 0
                logger.info("No existing FAISS index found, will create new one")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Reset to empty state
            self._index = None
            self._metadata = {}
            self._doc_to_chunks = {}
            self._chunk_counter = 0
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self._index is not None:
                # Save FAISS index
                faiss.write_index(self._index, str(self._index_file))
                
                # Save metadata
                with open(self._metadata_file, 'w') as f:
                    json.dump(self._metadata, f, indent=2)
                
                # Save document mapping
                with open(self._doc_mapping_file, 'w') as f:
                    json.dump(self._doc_to_chunks, f, indent=2)
                
                logger.debug("Saved FAISS index and metadata")
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def index_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: np.ndarray
    ) -> bool:
        """Index document chunks with their embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if len(chunks) == 0:
            return True
        
        try:
            # Initialize index if needed
            if self._index is None:
                dimension = embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                logger.info(f"Created new FAISS index with dimension {dimension}")
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # Add to index
            self._index.add(embeddings_normalized)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                chunk_idx = self._chunk_counter + i
                
                # Store chunk metadata
                self._metadata[str(chunk_idx)] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_text": chunk.chunk_text,
                    "page_no": chunk.page_no,
                    "heading_path": chunk.heading_path,
                    "roles": chunk.roles,
                    "equipment": chunk.equipment,
                    "step_ids": chunk.step_ids,
                    "risk_ids": chunk.risk_ids,
                    "control_ids": chunk.control_ids,
                    "timestamp": time.time()
                }
                
                # Update document to chunks mapping
                if chunk.doc_id not in self._doc_to_chunks:
                    self._doc_to_chunks[chunk.doc_id] = []
                self._doc_to_chunks[chunk.doc_id].append(chunk_idx)
            
            self._chunk_counter += len(chunks)
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks with optional metadata filtering"""
        if self._index is None or self._index.ntotal == 0:
            logger.warning("No vectors in index")
            return []
        
        try:
            # Normalize query embedding
            query_normalized = query_embedding.copy().reshape(1, -1)
            faiss.normalize_L2(query_normalized)
            
            # Search with larger k to allow for filtering
            search_k = min(top_k * 3, self._index.ntotal)  # Get more results for filtering
            scores, indices = self._index.search(query_normalized, search_k)
            
            # Convert results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                metadata = self._metadata.get(str(idx))
                if not metadata:
                    continue
                
                # Apply filters
                if filters and not self._matches_filters(metadata, filters):
                    continue
                
                result = SearchResult(
                    chunk_id=metadata["chunk_id"],
                    doc_id=metadata["doc_id"],
                    chunk_text=metadata["chunk_text"],
                    score=float(score),
                    metadata={
                        "page_no": metadata.get("page_no"),
                        "heading_path": metadata.get("heading_path"),
                        "roles": metadata.get("roles", []),
                        "equipment": metadata.get("equipment", []),
                        "step_ids": metadata.get("step_ids", []),
                        "risk_ids": metadata.get("risk_ids", []),
                        "control_ids": metadata.get("control_ids", [])
                    }
                )
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
            
            logger.debug(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters"""
        try:
            # Check doc_id filter
            if "doc_id" in filters and metadata.get("doc_id") != filters["doc_id"]:
                return False
            
            # Check roles filter (any role in filter list must be in metadata roles)
            if "roles" in filters:
                filter_roles = filters["roles"]
                metadata_roles = metadata.get("roles", [])
                if not any(role in metadata_roles for role in filter_roles):
                    return False
            
            # Check equipment filter
            if "equipment" in filters:
                filter_equipment = filters["equipment"]
                metadata_equipment = metadata.get("equipment", [])
                if not any(eq in metadata_equipment for eq in filter_equipment):
                    return False
            
            # Check step_ids filter
            if "step_ids" in filters:
                filter_steps = filters["step_ids"]
                metadata_steps = metadata.get("step_ids", [])
                if not any(step in metadata_steps for step in filter_steps):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter matching failed: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        if doc_id not in self._doc_to_chunks:
            logger.warning(f"Document {doc_id} not found in index")
            return False
        
        try:
            chunk_indices = self._doc_to_chunks[doc_id]
            
            # Remove from metadata
            for chunk_idx in chunk_indices:
                if str(chunk_idx) in self._metadata:
                    del self._metadata[str(chunk_idx)]
            
            # Remove from document mapping
            del self._doc_to_chunks[doc_id]
            
            # Note: FAISS doesn't support efficient deletion of individual vectors
            # For now, we just remove from metadata. In a production system,
            # you might want to rebuild the index periodically to reclaim space.
            
            # Save updated metadata
            self._save_index()
            
            logger.info(f"Deleted {len(chunk_indices)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def get_stats(self) -> IndexStats:
        """Get index statistics"""
        try:
            total_chunks = self._index.ntotal if self._index else 0
            total_documents = len(self._doc_to_chunks)
            
            # Calculate index size
            index_size_mb = 0.0
            if self._index_file.exists():
                index_size_mb += self._index_file.stat().st_size / (1024 * 1024)
            if self._metadata_file.exists():
                index_size_mb += self._metadata_file.stat().st_size / (1024 * 1024)
            if self._doc_mapping_file.exists():
                index_size_mb += self._doc_mapping_file.stat().st_size / (1024 * 1024)
            
            # Get last update time
            last_updated = None
            if self._metadata_file.exists():
                last_updated = self._metadata_file.stat().st_mtime
            
            return IndexStats(
                total_documents=total_documents,
                total_chunks=total_chunks,
                index_size_mb=index_size_mb,
                last_updated=last_updated,
                model_name=self.settings.hf_model_path,
                dimensions=self._index.d if self._index else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                index_size_mb=0.0
            )
    
    async def clear_index(self) -> bool:
        """Clear the entire index"""
        try:
            # Reset in-memory structures
            self._index = None
            self._metadata = {}
            self._doc_to_chunks = {}
            self._chunk_counter = 0
            
            # Remove files
            for file_path in [self._index_file, self._metadata_file, self._doc_mapping_file]:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("Cleared FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False


class VectorStoreService:
    """
    Main vector storage service that provides a unified interface
    for both AWS (OpenSearch) and local (FAISS) modes.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the appropriate vector store based on mode"""
        try:
            if self.settings.is_aws_mode():
                self._store = OpenSearchVectorStore()
                logger.info("Initialized OpenSearch vector store")
            else:
                self._store = FAISSVectorStore()
                logger.info("Initialized FAISS vector store")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def index_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: np.ndarray
    ) -> bool:
        """Index document chunks with their embeddings"""
        return await self._store.index_chunks(chunks, embeddings)
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar chunks"""
        return await self._store.search(query_embedding, filters, top_k)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        return await self._store.delete_document(doc_id)
    
    async def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return await self._store.get_stats()
    
    async def clear_index(self) -> bool:
        """Clear the entire index"""
        return await self._store.clear_index()
    
    def get_store_type(self) -> str:
        """Get the type of vector store being used"""
        store_class_name = self._store.__class__.__name__
        if store_class_name == "OpenSearchVectorStore":
            return "opensearch"
        elif store_class_name == "FAISSVectorStore":
            return "faiss"
        else:
            return "unknown"