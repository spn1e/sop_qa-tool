"""
Unit tests for Vector Storage Service

Tests both OpenSearch Serverless and FAISS implementations with metadata filtering,
document deletion, and index management functionality.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
import numpy as np

from sop_qa_tool.services.vectorstore import (
    VectorStoreService, 
    OpenSearchVectorStore, 
    FAISSVectorStore,
    SearchResult,
    IndexStats
)
from sop_qa_tool.models.sop_models import DocumentChunk, SourceInfo
from sop_qa_tool.config.settings import Settings, OperationMode


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing"""
    chunks = [
        DocumentChunk(
            chunk_id="doc1_chunk1",
            doc_id="doc1",
            chunk_text="Step 1: Prepare the equipment by checking temperature settings.",
            chunk_index=0,
            page_no=1,
            heading_path="1. Preparation",
            step_ids=["1.1"],
            roles=["Operator"],
            equipment=["Temperature Controller"]
        ),
        DocumentChunk(
            chunk_id="doc1_chunk2", 
            doc_id="doc1",
            chunk_text="Step 2: Verify safety protocols are in place before starting.",
            chunk_index=1,
            page_no=1,
            heading_path="2. Safety",
            step_ids=["2.1"],
            roles=["Safety Officer", "Operator"],
            equipment=["Safety Equipment"]
        ),
        DocumentChunk(
            chunk_id="doc2_chunk1",
            doc_id="doc2", 
            chunk_text="Quality check: Inspect product for defects using visual inspection.",
            chunk_index=0,
            page_no=2,
            heading_path="3. Quality Control",
            step_ids=["3.1"],
            roles=["QA Inspector"],
            equipment=["Inspection Tools"]
        )
    ]
    return chunks


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing"""
    # Create 3 embeddings with 384 dimensions (local mode)
    embeddings = np.random.rand(3, 384).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def sample_embeddings_aws():
    """Create sample embeddings for AWS mode (768 dimensions)"""
    embeddings = np.random.rand(3, 768).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFAISSVectorStore:
    """Test FAISS vector store implementation"""
    
    @pytest.fixture
    def faiss_store(self, temp_dir):
        """Create FAISS store with temporary directory"""
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings:
            settings = Mock()
            settings.faiss_index_path = temp_dir / "faiss_index"
            settings.hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value = settings
            
            store = FAISSVectorStore()
            return store
    
    @pytest.mark.asyncio
    async def test_index_chunks_success(self, faiss_store, sample_chunks, sample_embeddings):
        """Test successful chunk indexing"""
        result = await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        assert result is True
        assert faiss_store._index is not None
        assert faiss_store._index.ntotal == 3
        assert len(faiss_store._metadata) == 3
        assert len(faiss_store._doc_to_chunks) == 2  # 2 unique documents
    
    @pytest.mark.asyncio
    async def test_index_chunks_mismatch_error(self, faiss_store, sample_chunks, sample_embeddings):
        """Test error when chunks and embeddings count mismatch"""
        with pytest.raises(ValueError, match="Number of chunks must match number of embeddings"):
            await faiss_store.index_chunks(sample_chunks, sample_embeddings[:2])
    
    @pytest.mark.asyncio
    async def test_search_basic(self, faiss_store, sample_chunks, sample_embeddings):
        """Test basic vector search"""
        # Index chunks first
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        # Search with first embedding
        query_embedding = sample_embeddings[0]
        results = await faiss_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0 for r in results)  # Cosine similarity scores
        
        # First result should be the exact match
        if results:
            assert results[0].chunk_id == "doc1_chunk1"
            assert results[0].doc_id == "doc1"
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, faiss_store, sample_chunks, sample_embeddings):
        """Test search with metadata filters"""
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        # Search with role filter
        query_embedding = sample_embeddings[0]
        results = await faiss_store.search(
            query_embedding, 
            filters={"roles": ["QA Inspector"]},
            top_k=5
        )
        
        # Should only return chunks with QA Inspector role
        assert len(results) == 1
        assert results[0].chunk_id == "doc2_chunk1"
        assert "QA Inspector" in results[0].metadata["roles"]
    
    @pytest.mark.asyncio
    async def test_search_with_equipment_filter(self, faiss_store, sample_chunks, sample_embeddings):
        """Test search with equipment filter"""
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = await faiss_store.search(
            query_embedding,
            filters={"equipment": ["Temperature Controller"]},
            top_k=5
        )
        
        assert len(results) == 1
        assert results[0].chunk_id == "doc1_chunk1"
        assert "Temperature Controller" in results[0].metadata["equipment"]
    
    @pytest.mark.asyncio
    async def test_search_with_doc_id_filter(self, faiss_store, sample_chunks, sample_embeddings):
        """Test search with document ID filter"""
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = await faiss_store.search(
            query_embedding,
            filters={"doc_id": "doc1"},
            top_k=5
        )
        
        assert len(results) == 2  # doc1 has 2 chunks
        assert all(r.doc_id == "doc1" for r in results)
    
    @pytest.mark.asyncio
    async def test_search_empty_index(self, faiss_store):
        """Test search on empty index"""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = await faiss_store.search(query_embedding)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_delete_document(self, faiss_store, sample_chunks, sample_embeddings):
        """Test document deletion"""
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        # Delete doc1
        result = await faiss_store.delete_document("doc1")
        assert result is True
        
        # Verify doc1 chunks are removed from metadata
        remaining_chunks = [
            chunk_id for chunk_id, metadata in faiss_store._metadata.items()
            if metadata["doc_id"] == "doc1"
        ]
        assert len(remaining_chunks) == 0
        
        # Verify doc1 is removed from document mapping
        assert "doc1" not in faiss_store._doc_to_chunks
        
        # Verify doc2 is still there
        assert "doc2" in faiss_store._doc_to_chunks
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, faiss_store):
        """Test deletion of non-existent document"""
        result = await faiss_store.delete_document("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_stats(self, faiss_store, sample_chunks, sample_embeddings):
        """Test getting index statistics"""
        # Empty index stats
        stats = await faiss_store.get_stats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        
        # Index some chunks
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        stats = await faiss_store.get_stats()
        assert stats.total_documents == 2
        assert stats.total_chunks == 3
        assert stats.dimensions == 384
        assert stats.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert stats.index_size_mb >= 0
    
    @pytest.mark.asyncio
    async def test_clear_index(self, faiss_store, sample_chunks, sample_embeddings):
        """Test clearing the entire index"""
        await faiss_store.index_chunks(sample_chunks, sample_embeddings)
        
        # Verify index has data
        assert faiss_store._index.ntotal == 3
        assert len(faiss_store._metadata) == 3
        
        # Clear index
        result = await faiss_store.clear_index()
        assert result is True
        
        # Verify index is empty
        assert faiss_store._index is None
        assert len(faiss_store._metadata) == 0
        assert len(faiss_store._doc_to_chunks) == 0
        assert faiss_store._chunk_counter == 0
    
    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir, sample_chunks, sample_embeddings):
        """Test that index persists across instances"""
        # Create first store and index data
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings:
            settings = Mock()
            settings.faiss_index_path = temp_dir / "faiss_index"
            settings.hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value = settings
            
            store1 = FAISSVectorStore()
            await store1.index_chunks(sample_chunks, sample_embeddings)
            
            # Verify files exist
            assert (temp_dir / "faiss_index" / "vector.index").exists()
            assert (temp_dir / "faiss_index" / "metadata.json").exists()
            assert (temp_dir / "faiss_index" / "doc_mapping.json").exists()
        
        # Create second store and verify data is loaded
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings:
            settings = Mock()
            settings.faiss_index_path = temp_dir / "faiss_index"
            settings.hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value = settings
            
            store2 = FAISSVectorStore()
            
            assert store2._index.ntotal == 3
            assert len(store2._metadata) == 3
            assert len(store2._doc_to_chunks) == 2


class TestOpenSearchVectorStore:
    """Test OpenSearch vector store implementation"""
    
    @pytest.fixture
    def mock_opensearch_client(self):
        """Mock OpenSearch client"""
        client = Mock()
        client.info.return_value = {"version": {"number": "2.0.0"}}
        client.indices.exists.return_value = False
        client.indices.create.return_value = {"acknowledged": True}
        client.bulk.return_value = {"errors": False, "items": []}
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "chunk_id": "doc1_chunk1",
                            "doc_id": "doc1", 
                            "chunk_text": "Test chunk",
                            "metadata": {
                                "roles": ["Operator"],
                                "equipment": ["Test Equipment"]
                            }
                        }
                    }
                ]
            }
        }
        client.delete_by_query.return_value = {"deleted": 2}
        client.count.return_value = {"count": 3}
        client.indices.stats.return_value = {
            "indices": {
                "sop-chunks": {
                    "total": {
                        "store": {"size_in_bytes": 1024000}
                    }
                }
            }
        }
        return client
    
    @pytest.fixture
    def opensearch_store(self, mock_opensearch_client):
        """Create OpenSearch store with mocked client"""
        with patch('sop_qa_tool.services.vectorstore.AWS_AVAILABLE', True), \
             patch('boto3.Session') as mock_boto3_session, \
             patch('sop_qa_tool.services.vectorstore.OpenSearch') as mock_os, \
             patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings:
            
            # Mock settings
            settings = Mock()
            settings.is_aws_mode.return_value = True
            settings.aws_profile = "default"
            settings.aws_region = "us-east-1"
            settings.opensearch_endpoint = "https://test.us-east-1.aoss.amazonaws.com"
            mock_settings.return_value = settings
            
            # Mock boto3 session
            mock_session = Mock()
            mock_session.get_credentials.return_value = Mock()
            mock_session.client.return_value = Mock()
            mock_boto3_session.return_value = mock_session
            
            # Mock OpenSearch client
            mock_os.return_value = mock_opensearch_client
            
            store = OpenSearchVectorStore()
            store._client = mock_opensearch_client
            return store
    
    @pytest.mark.asyncio
    async def test_index_chunks_success(self, opensearch_store, sample_chunks, sample_embeddings_aws):
        """Test successful chunk indexing in OpenSearch"""
        result = await opensearch_store.index_chunks(sample_chunks, sample_embeddings_aws)
        
        assert result is True
        opensearch_store._client.bulk.assert_called_once()
        
        # Verify bulk data structure
        call_args = opensearch_store._client.bulk.call_args[1]
        bulk_data = call_args["body"]
        
        # Should have index action + document for each chunk
        assert len(bulk_data) == 6  # 3 chunks * 2 entries each
        
        # Check first document structure
        doc_data = bulk_data[1]  # First document (after index action)
        assert doc_data["chunk_id"] == "doc1_chunk1"
        assert doc_data["doc_id"] == "doc1"
        assert "embedding" in doc_data
        assert "metadata" in doc_data
    
    @pytest.mark.asyncio
    async def test_index_chunks_with_errors(self, opensearch_store, sample_chunks, sample_embeddings_aws):
        """Test indexing with bulk errors"""
        opensearch_store._client.bulk.return_value = {
            "errors": True,
            "items": [{"index": {"error": "Test error"}}]
        }
        
        result = await opensearch_store.index_chunks(sample_chunks, sample_embeddings_aws)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_basic(self, opensearch_store, sample_embeddings_aws):
        """Test basic vector search in OpenSearch"""
        query_embedding = sample_embeddings_aws[0]
        results = await opensearch_store.search(query_embedding, top_k=5)
        
        assert len(results) == 1
        assert results[0].chunk_id == "doc1_chunk1"
        assert results[0].doc_id == "doc1"
        assert results[0].score == 0.95
        
        # Verify search query structure
        opensearch_store._client.search.assert_called_once()
        call_args = opensearch_store._client.search.call_args[1]
        query = call_args["body"]
        
        assert query["size"] == 5
        assert "knn" in query["query"]
        assert "embedding" in query["query"]["knn"]
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, opensearch_store, sample_embeddings_aws):
        """Test search with metadata filters in OpenSearch"""
        query_embedding = sample_embeddings_aws[0]
        filters = {
            "roles": ["Operator"],
            "equipment": ["Test Equipment"],
            "doc_id": "doc1"
        }
        
        await opensearch_store.search(query_embedding, filters=filters, top_k=5)
        
        # Verify filter structure in query
        call_args = opensearch_store._client.search.call_args[1]
        query = call_args["body"]["query"]
        
        assert "bool" in query
        assert "must" in query["bool"]
        
        # Should have original knn query plus filters
        must_clauses = query["bool"]["must"]
        assert len(must_clauses) == 4  # knn + 3 filters
    
    @pytest.mark.asyncio
    async def test_delete_document(self, opensearch_store):
        """Test document deletion in OpenSearch"""
        result = await opensearch_store.delete_document("doc1")
        
        assert result is True
        opensearch_store._client.delete_by_query.assert_called_once()
        
        # Verify delete query structure
        call_args = opensearch_store._client.delete_by_query.call_args[1]
        delete_query = call_args["body"]
        
        assert "query" in delete_query
        assert "term" in delete_query["query"]
        assert delete_query["query"]["term"]["doc_id"] == "doc1"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, opensearch_store):
        """Test getting index statistics from OpenSearch"""
        # Mock aggregation response
        opensearch_store._client.search.return_value = {
            "aggregations": {
                "unique_docs": {"value": 2}
            }
        }
        
        stats = await opensearch_store.get_stats()
        
        assert stats.total_documents == 2
        assert stats.total_chunks == 3
        assert abs(stats.index_size_mb - 1.0) < 0.1  # 1024000 bytes = ~1MB
        assert stats.model_name == "amazon.titan-embed-text-v2:0"
        assert stats.dimensions == 768
    
    @pytest.mark.asyncio
    async def test_clear_index(self, opensearch_store):
        """Test clearing the entire OpenSearch index"""
        result = await opensearch_store.clear_index()
        
        assert result is True
        opensearch_store._client.delete_by_query.assert_called_once()
        
        # Verify delete all query
        call_args = opensearch_store._client.delete_by_query.call_args[1]
        delete_query = call_args["body"]
        
        assert delete_query["query"]["match_all"] == {}


class TestVectorStoreService:
    """Test the main vector store service"""
    
    @pytest.mark.asyncio
    async def test_service_initialization_local_mode(self):
        """Test service initializes FAISS store in local mode"""
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings, \
             patch('sop_qa_tool.services.vectorstore.FAISSVectorStore') as mock_faiss:
            
            settings = Mock()
            settings.is_aws_mode.return_value = False
            mock_settings.return_value = settings
            
            # Create a mock store with the right class name
            mock_store = Mock()
            mock_store.__class__.__name__ = "FAISSVectorStore"
            mock_faiss.return_value = mock_store
            
            service = VectorStoreService()
            
            mock_faiss.assert_called_once()
            assert service.get_store_type() == "faiss"
    
    @pytest.mark.asyncio
    async def test_service_initialization_aws_mode(self):
        """Test service initializes OpenSearch store in AWS mode"""
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings, \
             patch('sop_qa_tool.services.vectorstore.OpenSearchVectorStore') as mock_os, \
             patch('sop_qa_tool.services.vectorstore.AWS_AVAILABLE', True):
            
            settings = Mock()
            settings.is_aws_mode.return_value = True
            mock_settings.return_value = settings
            
            # Create a mock store with the right class name
            mock_store = Mock()
            mock_store.__class__.__name__ = "OpenSearchVectorStore"
            mock_os.return_value = mock_store
            
            service = VectorStoreService()
            
            mock_os.assert_called_once()
            assert service.get_store_type() == "opensearch"
    
    @pytest.mark.asyncio
    async def test_service_delegates_to_store(self, sample_chunks, sample_embeddings):
        """Test that service properly delegates calls to underlying store"""
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings, \
             patch('sop_qa_tool.services.vectorstore.FAISSVectorStore') as mock_faiss_class:
            
            settings = Mock()
            settings.is_aws_mode.return_value = False
            mock_settings.return_value = settings
            
            # Mock store instance
            mock_store = Mock()
            mock_store.index_chunks = AsyncMock(return_value=True)
            mock_store.search = AsyncMock(return_value=[])
            mock_store.delete_document = AsyncMock(return_value=True)
            mock_store.get_stats = AsyncMock(return_value=IndexStats(0, 0, 0.0))
            mock_store.clear_index = AsyncMock(return_value=True)
            
            mock_faiss_class.return_value = mock_store
            
            service = VectorStoreService()
            
            # Test all methods delegate properly
            await service.index_chunks(sample_chunks, sample_embeddings)
            mock_store.index_chunks.assert_called_once_with(sample_chunks, sample_embeddings)
            
            await service.search(sample_embeddings[0])
            mock_store.search.assert_called_once()
            
            await service.delete_document("doc1")
            mock_store.delete_document.assert_called_once_with("doc1")
            
            await service.get_stats()
            mock_store.get_stats.assert_called_once()
            
            await service.clear_index()
            mock_store.clear_index.assert_called_once()


class TestIntegration:
    """Integration tests for vector storage functionality"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_faiss(self, temp_dir, sample_chunks, sample_embeddings):
        """Test complete workflow with FAISS store"""
        with patch('sop_qa_tool.services.vectorstore.get_settings') as mock_settings:
            settings = Mock()
            settings.is_aws_mode.return_value = False
            settings.faiss_index_path = temp_dir / "faiss_index"
            settings.hf_model_path = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value = settings
            
            service = VectorStoreService()
            
            # 1. Index chunks
            result = await service.index_chunks(sample_chunks, sample_embeddings)
            assert result is True
            
            # 2. Search without filters
            query_embedding = sample_embeddings[0]
            results = await service.search(query_embedding, top_k=3)
            assert len(results) > 0
            assert results[0].chunk_id == "doc1_chunk1"  # Should be exact match
            
            # 3. Search with filters
            filtered_results = await service.search(
                query_embedding,
                filters={"roles": ["QA Inspector"]},
                top_k=3
            )
            assert len(filtered_results) == 1
            assert filtered_results[0].chunk_id == "doc2_chunk1"
            
            # 4. Get stats
            stats = await service.get_stats()
            assert stats.total_documents == 2
            assert stats.total_chunks == 3
            
            # 5. Delete document
            delete_result = await service.delete_document("doc1")
            assert delete_result is True
            
            # 6. Verify deletion
            stats_after_delete = await service.get_stats()
            assert stats_after_delete.total_documents == 1
            
            # 7. Clear index
            clear_result = await service.clear_index()
            assert clear_result is True
            
            # 8. Verify clearing
            final_stats = await service.get_stats()
            assert final_stats.total_documents == 0
            assert final_stats.total_chunks == 0


if __name__ == "__main__":
    pytest.main([__file__])
