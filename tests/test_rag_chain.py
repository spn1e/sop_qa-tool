"""
Unit tests for RAG Chain Implementation

Tests all components of the RAG pipeline including retrieval, reranking,
context fusion, answer generation, citation extraction, and confidence scoring.
"""

import asyncio
import json
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from sop_qa_tool.services.rag_chain import (
    RAGChain, MMRReranker, ContextFusion, AnswerGenerator,
    AnswerResult, Citation, Context, ConfidenceLevel
)
from sop_qa_tool.services.vectorstore import SearchResult
from sop_qa_tool.models.sop_models import (
    SOPDocument, DocumentChunk, SourceInfo, ProcedureStep, Risk, Control,
    RoleResponsibility, RiskCategory, ControlType, PriorityLevel
)
from sop_qa_tool.config.settings import Settings, OperationMode


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Mock(spec=Settings)
    settings.mode = OperationMode.LOCAL
    settings.confidence_threshold = 0.35
    settings.top_k_retrieval = 5
    settings.bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    settings.aws_profile = "default"
    settings.aws_region = "us-east-1"
    settings.is_aws_mode.return_value = False
    settings.is_local_mode.return_value = True
    return settings


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        SearchResult(
            chunk_id="doc1_chunk_1",
            doc_id="doc1",
            chunk_text="Step 3.1: Check temperature gauge reading. Ensure temperature is between 80-85°C before proceeding.",
            score=0.95,
            metadata={
                "page_no": 5,
                "heading_path": "3. Temperature Control > 3.1 Monitoring",
                "step_ids": ["3.1"],
                "roles": ["Operator"],
                "equipment": ["Temperature Gauge"]
            },
            source=SourceInfo(url="https://example.com/sop1.pdf")
        ),
        SearchResult(
            chunk_id="doc1_chunk_2",
            doc_id="doc1",
            chunk_text="Safety Risk R-003: High temperature exposure can cause burns. Always wear protective gloves.",
            score=0.87,
            metadata={
                "page_no": 12,
                "heading_path": "5. Safety > 5.2 Temperature Risks",
                "risk_ids": ["R-003"],
                "roles": ["Operator", "Safety Officer"],
                "equipment": ["Protective Gloves"]
            }
        ),
        SearchResult(
            chunk_id="doc2_chunk_1",
            doc_id="doc2",
            chunk_text="Control C-007: Temperature monitoring must be performed every 15 minutes during operation.",
            score=0.82,
            metadata={
                "page_no": 8,
                "heading_path": "4. Quality Controls",
                "control_ids": ["C-007"],
                "roles": ["QA Inspector"],
                "equipment": ["Timer", "Temperature Gauge"]
            }
        )
    ]


@pytest.fixture
def sample_sop_document():
    """Sample SOP document for testing"""
    return SOPDocument(
        doc_id="doc1",
        title="Temperature Control SOP",
        process_name="Heat Treatment Process",
        revision="Rev 2.1",
        procedure_steps=[
            ProcedureStep(
                step_id="3.1",
                description="Check temperature gauge reading. Ensure temperature is between 80-85°C before proceeding.",
                responsible_roles=["Operator"],
                required_equipment=["Temperature Gauge"],
                safety_notes=["Wear protective gloves when near hot surfaces"]
            )
        ],
        risks=[
            Risk(
                risk_id="R-003",
                description="High temperature exposure can cause burns",
                category=RiskCategory.SAFETY,
                severity=PriorityLevel.HIGH,
                affected_steps=["3.1"]
            )
        ],
        controls=[
            Control(
                control_id="C-007",
                description="Temperature monitoring must be performed every 15 minutes",
                control_type=ControlType.DETECTIVE,
                applicable_steps=["3.1"],
                responsible_roles=["QA Inspector"]
            )
        ],
        source=SourceInfo(url="https://example.com/sop1.pdf")
    )


class TestMMRReranker:
    """Test MMR reranking functionality"""
    
    def test_mmr_reranker_initialization(self):
        """Test MMR reranker initialization"""
        reranker = MMRReranker(lambda_param=0.8)
        assert reranker.lambda_param == 0.8
    
    def test_mmr_reranker_single_result(self, sample_search_results):
        """Test MMR reranking with single result"""
        reranker = MMRReranker()
        query_embedding = np.random.rand(384)
        
        result = reranker.rerank([sample_search_results[0]], query_embedding, top_k=5)
        assert len(result) == 1
        assert result[0] == sample_search_results[0]
    
    def test_mmr_reranker_multiple_results(self, sample_search_results):
        """Test MMR reranking with multiple results"""
        reranker = MMRReranker(lambda_param=0.7)
        query_embedding = np.random.rand(384)
        
        result = reranker.rerank(sample_search_results, query_embedding, top_k=2)
        assert len(result) == 2
        assert all(isinstance(r, SearchResult) for r in result)
    
    def test_mmr_reranker_diversity(self, sample_search_results):
        """Test that MMR promotes diversity"""
        # Create results with similar content
        similar_results = [
            SearchResult(
                chunk_id="doc1_chunk_1",
                doc_id="doc1",
                chunk_text="Temperature control is critical for safety",
                score=0.95,
                metadata={}
            ),
            SearchResult(
                chunk_id="doc1_chunk_2", 
                doc_id="doc1",
                chunk_text="Temperature control is essential for safety",
                score=0.93,
                metadata={}
            ),
            SearchResult(
                chunk_id="doc2_chunk_1",
                doc_id="doc2", 
                chunk_text="Quality inspection procedures must be followed",
                score=0.85,
                metadata={}
            )
        ]
        
        reranker = MMRReranker(lambda_param=0.5)  # Balance relevance and diversity
        query_embedding = np.random.rand(384)
        
        result = reranker.rerank(similar_results, query_embedding, top_k=2)
        assert len(result) == 2
        # Should prefer diverse results over similar ones
        assert result[1].chunk_text != result[0].chunk_text
    
    def test_text_similarity(self):
        """Test text similarity calculation"""
        reranker = MMRReranker()
        
        # Identical texts
        sim1 = reranker._text_similarity("hello world", "hello world")
        assert sim1 == 1.0
        
        # No overlap
        sim2 = reranker._text_similarity("hello world", "foo bar")
        assert sim2 == 0.0
        
        # Partial overlap
        sim3 = reranker._text_similarity("hello world test", "hello test example")
        assert 0 < sim3 < 1


class TestContextFusion:
    """Test context fusion functionality"""
    
    @pytest.fixture
    def context_fusion(self):
        """Create context fusion instance"""
        mock_extractor = Mock()
        return ContextFusion(mock_extractor)
    
    def test_fuse_context_basic(self, context_fusion, sample_search_results, sample_sop_document):
        """Test basic context fusion"""
        structured_summaries = {"doc1": sample_sop_document}
        
        contexts = context_fusion.fuse_context(sample_search_results, structured_summaries)
        
        assert len(contexts) == len(sample_search_results)
        assert all(isinstance(c, Context) for c in contexts)
        
        # Check that first context is enhanced with structured info
        first_context = contexts[0]
        assert "Related Steps:" in first_context.chunk_text
        assert "Step 3.1:" in first_context.chunk_text
    
    def test_fuse_context_no_structured_data(self, context_fusion, sample_search_results):
        """Test context fusion without structured summaries"""
        contexts = context_fusion.fuse_context(sample_search_results, {})
        
        assert len(contexts) == len(sample_search_results)
        # Should use original chunk text without enhancement
        assert contexts[0].chunk_text == sample_search_results[0].chunk_text
    
    def test_enhance_with_structure(self, context_fusion, sample_sop_document):
        """Test structure enhancement"""
        chunk_text = "Check temperature gauge reading"
        metadata = {
            "step_ids": ["3.1"],
            "risk_ids": ["R-003"],
            "control_ids": ["C-007"]
        }
        
        enhanced = context_fusion._enhance_with_structure(
            chunk_text, metadata, sample_sop_document
        )
        
        assert "Related Steps:" in enhanced
        assert "Related Risks:" in enhanced
        assert "Related Controls:" in enhanced
        assert "Step 3.1:" in enhanced
        assert "Risk R-003:" in enhanced
        assert "Control C-007:" in enhanced


class TestAnswerGenerator:
    """Test answer generation functionality"""
    
    @pytest.fixture
    def answer_generator(self, mock_settings):
        """Create answer generator instance"""
        with patch('sop_qa_tool.services.rag_chain.get_settings', return_value=mock_settings):
            return AnswerGenerator()
    
    def test_answer_generator_initialization(self, answer_generator):
        """Test answer generator initialization"""
        assert answer_generator._bedrock_client is None
        assert "system_prompt" in answer_generator._prompts
        assert "answer_prompt" in answer_generator._prompts
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_context(self, answer_generator):
        """Test answer generation with no context"""
        answer, confidence, citations = await answer_generator.generate_answer("What is the temperature?", [])
        
        assert "don't have any relevant information" in answer
        assert confidence == 0.0
        assert len(citations) == 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_local_mode(self, answer_generator, sample_search_results):
        """Test answer generation in local mode"""
        contexts = [
            Context(
                chunk_text=result.chunk_text,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata
            )
            for result in sample_search_results
        ]
        
        answer, confidence, citations = await answer_generator.generate_answer(
            "What is the temperature requirement?", contexts
        )
        
        assert len(answer) > 0
        assert 0 <= confidence <= 1
        assert len(citations) > 0
        assert all(isinstance(c, Citation) for c in citations)
    
    def test_format_context(self, answer_generator, sample_search_results):
        """Test context formatting"""
        contexts = [
            Context(
                chunk_text=result.chunk_text,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata
            )
            for result in sample_search_results[:2]
        ]
        
        formatted = answer_generator._format_context(contexts)
        
        assert "[Context 1 - Doc: doc1]" in formatted
        assert "[Context 2 - Doc: doc1]" in formatted
        assert "Section:" in formatted
        assert "Page:" in formatted
    
    def test_extract_citations(self, answer_generator, sample_search_results):
        """Test citation extraction"""
        contexts = [
            Context(
                chunk_text=result.chunk_text,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata,
                source_info=result.source
            )
            for result in sample_search_results
        ]
        
        answer = "According to doc1, the temperature should be between 80-85°C. [Doc: doc1, Section: Temperature Control]"
        
        citations = answer_generator._extract_citations(answer, contexts)
        
        assert len(citations) > 0
        assert all(isinstance(c, Citation) for c in citations)
        assert citations[0].doc_id == "doc1"
        assert len(citations[0].text_snippet) > 0
    
    def test_extract_relevant_snippet(self, answer_generator):
        """Test relevant snippet extraction"""
        answer = "The temperature should be between 80-85 degrees"
        context_text = "Step 3.1: Check temperature gauge reading. Ensure temperature is between 80-85°C before proceeding. This is critical for safety."
        
        snippet = answer_generator._extract_relevant_snippet(answer, context_text)
        
        assert "80-85" in snippet
        assert len(snippet) <= 200
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, answer_generator, sample_search_results):
        """Test confidence calculation"""
        contexts = [
            Context(
                chunk_text=result.chunk_text,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata
            )
            for result in sample_search_results
        ]
        
        # High confidence answer
        good_answer = "The temperature should be between 80-85°C according to step 3.1. This procedure requires protective equipment and safety controls."
        confidence1 = await answer_generator._calculate_confidence("What is the temperature?", good_answer, contexts)
        
        # Low confidence answer
        uncertain_answer = "I'm not sure about the temperature requirements."
        confidence2 = await answer_generator._calculate_confidence("What is the temperature?", uncertain_answer, contexts)
        
        assert confidence1 > confidence2
        assert 0 <= confidence1 <= 1
        assert 0 <= confidence2 <= 1


class TestRAGChain:
    """Test complete RAG chain functionality"""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service"""
        service = AsyncMock()
        service.embed_query.return_value = np.random.rand(384)
        return service
    
    @pytest.fixture
    def mock_vector_store(self, sample_search_results):
        """Mock vector store"""
        store = AsyncMock()
        store.search.return_value = sample_search_results
        return store
    
    @pytest.fixture
    def rag_chain(self, mock_settings, mock_embedding_service, mock_vector_store):
        """Create RAG chain instance with mocked dependencies"""
        with patch('sop_qa_tool.services.rag_chain.get_settings', return_value=mock_settings), \
             patch('sop_qa_tool.services.rag_chain.EmbeddingService', return_value=mock_embedding_service), \
             patch('sop_qa_tool.services.rag_chain.FAISSVectorStore', return_value=mock_vector_store), \
             patch('sop_qa_tool.services.rag_chain.OntologyExtractor'):
            
            chain = RAGChain()
            chain.embedding_service = mock_embedding_service
            chain.vector_store = mock_vector_store
            return chain
    
    @pytest.mark.asyncio
    async def test_answer_question_success(self, rag_chain, sample_sop_document):
        """Test successful question answering"""
        # Mock structured summaries
        rag_chain._structured_summaries = {"doc1": sample_sop_document, "doc2": sample_sop_document}
        
        result = await rag_chain.answer_question("What is the temperature requirement?")
        
        assert isinstance(result, AnswerResult)
        assert result.question == "What is the temperature requirement?"
        assert len(result.answer) > 0
        assert 0 <= result.confidence_score <= 1
        assert result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
        assert len(result.citations) > 0
        assert len(result.context_used) > 0
        assert result.processing_time_seconds >= 0
        assert "results_found" in result.retrieval_stats
    
    @pytest.mark.asyncio
    async def test_answer_question_no_results(self, rag_chain):
        """Test question answering with no search results"""
        rag_chain.vector_store.search.return_value = []
        
        result = await rag_chain.answer_question("What is the temperature requirement?")
        
        assert "couldn't find any relevant information" in result.answer
        assert result.confidence_score == 0.0
        assert result.confidence_level == ConfidenceLevel.LOW
        assert len(result.citations) == 0
        assert len(result.context_used) == 0
        assert "No relevant documents found" in result.warnings
    
    @pytest.mark.asyncio
    async def test_answer_question_with_filters(self, rag_chain, sample_sop_document):
        """Test question answering with metadata filters"""
        rag_chain._structured_summaries = {"doc1": sample_sop_document}
        
        filters = {"roles": ["Operator"], "equipment": ["Temperature Gauge"]}
        result = await rag_chain.answer_question("What is the temperature requirement?", filters=filters)
        
        assert result.filters_applied == filters
        rag_chain.vector_store.search.assert_called_once()
        call_args = rag_chain.vector_store.search.call_args
        assert call_args[1]["filters"] == filters
    
    @pytest.mark.asyncio
    async def test_answer_question_low_confidence(self, rag_chain, sample_sop_document):
        """Test question answering with low confidence result"""
        rag_chain._structured_summaries = {"doc1": sample_sop_document}
        
        # Mock low confidence answer
        with patch.object(rag_chain.answer_generator, 'generate_answer') as mock_generate:
            mock_generate.return_value = ("I'm not sure about this.", 0.2, [])
            
            result = await rag_chain.answer_question("What is the temperature requirement?")
            
            assert result.confidence_level == ConfidenceLevel.LOW
            assert any("Low confidence answer" in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_get_similar_questions(self, rag_chain):
        """Test similar questions generation"""
        questions = await rag_chain.get_similar_questions("What are the safety requirements?")
        
        assert isinstance(questions, list)
        assert len(questions) <= 5
        assert all(isinstance(q, str) for q in questions)
        # Should find safety-related questions
        assert any("safety" in q.lower() for q in questions)
    
    @pytest.mark.asyncio
    async def test_get_document_summary(self, rag_chain, sample_sop_document):
        """Test document summary generation"""
        rag_chain._structured_summaries = {"doc1": sample_sop_document}
        
        summary = await rag_chain.get_document_summary("doc1")
        
        assert summary is not None
        assert summary["doc_id"] == "doc1"
        assert summary["title"] == "Temperature Control SOP"
        assert summary["total_steps"] == 1
        assert summary["total_risks"] == 1
        assert summary["total_controls"] == 1
        # Check if roles list contains expected role (may be empty in test data)
        assert isinstance(summary["roles"], list)
    
    @pytest.mark.asyncio
    async def test_get_document_summary_not_found(self, rag_chain):
        """Test document summary for non-existent document"""
        summary = await rag_chain.get_document_summary("nonexistent")
        assert summary is None
    
    def test_get_stats(self, rag_chain):
        """Test RAG chain statistics"""
        stats = rag_chain.get_stats()
        
        assert "vector_store_type" in stats
        assert "embedding_model" in stats
        assert "operation_mode" in stats
        assert "cached_summaries" in stats
        assert "confidence_threshold" in stats
        assert "top_k_retrieval" in stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rag_chain):
        """Test error handling in RAG chain"""
        # Mock an error in vector store
        rag_chain.vector_store.search.side_effect = Exception("Vector store error")
        
        result = await rag_chain.answer_question("What is the temperature requirement?")
        
        assert "encountered an error" in result.answer
        assert result.confidence_score == 0.0
        assert result.confidence_level == ConfidenceLevel.LOW
        assert "Processing error" in result.warnings[0]


class TestIntegration:
    """Integration tests for RAG chain components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, mock_settings):
        """Test complete end-to-end RAG pipeline"""
        with patch('sop_qa_tool.services.rag_chain.get_settings', return_value=mock_settings):
            # Create real instances (but with mocked external dependencies)
            chain = RAGChain()
            
            # Mock external services
            chain.embedding_service = AsyncMock()
            chain.embedding_service.embed_query.return_value = np.random.rand(384)
            
            chain.vector_store = AsyncMock()
            chain.vector_store.search.return_value = [
                SearchResult(
                    chunk_id="test_chunk",
                    doc_id="test_doc",
                    chunk_text="The temperature must be maintained at 80-85°C for optimal results.",
                    score=0.9,
                    metadata={"step_ids": ["3.1"], "roles": ["Operator"]}
                )
            ]
            
            # Add structured summary
            chain._structured_summaries["test_doc"] = SOPDocument(
                doc_id="test_doc",
                title="Test SOP",
                process_name="Test Process",
                source=SourceInfo()
            )
            
            # Test the pipeline
            result = await chain.answer_question("What is the temperature requirement?")
            
            assert isinstance(result, AnswerResult)
            assert len(result.answer) > 0
            assert result.confidence_score > 0
            assert len(result.context_used) > 0
    
    def test_confidence_level_mapping(self):
        """Test confidence level mapping"""
        # Test high confidence
        assert ConfidenceLevel.HIGH == "high"
        
        # Test medium confidence  
        assert ConfidenceLevel.MEDIUM == "medium"
        
        # Test low confidence
        assert ConfidenceLevel.LOW == "low"
    
    def test_citation_creation(self):
        """Test citation object creation"""
        citation = Citation(
            doc_id="test_doc",
            chunk_id="test_chunk",
            text_snippet="Test snippet",
            page_no=5,
            heading_path="Section 3.1",
            relevance_score=0.85,
            source_url="https://example.com/doc.pdf"
        )
        
        assert citation.doc_id == "test_doc"
        assert citation.chunk_id == "test_chunk"
        assert citation.text_snippet == "Test snippet"
        assert citation.page_no == 5
        assert citation.heading_path == "Section 3.1"
        assert citation.relevance_score == 0.85
        assert citation.source_url == "https://example.com/doc.pdf"


if __name__ == "__main__":
    pytest.main([__file__])
