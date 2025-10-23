"""
RAG Chain Implementation

Provides retrieval-augmented generation capabilities for SOP question answering.
Includes vector search, metadata filtering, MMR reranking, context fusion,
LLM-powered answer generation, citation extraction, and confidence scoring.

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import re
import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

from ..config.settings import get_settings
from ..models.sop_models import SOPDocument, DocumentChunk, SourceInfo
from .vectorstore import VectorStore, SearchResult, OpenSearchVectorStore, FAISSVectorStore
from .embedder import EmbeddingService
from .ontology_extractor import OntologyExtractor


logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for answers"""
    HIGH = "high"      # > 0.7
    MEDIUM = "medium"  # 0.35 - 0.7
    LOW = "low"        # < 0.35


@dataclass
class Citation:
    """Citation linking answer claims to source documents"""
    doc_id: str
    chunk_id: str
    text_snippet: str
    page_no: Optional[int] = None
    heading_path: Optional[str] = None
    relevance_score: float = 0.0
    source_url: Optional[str] = None


@dataclass
class Context:
    """Context information for answer generation"""
    chunk_text: str
    doc_id: str
    chunk_id: str
    relevance_score: float
    metadata: Dict[str, Any]
    source_info: Optional[SourceInfo] = None


@dataclass
class AnswerResult:
    """Complete result from RAG chain processing"""
    question: str
    answer: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    citations: List[Citation]
    context_used: List[Context]
    processing_time_seconds: float
    retrieval_stats: Dict[str, Any]
    filters_applied: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MMRReranker:
    """Maximal Marginal Relevance reranker for result diversity"""
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR reranker.
        
        Args:
            lambda_param: Balance between relevance and diversity (0-1)
                         Higher values favor relevance, lower favor diversity
        """
        self.lambda_param = lambda_param
    
    def rerank(
        self, 
        results: List[SearchResult], 
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results using MMR for diversity.
        
        Args:
            results: Initial search results
            query_embedding: Original query embedding
            top_k: Number of results to return
            
        Returns:
            Reranked results with improved diversity
        """
        if len(results) <= 1:
            return results
        
        try:
            # Convert results to embeddings (we'll need to re-embed for MMR)
            # For now, use a simplified approach based on text similarity
            selected = []
            remaining = results.copy()
            
            # Select first result (highest relevance)
            if remaining:
                selected.append(remaining.pop(0))
            
            # Select remaining results using MMR
            while len(selected) < top_k and remaining:
                best_score = -float('inf')
                best_idx = 0
                
                for i, candidate in enumerate(remaining):
                    # Relevance score (already computed)
                    relevance = candidate.score
                    
                    # Diversity score (similarity to already selected)
                    max_similarity = 0.0
                    for selected_result in selected:
                        # Simple text-based similarity as approximation
                        similarity = self._text_similarity(
                            candidate.chunk_text, 
                            selected_result.chunk_text
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR score
                    mmr_score = (
                        self.lambda_param * relevance - 
                        (1 - self.lambda_param) * max_similarity
                    )
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                selected.append(remaining.pop(best_idx))
            
            logger.debug(f"MMR reranking: {len(results)} -> {len(selected)} results")
            return selected
            
        except Exception as e:
            logger.warning(f"MMR reranking failed, using original order: {e}")
            return results[:top_k]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class ContextFusion:
    """Combines chunks with structured summaries for enhanced context"""
    
    def __init__(self, ontology_extractor: OntologyExtractor):
        self.ontology_extractor = ontology_extractor
    
    def fuse_context(
        self, 
        search_results: List[SearchResult],
        structured_summaries: Dict[str, SOPDocument]
    ) -> List[Context]:
        """
        Combine search results with structured document summaries.
        
        Args:
            search_results: Vector search results
            structured_summaries: Extracted SOP documents by doc_id
            
        Returns:
            Enhanced context with both chunk text and structured info
        """
        contexts = []
        
        for result in search_results:
            # Base context from search result
            context = Context(
                chunk_text=result.chunk_text,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata,
                source_info=result.source
            )
            
            # Enhance with structured information if available
            if result.doc_id in structured_summaries:
                sop_doc = structured_summaries[result.doc_id]
                enhanced_text = self._enhance_with_structure(
                    result.chunk_text, 
                    result.metadata, 
                    sop_doc
                )
                context.chunk_text = enhanced_text
            
            contexts.append(context)
        
        return contexts
    
    def _enhance_with_structure(
        self, 
        chunk_text: str, 
        metadata: Dict[str, Any], 
        sop_doc: SOPDocument
    ) -> str:
        """Enhance chunk text with relevant structured information"""
        enhancements = []
        
        # Add relevant procedure steps
        step_ids = metadata.get('step_ids', [])
        if step_ids:
            relevant_steps = [
                step for step in sop_doc.procedure_steps 
                if step.step_id in step_ids
            ]
            if relevant_steps:
                step_info = []
                for step in relevant_steps:
                    step_details = f"Step {step.step_id}: {step.description}"
                    if step.responsible_roles:
                        step_details += f" (Roles: {', '.join(step.responsible_roles)})"
                    if step.required_equipment:
                        step_details += f" (Equipment: {', '.join(step.required_equipment)})"
                    step_info.append(step_details)
                enhancements.append("Related Steps:\n" + "\n".join(step_info))
        
        # Add relevant risks
        risk_ids = metadata.get('risk_ids', [])
        if risk_ids:
            relevant_risks = [
                risk for risk in sop_doc.risks 
                if risk.risk_id in risk_ids
            ]
            if relevant_risks:
                risk_info = [
                    f"Risk {risk.risk_id}: {risk.description} (Category: {risk.category})"
                    for risk in relevant_risks
                ]
                enhancements.append("Related Risks:\n" + "\n".join(risk_info))
        
        # Add relevant controls
        control_ids = metadata.get('control_ids', [])
        if control_ids:
            relevant_controls = [
                control for control in sop_doc.controls 
                if control.control_id in control_ids
            ]
            if relevant_controls:
                control_info = [
                    f"Control {control.control_id}: {control.description} (Type: {control.control_type})"
                    for control in relevant_controls
                ]
                enhancements.append("Related Controls:\n" + "\n".join(control_info))
        
        # Combine original text with enhancements
        if enhancements:
            return chunk_text + "\n\n" + "\n\n".join(enhancements)
        else:
            return chunk_text


class AnswerGenerator:
    """LLM-powered answer generation with citation extraction"""
    
    def __init__(self):
        self.settings = get_settings()
        self._bedrock_client = None
        self._prompts = self._load_prompts()
    
    def _get_bedrock_client(self):
        """Get or create Bedrock client for AWS mode"""
        if self._bedrock_client is None and self.settings.is_aws_mode():
            if not AWS_AVAILABLE:
                raise ImportError("boto3 is required for AWS mode")
            try:
                session = boto3.Session(
                    profile_name=self.settings.aws_profile,
                    region_name=self.settings.aws_region
                )
                self._bedrock_client = session.client('bedrock-runtime')
                logger.info("Initialized Bedrock client for answer generation")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        return self._bedrock_client
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts for answer generation"""
        return {
            "system_prompt": """You are an expert assistant for manufacturing Standard Operating Procedures (SOPs) and Work Instructions. Your role is to provide accurate, helpful answers to questions about factory procedures, safety protocols, quality controls, and operational processes.

Key guidelines:
1. Base your answers strictly on the provided context from SOP documents
2. Include specific citations with document IDs and relevant details
3. If information is not in the context, clearly state "I don't have information about that"
4. Focus on practical, actionable information for factory workers and auditors
5. Highlight safety considerations and quality requirements when relevant
6. Use clear, professional language appropriate for manufacturing environments
7. When uncertain, indicate your confidence level and suggest consulting source documents

Always structure your response with:
- Direct answer to the question
- Supporting details from the context
- Relevant safety or quality notes if applicable
- Clear citations to source documents""",
            
            "answer_prompt": """Based on the following context from SOP documents, please answer the user's question.

Question: {question}

Context:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Uses specific information from the context
3. Includes citations in the format [Doc: document_id, Section: heading_path]
4. Indicates if any information is missing or uncertain
5. Highlights relevant safety or quality considerations

Answer:""",
            
            "confidence_prompt": """Evaluate the confidence level for this answer based on the question and available context.

Question: {question}
Answer: {answer}
Context Quality: {context_summary}

Consider:
- How well the context addresses the question
- Completeness of information
- Clarity and specificity of the answer
- Any gaps or uncertainties

Provide a confidence score from 0.0 to 1.0 and brief explanation."""
        }
    
    async def generate_answer(
        self, 
        question: str, 
        contexts: List[Context]
    ) -> Tuple[str, float, List[Citation]]:
        """
        Generate answer from question and context.
        
        Args:
            question: User's question
            contexts: List of context information
            
        Returns:
            Tuple of (answer, confidence_score, citations)
        """
        if not contexts:
            return (
                "I don't have any relevant information to answer your question. "
                "Please try rephrasing your question or check if the relevant documents have been ingested.",
                0.0,
                []
            )
        
        try:
            # Prepare context text
            context_text = self._format_context(contexts)
            
            # Generate answer
            if self.settings.is_aws_mode():
                answer = await self._generate_with_bedrock(question, context_text)
            else:
                answer = await self._generate_with_local_model(question, context_text)
            
            # Extract citations
            citations = self._extract_citations(answer, contexts)
            
            # Calculate confidence score
            confidence = await self._calculate_confidence(question, answer, contexts)
            
            return answer, confidence, citations
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return (
                "I encountered an error while processing your question. Please try again.",
                0.0,
                []
            )
    
    def _format_context(self, contexts: List[Context]) -> str:
        """Format contexts into a single text for the LLM"""
        formatted_parts = []
        
        for i, context in enumerate(contexts, 1):
            part = f"[Context {i} - Doc: {context.doc_id}]"
            
            if context.metadata.get('heading_path'):
                part += f"\nSection: {context.metadata['heading_path']}"
            
            if context.metadata.get('page_no'):
                part += f"\nPage: {context.metadata['page_no']}"
            
            part += f"\nContent: {context.chunk_text}\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    async def _generate_with_bedrock(self, question: str, context: str) -> str:
        """Generate answer using AWS Bedrock"""
        try:
            client = self._get_bedrock_client()
            
            prompt = self._prompts["answer_prompt"].format(
                question=question,
                context=context
            )
            
            response = client.invoke_model(
                modelId=self.settings.bedrock_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "system": self._prompts["system_prompt"],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Bedrock answer generation failed: {e}")
            raise
    
    async def _generate_with_local_model(self, question: str, context: str) -> str:
        """Generate answer using local model (simplified fallback)"""
        # For local mode, provide a template-based response
        # In a production system, you might use a local LLM like Ollama
        
        # Extract key information from context
        key_info = []
        for line in context.split('\n'):
            if any(keyword in line.lower() for keyword in ['step', 'procedure', 'safety', 'risk', 'control']):
                key_info.append(line.strip())
        
        if key_info:
            answer = f"Based on the available SOP documentation:\n\n"
            answer += "\n".join(key_info[:5])  # Limit to top 5 relevant lines
            answer += "\n\nPlease refer to the source documents for complete details."
        else:
            answer = "I found some relevant information in the SOP documents, but cannot provide a detailed answer in local mode. Please refer to the source documents for complete information."
        
        return answer
    
    def _extract_citations(self, answer: str, contexts: List[Context]) -> List[Citation]:
        """Extract citations from answer text and context"""
        citations = []
        
        # Look for explicit citation patterns in the answer
        citation_patterns = [
            r'\[Doc:\s*([^,\]]+)(?:,\s*Section:\s*([^\]]+))?\]',
            r'\(Doc:\s*([^,\)]+)(?:,\s*Section:\s*([^\)]+))?\)',
            r'document\s+([^\s,]+)',
            r'section\s+([^\s,]+)'
        ]
        
        cited_docs = set()
        for pattern in citation_patterns:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                doc_ref = match.group(1).strip()
                cited_docs.add(doc_ref)
        
        # Create citations for referenced contexts
        for context in contexts:
            # Check if this context is referenced in the answer
            is_cited = (
                context.doc_id in cited_docs or
                any(keyword in answer.lower() for keyword in [
                    context.doc_id.lower(),
                    context.metadata.get('heading_path', '').lower()
                ])
            )
            
            if is_cited or len(citations) < 3:  # Include top contexts as citations
                # Extract relevant snippet from context
                snippet = self._extract_relevant_snippet(answer, context.chunk_text)
                
                citation = Citation(
                    doc_id=context.doc_id,
                    chunk_id=context.chunk_id,
                    text_snippet=snippet,
                    page_no=context.metadata.get('page_no'),
                    heading_path=context.metadata.get('heading_path'),
                    relevance_score=context.relevance_score,
                    source_url=context.source_info.url if context.source_info else None
                )
                citations.append(citation)
        
        return citations[:5]  # Limit to top 5 citations
    
    def _extract_relevant_snippet(self, answer: str, context_text: str, max_length: int = 200) -> str:
        """Extract most relevant snippet from context for citation"""
        # Simple approach: find sentences in context that share words with answer
        answer_words = set(answer.lower().split())
        sentences = re.split(r'[.!?]+', context_text)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(answer_words.intersection(sentence_words))
            score = overlap / len(sentence_words) if sentence_words else 0
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # Truncate if too long
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length] + "..."
        
        return best_sentence or context_text[:max_length] + "..."
    
    async def _calculate_confidence(
        self, 
        question: str, 
        answer: str, 
        contexts: List[Context]
    ) -> float:
        """Calculate confidence score for the answer"""
        try:
            # Factors for confidence calculation
            factors = []
            
            # 1. Context relevance (average of context scores)
            if contexts:
                avg_relevance = sum(c.relevance_score for c in contexts) / len(contexts)
                factors.append(min(avg_relevance, 1.0))
            else:
                factors.append(0.0)
            
            # 2. Answer completeness (length and detail)
            answer_length_score = min(len(answer) / 500, 1.0)  # Normalize to 500 chars
            factors.append(answer_length_score)
            
            # 3. Specificity (presence of specific terms)
            specific_terms = ['step', 'procedure', 'safety', 'control', 'equipment', 'role']
            specificity_score = sum(1 for term in specific_terms if term.lower() in answer.lower()) / len(specific_terms)
            factors.append(specificity_score)
            
            # 4. Uncertainty indicators (negative factor)
            uncertainty_phrases = ['i don\'t know', 'not sure', 'unclear', 'might be', 'possibly']
            uncertainty_penalty = sum(1 for phrase in uncertainty_phrases if phrase in answer.lower()) * 0.2
            
            # Calculate weighted average
            base_confidence = sum(factors) / len(factors)
            final_confidence = max(0.0, min(1.0, base_confidence - uncertainty_penalty))
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence


class RAGChain:
    """
    Complete RAG chain for SOP question answering.
    
    Orchestrates retrieval, reranking, context fusion, answer generation,
    and citation extraction with confidence scoring.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.ontology_extractor = OntologyExtractor()
        
        # Initialize vector store based on mode
        if self.settings.is_aws_mode():
            self.vector_store = OpenSearchVectorStore()
        else:
            self.vector_store = FAISSVectorStore()
        
        # Initialize components
        self.mmr_reranker = MMRReranker(lambda_param=0.7)
        self.context_fusion = ContextFusion(self.ontology_extractor)
        self.answer_generator = AnswerGenerator()
        
        # Cache for structured summaries
        self._structured_summaries: Dict[str, SOPDocument] = {}
    
    async def answer_question(
        self, 
        question: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> AnswerResult:
        """
        Answer a question using the complete RAG pipeline.
        
        Args:
            question: User's question
            filters: Optional metadata filters (roles, equipment, doc_id, etc.)
            top_k: Number of results to retrieve (uses config default if None)
            
        Returns:
            Complete answer result with citations and metadata
        """
        start_time = time.time()
        top_k = top_k or self.settings.top_k_retrieval
        warnings = []
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Query Analysis and Embedding
            query_embedding = await self.embedding_service.embed_query(question)
            
            # Step 2: Vector Retrieval with Filtering
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k * 2  # Get more results for reranking
            )
            
            if not search_results:
                return AnswerResult(
                    question=question,
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your question or check if the relevant documents have been ingested.",
                    confidence_score=0.0,
                    confidence_level=ConfidenceLevel.LOW,
                    citations=[],
                    context_used=[],
                    processing_time_seconds=time.time() - start_time,
                    retrieval_stats={"results_found": 0, "filters_applied": filters},
                    filters_applied=filters,
                    warnings=["No relevant documents found"]
                )
            
            # Step 3: MMR Reranking for Diversity
            reranked_results = self.mmr_reranker.rerank(
                search_results, 
                query_embedding, 
                top_k
            )
            
            # Step 4: Load Structured Summaries
            await self._ensure_structured_summaries(reranked_results)
            
            # Step 5: Context Fusion
            contexts = self.context_fusion.fuse_context(
                reranked_results, 
                self._structured_summaries
            )
            
            # Step 6: Answer Generation with Citations
            answer, confidence_score, citations = await self.answer_generator.generate_answer(
                question, 
                contexts
            )
            
            # Step 7: Determine Confidence Level
            if confidence_score >= 0.7:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence_score >= self.settings.confidence_threshold:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW
                warnings.append("Low confidence answer - please verify with source documents")
            
            # Compile retrieval statistics
            retrieval_stats = {
                "results_found": len(search_results),
                "results_after_reranking": len(reranked_results),
                "contexts_used": len(contexts),
                "avg_relevance_score": sum(r.score for r in reranked_results) / len(reranked_results),
                "filters_applied": filters
            }
            
            processing_time = time.time() - start_time
            
            logger.info(f"Question answered in {processing_time:.2f}s with {confidence_level} confidence")
            
            return AnswerResult(
                question=question,
                answer=answer,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                citations=citations,
                context_used=contexts,
                processing_time_seconds=processing_time,
                retrieval_stats=retrieval_stats,
                filters_applied=filters,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"RAG chain processing failed: {e}")
            return AnswerResult(
                question=question,
                answer="I encountered an error while processing your question. Please try again.",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.LOW,
                citations=[],
                context_used=[],
                processing_time_seconds=time.time() - start_time,
                retrieval_stats={"error": str(e)},
                filters_applied=filters,
                warnings=[f"Processing error: {str(e)}"]
            )
    
    async def _ensure_structured_summaries(self, search_results: List[SearchResult]):
        """Ensure structured summaries are loaded for search results"""
        doc_ids = {result.doc_id for result in search_results}
        missing_docs = doc_ids - set(self._structured_summaries.keys())
        
        if missing_docs:
            logger.debug(f"Loading structured summaries for {len(missing_docs)} documents")
            # In a full implementation, you would load these from storage
            # For now, we'll create minimal summaries
            for doc_id in missing_docs:
                self._structured_summaries[doc_id] = SOPDocument(
                    doc_id=doc_id,
                    title=f"SOP Document {doc_id}",
                    process_name="Unknown Process",
                    source=SourceInfo()
                )
    
    async def get_similar_questions(
        self, 
        question: str, 
        limit: int = 5
    ) -> List[str]:
        """
        Get similar questions that might help the user.
        
        Args:
            question: User's question
            limit: Maximum number of suggestions
            
        Returns:
            List of similar question suggestions
        """
        try:
            # This is a simplified implementation
            # In practice, you might maintain a database of common questions
            # or use the vector store to find similar question patterns
            
            common_questions = [
                "What are the safety requirements for this procedure?",
                "Who is responsible for performing this step?",
                "What equipment is needed for this process?",
                "What are the quality checkpoints?",
                "What risks are associated with this procedure?",
                "How long does this process take?",
                "What are the prerequisites for starting?",
                "What controls are in place for this risk?"
            ]
            
            # Simple keyword matching for suggestions
            question_words = set(question.lower().split())
            scored_questions = []
            
            for q in common_questions:
                q_words = set(q.lower().split())
                overlap = len(question_words.intersection(q_words))
                if overlap > 0:
                    scored_questions.append((overlap, q))
            
            # Sort by relevance and return top suggestions
            scored_questions.sort(reverse=True)
            return [q for _, q in scored_questions[:limit]]
            
        except Exception as e:
            logger.warning(f"Similar questions generation failed: {e}")
            return []
    
    async def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a specific document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document summary with key information
        """
        try:
            if doc_id in self._structured_summaries:
                sop_doc = self._structured_summaries[doc_id]
                
                return {
                    "doc_id": sop_doc.doc_id,
                    "title": sop_doc.title,
                    "process_name": sop_doc.process_name,
                    "revision": sop_doc.revision,
                    "effective_date": sop_doc.effective_date,
                    "total_steps": len(sop_doc.procedure_steps),
                    "total_risks": len(sop_doc.risks),
                    "total_controls": len(sop_doc.controls),
                    "roles": list(set(role.role for role in sop_doc.roles_responsibilities)),
                    "equipment": sop_doc.materials_equipment[:10],  # Top 10 equipment items
                    "high_priority_risks": len([r for r in sop_doc.risks if r.overall_rating in ["high", "critical"]])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Document summary generation failed for {doc_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG chain statistics"""
        return {
            "vector_store_type": type(self.vector_store).__name__,
            "embedding_model": self.embedding_service._get_model_name(),
            "operation_mode": self.settings.mode.value,
            "cached_summaries": len(self._structured_summaries),
            "confidence_threshold": self.settings.confidence_threshold,
            "top_k_retrieval": self.settings.top_k_retrieval
        }