"""
RAG Chain Demo

Demonstrates the complete RAG (Retrieval-Augmented Generation) pipeline
for SOP question answering including vector search, reranking, context fusion,
answer generation, and citation extraction.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sop_qa_tool.services.rag_chain import RAGChain, ConfidenceLevel
from sop_qa_tool.services.vectorstore import SearchResult
from sop_qa_tool.models.sop_models import (
    SOPDocument, DocumentChunk, SourceInfo, ProcedureStep, Risk, Control,
    RoleResponsibility, RiskCategory, ControlType, PriorityLevel
)
from sop_qa_tool.config.settings import get_settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample SOP data for demonstration"""
    
    # Sample document chunks
    chunks = [
        DocumentChunk(
            chunk_id="sop_001_chunk_1",
            doc_id="SOP-FILL-001",
            chunk_text="""
            Step 3.1: Temperature Verification
            Before starting the filling process, verify that the filler temperature is maintained between 80-85°C.
            Use the digital temperature gauge (TG-001) to check the reading.
            If temperature is outside range, notify the Line Supervisor immediately.
            Required PPE: Heat-resistant gloves, safety glasses.
            """,
            chunk_index=1,
            page_no=5,
            heading_path="3. Filling Process > 3.1 Temperature Control",
            step_ids=["3.1"],
            roles=["Operator", "Line Supervisor"],
            equipment=["Digital Temperature Gauge TG-001", "Heat-resistant Gloves"]
        ),
        DocumentChunk(
            chunk_id="sop_001_chunk_2", 
            doc_id="SOP-FILL-001",
            chunk_text="""
            Risk R-003: High Temperature Exposure
            Description: Exposure to high temperature surfaces during filling operation can cause severe burns.
            Probability: Medium | Severity: High | Overall Rating: High
            Affected Steps: 3.1, 3.2, 3.3
            Potential Consequences: First/second degree burns, production delays, worker compensation claims
            Triggers: Equipment malfunction, inadequate PPE, procedural non-compliance
            """,
            chunk_index=2,
            page_no=12,
            heading_path="5. Risk Assessment > 5.2 Temperature Risks",
            risk_ids=["R-003"],
            roles=["Safety Officer", "Operator"],
            equipment=["PPE", "Temperature Monitoring Equipment"]
        ),
        DocumentChunk(
            chunk_id="sop_001_chunk_3",
            doc_id="SOP-FILL-001", 
            chunk_text="""
            Control C-007: Temperature Monitoring Protocol
            Description: Continuous temperature monitoring during filling operations with automated alerts.
            Type: Detective Control | Effectiveness: High
            Applicable Risks: R-003, R-004
            Applicable Steps: 3.1, 3.2, 3.3
            Responsible Roles: QA Inspector, Process Technician
            Verification Method: Digital monitoring system with 15-minute interval checks
            Frequency: Continuous during operation
            """,
            chunk_index=3,
            page_no=18,
            heading_path="6. Control Measures > 6.3 Temperature Controls",
            control_ids=["C-007"],
            roles=["QA Inspector", "Process Technician"],
            equipment=["Digital Monitoring System", "Temperature Sensors"]
        ),
        DocumentChunk(
            chunk_id="sop_002_chunk_1",
            doc_id="SOP-QC-002",
            chunk_text="""
            Step 2.1: Quality Inspection Setup
            Prepare the inspection station with all required tools and documentation.
            Ensure proper lighting (minimum 500 lux) and clean work surface.
            Calibrate measuring instruments according to schedule QC-CAL-001.
            Required Equipment: Calipers, Micrometers, Go/No-Go Gauges, Inspection Forms
            Duration: 15 minutes
            """,
            chunk_index=1,
            page_no=3,
            heading_path="2. Inspection Setup > 2.1 Station Preparation",
            step_ids=["2.1"],
            roles=["QC Inspector"],
            equipment=["Calipers", "Micrometers", "Go/No-Go Gauges"]
        )
    ]
    
    # Sample SOP documents
    sop_documents = {
        "SOP-FILL-001": SOPDocument(
            doc_id="SOP-FILL-001",
            title="Bottle Filling Process SOP",
            process_name="Automated Bottle Filling",
            revision="Rev 3.2",
            owner_role="Production Manager",
            procedure_steps=[
                ProcedureStep(
                    step_id="3.1",
                    title="Temperature Verification",
                    description="Verify filler temperature is between 80-85°C using digital gauge TG-001",
                    responsible_roles=["Operator"],
                    required_equipment=["Digital Temperature Gauge TG-001", "Heat-resistant Gloves"],
                    duration_minutes=5,
                    safety_notes=["Wear heat-resistant gloves", "Maintain safe distance from hot surfaces"],
                    acceptance_criteria=["Temperature reading between 80-85°C", "Gauge calibration current"]
                )
            ],
            risks=[
                Risk(
                    risk_id="R-003",
                    description="High temperature exposure can cause severe burns",
                    category=RiskCategory.SAFETY,
                    probability=PriorityLevel.MEDIUM,
                    severity=PriorityLevel.HIGH,
                    overall_rating=PriorityLevel.HIGH,
                    affected_steps=["3.1", "3.2", "3.3"],
                    potential_consequences=["First/second degree burns", "Production delays"],
                    triggers=["Equipment malfunction", "Inadequate PPE", "Procedural non-compliance"]
                )
            ],
            controls=[
                Control(
                    control_id="C-007",
                    description="Continuous temperature monitoring with automated alerts",
                    control_type=ControlType.DETECTIVE,
                    effectiveness=PriorityLevel.HIGH,
                    applicable_risks=["R-003"],
                    applicable_steps=["3.1", "3.2", "3.3"],
                    responsible_roles=["QA Inspector", "Process Technician"],
                    verification_method="Digital monitoring system with 15-minute checks",
                    frequency="Continuous during operation"
                )
            ],
            roles_responsibilities=[
                RoleResponsibility(
                    role="Operator",
                    responsibilities=[
                        "Perform temperature verification",
                        "Wear required PPE",
                        "Report temperature deviations immediately"
                    ],
                    qualifications=["Basic training certification", "PPE training"],
                    authority_level="operator"
                ),
                RoleResponsibility(
                    role="QA Inspector", 
                    responsibilities=[
                        "Monitor temperature control system",
                        "Verify control effectiveness",
                        "Document inspection results"
                    ],
                    qualifications=["QC certification", "5+ years experience"],
                    authority_level="supervisor"
                )
            ],
            materials_equipment=[
                "Digital Temperature Gauge TG-001",
                "Heat-resistant Gloves",
                "Safety Glasses",
                "Temperature Monitoring System",
                "Automated Filling Equipment"
            ],
            source=SourceInfo(
                url="https://factory.local/sops/SOP-FILL-001.pdf",
                page_range=[1, 25]
            )
        ),
        "SOP-QC-002": SOPDocument(
            doc_id="SOP-QC-002",
            title="Quality Control Inspection SOP",
            process_name="Product Quality Inspection",
            revision="Rev 2.1",
            owner_role="Quality Manager",
            procedure_steps=[
                ProcedureStep(
                    step_id="2.1",
                    title="Inspection Setup",
                    description="Prepare inspection station with required tools and documentation",
                    responsible_roles=["QC Inspector"],
                    required_equipment=["Calipers", "Micrometers", "Go/No-Go Gauges"],
                    duration_minutes=15,
                    acceptance_criteria=["All tools calibrated", "Work area clean", "Proper lighting verified"]
                )
            ],
            source=SourceInfo(
                url="https://factory.local/sops/SOP-QC-002.pdf",
                page_range=[1, 15]
            )
        )
    }
    
    return chunks, sop_documents


class MockVectorStore:
    """Mock vector store for demonstration"""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.embedding_service = None
    
    async def search(self, query_embedding, filters=None, top_k=5):
        """Mock search that returns relevant results based on keywords"""
        import numpy as np
        
        # Simple keyword-based matching for demo
        # Get query text from the embedding service
        query_text = ""
        if self.embedding_service and hasattr(self.embedding_service, 'query_text'):
            query_text = self.embedding_service.query_text
        results = []
        
        for chunk in self.chunks:
            # Calculate simple relevance score based on keyword overlap
            chunk_words = set(chunk.chunk_text.lower().split())
            query_words = set(query_text.lower().split())
            overlap = len(chunk_words.intersection(query_words))
            
            if overlap > 0:
                score = min(overlap / len(query_words), 1.0) if query_words else 0.5
                
                # Apply filters if provided
                if filters:
                    if 'roles' in filters and not any(role in chunk.roles for role in filters['roles']):
                        continue
                    if 'equipment' in filters and not any(eq in chunk.equipment for eq in filters['equipment']):
                        continue
                    if 'doc_id' in filters and chunk.doc_id != filters['doc_id']:
                        continue
                
                results.append(SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    chunk_text=chunk.chunk_text,
                    score=score,
                    metadata={
                        'page_no': chunk.page_no,
                        'heading_path': chunk.heading_path,
                        'step_ids': chunk.step_ids,
                        'risk_ids': chunk.risk_ids,
                        'control_ids': chunk.control_ids,
                        'roles': chunk.roles,
                        'equipment': chunk.equipment
                    },
                    source=SourceInfo(url="https://factory.local/demo.pdf")
                ))
        
        # Sort by relevance score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class MockEmbeddingService:
    """Mock embedding service for demonstration"""
    
    def __init__(self):
        self.query_text = ""
    
    async def embed_query(self, query: str):
        """Mock embedding that includes query text for keyword matching"""
        import numpy as np
        self.query_text = query  # Store query text for mock search
        embedding = np.random.rand(384)
        return embedding
    
    def _get_model_name(self):
        """Mock method for getting model name"""
        return "mock-embedding-model"


async def demonstrate_rag_chain():
    """Demonstrate the complete RAG chain functionality"""
    
    print("🔧 RAG Chain Demo - SOP Question Answering System")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample SOP data...")
    chunks, sop_documents = create_sample_data()
    print(f"✅ Created {len(chunks)} document chunks and {len(sop_documents)} SOP documents")
    
    # Initialize RAG chain with mock services
    print("\n🚀 Initializing RAG chain...")
    rag_chain = RAGChain()
    
    # Replace with mock services for demo
    mock_vector_store = MockVectorStore(chunks)
    mock_embedding_service = MockEmbeddingService()
    
    rag_chain.vector_store = mock_vector_store
    rag_chain.embedding_service = mock_embedding_service
    rag_chain._structured_summaries = sop_documents
    
    # Connect the services so vector store can access query text
    mock_vector_store.embedding_service = mock_embedding_service
    
    print("✅ RAG chain initialized with mock services")
    
    # Demo questions
    demo_questions = [
        {
            "question": "What is the temperature requirement for the filling process?",
            "filters": None,
            "description": "Basic question about temperature requirements"
        },
        {
            "question": "What safety risks are associated with high temperature?",
            "filters": {"roles": ["Safety Officer"]},
            "description": "Safety-focused question with role filter"
        },
        {
            "question": "What equipment is needed for quality inspection?",
            "filters": {"doc_id": "SOP-QC-002"},
            "description": "Equipment question filtered by specific document"
        },
        {
            "question": "How often should temperature monitoring be performed?",
            "filters": {"equipment": ["Temperature Monitoring System"]},
            "description": "Frequency question with equipment filter"
        }
    ]
    
    # Process each demo question
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Demo Question {i}: {demo['description']}")
        print(f"Question: {demo['question']}")
        
        if demo['filters']:
            print(f"Filters: {demo['filters']}")
        
        print("\n⏳ Processing question through RAG pipeline...")
        
        try:
            # Answer the question
            result = await rag_chain.answer_question(
                demo['question'], 
                filters=demo['filters']
            )
            
            # Display results
            print(f"\n📝 Answer:")
            print(f"{result.answer}")
            
            print(f"\n📊 Confidence: {result.confidence_score:.2f} ({result.confidence_level.value})")
            
            if result.confidence_level == ConfidenceLevel.LOW:
                print("⚠️  Low confidence - please verify with source documents")
            elif result.confidence_level == ConfidenceLevel.HIGH:
                print("✅ High confidence answer")
            
            print(f"\n📚 Citations ({len(result.citations)}):")
            for j, citation in enumerate(result.citations, 1):
                print(f"  {j}. Doc: {citation.doc_id}")
                if citation.heading_path:
                    print(f"     Section: {citation.heading_path}")
                if citation.page_no:
                    print(f"     Page: {citation.page_no}")
                print(f"     Snippet: {citation.text_snippet[:100]}...")
                print(f"     Relevance: {citation.relevance_score:.2f}")
            
            print(f"\n🔍 Retrieval Stats:")
            for key, value in result.retrieval_stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n⏱️  Processing Time: {result.processing_time_seconds:.2f} seconds")
            
            if result.warnings:
                print(f"\n⚠️  Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            logger.exception("Question processing failed")
    
    # Demonstrate additional features
    print(f"\n{'='*60}")
    print("🔧 Additional Features Demo")
    
    # Similar questions
    print("\n🔍 Similar Questions:")
    similar = await rag_chain.get_similar_questions("What are the safety requirements?")
    for i, q in enumerate(similar, 1):
        print(f"  {i}. {q}")
    
    # Document summary
    print("\n📄 Document Summary (SOP-FILL-001):")
    summary = await rag_chain.get_document_summary("SOP-FILL-001")
    if summary:
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(map(str, value[:3]))}{'...' if len(value) > 3 else ''}")
            else:
                print(f"  {key}: {value}")
    
    # RAG chain statistics
    print("\n📊 RAG Chain Statistics:")
    stats = rag_chain.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
    print("✅ RAG Chain Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Vector search with metadata filtering")
    print("  • MMR reranking for result diversity")
    print("  • Context fusion with structured SOP data")
    print("  • LLM-powered answer generation")
    print("  • Citation extraction and linking")
    print("  • Confidence scoring and assessment")
    print("  • Similar question suggestions")
    print("  • Document summaries")


async def demonstrate_components():
    """Demonstrate individual RAG components"""
    
    print("\n🔧 Individual Component Demos")
    print("=" * 40)
    
    from sop_qa_tool.services.rag_chain import MMRReranker, ContextFusion, AnswerGenerator
    import numpy as np
    
    # MMR Reranker Demo
    print("\n1. 🔄 MMR Reranker Demo")
    chunks, sop_documents = create_sample_data()
    
    # Create sample search results
    search_results = [
        SearchResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_text=chunk.chunk_text,
            score=0.9 - i * 0.1,  # Decreasing scores
            metadata={'roles': chunk.roles, 'equipment': chunk.equipment}
        )
        for i, chunk in enumerate(chunks[:3])
    ]
    
    reranker = MMRReranker(lambda_param=0.7)
    query_embedding = np.random.rand(384)
    
    print(f"Original order: {[r.chunk_id for r in search_results]}")
    reranked = reranker.rerank(search_results, query_embedding, top_k=2)
    print(f"After MMR reranking: {[r.chunk_id for r in reranked]}")
    
    # Context Fusion Demo
    print("\n2. 🔗 Context Fusion Demo")
    from unittest.mock import Mock
    context_fusion = ContextFusion(Mock())
    
    contexts = context_fusion.fuse_context(search_results[:2], sop_documents)
    print(f"Enhanced {len(contexts)} contexts with structured information")
    print(f"First context preview: {contexts[0].chunk_text[:100]}...")
    
    # Answer Generator Demo (Local Mode)
    print("\n3. 💬 Answer Generator Demo")
    from unittest.mock import Mock
    
    # Create mock settings for local mode
    mock_settings = Mock()
    mock_settings.is_aws_mode.return_value = False
    mock_settings.is_local_mode.return_value = True
    
    # Create answer generator with mock settings
    with patch('sop_qa_tool.services.rag_chain.get_settings', return_value=mock_settings):
        answer_gen = AnswerGenerator()
        
        answer, confidence, citations = await answer_gen.generate_answer(
            "What is the temperature requirement?",
            contexts[:2]
        )
        
        print(f"Generated answer: {answer[:100]}...")
        print(f"Confidence: {confidence:.2f}")
        print(f"Citations: {len(citations)}")


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(demonstrate_rag_chain())
        asyncio.run(demonstrate_components())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo execution failed")