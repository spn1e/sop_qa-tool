"""
FastAPI Backend Implementation

Main FastAPI application with endpoints for document ingestion, question answering,
source management, and system administration.

Requirements: 6.1, 6.2, 6.3, 5.4
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config.settings import get_settings, Settings
from ..config.logging_config import setup_application_logging, get_logger, LogContext, PerformanceLogger
from ..services.document_ingestion import DocumentIngestionService, IngestResult, DocumentText
from ..services.rag_chain import RAGChain, AnswerResult, ConfidenceLevel
from ..services.vectorstore import VectorStoreService
from ..services.security import SecurityValidator
from ..models.sop_models import SOPDocument, SourceInfo


logger = logging.getLogger(__name__)

# Global services - initialized on startup
ingestion_service: Optional[DocumentIngestionService] = None
rag_chain: Optional[RAGChain] = None
vector_store: Optional[VectorStoreService] = None
security_validator: Optional[SecurityValidator] = None

# Background task tracking
active_tasks: Dict[str, Dict[str, Any]] = {}


class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    urls: List[str] = Field(default_factory=list, description="List of URLs to ingest")
    use_ocr: bool = Field(default=True, description="Whether to use OCR for scanned documents")
    extract_ontology: bool = Field(default=True, description="Whether to extract SOP ontology")


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    task_id: str = Field(..., description="Unique task identifier for tracking progress")
    message: str = Field(..., description="Status message")
    estimated_time_minutes: Optional[int] = Field(None, description="Estimated completion time")


class IngestStatus(BaseModel):
    """Status model for ingestion progress tracking"""
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    documents_processed: int
    documents_total: int
    errors: List[str]
    started_at: str
    completed_at: Optional[str] = None
    result: Optional[IngestResult] = None


class AskRequest(BaseModel):
    """Request model for question answering"""
    question: str = Field(..., description="Question to ask about the SOPs")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to search")
    top_k: int = Field(default=5, description="Number of top results to retrieve", ge=1, le=20)


class AskResponse(BaseModel):
    """Response model for question answering"""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    confidence_level: str = Field(..., description="Confidence level: high, medium, low")
    citations: List[Dict[str, Any]] = Field(..., description="Source citations")
    context_used: List[Dict[str, Any]] = Field(..., description="Context chunks used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class SourceInfo(BaseModel):
    """Information about a document source"""
    doc_id: str
    title: str
    source_type: str  # "url" or "file"
    source_value: str
    ingested_at: str
    chunk_count: int
    has_ontology: bool
    file_size_bytes: Optional[int] = None


class SourcesResponse(BaseModel):
    """Response model for sources listing"""
    sources: List[SourceInfo] = Field(..., description="List of ingested sources")
    total_count: int = Field(..., description="Total number of sources")
    total_chunks: int = Field(..., description="Total number of chunks across all sources")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Overall system status")
    mode: str = Field(..., description="Operation mode (aws/local)")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    version: str = Field(default="1.0.0", description="API version")


# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global ingestion_service, rag_chain, vector_store, security_validator
    
    # Initialize application logging first
    app_logger = setup_application_logging()
    app_logger.info("Starting SOP Q&A Tool API...")
    
    settings = get_settings()
    
    try:
        # Initialize services
        with LogContext(app_logger, component="startup", operation="service_initialization"):
            app_logger.info(f"Initializing services in {settings.mode} mode...")
            
            # Initialize security validator first
            security_validator = SecurityValidator()
            app_logger.info("Security validator initialized")
            
            # Initialize vector store
            vector_store = VectorStoreService()
            app_logger.info("Vector store service initialized")
            
            # Initialize ingestion service
            ingestion_service = DocumentIngestionService()
            app_logger.info("Document ingestion service initialized")
            
            # Initialize RAG chain
            rag_chain = RAGChain()
            app_logger.info("RAG chain initialized")
            
            app_logger.info("All services initialized successfully")
        
    except Exception as e:
        app_logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise
    
    # Record startup time
    app.state.startup_time = time.time()
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down SOP Q&A Tool API...")
    
    # Cancel any running background tasks
    for task_id, task_info in active_tasks.items():
        if task_info.get("task") and not task_info["task"].done():
            task_info["task"].cancel()
    
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="SOP Q&A Tool API",
    description="Automated Research & Q/A Tool for Factory SOPs",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    if security_validator:
        security_headers = security_validator.create_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
    
    return response


def get_settings_dependency() -> Settings:
    """Dependency to get settings"""
    return get_settings()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "SOP Q&A Tool API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/ingest/urls", response_model=IngestResponse)
async def ingest_urls(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Ingest documents from URLs with progress tracking.
    
    Requirements: 6.1, 6.2, 8.1
    """
    if not ingestion_service or not security_validator:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    if not request.urls:
        raise HTTPException(
            status_code=400, 
            detail="URLs must be provided"
        )
    
    # Validate URLs for security
    url_validation = security_validator.validate_url_batch(request.urls)
    
    if url_validation['invalid_urls']:
        security_validator.log_security_event(
            "blocked_urls",
            {
                "invalid_urls": url_validation['invalid_urls'],
                "errors": url_validation['errors']
            }
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid URLs detected: {', '.join(url_validation['invalid_urls'])}"
        )
    
    if not url_validation['valid_urls']:
        raise HTTPException(
            status_code=400,
            detail="No valid URLs provided"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    active_tasks[task_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting URL ingestion...",
        "documents_processed": 0,
        "documents_total": len(request.urls),
        "errors": [],
        "started_at": time.time(),
        "completed_at": None,
        "result": None,
        "task": None
    }
    
    # Start background ingestion
    task = asyncio.create_task(
        _run_ingestion_task(task_id, request, [])
    )
    active_tasks[task_id]["task"] = task
    background_tasks.add_task(_cleanup_completed_task, task_id)
    
    # Estimate completion time (rough estimate: 30 seconds per document)
    estimated_minutes = max(1, len(request.urls) * 0.5)
    
    return IngestResponse(
        task_id=task_id,
        message="URL ingestion started successfully",
        estimated_time_minutes=int(estimated_minutes)
    )


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Ingest documents from file uploads with progress tracking.
    
    Requirements: 6.1, 6.2, 8.2
    """
    if not ingestion_service or not security_validator:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="Files must be provided"
        )
    
    # Validate files for security
    file_validation = security_validator.validate_batch_upload(files)
    
    if file_validation['invalid_files']:
        security_validator.log_security_event(
            "blocked_files",
            {
                "invalid_files": file_validation['invalid_files'],
                "errors": file_validation['errors']
            }
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid files detected: {', '.join(file_validation['invalid_files'])}"
        )
    
    if not file_validation['valid_files']:
        raise HTTPException(
            status_code=400,
            detail="No valid files provided"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    active_tasks[task_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting file ingestion...",
        "documents_processed": 0,
        "documents_total": len(files),
        "errors": [],
        "started_at": time.time(),
        "completed_at": None,
        "result": None,
        "task": None
    }
    
    # Start background ingestion
    task = asyncio.create_task(
        _run_ingestion_task(task_id, IngestRequest(), files)
    )
    active_tasks[task_id]["task"] = task
    background_tasks.add_task(_cleanup_completed_task, task_id)
    
    # Estimate completion time (rough estimate: 30 seconds per document)
    estimated_minutes = max(1, len(files) * 0.5)
    
    return IngestResponse(
        task_id=task_id,
        message="File ingestion started successfully",
        estimated_time_minutes=int(estimated_minutes)
    )


# Legacy endpoint for backward compatibility
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents_legacy(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Legacy ingest endpoint - redirects to URL ingestion.
    
    Requirements: 6.1, 6.2
    """
    return await ingest_urls(request, background_tasks, settings)


@app.get("/ingest/{task_id}/status", response_model=IngestStatus)
async def get_ingestion_status(task_id: str):
    """
    Get the status of a document ingestion task.
    
    Requirements: 6.1
    """
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    return IngestStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        message=task_info["message"],
        documents_processed=task_info["documents_processed"],
        documents_total=task_info["documents_total"],
        errors=task_info["errors"],
        started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task_info["started_at"])),
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task_info["completed_at"])) if task_info["completed_at"] else None,
        result=task_info["result"]
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    settings: Settings = Depends(get_settings_dependency)
):
    """
    Ask a question about the SOPs with filtering support.
    
    Requirements: 6.1, 5.4, 8.4
    """
    if not rag_chain or not security_validator:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    # Validate and sanitize the question
    if not security_validator.validate_query(request.question):
        security_validator.log_security_event(
            "blocked_query",
            {"question": request.question[:100] + "..." if len(request.question) > 100 else request.question}
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid or potentially malicious query detected"
        )
    
    # Sanitize the question
    sanitized_question = security_validator.sanitize_input(request.question)
    
    # Apply PII redaction if enabled
    processed_question = security_validator.redact_pii(sanitized_question)
    
    start_time = time.time()
    
    try:
        # Process the question
        result = await rag_chain.answer_question(
            question=processed_question,
            filters=request.filters,
            top_k=request.top_k
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return AskResponse(
            answer=result.answer,
            confidence=result.confidence_score,
            confidence_level=result.confidence_level.value,
            citations=[
                {
                    "doc_id": citation.doc_id,
                    "chunk_id": citation.chunk_id,
                    "text_snippet": citation.text_snippet
                }
                for citation in result.citations
            ],
            context_used=[
                {
                    "chunk_id": context.chunk_id,
                    "doc_id": context.doc_id,
                    "text": context.chunk_text,
                    "metadata": context.metadata,
                    "relevance_score": context.relevance_score
                }
                for context in result.context_used
            ],
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/sources", response_model=SourcesResponse)
async def list_sources():
    """
    List all ingested document sources.
    
    Requirements: 6.2, 6.3
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store service not available")
    
    try:
        # Get index statistics
        stats = await vector_store.get_stats()
        
        # For now, return basic stats - this would need to be enhanced
        # to track individual document metadata
        sources = []  # Would need to implement document tracking
        
        return SourcesResponse(
            sources=sources,
            total_count=0,
            total_chunks=stats.total_chunks if stats else 0
        )
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing sources: {str(e)}")


@app.delete("/sources/{doc_id}")
async def delete_source(doc_id: str):
    """
    Delete a document source and its associated data.
    
    Requirements: 6.2, 6.3
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store service not available")
    
    try:
        success = await vector_store.delete_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting source: {str(e)}")


@app.post("/reindex")
async def reindex_documents(background_tasks: BackgroundTasks):
    """
    Rebuild the document index from stored data.
    
    Requirements: 6.3
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store service not available")
    
    # Generate task ID for tracking
    task_id = str(uuid.uuid4())
    
    # Initialize task tracking
    active_tasks[task_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting reindex...",
        "documents_processed": 0,
        "documents_total": 0,  # Will be updated during reindex
        "errors": [],
        "started_at": time.time(),
        "completed_at": None,
        "result": None,
        "task": None
    }
    
    # Start background reindex
    task = asyncio.create_task(_run_reindex_task(task_id))
    active_tasks[task_id]["task"] = task
    background_tasks.add_task(_cleanup_completed_task, task_id)
    
    return {
        "task_id": task_id,
        "message": "Reindex started successfully"
    }


@app.get("/reindex/{task_id}/status")
async def get_reindex_status(task_id: str):
    """Get the status of a reindex operation"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return active_tasks[task_id]


@app.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings_dependency)):
    """
    System health check endpoint.
    
    Requirements: 6.3
    """
    components = {}
    overall_status = "healthy"
    
    # Check vector store
    try:
        if vector_store:
            stats = await vector_store.get_stats()
            components["vector_store"] = {"status": "healthy", "details": f"Total chunks: {stats.total_chunks if stats else 0}"}
        else:
            components["vector_store"] = {"status": "unhealthy", "details": "Not initialized"}
            overall_status = "unhealthy"
    except Exception as e:
        components["vector_store"] = {"status": "unhealthy", "details": str(e)}
        overall_status = "unhealthy"
    
    # Check ingestion service
    try:
        if ingestion_service:
            components["ingestion_service"] = {"status": "healthy", "details": "Available"}
        else:
            components["ingestion_service"] = {"status": "unhealthy", "details": "Not initialized"}
            overall_status = "unhealthy"
    except Exception as e:
        components["ingestion_service"] = {"status": "unhealthy", "details": str(e)}
        overall_status = "unhealthy"
    
    # Check RAG chain
    try:
        if rag_chain:
            components["rag_chain"] = {"status": "healthy", "details": "Available"}
        else:
            components["rag_chain"] = {"status": "unhealthy", "details": "Not initialized"}
            overall_status = "unhealthy"
    except Exception as e:
        components["rag_chain"] = {"status": "unhealthy", "details": str(e)}
        overall_status = "unhealthy"
    
    # Calculate uptime
    uptime_seconds = int(time.time() - getattr(app.state, 'startup_time', time.time()))
    
    return HealthResponse(
        status=overall_status,
        mode=settings.mode.value,
        components=components,
        uptime_seconds=uptime_seconds
    )


@app.get("/security/config")
async def get_security_config():
    """
    Get current security configuration.
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    if not security_validator:
        raise HTTPException(status_code=503, detail="Security validator not available")
    
    return security_validator.get_security_summary()


# Background task implementations
async def _run_ingestion_task(task_id: str, request: IngestRequest, files: List[UploadFile]):
    """Run document ingestion in background"""
    try:
        task_info = active_tasks[task_id]
        
        # Process URLs
        url_results = []
        if request.urls:
            task_info["message"] = "Processing URLs..."
            for i, url in enumerate(request.urls):
                try:
                    result = await ingestion_service.ingest_url(url)
                    url_results.append(result)
                    task_info["documents_processed"] += 1
                    task_info["progress"] = task_info["documents_processed"] / task_info["documents_total"]
                    task_info["message"] = f"Processed URL {i+1}/{len(request.urls)}"
                except Exception as e:
                    error_msg = f"Failed to process URL {url}: {str(e)}"
                    task_info["errors"].append(error_msg)
                    logger.error(error_msg)
        
        # Process files
        file_results = []
        if files:
            task_info["message"] = "Processing files..."
            for i, file in enumerate(files):
                try:
                    result = await ingestion_service.ingest_file(file)
                    file_results.append(result)
                    task_info["documents_processed"] += 1
                    task_info["progress"] = task_info["documents_processed"] / task_info["documents_total"]
                    task_info["message"] = f"Processed file {i+1}/{len(files)}"
                except Exception as e:
                    error_msg = f"Failed to process file {file.filename}: {str(e)}"
                    task_info["errors"].append(error_msg)
                    logger.error(error_msg)
        
        # Combine results
        all_results = url_results + file_results
        
        # Create final result
        final_result = {
            "success": len(all_results) > 0,
            "documents_processed": len(all_results),
            "documents_failed": len(task_info["errors"]),
            "processing_time_seconds": time.time() - task_info["started_at"],
            "errors": task_info["errors"]
        }
        
        # Update task status
        task_info["status"] = "completed"
        task_info["progress"] = 1.0
        task_info["message"] = f"Completed: {final_result['documents_processed']} documents processed"
        task_info["completed_at"] = time.time()
        task_info["result"] = final_result
        
    except Exception as e:
        logger.error(f"Ingestion task {task_id} failed: {e}")
        task_info["status"] = "failed"
        task_info["message"] = f"Task failed: {str(e)}"
        task_info["completed_at"] = time.time()
        task_info["errors"].append(str(e))


async def _run_reindex_task(task_id: str):
    """Run document reindexing in background"""
    try:
        task_info = active_tasks[task_id]
        
        task_info["message"] = "Clearing and rebuilding index..."
        await vector_store.clear_index()
        
        task_info["status"] = "completed"
        task_info["progress"] = 1.0
        task_info["message"] = "Index rebuilt successfully"
        task_info["completed_at"] = time.time()
        
    except Exception as e:
        logger.error(f"Reindex task {task_id} failed: {e}")
        task_info["status"] = "failed"
        task_info["message"] = f"Reindex failed: {str(e)}"
        task_info["completed_at"] = time.time()
        task_info["errors"].append(str(e))


async def _cleanup_completed_task(task_id: str):
    """Clean up completed tasks after a delay"""
    await asyncio.sleep(3600)  # Keep task info for 1 hour
    if task_id in active_tasks:
        del active_tasks[task_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)