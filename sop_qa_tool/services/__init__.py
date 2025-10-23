"""
Services package for SOP Q&A Tool

Contains business logic services for document processing, text extraction,
security validation, and other core functionality.
"""

from .document_ingestion import (
    DocumentIngestionService,
    DocumentSource,
    DocumentText,
    IngestResult,
    BatchIngestResult,
    ingest_url,
    ingest_urls,
    ingest_file,
    ingest_files
)

from .text_extraction import (
    TextExtractor,
    OCRService,
    TextExtractionError
)

from .security import SecurityValidator

from .text_chunker import (
    TextChunker,
    HeadingInfo,
    ChunkMetadata
)

from .ontology_extractor import OntologyExtractor

from .summarizer import SOPSummarizer

from .embedder import (
    EmbeddingService,
    EmbeddingResult,
    EmbeddingCache,
    RateLimiter
)

__all__ = [
    # Document Ingestion
    'DocumentIngestionService',
    'DocumentSource',
    'DocumentText',
    'IngestResult',
    'BatchIngestResult',
    'ingest_url',
    'ingest_urls',
    'ingest_file',
    'ingest_files',
    
    # Text Extraction
    'TextExtractor',
    'OCRService',
    'TextExtractionError',
    
    # Text Chunking
    'TextChunker',
    'HeadingInfo',
    'ChunkMetadata',
    
    # Security
    'SecurityValidator',
    
    # Ontology Extraction
    'OntologyExtractor',
    
    # Summarization
    'SOPSummarizer',
    
    # Embeddings
    'EmbeddingService',
    'EmbeddingResult',
    'EmbeddingCache',
    'RateLimiter',
]