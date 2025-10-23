# Architecture Documentation

## System Overview

The SOP QA Tool is designed as a modular, dual-mode system that can operate either with AWS cloud services or in a completely local environment. The architecture follows microservices principles with clear separation of concerns.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web UI]
        API[FastAPI REST API]
    end
    
    subgraph "Processing Services"
        ING[Document Ingestion]
        TXT[Text Extraction]
        CHK[Text Chunking]
        ONT[Ontology Extraction]
        EMB[Embeddings Service]
        RAG[RAG Chain]
    end
    
    subgraph "AI Services"
        subgraph "AWS Mode"
            BR[Bedrock Claude]
            TIT[Titan Embeddings]
            TX[Textract OCR]
        end
        subgraph "Local Mode"
            HF[Local LLM]
            ST[SentenceTransformers]
            OCR[Local OCR]
        end
    end
    
    subgraph "Storage Layer"
        subgraph "AWS Storage"
            S3[S3 Buckets]
            OS[OpenSearch Serverless]
        end
        subgraph "Local Storage"
            FS[File System]
            FAISS[FAISS Vector DB]
        end
    end
    
    UI --> API
    API --> ING
    API --> RAG
    
    ING --> TXT
    TXT --> CHK
    CHK --> ONT
    ONT --> EMB
    EMB --> OS
    EMB --> FAISS
    
    RAG --> OS
    RAG --> FAISS
    
    BR -.-> ONT
    BR -.-> RAG
    TIT -.-> EMB
    TX -.-> TXT
    
    HF -.-> ONT
    HF -.-> RAG
    ST -.-> EMB
    OCR -.-> TXT
    
    OS --> S3
    FAISS --> FS
```

## Component Architecture

### 1. User Interface Layer

#### Streamlit Web UI (`sop_qa_tool/ui/`)
- **Purpose**: Provides conversational interface for document upload and querying
- **Key Components**:
  - Chat interface with message history
  - Document upload forms
  - Filtering controls (role, equipment, document type)
  - Export functionality (CSV, Markdown)
  - SOP card display with structured data

#### FastAPI REST API (`sop_qa_tool/api/`)
- **Purpose**: Provides programmatic access to all system functionality
- **Endpoints**:
  - `POST /ingest` - Document ingestion
  - `POST /ask` - Question answering
  - `GET /sources` - Document management
  - `GET /health` - System health check
  - `POST /reindex` - Index rebuilding

### 2. Processing Services

#### Document Ingestion Service (`sop_qa_tool/services/document_ingestion.py`)
```mermaid
flowchart LR
    A[URL/File Input] --> B[Security Validation]
    B --> C[Content Fetching]
    C --> D[Format Detection]
    D --> E[Text Extraction]
    E --> F[OCR Processing]
    F --> G[Content Storage]
    G --> H[Metadata Creation]
```

**Key Features**:
- SSRF protection for URL inputs
- File type validation and size limits
- Retry logic with exponential backoff
- Progress tracking for bulk operations

#### Text Processing Pipeline
```mermaid
flowchart TD
    A[Raw Document] --> B[Text Extraction]
    B --> C[Structure Detection]
    C --> D[Chunking Strategy]
    D --> E[Chunk Creation]
    E --> F[Metadata Enrichment]
    F --> G[Validation]
    G --> H[Storage]
    
    subgraph "Chunking Strategy"
        D1[Heading-based]
        D2[Semantic]
        D3[Fixed-size]
    end
    
    D --> D1
    D --> D2
    D --> D3
```

#### Ontology Extraction Service (`sop_qa_tool/services/ontology_extractor.py`)
```mermaid
flowchart LR
    A[Document Chunks] --> B[LLM Processing]
    B --> C[Schema Validation]
    C --> D[Partial Extraction]
    D --> E[Merge Logic]
    E --> F[Final Validation]
    F --> G[Structured Data]
    
    subgraph "Extracted Elements"
        G1[Procedure Steps]
        G2[Risks & Controls]
        G3[Roles & Responsibilities]
        G4[Equipment & Materials]
    end
    
    G --> G1
    G --> G2
    G --> G3
    G --> G4
```

### 3. RAG (Retrieval-Augmented Generation) Pipeline

```mermaid
flowchart TD
    A[User Query] --> B[Query Analysis]
    B --> C[Vector Search]
    C --> D[Metadata Filtering]
    D --> E[Result Reranking]
    E --> F[Context Fusion]
    F --> G[Answer Generation]
    G --> H[Citation Extraction]
    H --> I[Confidence Scoring]
    I --> J[Response Formatting]
    
    subgraph "Search Strategy"
        C1[Semantic Similarity]
        C2[Keyword Matching]
        C3[Hybrid Search]
    end
    
    C --> C1
    C --> C2
    C --> C3
    
    subgraph "Reranking Methods"
        E1[MMR Diversity]
        E2[Relevance Scoring]
        E3[Freshness Weighting]
    end
    
    E --> E1
    E --> E2
    E --> E3
```

## Data Flow Architecture

### Document Ingestion Flow
```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant ING as Ingestion Service
    participant TXT as Text Extraction
    participant CHK as Chunker
    participant ONT as Ontology Extractor
    participant EMB as Embeddings
    participant VS as Vector Store
    
    U->>API: Upload Document/URL
    API->>ING: Process Document
    ING->>TXT: Extract Text
    TXT->>CHK: Create Chunks
    CHK->>ONT: Extract Structure
    ONT->>EMB: Generate Embeddings
    EMB->>VS: Store Vectors
    VS-->>API: Confirmation
    API-->>U: Success Response
```

### Query Processing Flow
```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant RAG as RAG Chain
    participant VS as Vector Store
    participant LLM as Language Model
    participant CIT as Citation Tracker
    
    U->>API: Ask Question
    API->>RAG: Process Query
    RAG->>VS: Vector Search
    VS-->>RAG: Relevant Chunks
    RAG->>LLM: Generate Answer
    LLM-->>RAG: Raw Answer
    RAG->>CIT: Extract Citations
    CIT-->>RAG: Cited Answer
    RAG-->>API: Final Response
    API-->>U: Answer with Citations
```

## Storage Architecture

### AWS Mode Storage
```mermaid
graph TB
    subgraph "S3 Storage"
        S3R[Raw Documents Bucket]
        S3C[Processed Chunks Bucket]
    end
    
    subgraph "OpenSearch Serverless"
        OSI[Vector Index]
        OSM[Metadata Index]
    end
    
    subgraph "Data Organization"
        S3R --> |"/{doc_id}/original.{ext}"| D1[Document Files]
        S3R --> |"/{doc_id}/metadata.json"| D2[Document Metadata]
        S3C --> |"/{doc_id}/chunks.jsonl"| D3[Text Chunks]
        S3C --> |"/{doc_id}/summary.json"| D4[Structured Summary]
    end
    
    OSI --> |Vector Embeddings| V1[768-dim Titan Vectors]
    OSM --> |Searchable Metadata| M1[Roles, Equipment, etc.]
```

### Local Mode Storage
```mermaid
graph TB
    subgraph "File System"
        FS[./data/]
    end
    
    subgraph "FAISS Vector Store"
        FI[Index Files]
        FM[Metadata JSON]
    end
    
    subgraph "Data Organization"
        FS --> |raw_docs/| R1[Original Documents]
        FS --> |chunks/| R2[Processed Chunks]
        FS --> |embeddings_cache.pkl| R3[Embedding Cache]
        FS --> |faiss_index/| FI
        FS --> |faiss_index/metadata.json| FM
    end
    
    FI --> |384-dim Vectors| V2[SentenceTransformer Vectors]
    FM --> |Document Metadata| M2[Searchable Attributes]
```

## Security Architecture

```mermaid
graph TB
    subgraph "Input Validation Layer"
        IV1[URL Validation]
        IV2[File Type Checking]
        IV3[Size Limit Enforcement]
        IV4[Content Scanning]
    end
    
    subgraph "Processing Security"
        PS1[SSRF Protection]
        PS2[Sandbox Execution]
        PS3[Resource Limits]
        PS4[Error Handling]
    end
    
    subgraph "Data Protection"
        DP1[PII Redaction]
        DP2[Access Control]
        DP3[Audit Logging]
        DP4[Encryption at Rest]
    end
    
    subgraph "Network Security"
        NS1[CORS Configuration]
        NS2[Rate Limiting]
        NS3[Input Sanitization]
        NS4[Output Filtering]
    end
    
    IV1 --> PS1
    IV2 --> PS2
    IV3 --> PS3
    IV4 --> PS4
    
    PS1 --> DP1
    PS2 --> DP2
    PS3 --> DP3
    PS4 --> DP4
    
    DP1 --> NS1
    DP2 --> NS2
    DP3 --> NS3
    DP4 --> NS4
```

## Deployment Architecture

### Local Deployment
```mermaid
graph TB
    subgraph "Windows 11 Machine"
        subgraph "Python Environment"
            APP[SOP QA Application]
            API[FastAPI Server :8000]
            UI[Streamlit Server :8501]
        end
        
        subgraph "Local Storage"
            FS[File System Data]
            FAISS[FAISS Vector Index]
            CACHE[Embedding Cache]
        end
        
        subgraph "Local Models"
            ST[SentenceTransformers]
            OCR[Tesseract OCR]
        end
    end
    
    APP --> API
    APP --> UI
    API --> FS
    API --> FAISS
    API --> ST
    API --> OCR
```

### AWS Deployment
```mermaid
graph TB
    subgraph "AWS Cloud"
        subgraph "Compute"
            EC2[EC2 Instance]
            ECS[ECS Fargate]
        end
        
        subgraph "AI Services"
            BR[Bedrock]
            TX[Textract]
        end
        
        subgraph "Storage"
            S3[S3 Buckets]
            OS[OpenSearch Serverless]
        end
        
        subgraph "Monitoring"
            CW[CloudWatch]
            XR[X-Ray]
        end
    end
    
    subgraph "Application"
        APP[SOP QA Tool]
        API[FastAPI]
        UI[Streamlit]
    end
    
    EC2 --> APP
    ECS --> APP
    APP --> BR
    APP --> TX
    APP --> S3
    APP --> OS
    APP --> CW
    APP --> XR
```

## Configuration Management

```mermaid
graph LR
    subgraph "Configuration Sources"
        ENV[.env File]
        ARGS[CLI Arguments]
        DEFAULTS[Default Values]
    end
    
    subgraph "Configuration Processing"
        LOAD[Config Loader]
        VALID[Validation]
        MERGE[Merge Strategy]
    end
    
    subgraph "Runtime Configuration"
        SETTINGS[Settings Object]
        MODE[Operation Mode]
        PARAMS[Processing Parameters]
    end
    
    ENV --> LOAD
    ARGS --> LOAD
    DEFAULTS --> LOAD
    
    LOAD --> VALID
    VALID --> MERGE
    MERGE --> SETTINGS
    
    SETTINGS --> MODE
    SETTINGS --> PARAMS
```

## Performance Considerations

### Scalability Patterns
- **Horizontal Scaling**: Multiple API instances behind load balancer
- **Vertical Scaling**: Increased memory/CPU for local processing
- **Caching Strategy**: Multi-level caching (embeddings, results, metadata)
- **Async Processing**: Background document ingestion with progress tracking

### Memory Management
- **Streaming Processing**: Process large documents in chunks
- **Lazy Loading**: Load models and data on-demand
- **Cache Eviction**: LRU cache for embeddings and results
- **Resource Monitoring**: Track memory usage and implement limits

### Optimization Strategies
- **Batch Processing**: Group similar operations for efficiency
- **Connection Pooling**: Reuse database and API connections
- **Compression**: Compress stored embeddings and metadata
- **Indexing**: Optimize vector search with appropriate algorithms

## Error Handling and Resilience

```mermaid
graph TB
    subgraph "Error Types"
        E1[Network Errors]
        E2[Processing Errors]
        E3[Storage Errors]
        E4[Model Errors]
    end
    
    subgraph "Handling Strategies"
        H1[Retry with Backoff]
        H2[Circuit Breaker]
        H3[Graceful Degradation]
        H4[Fallback Mechanisms]
    end
    
    subgraph "Recovery Actions"
        R1[Log and Alert]
        R2[Partial Results]
        R3[Queue for Retry]
        R4[User Notification]
    end
    
    E1 --> H1
    E2 --> H2
    E3 --> H3
    E4 --> H4
    
    H1 --> R1
    H2 --> R2
    H3 --> R3
    H4 --> R4
```

## Monitoring and Observability

### Metrics Collection
- **Performance Metrics**: Response times, throughput, error rates
- **Resource Metrics**: Memory usage, CPU utilization, disk space
- **Business Metrics**: Documents processed, queries answered, user satisfaction
- **Health Metrics**: Service availability, dependency status

### Logging Strategy
- **Structured Logging**: JSON format with consistent fields
- **Log Levels**: DEBUG, INFO, WARN, ERROR with appropriate filtering
- **Correlation IDs**: Track requests across service boundaries
- **Audit Trail**: Security events and data access patterns

### Alerting Rules
- **High Error Rate**: >5% errors in 5-minute window
- **High Latency**: P95 response time >10 seconds
- **Resource Exhaustion**: Memory usage >80% or disk space >90%
- **Service Unavailability**: Health check failures

This architecture provides a robust, scalable foundation for the SOP QA Tool while maintaining flexibility for both cloud and local deployments.