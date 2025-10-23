# SOP QA Tool - Project Structure

This document describes the organization of the SOP QA Tool project.

## Directory Structure

```
sop-qa-tool/
├── sop_qa_tool/                 # Main application package
│   ├── __init__.py
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py          # Settings with validation
│   ├── models/                  # Data models and schemas
│   │   └── __init__.py
│   ├── services/                # Business logic services
│   │   └── __init__.py
│   ├── api/                     # FastAPI backend
│   │   └── __init__.py
│   └── ui/                      # Streamlit frontend
│       └── __init__.py
├── tests/                       # Test suite
│   └── __init__.py
├── scripts/                     # Utility scripts
│   └── __init__.py
├── data/                        # Local data storage (created by bootstrap)
│   ├── raw_docs/               # Original documents
│   ├── chunks/                 # Processed chunks
│   ├── faiss_index/           # FAISS vector index
│   └── logs/                  # Application logs
├── bootstrap.ps1               # Windows setup script
├── requirements.txt            # Python dependencies
├── .env                       # Environment configuration
└── PROJECT_STRUCTURE.md      # This file
```

## Module Organization

### `sop_qa_tool/config/`
- **settings.py**: Centralized configuration management with Pydantic validation
- Supports both AWS and local operation modes
- Environment variable parsing and validation
- Directory creation for local mode

### `sop_qa_tool/models/`
- Data models and Pydantic schemas
- SOP ontology definitions
- Request/response models for API

### `sop_qa_tool/services/`
- Core business logic services
- Document ingestion, processing, and retrieval
- LLM integration and RAG pipeline

### `sop_qa_tool/api/`
- FastAPI backend implementation
- REST endpoints for document management and Q&A
- Authentication and security middleware

### `sop_qa_tool/ui/`
- Streamlit frontend application
- Chat interface and document management UI
- Visualization and export functionality

### `tests/`
- Unit tests, integration tests, and performance tests
- Test fixtures and utilities
- RAGAS evaluation framework

### `scripts/`
- Utility scripts for operations and maintenance
- Bulk ingestion, index management, and monitoring tools

## Configuration Management

The application uses a hierarchical configuration system:

1. **Default values** in `settings.py`
2. **Environment variables** (prefixed or direct)
3. **.env file** for local development
4. **Runtime validation** with Pydantic

## Getting Started

1. Run the bootstrap script: `.\bootstrap.ps1`
2. Activate the virtual environment: `.\sop-qa-venv\Scripts\Activate.ps1`
3. Review and modify `.env` file as needed
4. Install dependencies: `pip install -r requirements.txt`
5. Run configuration validation: `python -c "from sop_qa_tool.config.settings import validate_settings; validate_settings()"`

## Development Workflow

1. **Local Mode**: Default configuration for development and testing
2. **AWS Mode**: Production configuration with cloud services
3. **Testing**: Run `python -m pytest tests/` for test suite
4. **Code Quality**: Use `black`, `flake8`, and `mypy` for code formatting and linting