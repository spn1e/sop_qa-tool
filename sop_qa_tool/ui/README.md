# SOP Q&A Tool - Streamlit Frontend

This directory contains the Streamlit-based frontend implementation for the SOP Q&A Tool.

## Requirements Implemented

- **5.1**: Chat-based interaction model with message history
- **5.2**: Structured SOP card display with ontology information  
- **5.3**: Confidence scores and source citations display
- **5.4**: CSV and Markdown export functionality
- **4.2**: Filtering controls for role, equipment, and document type
- **4.3**: Version comparison capabilities
- **7.3**: Mode switching between AWS and local operation

## Components

### Main Application (`streamlit_app.py`)
- **APIClient**: HTTP client for communicating with FastAPI backend
- **Chat Interface**: Real-time Q&A with response streaming
- **Document Management**: Upload and URL ingestion with progress tracking
- **Filtering System**: Advanced search filters with real-time application
- **Export Functionality**: CSV and Markdown export of Q&A sessions

### SOP Cards (`sop_cards.py`)
- **render_sop_card()**: Complete SOP document display with tabs
- **render_procedure_steps()**: Step-by-step procedure visualization
- **render_risks_and_controls()**: Risk assessment and control measures
- **render_roles_responsibilities()**: Role-based responsibility matrix
- **render_equipment_materials()**: Categorized equipment and materials
- **render_sop_comparison()**: Side-by-side SOP version comparison

### UI Components (`components.py`)
- **Data Visualization**: Confidence charts, response time graphs, citation networks
- **Interactive Elements**: Search interfaces, pagination, file uploaders
- **Export Tools**: Multi-format export buttons with data processing
- **Utility Functions**: Status indicators, metric cards, filter sidebars

## Features

### Chat Interface
- Natural language question input
- Real-time response streaming
- Message history persistence
- Confidence score display
- Source citation linking

### Document Management
- **URL Ingestion**: Batch processing of document URLs
- **File Upload**: Multi-file upload with validation
- **Progress Tracking**: Real-time ingestion status
- **Source Management**: View and delete ingested documents

### Advanced Filtering
- **Role-based**: Filter by operator, QA inspector, supervisor, etc.
- **Equipment-based**: Filter by specific equipment or machinery
- **Document Type**: Filter by SOP, work instruction, safety procedure
- **Confidence Threshold**: Minimum confidence score filtering
- **Result Limits**: Configurable maximum results

### SOP Visualization
- **Structured Display**: Organized tabs for different SOP sections
- **Interactive Cards**: Expandable sections with detailed information
- **Visual Indicators**: Color-coded risk levels and control types
- **Metadata Display**: Complete document metadata and change history

### Export Capabilities
- **CSV Export**: Tabular data with flattened citations
- **Markdown Export**: Formatted reports with full context
- **Real-time Preview**: Data preview before export
- **Timestamp Tracking**: Automatic timestamp and metadata inclusion

### Mode Switching
- **AWS Mode**: Full cloud integration with Bedrock and OpenSearch
- **Local Mode**: Offline operation with local models and FAISS
- **Seamless Transition**: Consistent UI regardless of backend mode
- **Status Indicators**: Clear mode indication and health status

## Usage

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run both backend and frontend
python run_ui.py

# Or run Streamlit directly
streamlit run sop_qa_tool/ui/streamlit_app.py
```

### Configuration

The UI automatically detects the operation mode from the backend configuration:

```python
# AWS Mode
MODE=aws
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint
S3_RAW_BUCKET=your-raw-bucket
S3_CHUNKS_BUCKET=your-chunks-bucket

# Local Mode  
MODE=local
LOCAL_DATA_PATH=./data
FAISS_INDEX_PATH=./data/faiss_index
```

### API Integration

The frontend communicates with the FastAPI backend through these endpoints:

- `GET /health` - System health check
- `POST /ask` - Question answering
- `POST /ingest/urls` - URL document ingestion
- `POST /ingest/files` - File upload ingestion
- `GET /ingest/{task_id}/status` - Ingestion progress
- `GET /sources` - List document sources
- `DELETE /sources/{doc_id}` - Delete document
- `POST /reindex` - Rebuild search index

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   Backend       │
│                 │    │                 │    │   Services      │
│ • Chat Interface│    │ • /ask          │    │ • RAG Chain     │
│ • File Upload   │    │ • /ingest       │    │ • Vector Store  │
│ • Filtering     │    │ • /sources      │    │ • Embeddings    │
│ • Export        │    │ • /health       │    │ • Security      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Testing

The UI components are tested for:

- Data processing logic
- Filter configuration validation
- Export functionality
- SOP data structure handling
- API response structure validation

```bash
# Run UI tests
python -m pytest tests/test_ui.py -v
```

## Security Features

- **Input Validation**: All user inputs are validated and sanitized
- **File Type Restrictions**: Only allowed file types can be uploaded
- **Size Limits**: File size restrictions prevent resource exhaustion
- **PII Redaction**: Optional personally identifiable information removal
- **CORS Configuration**: Proper cross-origin resource sharing setup

## Performance Considerations

- **Lazy Loading**: Components load data on demand
- **Caching**: Streamlit caching for expensive operations
- **Pagination**: Large datasets are paginated for performance
- **Progress Indicators**: Long-running operations show progress
- **Error Handling**: Graceful degradation on failures

## Customization

### Themes
The UI supports custom themes through `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Components
All UI components are modular and can be customized:

```python
from sop_qa_tool.ui.components import create_metric_card, create_filter_sidebar
from sop_qa_tool.ui.sop_cards import render_sop_card

# Custom metric display
create_metric_card("Custom Metric", "100", delta="10")

# Custom SOP card
render_sop_card(sop_data, expanded=True)
```

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure FastAPI backend is running on port 8000
   - Check CORS configuration in backend

2. **File Upload Errors**
   - Verify file types are in allowed list
   - Check file size limits

3. **Slow Response Times**
   - Check backend mode (AWS vs Local)
   - Verify network connectivity for AWS mode

4. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure Streamlit version compatibility

### Debug Mode

Enable debug mode for detailed error information:

```bash
streamlit run sop_qa_tool/ui/streamlit_app.py --logger.level=debug
```