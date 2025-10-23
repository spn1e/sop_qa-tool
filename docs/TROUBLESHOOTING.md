# Troubleshooting Guide

This guide covers common issues and their solutions for the SOP QA Tool.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [AWS-Specific Issues](#aws-specific-issues)
6. [Local Mode Issues](#local-mode-issues)
7. [API and UI Issues](#api-and-ui-issues)
8. [Document Processing Issues](#document-processing-issues)
9. [Diagnostic Tools](#diagnostic-tools)

## Installation Issues

### Python Not Found

**Symptoms:**
- `python` command not recognized
- Bootstrap script fails with "Python not found"

**Solutions:**
```powershell
# Option 1: Install via winget
winget install Python.Python.3.11

# Option 2: Download from python.org
# Visit https://python.org and download Python 3.11+

# Option 3: Check if Python is installed but not in PATH
where python
# If found, add the directory to your PATH environment variable
```

**Verification:**
```powershell
python --version
# Should show Python 3.11.x or higher
```

### PowerShell Execution Policy

**Symptoms:**
- "Execution of scripts is disabled on this system"
- Bootstrap script won't run

**Solutions:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user (recommended)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Bypass for single script
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

### Virtual Environment Issues

**Symptoms:**
- Virtual environment creation fails
- Dependencies not installing correctly

**Solutions:**
```powershell
# Clean up and recreate
Remove-Item -Recurse -Force sop-qa-venv -ErrorAction SilentlyContinue
python -m venv sop-qa-venv
.\sop-qa-venv\Scripts\Activate.ps1

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Dependency Installation Failures

**Symptoms:**
- Package installation errors
- Missing system dependencies

**Solutions:**
```powershell
# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install with verbose output to see errors
pip install -r requirements.txt -v

# For specific packages that fail:
# Install Microsoft C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# For torch/transformers issues:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configuration Problems

### Environment File Issues

**Symptoms:**
- Configuration not loading
- "Settings validation error"

**Solutions:**
```powershell
# Check if .env file exists
if (!(Test-Path .env)) {
    Copy-Item .env.template .env
    Write-Host "Created .env file from template"
}

# Validate configuration
python -c "from sop_qa_tool.config.settings import settings; print('Config loaded successfully')"

# Check for syntax errors in .env
Get-Content .env | ForEach-Object { 
    if ($_ -match '^[^#].*=' -and $_ -notmatch '^[A-Z_]+=.*$') {
        Write-Host "Potential syntax error: $_"
    }
}
```

### Mode Configuration Issues

**Symptoms:**
- Wrong mode being used
- AWS services not accessible in local mode

**Solutions:**
```powershell
# Check current mode
python -c "from sop_qa_tool.config.settings import settings; print(f'Mode: {settings.mode}')"

# Set mode explicitly in .env
echo "MODE=local" >> .env
# or
echo "MODE=aws" >> .env

# Validate mode-specific settings
python validate_setup.py
```

### Path Configuration Issues

**Symptoms:**
- "Directory not found" errors
- Data not persisting

**Solutions:**
```powershell
# Check data directory
if (!(Test-Path data)) {
    New-Item -ItemType Directory -Path data
    New-Item -ItemType Directory -Path data/chunks
    New-Item -ItemType Directory -Path data/raw_docs
    New-Item -ItemType Directory -Path data/faiss_index
}

# Verify paths in configuration
python -c "
from sop_qa_tool.config.settings import settings
print(f'Data path: {settings.local_data_path}')
print(f'FAISS path: {settings.faiss_index_path}')
"
```

## Runtime Errors

### Port Already in Use

**Symptoms:**
- "Address already in use" when starting servers
- Cannot access UI or API

**Solutions:**
```powershell
# Check what's using the ports
netstat -ano | findstr :8501  # Streamlit
netstat -ano | findstr :8000  # FastAPI

# Kill processes using the ports
# Get PID from netstat output, then:
taskkill /PID <PID> /F

# Use different ports
streamlit run run_ui.py --server.port 8502
uvicorn sop_qa_tool.api.main:app --port 8001
```

### Memory Issues

**Symptoms:**
- "Out of memory" errors
- System becomes unresponsive during processing

**Solutions:**
```powershell
# Check memory usage
Get-Process python | Select-Object ProcessName, WorkingSet, VirtualMemorySize

# Reduce batch sizes in .env
echo "EMBEDDING_BATCH_SIZE=10" >> .env
echo "CHUNK_SIZE=500" >> .env

# Close other applications
# Restart with more virtual memory if needed
```

### Import Errors

**Symptoms:**
- "ModuleNotFoundError"
- "ImportError: cannot import name"

**Solutions:**
```powershell
# Ensure virtual environment is activated
.\sop-qa-venv\Scripts\Activate.ps1

# Reinstall packages
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install in development mode
pip install -e .
```

## Performance Issues

### Slow Response Times

**Symptoms:**
- Queries taking longer than expected
- UI becomes unresponsive

**Diagnostic Steps:**
```powershell
# Check system resources
Get-Process python | Select-Object ProcessName, CPU, WorkingSet

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# Check logs for bottlenecks
Get-Content data/logs/app.log | Select-String "slow\|timeout\|error"
```

**Solutions:**
```powershell
# Optimize configuration
echo "TOP_K_RETRIEVAL=3" >> .env
echo "CHUNK_SIZE=600" >> .env
echo "ENABLE_EMBEDDING_CACHE=true" >> .env

# Clear and rebuild index
.\scripts\rebuild-index.ps1

# Use SSD storage for better I/O
# Move data directory to SSD if on HDD
```

### High Memory Usage

**Symptoms:**
- Memory usage continuously increasing
- System swapping to disk

**Solutions:**
```powershell
# Enable garbage collection logging
$env:PYTHONMALLOC="debug"

# Reduce model sizes
echo "USE_SMALL_MODELS=true" >> .env

# Implement memory limits
echo "MAX_MEMORY_GB=4" >> .env

# Restart services periodically
# Set up scheduled task to restart daily
```

## AWS-Specific Issues

### Authentication Problems

**Symptoms:**
- "Unable to locate credentials"
- "Access denied" errors

**Solutions:**
```powershell
# Check AWS credentials
aws sts get-caller-identity

# Configure credentials if needed
aws configure

# Check IAM permissions
aws iam get-user
aws iam list-attached-user-policies --user-name <username>

# Verify .env settings
python -c "
import boto3
print('AWS Region:', boto3.Session().region_name)
print('AWS Profile:', boto3.Session().profile_name)
"
```

### Bedrock Access Issues

**Symptoms:**
- "Model not found" errors
- "Quota exceeded" errors

**Solutions:**
```powershell
# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1

# Request model access in AWS Console
# Go to Bedrock > Model access > Request access

# Check quotas
aws service-quotas get-service-quota --service-code bedrock --quota-code L-12345

# Use alternative model
echo "BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0" >> .env
```

### OpenSearch Connection Issues

**Symptoms:**
- "Connection timeout" to OpenSearch
- "Index not found" errors

**Solutions:**
```powershell
# Test OpenSearch connectivity
curl -X GET "https://your-opensearch-endpoint.us-east-1.aoss.amazonaws.com/"

# Check collection status
aws opensearchserverless list-collections

# Verify network policies
aws opensearchserverless list-security-policies --type network

# Update endpoint in .env
echo "OPENSEARCH_ENDPOINT=https://your-actual-endpoint.us-east-1.aoss.amazonaws.com" >> .env
```

### S3 Access Issues

**Symptoms:**
- "Bucket does not exist"
- "Access denied" for S3 operations

**Solutions:**
```powershell
# Check bucket existence
aws s3 ls s3://your-bucket-name

# Test bucket access
aws s3 cp test.txt s3://your-bucket-name/test.txt
aws s3 rm s3://your-bucket-name/test.txt

# Create buckets if needed
.\scripts\setup-s3-buckets.ps1

# Update bucket names in .env
echo "S3_RAW_BUCKET=your-actual-raw-bucket" >> .env
echo "S3_CHUNKS_BUCKET=your-actual-chunks-bucket" >> .env
```

## Local Mode Issues

### FAISS Index Problems

**Symptoms:**
- "Index file not found"
- "Dimension mismatch" errors

**Solutions:**
```powershell
# Check index files
Get-ChildItem data/faiss_index/

# Rebuild index
Remove-Item -Recurse -Force data/faiss_index -ErrorAction SilentlyContinue
.\scripts\rebuild-index.ps1

# Check embedding dimensions
python -c "
from sop_qa_tool.services.embedder import EmbeddingService
embedder = EmbeddingService()
test_embedding = embedder.embed_query('test')
print(f'Embedding dimension: {len(test_embedding)}')
"
```

### Model Download Issues

**Symptoms:**
- "Model not found" for sentence-transformers
- Slow first-time loading

**Solutions:**
```powershell
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded successfully')
"

# Check model cache
Get-ChildItem $env:USERPROFILE\.cache\huggingface\transformers\

# Use offline mode if needed
$env:TRANSFORMERS_OFFLINE=1
```

### OCR Issues

**Symptoms:**
- "Tesseract not found"
- Poor OCR quality

**Solutions:**
```powershell
# Install Tesseract
winget install UB-Mannheim.TesseractOCR

# Add to PATH
$env:PATH += ";C:\Program Files\Tesseract-OCR"

# Test OCR
python -c "
import pytesseract
print('Tesseract version:', pytesseract.get_tesseract_version())
"

# Improve OCR quality
echo "OCR_DPI=300" >> .env
echo "OCR_PSM=6" >> .env
```

## API and UI Issues

### API Server Won't Start

**Symptoms:**
- FastAPI server crashes on startup
- Import errors in API modules

**Solutions:**
```powershell
# Check for syntax errors
python -m py_compile sop_qa_tool/api/main.py

# Start with debug mode
uvicorn sop_qa_tool.api.main:app --reload --log-level debug

# Check dependencies
pip list | findstr fastapi
pip list | findstr uvicorn

# Test minimal API
python -c "
from sop_qa_tool.api.main import app
print('API module loaded successfully')
"
```

### Streamlit UI Issues

**Symptoms:**
- UI not loading
- "Connection error" messages

**Solutions:**
```powershell
# Check Streamlit installation
streamlit --version

# Start with debug output
streamlit run run_ui.py --logger.level debug

# Clear Streamlit cache
Remove-Item -Recurse -Force $env:USERPROFILE\.streamlit -ErrorAction SilentlyContinue

# Test UI components
python -c "
from sop_qa_tool.ui.streamlit_app import main
print('UI module loaded successfully')
"
```

### CORS Issues

**Symptoms:**
- "CORS policy" errors in browser
- API calls failing from UI

**Solutions:**
```powershell
# Check CORS configuration
python -c "
from sop_qa_tool.api.main import app
print('CORS middleware configured')
"

# Update CORS origins in settings
echo "CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501" >> .env

# Test API directly
curl -H "Origin: http://localhost:8501" -H "Access-Control-Request-Method: POST" -X OPTIONS http://localhost:8000/ask
```

## Document Processing Issues

### File Upload Failures

**Symptoms:**
- "File type not supported"
- "File too large" errors

**Solutions:**
```powershell
# Check file type restrictions
python -c "
from sop_qa_tool.config.settings import settings
print('Allowed types:', settings.allowed_file_types)
print('Max size MB:', settings.max_file_size_mb)
"

# Update file restrictions
echo "ALLOWED_FILE_TYPES=pdf,docx,html,txt,doc" >> .env
echo "MAX_FILE_SIZE_MB=100" >> .env

# Test file processing
python -c "
from sop_qa_tool.services.document_ingestion import DocumentIngestionService
service = DocumentIngestionService()
print('Document service initialized')
"
```

### Text Extraction Issues

**Symptoms:**
- Empty text from PDFs
- OCR not working

**Solutions:**
```powershell
# Test text extraction
python examples/document_ingestion_demo.py

# Check PDF processing
python -c "
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf('test.pdf')
print(f'Extracted {len(elements)} elements')
"

# Enable OCR fallback
echo "ENABLE_OCR_FALLBACK=true" >> .env
echo "OCR_CONFIDENCE_THRESHOLD=60" >> .env
```

### Chunking Problems

**Symptoms:**
- Chunks too large or small
- Important context lost

**Solutions:**
```powershell
# Adjust chunking parameters
echo "CHUNK_SIZE=1000" >> .env
echo "CHUNK_OVERLAP=200" >> .env
echo "PRESERVE_HEADINGS=true" >> .env

# Test chunking
python examples/text_chunking_demo.py

# Debug chunk creation
python -c "
from sop_qa_tool.services.text_chunker import TextChunker
chunker = TextChunker()
chunks = chunker.chunk_text('Your test text here')
print(f'Created {len(chunks)} chunks')
for i, chunk in enumerate(chunks[:3]):
    print(f'Chunk {i}: {len(chunk.text)} chars')
"
```

## Diagnostic Tools

### Health Check Script

```powershell
# Create health check script
@"
# health_check.ps1
Write-Host "=== SOP QA Tool Health Check ==="

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "✓ Python: $pythonVersion"
} catch {
    Write-Host "✗ Python not found"
}

# Check virtual environment
if (Test-Path "sop-qa-venv\Scripts\Activate.ps1") {
    Write-Host "✓ Virtual environment exists"
} else {
    Write-Host "✗ Virtual environment missing"
}

# Check configuration
try {
    $config = python -c "from sop_qa_tool.config.settings import settings; print(settings.mode)"
    Write-Host "✓ Configuration loaded: $config"
} catch {
    Write-Host "✗ Configuration error"
}

# Check data directory
if (Test-Path "data") {
    $dataSize = (Get-ChildItem data -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "✓ Data directory: $([math]::Round($dataSize, 2)) MB"
} else {
    Write-Host "✗ Data directory missing"
}

# Check API health
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "✓ API health: $($response.status)"
} catch {
    Write-Host "✗ API not responding"
}

Write-Host "=== End Health Check ==="
"@ | Out-File -FilePath health_check.ps1 -Encoding UTF8

# Run health check
.\health_check.ps1
```

### Log Analysis

```powershell
# Check recent errors
Get-Content data/logs/app.log -Tail 50 | Select-String "ERROR\|CRITICAL"

# Monitor real-time logs
Get-Content data/logs/app.log -Wait -Tail 10

# Analyze performance
Get-Content data/logs/app.log | Select-String "duration\|latency" | Select-Object -Last 20
```

### System Resource Monitoring

```powershell
# Monitor Python processes
while ($true) {
    Get-Process python -ErrorAction SilentlyContinue | 
    Select-Object ProcessName, CPU, WorkingSet, VirtualMemorySize |
    Format-Table -AutoSize
    Start-Sleep 5
}

# Check disk space
Get-WmiObject -Class Win32_LogicalDisk | 
Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, 
@{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}
```

### Network Connectivity Tests

```powershell
# Test API endpoints
$endpoints = @(
    "http://localhost:8000/health",
    "http://localhost:8000/sources",
    "http://localhost:8501"
)

foreach ($endpoint in $endpoints) {
    try {
        $response = Invoke-WebRequest -Uri $endpoint -TimeoutSec 5
        Write-Host "✓ $endpoint - Status: $($response.StatusCode)"
    } catch {
        Write-Host "✗ $endpoint - Error: $($_.Exception.Message)"
    }
}

# Test AWS connectivity (if in AWS mode)
if ($env:MODE -eq "aws") {
    aws sts get-caller-identity
    aws bedrock list-foundation-models --region us-east-1 --max-items 1
}
```

## Getting Additional Help

### Enable Debug Logging

```powershell
# Add to .env file
echo "LOG_LEVEL=DEBUG" >> .env
echo "ENABLE_DETAILED_LOGGING=true" >> .env

# Restart services to apply changes
```

### Collect System Information

```powershell
# Create system info report
@"
=== System Information ===
OS: $(Get-WmiObject Win32_OperatingSystem | Select-Object -ExpandProperty Caption)
Python: $(python --version 2>&1)
Memory: $([math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory/1GB, 2)) GB
Disk: $(Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'" | ForEach-Object { [math]::Round($_.FreeSpace/1GB, 2) }) GB free

=== Configuration ===
$(Get-Content .env 2>$null | Where-Object { $_ -notmatch '^#' -and $_ -ne '' })

=== Recent Errors ===
$(Get-Content data/logs/app.log -Tail 20 2>$null | Select-String "ERROR\|CRITICAL")
"@ | Out-File -FilePath system_info.txt -Encoding UTF8

Write-Host "System information saved to system_info.txt"
```

### Contact Support

If issues persist after trying these solutions:

1. Run the health check script
2. Collect system information
3. Check the GitHub issues page
4. Create a new issue with:
   - System information
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior

Remember to remove any sensitive information (API keys, personal data) before sharing logs or configuration files.