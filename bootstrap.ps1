# SOP QA Tool - Windows 11 Bootstrap Script
# This script sets up Python 3.11+, virtual environment, and dependencies

param(
    [switch]$SkipPythonInstall,
    [switch]$SkipVenvCreation,
    [string]$VenvName = "sop-qa-venv",
    [string]$PythonVersion = "3.11"
)

Write-Host "=== SOP QA Tool Bootstrap Script ===" -ForegroundColor Green
Write-Host "Setting up Python environment on Windows 11..." -ForegroundColor Yellow

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Function to get Python version
function Get-PythonVersion {
    param($PythonCmd)
    try {
        $version = & $PythonCmd --version 2>&1
        if ($version -match "Python (\d+\.\d+)") {
            return [version]$matches[1]
        }
    }
    catch {
        return $null
    }
    return $null
}

# Check for Python 3.11+
Write-Host "Checking Python installation..." -ForegroundColor Yellow

$pythonCommands = @("python", "python3", "py")
$validPython = $null
$requiredVersion = [version]$PythonVersion

foreach ($cmd in $pythonCommands) {
    if (Test-Command $cmd) {
        $version = Get-PythonVersion $cmd
        if ($version -and $version -ge $requiredVersion) {
            $validPython = $cmd
            Write-Host "Found Python $version at: $cmd" -ForegroundColor Green
            break
        }
    }
}

if (-not $validPython -and -not $SkipPythonInstall) {
    Write-Host "Python $PythonVersion+ not found. Installing Python..." -ForegroundColor Red
    
    # Check if winget is available
    if (Test-Command "winget") {
        Write-Host "Installing Python using winget..." -ForegroundColor Yellow
        winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Re-check for Python
        foreach ($cmd in $pythonCommands) {
            if (Test-Command $cmd) {
                $version = Get-PythonVersion $cmd
                if ($version -and $version -ge $requiredVersion) {
                    $validPython = $cmd
                    Write-Host "Python installation successful: $cmd (version $version)" -ForegroundColor Green
                    break
                }
            }
        }
    }
    else {
        Write-Host "winget not available. Please install Python $PythonVersion+ manually from https://python.org" -ForegroundColor Red
        Write-Host "After installation, run this script again with -SkipPythonInstall" -ForegroundColor Yellow
        exit 1
    }
}

if (-not $validPython) {
    Write-Host "ERROR: Python $PythonVersion+ is required but not found." -ForegroundColor Red
    Write-Host "Please install Python $PythonVersion+ and ensure it's in your PATH." -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment already exists
$venvPath = Join-Path $PWD $VenvName
if (Test-Path $venvPath) {
    if (-not $SkipVenvCreation) {
        $response = Read-Host "Virtual environment '$VenvName' already exists. Recreate it? (y/N)"
        if ($response -eq "y" -or $response -eq "Y") {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $venvPath
        }
        else {
            Write-Host "Using existing virtual environment..." -ForegroundColor Green
            $SkipVenvCreation = $true
        }
    }
}

# Create virtual environment
if (-not $SkipVenvCreation) {
    Write-Host "Creating virtual environment '$VenvName'..." -ForegroundColor Yellow
    & $validPython -m venv $VenvName
    
    if (-not $?) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript
    
    # Verify activation
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Host "Virtual environment activated successfully." -ForegroundColor Green
        Write-Host "Python location: $venvPython" -ForegroundColor Cyan
    }
}
else {
    Write-Host "ERROR: Could not find activation script at $activateScript" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip

# Install requirements if requirements.txt exists
$requirementsFile = Join-Path $PWD "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & python -m pip install -r $requirementsFile
    
    if ($?) {
        Write-Host "Dependencies installed successfully." -ForegroundColor Green
    }
    else {
        Write-Host "WARNING: Some dependencies may have failed to install." -ForegroundColor Yellow
    }
}
else {
    Write-Host "No requirements.txt found. Skipping dependency installation." -ForegroundColor Yellow
    Write-Host "You can install dependencies later with: pip install -r requirements.txt" -ForegroundColor Cyan
}

# Create .env file if it doesn't exist
$envFile = Join-Path $PWD ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating default .env file..." -ForegroundColor Yellow
    
    $defaultEnv = @"
# SOP QA Tool Configuration
# Mode: aws or local
MODE=local

# Local Configuration (used when MODE=local)
LOCAL_DATA_PATH=./data
FAISS_INDEX_PATH=./data/faiss_index
HF_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2

# AWS Configuration (used when MODE=aws)
# AWS_PROFILE=default
# AWS_REGION=us-east-1
# BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
# TITAN_EMBEDDINGS_ID=amazon.titan-embed-text-v2:0
# OPENSEARCH_ENDPOINT=
# S3_RAW_BUCKET=
# S3_CHUNKS_BUCKET=

# Application Settings
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K_RETRIEVAL=5
CONFIDENCE_THRESHOLD=0.35

# Security Settings
ALLOWED_FILE_TYPES=pdf,docx,html,txt
ENABLE_PII_REDACTION=false
BLOCK_LOCALHOST_URLS=true

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT_SECONDS=30
EMBEDDING_BATCH_SIZE=32

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true
"@
    
    Set-Content -Path $envFile -Value $defaultEnv -Encoding UTF8
    Write-Host "Default .env file created. Please review and modify as needed." -ForegroundColor Green
}

# Validate configuration
Write-Host "Validating configuration..." -ForegroundColor Yellow
& python -c "from sop_qa_tool.config.settings import validate_settings; exit(0 if validate_settings() else 1)"

if ($?) {
    Write-Host "Configuration validation successful." -ForegroundColor Green
}
else {
    Write-Host "WARNING: Configuration validation failed. Please check your .env file." -ForegroundColor Yellow
}

Write-Host "" -ForegroundColor White
Write-Host "=== Bootstrap Complete ===" -ForegroundColor Green
Write-Host "Virtual environment: $VenvName" -ForegroundColor Cyan
Write-Host "Python executable: $venvPython" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "To activate the environment in future sessions, run:" -ForegroundColor Yellow
Write-Host "  .\$VenvName\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "To start development:" -ForegroundColor Yellow
Write-Host "  1. Review and modify .env file as needed" -ForegroundColor Cyan
Write-Host "  2. Install additional dependencies: pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host "  3. Run tests: python -m pytest tests/" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White