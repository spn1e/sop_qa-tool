# SOP QA Tool Deployment Script
# This script provides automated deployment options for the SOP QA Tool

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("local", "aws", "docker-dev", "docker-prod")]
    [string]$Mode = "local",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipValidation,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force,
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigFile = ".env"
)

# Color output functions
function Write-Success { param($Message) Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "✗ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠ $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ $Message" -ForegroundColor Cyan }

# Banner
Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║                    SOP QA Tool Deployment                   ║
║              Automated Research & Q/A Tool                  ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Blue

Write-Info "Deployment Mode: $Mode"
Write-Info "Configuration File: $ConfigFile"

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $prerequisites = @()
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.1[1-9]") {
            Write-Success "Python: $pythonVersion"
        } else {
            Write-Error "Python 3.11+ required. Found: $pythonVersion"
            $prerequisites += "Python 3.11+"
        }
    } catch {
        Write-Error "Python not found"
        $prerequisites += "Python 3.11+"
    }
    
    # Check Docker (for Docker modes)
    if ($Mode -like "docker-*") {
        try {
            $dockerVersion = docker --version 2>&1
            Write-Success "Docker: $dockerVersion"
            
            $composeVersion = docker-compose --version 2>&1
            Write-Success "Docker Compose: $composeVersion"
        } catch {
            Write-Error "Docker or Docker Compose not found"
            $prerequisites += "Docker Desktop"
        }
    }
    
    # Check AWS CLI (for AWS mode)
    if ($Mode -eq "aws") {
        try {
            $awsVersion = aws --version 2>&1
            Write-Success "AWS CLI: $awsVersion"
        } catch {
            Write-Warning "AWS CLI not found (optional for AWS mode)"
        }
    }
    
    if ($prerequisites.Count -gt 0 -and -not $Force) {
        Write-Error "Missing prerequisites: $($prerequisites -join ', ')"
        Write-Info "Install missing prerequisites or use -Force to continue"
        exit 1
    }
    
    Write-Success "Prerequisites check completed"
}

# Create configuration
function New-Configuration {
    Write-Info "Creating configuration..."
    
    if (Test-Path $ConfigFile) {
        if (-not $Force) {
            $overwrite = Read-Host "Configuration file exists. Overwrite? (y/N)"
            if ($overwrite -ne "y" -and $overwrite -ne "Y") {
                Write-Info "Using existing configuration"
                return
            }
        }
    }
    
    # Copy template
    if (Test-Path ".env.template") {
        Copy-Item ".env.template" $ConfigFile
        Write-Success "Configuration template copied to $ConfigFile"
    } else {
        Write-Error "Configuration template not found"
        exit 1
    }
    
    # Update configuration based on mode
    switch ($Mode) {
        "local" {
            (Get-Content $ConfigFile) -replace "^MODE=.*", "MODE=local" | Set-Content $ConfigFile
            Write-Success "Configuration set to local mode"
        }
        "aws" {
            (Get-Content $ConfigFile) -replace "^MODE=.*", "MODE=aws" | Set-Content $ConfigFile
            Write-Success "Configuration set to AWS mode"
            Write-Warning "Please update AWS-specific settings in $ConfigFile"
        }
        { $_ -like "docker-*" } {
            (Get-Content $ConfigFile) -replace "^MODE=.*", "MODE=local" | Set-Content $ConfigFile
            Write-Success "Configuration set for Docker deployment"
        }
    }
}

# Setup Python environment
function Initialize-PythonEnvironment {
    Write-Info "Setting up Python environment..."
    
    # Create virtual environment
    if (-not (Test-Path "sop-qa-venv")) {
        python -m venv sop-qa-venv
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists"
    }
    
    # Activate virtual environment
    & ".\sop-qa-venv\Scripts\Activate.ps1"
    Write-Success "Virtual environment activated"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    Write-Success "Pip upgraded"
    
    # Install dependencies
    pip install -r requirements.txt
    Write-Success "Dependencies installed"
}

# Setup data directories
function Initialize-DataDirectories {
    Write-Info "Setting up data directories..."
    
    $directories = @(
        "data",
        "data/chunks",
        "data/raw_docs", 
        "data/faiss_index",
        "data/logs",
        "data/evaluation"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
    }
}

# AWS setup
function Initialize-AWSServices {
    Write-Info "Setting up AWS services..."
    
    # Check AWS credentials
    try {
        $identity = aws sts get-caller-identity 2>&1
        Write-Success "AWS credentials configured"
    } catch {
        Write-Error "AWS credentials not configured"
        Write-Info "Run 'aws configure' to set up credentials"
        return
    }
    
    # Setup OpenSearch
    Write-Info "Setting up OpenSearch Serverless..."
    try {
        & ".\scripts\setup-opensearch.ps1"
        Write-Success "OpenSearch setup completed"
    } catch {
        Write-Warning "OpenSearch setup failed: $($_.Exception.Message)"
    }
    
    # Setup S3 buckets
    Write-Info "Setting up S3 buckets..."
    try {
        & ".\scripts\setup-s3-buckets.ps1"
        Write-Success "S3 setup completed"
    } catch {
        Write-Warning "S3 setup failed: $($_.Exception.Message)"
    }
}

# Docker deployment
function Start-DockerDeployment {
    param([string]$DockerMode)
    
    Write-Info "Starting Docker deployment..."
    
    switch ($DockerMode) {
        "docker-dev" {
            Write-Info "Starting development environment..."
            docker-compose -f docker-compose.dev.yml up -d
            
            Write-Success "Development environment started"
            Write-Info "Services available at:"
            Write-Info "  - UI: http://localhost:8501"
            Write-Info "  - API: http://localhost:8000"
            Write-Info "  - Jupyter: http://localhost:8888"
        }
        "docker-prod" {
            Write-Info "Starting production environment..."
            docker-compose up -d
            
            Write-Success "Production environment started"
            Write-Info "Services available at:"
            Write-Info "  - UI: http://localhost:8501"
            Write-Info "  - API: http://localhost:8000"
        }
    }
    
    # Wait for services to be ready
    Write-Info "Waiting for services to start..."
    Start-Sleep -Seconds 10
    
    # Check service health
    try {
        $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Success "API service is healthy"
    } catch {
        Write-Warning "API service not responding"
    }
    
    try {
        $uiResponse = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 5
        Write-Success "UI service is accessible"
    } catch {
        Write-Warning "UI service not responding"
    }
}

# Local deployment
function Start-LocalDeployment {
    Write-Info "Starting local deployment..."
    
    # Activate virtual environment
    & ".\sop-qa-venv\Scripts\Activate.ps1"
    
    # Start API in background
    Write-Info "Starting API server..."
    Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "sop_qa_tool.api.main:app", "--host", "0.0.0.0", "--port", "8000" -WindowStyle Hidden
    
    # Wait for API to start
    Start-Sleep -Seconds 5
    
    # Start UI
    Write-Info "Starting UI server..."
    Start-Process -FilePath "streamlit" -ArgumentList "run", "run_ui.py", "--server.port", "8501"
    
    Write-Success "Local deployment started"
    Write-Info "Services available at:"
    Write-Info "  - UI: http://localhost:8501"
    Write-Info "  - API: http://localhost:8000"
}

# Validation
function Test-Deployment {
    if ($SkipValidation) {
        Write-Info "Skipping validation"
        return
    }
    
    Write-Info "Validating deployment..."
    
    # Test configuration loading
    try {
        if ($Mode -like "docker-*") {
            # Test Docker services
            $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
            Write-Success "API health check passed"
            
            $uiResponse = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 10
            Write-Success "UI accessibility check passed"
        } else {
            # Test Python environment
            & ".\sop-qa-venv\Scripts\Activate.ps1"
            python -c "from sop_qa_tool.config.settings import settings; print(f'Configuration loaded: {settings.mode}')"
            Write-Success "Configuration validation passed"
            
            # Run validation script
            python validate_setup.py
            Write-Success "Setup validation passed"
        }
    } catch {
        Write-Error "Validation failed: $($_.Exception.Message)"
        Write-Info "Check logs and configuration for issues"
    }
}

# Main deployment logic
function Start-Deployment {
    Write-Info "Starting deployment process..."
    
    # Check prerequisites
    Test-Prerequisites
    
    # Create configuration
    New-Configuration
    
    # Deploy based on mode
    switch ($Mode) {
        "local" {
            Initialize-PythonEnvironment
            Initialize-DataDirectories
            Start-LocalDeployment
        }
        "aws" {
            Initialize-PythonEnvironment
            Initialize-DataDirectories
            Initialize-AWSServices
            Start-LocalDeployment
        }
        { $_ -like "docker-*" } {
            Initialize-DataDirectories
            Start-DockerDeployment -DockerMode $Mode
        }
    }
    
    # Validate deployment
    Test-Deployment
    
    Write-Success "Deployment completed successfully!"
    
    # Show next steps
    Write-Info "Next steps:"
    switch ($Mode) {
        "local" {
            Write-Info "1. Access the UI at http://localhost:8501"
            Write-Info "2. Upload documents or enter URLs to get started"
            Write-Info "3. Check the troubleshooting guide if you encounter issues"
        }
        "aws" {
            Write-Info "1. Verify AWS services are configured correctly"
            Write-Info "2. Update $ConfigFile with your AWS endpoints"
            Write-Info "3. Access the UI at http://localhost:8501"
            Write-Info "4. Review the cost optimization guide"
        }
        { $_ -like "docker-*" } {
            Write-Info "1. Access the UI at http://localhost:8501"
            Write-Info "2. Check container logs: docker-compose logs -f"
            Write-Info "3. Scale services as needed: docker-compose up -d --scale api=2"
        }
    }
}

# Error handling
try {
    Start-Deployment
} catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    Write-Info "Check the troubleshooting guide for common issues"
    Write-Info "Run with -Force to bypass some checks"
    exit 1
}

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                   Deployment Complete!                      ║
║         SOP QA Tool is ready for use                        ║
╚══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green