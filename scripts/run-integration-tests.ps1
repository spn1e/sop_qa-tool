# PowerShell script to run integration tests for the SOP Q&A Tool
# Usage: .\scripts\run-integration-tests.ps1 [TestType] [-IncludeSlow] [-ValidateEnv]

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "e2e", "performance", "mode-switching", "error-scenarios", "acceptance", "smoke", "report")]
    [string]$TestType = "all",
    
    [switch]$IncludeSlow,
    [switch]$ValidateEnv,
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host @"
Integration Test Runner for SOP Q&A Tool

Usage: .\scripts\run-integration-tests.ps1 [TestType] [Options]

Test Types:
  all              Run all integration tests (default)
  e2e              Run end-to-end integration tests
  performance      Run performance tests
  mode-switching   Run mode switching tests
  error-scenarios  Run error scenario tests
  acceptance       Run acceptance tests validating requirements
  smoke            Run quick smoke tests
  report           Generate comprehensive test report with coverage

Options:
  -IncludeSlow     Include slow-running tests
  -ValidateEnv     Validate test environment before running
  -Help            Show this help message

Examples:
  .\scripts\run-integration-tests.ps1 smoke
  .\scripts\run-integration-tests.ps1 all -IncludeSlow
  .\scripts\run-integration-tests.ps1 performance -ValidateEnv
"@
    exit 0
}

# Set error handling
$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "SOP Q&A Tool - Integration Test Runner" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot"
Write-Host "Test Type: $TestType"
Write-Host ""

# Change to project root
Set-Location $ProjectRoot

# Validate environment if requested
if ($ValidateEnv) {
    Write-Host "Validating test environment..." -ForegroundColor Yellow
    
    # Check Python installation
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Python not found. Please install Python 3.11+" -ForegroundColor Red
        exit 1
    }
    
    # Check if virtual environment is activated
    if ($env:VIRTUAL_ENV) {
        Write-Host "✓ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
    }
    else {
        Write-Host "⚠ No virtual environment detected. Consider activating one." -ForegroundColor Yellow
    }
    
    # Check required packages
    $requiredPackages = @("pytest", "pytest-asyncio", "pytest-cov", "fastapi", "streamlit")
    foreach ($package in $requiredPackages) {
        try {
            python -c "import $($package.Replace('-', '_'))" 2>$null
            Write-Host "✓ Package: $package" -ForegroundColor Green
        }
        catch {
            Write-Host "✗ Missing package: $package" -ForegroundColor Red
            Write-Host "  Install with: pip install -r requirements.txt" -ForegroundColor Yellow
            exit 1
        }
    }
    
    Write-Host "Environment validation passed!" -ForegroundColor Green
    Write-Host ""
}

# Set environment variables for testing
$env:TESTING = "true"
$env:LOG_LEVEL = "WARNING"

# Create test reports directory
$ReportsDir = Join-Path $ProjectRoot "test_reports"
if (-not (Test-Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
}

# Build pytest command based on test type
$pytestArgs = @()

switch ($TestType) {
    "all" {
        Write-Host "Running all integration tests..." -ForegroundColor Yellow
        $pytestArgs += @(
            "tests/test_integration_e2e.py",
            "tests/test_performance.py", 
            "tests/test_mode_switching.py",
            "tests/test_error_scenarios.py",
            "tests/test_acceptance_requirements.py"
        )
    }
    "e2e" {
        Write-Host "Running end-to-end integration tests..." -ForegroundColor Yellow
        $pytestArgs += "tests/test_integration_e2e.py"
    }
    "performance" {
        Write-Host "Running performance tests..." -ForegroundColor Yellow
        $pytestArgs += "tests/test_performance.py"
        $pytestArgs += @("-m", "performance")
    }
    "mode-switching" {
        Write-Host "Running mode switching tests..." -ForegroundColor Yellow
        $pytestArgs += "tests/test_mode_switching.py"
    }
    "error-scenarios" {
        Write-Host "Running error scenario tests..." -ForegroundColor Yellow
        $pytestArgs += "tests/test_error_scenarios.py"
    }
    "acceptance" {
        Write-Host "Running acceptance tests..." -ForegroundColor Yellow
        $pytestArgs += "tests/test_acceptance_requirements.py"
    }
    "smoke" {
        Write-Host "Running quick smoke tests..." -ForegroundColor Yellow
        $pytestArgs += @(
            "tests/test_integration_e2e.py::TestEndToEndIntegration::test_health_check_integration",
            "tests/test_mode_switching.py::TestModeSwitching::test_local_mode_functionality",
            "tests/test_error_scenarios.py::TestErrorScenarios::test_invalid_file_type_rejection"
        )
    }
    "report" {
        Write-Host "Generating comprehensive test report..." -ForegroundColor Yellow
        $pytestArgs += "tests/"
        $pytestArgs += @(
            "--cov=sop_qa_tool",
            "--cov-report=html:test_reports/coverage_html",
            "--cov-report=xml:test_reports/coverage.xml",
            "--cov-report=term-missing",
            "--junit-xml=test_reports/junit.xml"
        )
    }
}

# Add common pytest options
$pytestArgs += @(
    "-v",
    "--tb=short",
    "--maxfail=5",
    "--durations=10"
)

# Add slow test marker if not including slow tests
if (-not $IncludeSlow -and $TestType -ne "smoke") {
    $pytestArgs += @("-m", "not slow")
}

# Record start time
$startTime = Get-Date

# Run pytest
Write-Host "Executing: python -m pytest $($pytestArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

try {
    & python -m pytest @pytestArgs
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Host "Error running pytest: $_" -ForegroundColor Red
    $exitCode = 1
}

# Calculate duration
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

# Display results
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "TEST EXECUTION COMPLETED" -ForegroundColor Cyan
Write-Host "Duration: $([math]::Round($duration, 2)) seconds"
Write-Host "Exit code: $exitCode"
Write-Host "============================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "✅ All tests passed!" -ForegroundColor Green
    
    # Show report locations if generated
    if ($TestType -eq "report") {
        Write-Host ""
        Write-Host "Test reports generated:" -ForegroundColor Green
        Write-Host "  HTML Coverage: test_reports/coverage_html/index.html"
        Write-Host "  XML Coverage: test_reports/coverage.xml"
        Write-Host "  JUnit XML: test_reports/junit.xml"
    }
}
else {
    Write-Host "❌ Some tests failed. Check the output above for details." -ForegroundColor Red
}

# Clean up test artifacts
Write-Host ""
Write-Host "Cleaning up test artifacts..." -ForegroundColor Yellow
$testDirs = @("test_data", "test_faiss", "test_data_local", "test_data_aws")
foreach ($dir in $testDirs) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (Test-Path $fullPath) {
        Remove-Item -Recurse -Force $fullPath -ErrorAction SilentlyContinue
        Write-Host "  Removed: $dir"
    }
}

exit $exitCode