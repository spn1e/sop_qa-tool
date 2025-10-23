#!/usr/bin/env pwsh
<#
.SYNOPSIS
    System health monitoring script for SOP QA Tool
    
.DESCRIPTION
    Monitors system health with component status checks, performance metrics,
    and alerting capabilities. Supports continuous monitoring and one-time checks.
    
.PARAMETER ApiUrl
    Base URL of the SOP QA API (default: http://localhost:8000)
    
.PARAMETER Mode
    Operation mode: aws or local (default: local)
    
.PARAMETER Continuous
    Run continuous monitoring (default: single check)
    
.PARAMETER IntervalSeconds
    Monitoring interval in seconds for continuous mode (default: 60)
    
.PARAMETER LogFile
    Path to log file for monitoring output
    
.PARAMETER AlertThresholds
    JSON file with custom alert thresholds
    
.PARAMETER OutputFormat
    Output format: console, json, or csv (default: console)
    
.EXAMPLE
    .\health-monitor.ps1 -Mode local
    
.EXAMPLE
    .\health-monitor.ps1 -Continuous -IntervalSeconds 30 -LogFile "health.log"
    
.EXAMPLE
    .\health-monitor.ps1 -OutputFormat json -AlertThresholds "thresholds.json"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://localhost:8000",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("aws", "local")]
    [string]$Mode = "local",
    
    [Parameter(Mandatory=$false)]
    [switch]$Continuous,
    
    [Parameter(Mandatory=$false)]
    [int]$IntervalSeconds = 60,
    
    [Parameter(Mandatory=$false)]
    [string]$LogFile,
    
    [Parameter(Mandatory=$false)]
    [string]$AlertThresholds,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("console", "json", "csv")]
    [string]$OutputFormat = "console"
)

# Default alert thresholds
$defaultThresholds = @{
    response_time_ms = 5000
    memory_usage_percent = 80
    disk_usage_percent = 85
    error_rate_percent = 5
    index_age_hours = 24
}

# Load custom thresholds if provided
if ($AlertThresholds -and (Test-Path $AlertThresholds)) {
    try {
        $customThresholds = Get-Content $AlertThresholds | ConvertFrom-Json -AsHashtable
        foreach ($key in $customThresholds.Keys) {
            $defaultThresholds[$key] = $customThresholds[$key]
        }
        Write-Host "Loaded custom alert thresholds from $AlertThresholds"
    } catch {
        Write-Host "Failed to load custom thresholds: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Initialize logging
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    if ($OutputFormat -eq "console") {
        $color = switch ($Level) {
            "ERROR" { "Red" }
            "WARN" { "Yellow" }
            "SUCCESS" { "Green" }
            default { "White" }
        }
        Write-Host $logEntry -ForegroundColor $color
    }
    
    if ($LogFile) {
        Add-Content -Path $LogFile -Value $logEntry
    }
}

# Get system performance metrics
function Get-SystemMetrics {
    $metrics = @{}
    
    try {
        # Memory usage
        $memory = Get-WmiObject -Class Win32_OperatingSystem
        $totalMemory = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
        $freeMemory = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
        $usedMemory = $totalMemory - $freeMemory
        $memoryPercent = [math]::Round(($usedMemory / $totalMemory) * 100, 1)
        
        $metrics.memory = @{
            total_gb = $totalMemory
            used_gb = $usedMemory
            free_gb = $freeMemory
            usage_percent = $memoryPercent
        }
        
        # Disk usage
        $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3" | Where-Object { $_.DeviceID -eq "C:" }
        if ($disk) {
            $totalDisk = [math]::Round($disk.Size / 1GB, 2)
            $freeDisk = [math]::Round($disk.FreeSpace / 1GB, 2)
            $usedDisk = $totalDisk - $freeDisk
            $diskPercent = [math]::Round(($usedDisk / $totalDisk) * 100, 1)
            
            $metrics.disk = @{
                total_gb = $totalDisk
                used_gb = $usedDisk
                free_gb = $freeDisk
                usage_percent = $diskPercent
            }
        }
        
        # CPU usage (average over 5 seconds)
        $cpu = Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average
        $metrics.cpu = @{
            usage_percent = [math]::Round($cpu.Average, 1)
        }
        
    } catch {
        Write-Log "Failed to collect system metrics: $($_.Exception.Message)" "WARN"
    }
    
    return $metrics
}

# Check API health
function Test-ApiHealth {
    $healthCheck = @{
        status = "unknown"
        response_time_ms = 0
        components = @{}
        errors = @()
    }
    
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri "$ApiUrl/health" -Method GET -TimeoutSec 30
        $stopwatch.Stop()
        
        $healthCheck.status = $response.status
        $healthCheck.response_time_ms = $stopwatch.ElapsedMilliseconds
        $healthCheck.components = $response.components
        
        if ($response.errors) {
            $healthCheck.errors = $response.errors
        }
        
    } catch {
        $healthCheck.status = "error"
        $healthCheck.errors += "API request failed: $($_.Exception.Message)"
    }
    
    return $healthCheck
}

# Check index health
function Test-IndexHealth {
    $indexHealth = @{
        status = "unknown"
        document_count = 0
        last_updated = $null
        size_mb = 0
        errors = @()
    }
    
    try {
        $sourcesResponse = Invoke-RestMethod -Uri "$ApiUrl/sources" -Method GET -TimeoutSec 30
        
        $indexHealth.document_count = $sourcesResponse.sources.Count
        $indexHealth.size_mb = ($sourcesResponse.sources | Measure-Object -Property size_mb -Sum).Sum
        
        if ($sourcesResponse.sources.Count -gt 0) {
            $lastUpdated = $sourcesResponse.sources | Sort-Object last_updated -Descending | Select-Object -First 1 -ExpandProperty last_updated
            $indexHealth.last_updated = $lastUpdated
            
            # Check if index is stale
            $lastUpdateTime = [DateTime]::Parse($lastUpdated)
            $hoursSinceUpdate = ((Get-Date) - $lastUpdateTime).TotalHours
            
            if ($hoursSinceUpdate -gt $defaultThresholds.index_age_hours) {
                $indexHealth.errors += "Index is stale (last updated $([math]::Round($hoursSinceUpdate, 1)) hours ago)"
            }
        }
        
        $indexHealth.status = if ($indexHealth.errors.Count -eq 0) { "healthy" } else { "warning" }
        
    } catch {
        $indexHealth.status = "error"
        $indexHealth.errors += "Failed to check index: $($_.Exception.Message)"
    }
    
    return $indexHealth
}

# Evaluate health status and generate alerts
function Test-HealthThresholds {
    param(
        [hashtable]$SystemMetrics,
        [hashtable]$ApiHealth,
        [hashtable]$IndexHealth
    )
    
    $alerts = @()
    
    # Check response time
    if ($ApiHealth.response_time_ms -gt $defaultThresholds.response_time_ms) {
        $alerts += "High API response time: $($ApiHealth.response_time_ms)ms (threshold: $($defaultThresholds.response_time_ms)ms)"
    }
    
    # Check memory usage
    if ($SystemMetrics.memory -and $SystemMetrics.memory.usage_percent -gt $defaultThresholds.memory_usage_percent) {
        $alerts += "High memory usage: $($SystemMetrics.memory.usage_percent)% (threshold: $($defaultThresholds.memory_usage_percent)%)"
    }
    
    # Check disk usage
    if ($SystemMetrics.disk -and $SystemMetrics.disk.usage_percent -gt $defaultThresholds.disk_usage_percent) {
        $alerts += "High disk usage: $($SystemMetrics.disk.usage_percent)% (threshold: $($defaultThresholds.disk_usage_percent)%)"
    }
    
    # Check API status
    if ($ApiHealth.status -ne "healthy") {
        $alerts += "API health check failed: $($ApiHealth.status)"
    }
    
    # Check index status
    if ($IndexHealth.status -eq "error") {
        $alerts += "Index health check failed"
    }
    
    # Add component-specific errors
    if ($ApiHealth.errors) {
        $alerts += $ApiHealth.errors
    }
    
    if ($IndexHealth.errors) {
        $alerts += $IndexHealth.errors
    }
    
    return $alerts
}

# Format output based on selected format
function Format-HealthOutput {
    param(
        [hashtable]$SystemMetrics,
        [hashtable]$ApiHealth,
        [hashtable]$IndexHealth,
        [array]$Alerts
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $overallStatus = if ($Alerts.Count -eq 0) { "HEALTHY" } else { "UNHEALTHY" }
    
    switch ($OutputFormat) {
        "json" {
            $output = @{
                timestamp = $timestamp
                overall_status = $overallStatus
                system_metrics = $SystemMetrics
                api_health = $ApiHealth
                index_health = $IndexHealth
                alerts = $Alerts
            } | ConvertTo-Json -Depth 5
            
            Write-Output $output
        }
        
        "csv" {
            $csvData = [PSCustomObject]@{
                Timestamp = $timestamp
                OverallStatus = $overallStatus
                ApiStatus = $ApiHealth.status
                ApiResponseTime = $ApiHealth.response_time_ms
                IndexStatus = $IndexHealth.status
                DocumentCount = $IndexHealth.document_count
                MemoryUsage = if ($SystemMetrics.memory) { $SystemMetrics.memory.usage_percent } else { "N/A" }
                DiskUsage = if ($SystemMetrics.disk) { $SystemMetrics.disk.usage_percent } else { "N/A" }
                AlertCount = $Alerts.Count
                Alerts = ($Alerts -join "; ")
            }
            
            $csvData | ConvertTo-Csv -NoTypeInformation
        }
        
        default {
            Write-Log "=== SYSTEM HEALTH CHECK - $timestamp ===" "INFO"
            Write-Log "Overall Status: $overallStatus" $(if ($overallStatus -eq "HEALTHY") { "SUCCESS" } else { "ERROR" })
            
            # API Health
            Write-Log "API Health: $($ApiHealth.status) (Response: $($ApiHealth.response_time_ms)ms)" $(if ($ApiHealth.status -eq "healthy") { "SUCCESS" } else { "ERROR" })
            
            # Index Health
            Write-Log "Index Health: $($IndexHealth.status) ($($IndexHealth.document_count) documents, $([math]::Round($IndexHealth.size_mb, 2)) MB)" $(if ($IndexHealth.status -eq "healthy") { "SUCCESS" } else { "WARN" })
            
            # System Metrics
            if ($SystemMetrics.memory) {
                Write-Log "Memory: $($SystemMetrics.memory.usage_percent)% used ($($SystemMetrics.memory.used_gb)GB / $($SystemMetrics.memory.total_gb)GB)" "INFO"
            }
            
            if ($SystemMetrics.disk) {
                Write-Log "Disk: $($SystemMetrics.disk.usage_percent)% used ($($SystemMetrics.disk.used_gb)GB / $($SystemMetrics.disk.total_gb)GB)" "INFO"
            }
            
            if ($SystemMetrics.cpu) {
                Write-Log "CPU: $($SystemMetrics.cpu.usage_percent)% usage" "INFO"
            }
            
            # Alerts
            if ($Alerts.Count -gt 0) {
                Write-Log "ALERTS ($($Alerts.Count)):" "ERROR"
                foreach ($alert in $Alerts) {
                    Write-Log "  - $alert" "ERROR"
                }
            } else {
                Write-Log "No alerts detected" "SUCCESS"
            }
            
            Write-Log "================================" "INFO"
        }
    }
}

# Main health check function
function Invoke-HealthCheck {
    Write-Log "Starting health check (Mode: $Mode)" "INFO"
    
    # Collect metrics
    $systemMetrics = Get-SystemMetrics
    $apiHealth = Test-ApiHealth
    $indexHealth = Test-IndexHealth
    
    # Evaluate thresholds
    $alerts = Test-HealthThresholds -SystemMetrics $systemMetrics -ApiHealth $apiHealth -IndexHealth $indexHealth
    
    # Format and output results
    Format-HealthOutput -SystemMetrics $systemMetrics -ApiHealth $apiHealth -IndexHealth $indexHealth -Alerts $alerts
    
    # Return overall status for exit code
    return ($alerts.Count -eq 0)
}

# Main execution
if ($Continuous) {
    Write-Log "Starting continuous health monitoring (interval: $IntervalSeconds seconds)" "INFO"
    Write-Log "Press Ctrl+C to stop monitoring" "INFO"
    
    try {
        while ($true) {
            $isHealthy = Invoke-HealthCheck
            
            if (-not $isHealthy) {
                Write-Log "Health check detected issues" "WARN"
            }
            
            Start-Sleep -Seconds $IntervalSeconds
        }
    } catch {
        Write-Log "Monitoring interrupted: $($_.Exception.Message)" "INFO"
    }
} else {
    # Single health check
    $isHealthy = Invoke-HealthCheck
    
    if ($isHealthy) {
        exit 0
    } else {
        exit 1
    }
}