#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Index rebuild script for SOP QA Tool
    
.DESCRIPTION
    Rebuilds the vector index from existing documents with progress tracking and error handling.
    Supports both AWS and local mode operation.
    
.PARAMETER ApiUrl
    Base URL of the SOP QA API (default: http://localhost:8000)
    
.PARAMETER Mode
    Operation mode: aws or local (default: local)
    
.PARAMETER BackupIndex
    Create backup of existing index before rebuild
    
.PARAMETER LogFile
    Path to log file for operation tracking
    
.PARAMETER Force
    Force rebuild without confirmation prompt
    
.EXAMPLE
    .\rebuild-index.ps1 -BackupIndex -LogFile "rebuild.log"
    
.EXAMPLE
    .\rebuild-index.ps1 -Mode aws -Force -ApiUrl "http://localhost:8000"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://localhost:8000",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("aws", "local")]
    [string]$Mode = "local",
    
    [Parameter(Mandatory=$false)]
    [switch]$BackupIndex,
    
    [Parameter(Mandatory=$false)]
    [string]$LogFile = "rebuild-index-$(Get-Date -Format 'yyyyMMdd-HHmmss').log",
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Initialize logging
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    Write-Host $logEntry
    Add-Content -Path $LogFile -Value $logEntry
}

# Check system health before rebuild
function Test-SystemHealth {
    Write-Log "Checking system health before rebuild"
    
    try {
        $healthResponse = Invoke-RestMethod -Uri "$ApiUrl/health" -Method GET -TimeoutSec 30
        
        if ($healthResponse.status -eq "healthy") {
            Write-Log "System health check passed"
            return $true
        } else {
            Write-Log "System health check failed: $($healthResponse.message)" "ERROR"
            return $false
        }
    } catch {
        Write-Log "Failed to check system health: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Get current index statistics
function Get-IndexStats {
    Write-Log "Retrieving current index statistics"
    
    try {
        $sourcesResponse = Invoke-RestMethod -Uri "$ApiUrl/sources" -Method GET -TimeoutSec 30
        
        $stats = @{
            total_documents = $sourcesResponse.sources.Count
            total_size_mb = ($sourcesResponse.sources | Measure-Object -Property size_mb -Sum).Sum
            last_updated = $sourcesResponse.sources | Sort-Object last_updated -Descending | Select-Object -First 1 -ExpandProperty last_updated
        }
        
        Write-Log "Current index stats: $($stats.total_documents) documents, $([Math]::Round($stats.total_size_mb, 2)) MB"
        return $stats
    } catch {
        Write-Log "Failed to retrieve index statistics: $($_.Exception.Message)" "WARN"
        return $null
    }
}

# Create index backup
function Backup-Index {
    if (-not $BackupIndex) {
        return $true
    }
    
    Write-Log "Creating index backup"
    
    try {
        $backupDir = "backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        
        if ($Mode -eq "local") {
            # Backup local FAISS index and metadata
            $dataPath = "./data"
            $backupPath = "./backups/$backupDir"
            
            if (Test-Path $dataPath) {
                New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
                Copy-Item -Path "$dataPath/*" -Destination $backupPath -Recurse -Force
                Write-Log "Local index backed up to: $backupPath"
            } else {
                Write-Log "No local data directory found to backup" "WARN"
            }
        } else {
            # For AWS mode, we rely on S3 versioning and OpenSearch snapshots
            Write-Log "AWS mode: Relying on S3 versioning and OpenSearch automatic snapshots"
        }
        
        return $true
    } catch {
        Write-Log "Failed to create backup: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Monitor rebuild progress
function Monitor-RebuildProgress {
    param([string]$RebuildId)
    
    $maxWaitMinutes = 60
    $checkIntervalSeconds = 10
    $maxChecks = ($maxWaitMinutes * 60) / $checkIntervalSeconds
    $checks = 0
    
    Write-Log "Monitoring rebuild progress (max wait: $maxWaitMinutes minutes)"
    
    while ($checks -lt $maxChecks) {
        try {
            $statusResponse = Invoke-RestMethod -Uri "$ApiUrl/reindex/status/$RebuildId" -Method GET -TimeoutSec 30
            
            $status = $statusResponse.status
            $progress = $statusResponse.progress
            
            Write-Log "Rebuild status: $status, Progress: $($progress.processed)/$($progress.total) documents"
            
            if ($status -eq "completed") {
                Write-Log "Rebuild completed successfully"
                return $true
            } elseif ($status -eq "failed") {
                Write-Log "Rebuild failed: $($statusResponse.error)" "ERROR"
                return $false
            }
            
            Start-Sleep -Seconds $checkIntervalSeconds
            $checks++
            
        } catch {
            Write-Log "Failed to check rebuild status: $($_.Exception.Message)" "WARN"
            Start-Sleep -Seconds $checkIntervalSeconds
            $checks++
        }
    }
    
    Write-Log "Rebuild monitoring timed out after $maxWaitMinutes minutes" "ERROR"
    return $false
}

# Main execution
Write-Log "=== INDEX REBUILD STARTED ==="
Write-Log "Mode: $Mode"
Write-Log "API URL: $ApiUrl"
Write-Log "Backup enabled: $BackupIndex"

# Confirmation prompt
if (-not $Force) {
    Write-Host "This will rebuild the entire vector index. This operation may take significant time." -ForegroundColor Yellow
    $confirmation = Read-Host "Do you want to continue? (y/N)"
    
    if ($confirmation -ne "y" -and $confirmation -ne "Y") {
        Write-Log "Rebuild cancelled by user"
        exit 0
    }
}

$startTime = Get-Date

# Step 1: Health check
if (-not (Test-SystemHealth)) {
    Write-Log "System health check failed. Aborting rebuild." "ERROR"
    exit 1
}

# Step 2: Get current stats
$initialStats = Get-IndexStats

# Step 3: Create backup
if (-not (Backup-Index)) {
    Write-Log "Backup creation failed. Aborting rebuild." "ERROR"
    exit 1
}

# Step 4: Initiate rebuild
Write-Log "Initiating index rebuild"
try {
    $rebuildRequest = @{
        mode = $Mode
        force = $true
    } | ConvertTo-Json
    
    $rebuildResponse = Invoke-RestMethod -Uri "$ApiUrl/reindex" -Method POST -Body $rebuildRequest -ContentType "application/json" -TimeoutSec 60
    
    if ($rebuildResponse.success) {
        $rebuildId = $rebuildResponse.rebuild_id
        Write-Log "Rebuild initiated with ID: $rebuildId"
        
        # Step 5: Monitor progress
        $rebuildSuccess = Monitor-RebuildProgress -RebuildId $rebuildId
        
        if ($rebuildSuccess) {
            # Step 6: Verify rebuild
            Write-Log "Verifying rebuilt index"
            $finalStats = Get-IndexStats
            
            if ($finalStats) {
                Write-Log "Rebuild verification:"
                Write-Log "  Documents: $($initialStats.total_documents) -> $($finalStats.total_documents)"
                Write-Log "  Size (MB): $([Math]::Round($initialStats.total_size_mb, 2)) -> $([Math]::Round($finalStats.total_size_mb, 2))"
            }
            
            # Step 7: Health check after rebuild
            if (Test-SystemHealth) {
                $endTime = Get-Date
                $duration = $endTime - $startTime
                $durationStr = "{0:hh\:mm\:ss}" -f $duration
                
                Write-Log "=== INDEX REBUILD COMPLETED SUCCESSFULLY ==="
                Write-Log "Duration: $durationStr"
                Write-Log "Log file: $LogFile"
                exit 0
            } else {
                Write-Log "Post-rebuild health check failed" "ERROR"
                exit 1
            }
        } else {
            Write-Log "Index rebuild failed" "ERROR"
            exit 1
        }
    } else {
        Write-Log "Failed to initiate rebuild: $($rebuildResponse.message)" "ERROR"
        exit 1
    }
    
} catch {
    Write-Log "Rebuild request failed: $($_.Exception.Message)" "ERROR"
    exit 1
}