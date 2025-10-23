#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Bulk URL ingestion script for SOP QA Tool
    
.DESCRIPTION
    Reads URLs from a text file and ingests them into the SOP QA system.
    Supports progress tracking, error handling, and resumable operations.
    
.PARAMETER UrlFile
    Path to text file containing URLs (one per line)
    
.PARAMETER ApiUrl
    Base URL of the SOP QA API (default: http://localhost:8000)
    
.PARAMETER BatchSize
    Number of URLs to process in each batch (default: 5)
    
.PARAMETER LogFile
    Path to log file for operation tracking
    
.PARAMETER Resume
    Resume from last successful position (requires log file)
    
.EXAMPLE
    .\bulk-ingest-urls.ps1 -UrlFile "urls.txt" -LogFile "ingest.log"
    
.EXAMPLE
    .\bulk-ingest-urls.ps1 -UrlFile "urls.txt" -ApiUrl "http://localhost:8000" -BatchSize 3 -Resume
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$UrlFile,
    
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://localhost:8000",
    
    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 5,
    
    [Parameter(Mandatory=$false)]
    [string]$LogFile = "bulk-ingest-$(Get-Date -Format 'yyyyMMdd-HHmmss').log",
    
    [Parameter(Mandatory=$false)]
    [switch]$Resume
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

# Check if URL file exists
if (-not (Test-Path $UrlFile)) {
    Write-Log "ERROR: URL file '$UrlFile' not found" "ERROR"
    exit 1
}

# Read URLs from file
Write-Log "Reading URLs from '$UrlFile'"
$urls = Get-Content $UrlFile | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") }
$totalUrls = $urls.Count

if ($totalUrls -eq 0) {
    Write-Log "No valid URLs found in file" "WARN"
    exit 0
}

Write-Log "Found $totalUrls URLs to process"

# Resume logic
$startIndex = 0
if ($Resume -and (Test-Path $LogFile)) {
    Write-Log "Resume mode enabled, checking log file for last position"
    $logContent = Get-Content $LogFile
    $successPattern = "Successfully processed URL \d+/\d+:"
    $lastSuccess = $logContent | Where-Object { $_ -match $successPattern } | Select-Object -Last 1
    
    if ($lastSuccess) {
        $matches = [regex]::Matches($lastSuccess, "Successfully processed URL (\d+)/\d+:")
        if ($matches.Count -gt 0) {
            $startIndex = [int]$matches[0].Groups[1].Value
            Write-Log "Resuming from position $startIndex"
        }
    }
}

# Progress tracking
$processed = $startIndex
$successful = 0
$failed = 0
$startTime = Get-Date

Write-Log "Starting bulk ingestion (Batch size: $BatchSize)"
Write-Log "API URL: $ApiUrl"

# Process URLs in batches
for ($i = $startIndex; $i -lt $totalUrls; $i += $BatchSize) {
    $batchEnd = [Math]::Min($i + $BatchSize - 1, $totalUrls - 1)
    $batchUrls = $urls[$i..$batchEnd]
    
    Write-Log "Processing batch: URLs $($i+1) to $($batchEnd+1)"
    
    # Prepare batch request
    $requestBody = @{
        urls = $batchUrls
        source_type = "url"
    } | ConvertTo-Json -Depth 3
    
    try {
        # Make API request
        $response = Invoke-RestMethod -Uri "$ApiUrl/ingest" -Method POST -Body $requestBody -ContentType "application/json" -TimeoutSec 300
        
        if ($response.success) {
            $batchSuccessful = $batchUrls.Count
            $successful += $batchSuccessful
            
            foreach ($url in $batchUrls) {
                $processed++
                Write-Log "Successfully processed URL $processed/$totalUrls`: $url"
            }
            
            # Progress update
            $percentComplete = [Math]::Round(($processed / $totalUrls) * 100, 1)
            Write-Log "Progress: $percentComplete% ($processed/$totalUrls URLs processed)"
            
        } else {
            Write-Log "Batch failed: $($response.message)" "ERROR"
            $failed += $batchUrls.Count
            $processed += $batchUrls.Count
        }
        
    } catch {
        Write-Log "API request failed: $($_.Exception.Message)" "ERROR"
        
        # Try individual URLs in case of batch failure
        Write-Log "Attempting individual URL processing for failed batch"
        
        foreach ($url in $batchUrls) {
            try {
                $singleRequestBody = @{
                    urls = @($url)
                    source_type = "url"
                } | ConvertTo-Json -Depth 3
                
                $singleResponse = Invoke-RestMethod -Uri "$ApiUrl/ingest" -Method POST -Body $singleRequestBody -ContentType "application/json" -TimeoutSec 120
                
                if ($singleResponse.success) {
                    $successful++
                    Write-Log "Successfully processed URL $($processed+1)/$totalUrls`: $url"
                } else {
                    $failed++
                    Write-Log "Failed to process URL $($processed+1)/$totalUrls`: $url - $($singleResponse.message)" "ERROR"
                }
                
            } catch {
                $failed++
                Write-Log "Failed to process URL $($processed+1)/$totalUrls`: $url - $($_.Exception.Message)" "ERROR"
            }
            
            $processed++
        }
    }
    
    # Small delay between batches to avoid overwhelming the API
    if ($i + $BatchSize -lt $totalUrls) {
        Start-Sleep -Seconds 2
    }
}

# Final summary
$endTime = Get-Date
$duration = $endTime - $startTime
$durationStr = "{0:hh\:mm\:ss}" -f $duration

Write-Log "=== BULK INGESTION COMPLETE ==="
Write-Log "Total URLs: $totalUrls"
Write-Log "Successful: $successful"
Write-Log "Failed: $failed"
Write-Log "Duration: $durationStr"
Write-Log "Log file: $LogFile"

if ($failed -gt 0) {
    Write-Log "Some URLs failed to process. Check the log file for details." "WARN"
    exit 1
} else {
    Write-Log "All URLs processed successfully!"
    exit 0
}