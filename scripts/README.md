# Operational Scripts

This directory contains PowerShell scripts for system administration and maintenance of the SOP QA Tool.

## Scripts Overview

### 1. bulk-ingest-urls.ps1
Bulk URL ingestion script for processing multiple documents at once.

**Features:**
- Batch processing with configurable batch size
- Progress tracking and resumable operations
- Comprehensive error handling and logging
- Support for comment lines in URL files

**Usage:**
```powershell
# Basic usage
.\bulk-ingest-urls.ps1 -UrlFile "urls.txt"

# With custom settings
.\bulk-ingest-urls.ps1 -UrlFile "urls.txt" -ApiUrl "http://localhost:8000" -BatchSize 3 -LogFile "ingest.log"

# Resume from previous run
.\bulk-ingest-urls.ps1 -UrlFile "urls.txt" -Resume
```

**URL File Format:**
```
# Comments start with #
https://example.com/sop1.pdf
https://example.com/sop2.pdf
https://example.com/sop3.docx
```

### 2. rebuild-index.ps1
Index rebuild script for refreshing the vector index from existing documents.

**Features:**
- Health checks before and after rebuild
- Optional index backup creation
- Progress monitoring with timeout handling
- Support for both AWS and local modes

**Usage:**
```powershell
# Basic rebuild
.\rebuild-index.ps1

# With backup and custom settings
.\rebuild-index.ps1 -BackupIndex -Mode aws -LogFile "rebuild.log"

# Force rebuild without confirmation
.\rebuild-index.ps1 -Force
```

### 3. health-monitor.ps1
System health monitoring with component status checks and alerting.

**Features:**
- Comprehensive system metrics collection
- API health and performance monitoring
- Index health and staleness detection
- Multiple output formats (console, JSON, CSV)
- Continuous monitoring mode
- Configurable alert thresholds

**Usage:**
```powershell
# Single health check
.\health-monitor.ps1

# Continuous monitoring
.\health-monitor.ps1 -Continuous -IntervalSeconds 30

# JSON output for integration
.\health-monitor.ps1 -OutputFormat json

# Custom alert thresholds
.\health-monitor.ps1 -AlertThresholds "alert-thresholds.json"
```

## Configuration Files

### alert-thresholds.json
Custom alert thresholds for health monitoring:

```json
{
  "response_time_ms": 5000,
  "memory_usage_percent": 80,
  "disk_usage_percent": 85,
  "error_rate_percent": 5,
  "index_age_hours": 24
}
```

### sample-urls.txt
Example URL file format for bulk ingestion:

```
# Sample URLs for bulk ingestion
https://example.com/sop1.pdf
https://example.com/sop2.pdf
```

## Prerequisites

### PowerShell Requirements
- PowerShell 5.1 or PowerShell Core 7.0+
- Execution policy allowing script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### System Requirements
- Windows 10/11 or Windows Server 2016+
- .NET Framework 4.7.2+ (for PowerShell 5.1)
- Network access to SOP QA Tool API

### API Requirements
- SOP QA Tool API running and accessible
- Appropriate endpoints available:
  - `/health` - System health check
  - `/ingest` - Document ingestion
  - `/reindex` - Index rebuild
  - `/sources` - Document management

## Logging

All scripts support structured logging with the following features:

- **Timestamped entries** with log levels (INFO, WARN, ERROR)
- **Progress tracking** for long-running operations
- **Error details** with context information
- **Performance metrics** for operation timing
- **Resumable operations** using log file state

### Log File Locations
- Default: `{script-name}-{timestamp}.log`
- Custom: Specify with `-LogFile` parameter
- Structured format for easy parsing and analysis

## Error Handling

Scripts implement comprehensive error handling:

1. **Input Validation**
   - File existence checks
   - URL format validation
   - Parameter range validation

2. **Network Resilience**
   - Retry logic with exponential backoff
   - Timeout handling
   - Graceful degradation

3. **Resource Management**
   - Memory usage monitoring
   - Disk space checks
   - Process cleanup

4. **Recovery Mechanisms**
   - Resumable operations
   - Partial failure handling
   - State preservation

## Integration Examples

### Scheduled Tasks
Create Windows scheduled tasks for automated operations:

```powershell
# Daily health check
$action = New-ScheduledTaskAction -Execute "pwsh" -Argument "-File C:\path\to\health-monitor.ps1 -LogFile C:\logs\daily-health.log"
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"
Register-ScheduledTask -TaskName "SOP-QA-HealthCheck" -Action $action -Trigger $trigger
```

### Monitoring Integration
Use JSON output for integration with monitoring systems:

```powershell
# Export health data for monitoring
.\health-monitor.ps1 -OutputFormat json | Out-File -FilePath "health-status.json"
```

### Batch Processing
Combine scripts for complex workflows:

```powershell
# Bulk ingest followed by health check
.\bulk-ingest-urls.ps1 -UrlFile "new-sops.txt" -LogFile "ingest.log"
if ($LASTEXITCODE -eq 0) {
    .\health-monitor.ps1 -LogFile "post-ingest-health.log"
}
```

## Troubleshooting

### Common Issues

1. **PowerShell Execution Policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **API Connection Issues**
   - Verify API URL and port
   - Check firewall settings
   - Confirm API service is running

3. **File Permission Issues**
   - Ensure write access to log directories
   - Check file locks on URL input files
   - Verify backup directory permissions

4. **Memory/Performance Issues**
   - Reduce batch sizes for bulk operations
   - Increase timeout values for large datasets
   - Monitor system resources during operations

### Debug Mode
Enable verbose logging by setting PowerShell preference:

```powershell
$VerbosePreference = "Continue"
.\script-name.ps1 -Verbose
```

### Log Analysis
Use PowerShell to analyze log files:

```powershell
# Find errors in log files
Get-Content "script.log" | Where-Object { $_ -match "\[ERROR\]" }

# Count successful operations
(Get-Content "bulk-ingest.log" | Where-Object { $_ -match "Successfully processed" }).Count
```

## Security Considerations

1. **URL Validation**
   - Scripts validate URLs to prevent SSRF attacks
   - Block localhost and file:// schemes by default
   - Support allowlist/blocklist configuration

2. **File Handling**
   - Enforce file type restrictions
   - Implement size limits
   - Validate file content before processing

3. **Logging Security**
   - Avoid logging sensitive information
   - Implement log rotation to prevent disk exhaustion
   - Secure log file permissions

4. **Network Security**
   - Use HTTPS for API communications
   - Implement timeout and retry limits
   - Validate SSL certificates

## Performance Optimization

1. **Batch Processing**
   - Optimize batch sizes based on system resources
   - Implement parallel processing where appropriate
   - Use streaming for large datasets

2. **Resource Management**
   - Monitor memory usage during operations
   - Implement cleanup routines
   - Use efficient data structures

3. **Network Optimization**
   - Implement connection pooling
   - Use compression for large payloads
   - Optimize retry strategies

## Support

For issues with operational scripts:

1. Check log files for detailed error information
2. Verify system requirements and prerequisites
3. Test with sample data and configurations
4. Review API endpoint availability and responses
5. Consult troubleshooting section for common issues