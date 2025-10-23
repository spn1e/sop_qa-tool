# AWS Cost Optimization Guide

This guide provides strategies to minimize AWS costs while using the SOP QA Tool in AWS mode.

## Cost Overview

The SOP QA Tool uses several AWS services that incur costs:

### Primary Cost Drivers

1. **Amazon Bedrock** - LLM inference costs
2. **Amazon Titan Embeddings** - Text embedding generation
3. **OpenSearch Serverless** - Vector storage and search
4. **Amazon S3** - Document and chunk storage
5. **AWS Textract** - OCR processing (when needed)
6. **Data Transfer** - Between services and regions

### Typical Monthly Costs (Estimates)

| Usage Level | Documents | Queries/Month | Estimated Cost |
|-------------|-----------|---------------|----------------|
| Light | 100 docs | 500 queries | $15-30 |
| Medium | 500 docs | 2,000 queries | $50-100 |
| Heavy | 2,000 docs | 10,000 queries | $200-400 |

*Costs vary based on document size, query complexity, and AWS region*

## Service-Specific Optimization

### Amazon Bedrock Optimization

#### Model Selection
```bash
# Cost-effective models (in order of cost)
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0     # Lowest cost
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0    # Balanced
BEDROCK_MODEL_ID=anthropic.claude-3-opus-20240229-v1:0      # Highest quality/cost
```

#### Prompt Optimization
```python
# Reduce token usage in prompts
SYSTEM_PROMPT_MAX_TOKENS=500  # Default: 1000
CONTEXT_MAX_TOKENS=3000       # Default: 4000
RESPONSE_MAX_TOKENS=1000      # Default: 2000
```

#### Request Batching
```bash
# Process multiple queries in single request when possible
ENABLE_BATCH_PROCESSING=true
BATCH_SIZE=5
BATCH_TIMEOUT_SECONDS=30
```

#### Caching Strategy
```bash
# Cache LLM responses to avoid repeated calls
ENABLE_LLM_RESPONSE_CACHE=true
CACHE_TTL_HOURS=24
CACHE_MAX_SIZE_MB=100
```

### Amazon Titan Embeddings Optimization

#### Batch Processing
```bash
# Optimize embedding generation
EMBEDDING_BATCH_SIZE=25       # Max batch size for Titan
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL_DAYS=30
```

#### Dimension Optimization
```bash
# Use appropriate embedding dimensions
TITAN_EMBEDDING_DIMENSION=512  # Default: 1024 (costs 2x more)
# Note: Lower dimensions may affect search quality
```

#### Preprocessing
```bash
# Reduce text before embedding
MAX_TEXT_LENGTH_FOR_EMBEDDING=8000  # Titan limit
ENABLE_TEXT_PREPROCESSING=true
REMOVE_STOPWORDS=true
```

### OpenSearch Serverless Optimization

#### Collection Configuration
```bash
# Optimize collection settings
OPENSEARCH_COLLECTION_TYPE=search    # vs 'timeseries'
OPENSEARCH_STANDBY_REPLICAS=false    # Disable for dev/test
```

#### Index Management
```bash
# Optimize indexing
VECTOR_INDEX_REFRESH_INTERVAL=30s    # Default: 1s
ENABLE_INDEX_COMPRESSION=true
DELETE_OLD_INDEXES=true
INDEX_RETENTION_DAYS=90
```

#### Query Optimization
```bash
# Reduce search costs
DEFAULT_SEARCH_SIZE=5        # Default: 10
MAX_SEARCH_SIZE=20          # Limit large queries
ENABLE_RESULT_CACHING=true
SEARCH_CACHE_TTL_MINUTES=15
```

#### Data Lifecycle
```bash
# Manage data retention
ENABLE_DATA_LIFECYCLE=true
ARCHIVE_AFTER_DAYS=180
DELETE_AFTER_DAYS=365
```

### Amazon S3 Optimization

#### Storage Classes
```bash
# Use appropriate storage classes
S3_STORAGE_CLASS=STANDARD_IA         # For infrequently accessed data
S3_ARCHIVE_STORAGE_CLASS=GLACIER     # For long-term archive
```

#### Lifecycle Policies
```json
{
  "Rules": [
    {
      "Id": "SOPDocumentLifecycle",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ]
    }
  ]
}
```

#### Compression
```bash
# Enable compression for stored data
ENABLE_S3_COMPRESSION=true
COMPRESSION_FORMAT=gzip
COMPRESS_CHUNKS=true
COMPRESS_EMBEDDINGS=true
```

#### Multipart Upload Optimization
```bash
# Optimize large file uploads
S3_MULTIPART_THRESHOLD_MB=100
S3_MULTIPART_CHUNKSIZE_MB=10
S3_MAX_CONCURRENCY=4
```

### AWS Textract Optimization

#### Selective OCR
```bash
# Only use OCR when necessary
ENABLE_OCR_DETECTION=true           # Detect if OCR is needed
OCR_CONFIDENCE_THRESHOLD=85         # Skip low-confidence results
MAX_OCR_PAGES_PER_DOCUMENT=50       # Limit OCR processing
```

#### Document Preprocessing
```bash
# Optimize documents before OCR
ENABLE_IMAGE_PREPROCESSING=true
IMAGE_DPI_OPTIMIZATION=true
REMOVE_BLANK_PAGES=true
```

## Cost Monitoring and Alerts

### CloudWatch Cost Monitoring

#### Set Up Billing Alerts
```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "SOP-QA-Tool-Monthly-Cost" \
    --alarm-description "Alert when monthly costs exceed threshold" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --threshold 100 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=Currency,Value=USD \
    --evaluation-periods 1
```

#### Service-Specific Monitoring
```python
# Monitor costs by service
ENABLE_COST_TRACKING=true
COST_TRACKING_SERVICES=bedrock,opensearch,s3,textract
DAILY_COST_REPORT=true
COST_ALERT_THRESHOLD_USD=50
```

### Usage Analytics
```bash
# Track usage patterns
ENABLE_USAGE_ANALYTICS=true
TRACK_QUERY_PATTERNS=true
TRACK_DOCUMENT_PROCESSING=true
GENERATE_USAGE_REPORTS=true
```

## Development vs Production Optimization

### Development Environment
```bash
# Cost-optimized development settings
MODE=local                          # Use local mode for development
ENABLE_AWS_ONLY_FOR_TESTING=true   # Minimal AWS usage
USE_SMALLER_MODELS=true
LIMIT_DOCUMENT_SIZE_MB=10
MAX_DOCUMENTS_PER_SESSION=50
```

### Production Environment
```bash
# Production optimization
ENABLE_PRODUCTION_OPTIMIZATIONS=true
USE_RESERVED_CAPACITY=true          # For predictable workloads
ENABLE_AUTO_SCALING=true
SCALE_DOWN_DURING_OFF_HOURS=true
```

## Regional Cost Optimization

### Region Selection
```bash
# Choose cost-effective regions
AWS_REGION=us-east-1    # Often lowest cost
# Alternative regions:
# us-west-2, eu-west-1, ap-southeast-1
```

### Data Transfer Optimization
```bash
# Minimize cross-region transfers
KEEP_DATA_IN_SAME_REGION=true
USE_VPC_ENDPOINTS=true
ENABLE_CLOUDFRONT_CACHING=false  # Only if needed for global access
```

## Automated Cost Optimization

### Scheduled Operations
```powershell
# PowerShell script for cost optimization
# Schedule to run daily

# Scale down during off-hours
$currentHour = (Get-Date).Hour
if ($currentHour -lt 8 -or $currentHour -gt 18) {
    # Reduce OpenSearch capacity
    aws opensearchserverless update-collection --id $collectionId --type search
    
    # Pause non-critical processing
    $env:ENABLE_BACKGROUND_PROCESSING = "false"
}

# Weekly cleanup
if ((Get-Date).DayOfWeek -eq "Sunday") {
    # Clean up old data
    .\scripts\cleanup-old-data.ps1
    
    # Optimize indexes
    .\scripts\optimize-indexes.ps1
}
```

### Auto-scaling Configuration
```bash
# Configure auto-scaling for OpenSearch
OPENSEARCH_AUTO_SCALING_ENABLED=true
OPENSEARCH_MIN_CAPACITY=1
OPENSEARCH_MAX_CAPACITY=10
OPENSEARCH_TARGET_UTILIZATION=70

# S3 intelligent tiering
S3_INTELLIGENT_TIERING=true
S3_AUTO_DELETE_INCOMPLETE_UPLOADS=true
```

## Cost Optimization Checklist

### Initial Setup
- [ ] Choose cost-effective AWS region
- [ ] Select appropriate Bedrock model (start with Haiku)
- [ ] Configure S3 lifecycle policies
- [ ] Set up billing alerts
- [ ] Enable compression for stored data

### Ongoing Optimization
- [ ] Monitor daily costs via CloudWatch
- [ ] Review and optimize prompts monthly
- [ ] Clean up unused data quarterly
- [ ] Analyze usage patterns for optimization opportunities
- [ ] Update model selection based on cost/performance trade-offs

### Development Practices
- [ ] Use local mode for development
- [ ] Implement caching strategies
- [ ] Batch API requests when possible
- [ ] Optimize document preprocessing
- [ ] Monitor and limit resource usage

## Cost Estimation Tools

### Pre-deployment Estimation
```python
# Cost estimation script
def estimate_monthly_cost(
    num_documents: int,
    avg_document_size_mb: float,
    queries_per_month: int,
    avg_query_complexity: str = "medium"
):
    """
    Estimate monthly AWS costs for SOP QA Tool
    """
    
    # Bedrock costs (per 1K tokens)
    bedrock_costs = {
        "haiku": {"input": 0.00025, "output": 0.00125},
        "sonnet": {"input": 0.003, "output": 0.015},
        "opus": {"input": 0.015, "output": 0.075}
    }
    
    # Titan embedding costs (per 1K tokens)
    titan_cost_per_1k_tokens = 0.0001
    
    # OpenSearch Serverless (per OCU-hour)
    opensearch_cost_per_ocu_hour = 0.24
    
    # S3 costs (per GB-month)
    s3_standard_cost_per_gb = 0.023
    
    # Estimate calculations...
    # (Implementation details)
    
    return {
        "bedrock": bedrock_monthly_cost,
        "titan": titan_monthly_cost,
        "opensearch": opensearch_monthly_cost,
        "s3": s3_monthly_cost,
        "total": total_monthly_cost
    }

# Example usage
costs = estimate_monthly_cost(
    num_documents=500,
    avg_document_size_mb=2.0,
    queries_per_month=2000
)
print(f"Estimated monthly cost: ${costs['total']:.2f}")
```

### Real-time Cost Tracking
```bash
# Add to .env for cost tracking
ENABLE_REAL_TIME_COST_TRACKING=true
COST_TRACKING_INTERVAL_MINUTES=60
COST_ALERT_EMAIL=admin@company.com
DAILY_COST_REPORT_EMAIL=true
```

## Emergency Cost Controls

### Circuit Breakers
```bash
# Implement cost-based circuit breakers
MAX_DAILY_COST_USD=50
MAX_HOURLY_BEDROCK_REQUESTS=1000
MAX_DAILY_TEXTRACT_PAGES=500
ENABLE_COST_CIRCUIT_BREAKER=true
```

### Automatic Shutdowns
```bash
# Auto-shutdown on cost thresholds
COST_EMERGENCY_SHUTDOWN_THRESHOLD=200
ENABLE_EMERGENCY_SHUTDOWN=true
SHUTDOWN_NOTIFICATION_EMAIL=admin@company.com
```

## Best Practices Summary

1. **Start Small**: Begin with local mode, migrate to AWS gradually
2. **Monitor Continuously**: Set up alerts and daily cost reviews
3. **Optimize Regularly**: Review and adjust settings monthly
4. **Use Caching**: Implement multi-level caching strategies
5. **Batch Operations**: Group similar operations together
6. **Choose Right Models**: Balance cost vs. quality requirements
7. **Implement Lifecycle**: Automatically manage data retention
8. **Scale Appropriately**: Use auto-scaling and scheduled operations
9. **Test Thoroughly**: Validate optimizations don't impact functionality
10. **Document Changes**: Track optimization changes and their impact

## Cost Optimization Scripts

### Daily Cost Report
```powershell
# daily-cost-report.ps1
$startDate = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$endDate = (Get-Date).ToString("yyyy-MM-dd")

$costs = aws ce get-cost-and-usage `
    --time-period Start=$startDate,End=$endDate `
    --granularity DAILY `
    --metrics BlendedCost `
    --group-by Type=DIMENSION,Key=SERVICE

Write-Host "Daily AWS Costs for SOP QA Tool:"
$costs | ConvertFrom-Json | ForEach-Object {
    $_.ResultsByTime[0].Groups | ForEach-Object {
        $service = $_.Keys[0]
        $cost = [math]::Round([decimal]$_.Metrics.BlendedCost.Amount, 2)
        if ($cost -gt 0) {
            Write-Host "$service`: $cost USD"
        }
    }
}
```

### Weekly Optimization Review
```powershell
# weekly-optimization.ps1
Write-Host "=== Weekly Cost Optimization Review ==="

# Check for unused resources
aws opensearchserverless list-collections --query 'collectionSummaries[?status==`ACTIVE`]'
aws s3api list-buckets --query 'Buckets[?contains(Name, `sop-qa`)]'

# Analyze usage patterns
$logFile = "data/logs/usage.log"
if (Test-Path $logFile) {
    $queries = Get-Content $logFile | Select-String "query_processed" | Measure-Object
    $documents = Get-Content $logFile | Select-String "document_ingested" | Measure-Object
    
    Write-Host "This week: $($queries.Count) queries, $($documents.Count) documents processed"
}

# Suggest optimizations
Write-Host "Optimization suggestions:"
Write-Host "- Review model usage and consider switching to Haiku for simple queries"
Write-Host "- Check for duplicate documents in S3"
Write-Host "- Verify OpenSearch collection is right-sized"
```

By following this cost optimization guide, you can significantly reduce AWS costs while maintaining the functionality and performance of the SOP QA Tool.