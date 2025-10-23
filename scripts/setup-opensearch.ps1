# OpenSearch Serverless Collection Setup Script
# This script creates an OpenSearch Serverless collection with vector index configuration
# for the SOP Q&A Tool

param(
    [Parameter(Mandatory=$false)]
    [string]$CollectionName = "sop-qa-collection",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1",
    
    [Parameter(Mandatory=$false)]
    [string]$Profile = "default"
)

# Set AWS CLI profile and region
$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

Write-Host "Setting up OpenSearch Serverless collection: $CollectionName" -ForegroundColor Green
Write-Host "Region: $Region" -ForegroundColor Yellow
Write-Host "Profile: $Profile" -ForegroundColor Yellow

# Check if AWS CLI is installed
try {
    aws --version | Out-Null
} catch {
    Write-Error "AWS CLI is not installed or not in PATH. Please install AWS CLI first."
    exit 1
}

# Check if user is authenticated
try {
    aws sts get-caller-identity --profile $Profile | Out-Null
    Write-Host "✓ AWS authentication verified" -ForegroundColor Green
} catch {
    Write-Error "AWS authentication failed. Please configure your credentials."
    exit 1
}

# Create security policy for the collection
Write-Host "Creating security policy..." -ForegroundColor Cyan

$securityPolicyName = "$CollectionName-security-policy"
$securityPolicy = @{
    "Rules" = @(
        @{
            "ResourceType" = "collection"
            "Resource" = @("collection/$CollectionName")
        }
    )
    "AWSOSSLPolicy" = @{
        "TLSSecurityPolicy" = "Policy-Min-TLS-1-2-2019-07"
    }
} | ConvertTo-Json -Depth 10

try {
    aws opensearchserverless create-security-policy `
        --name $securityPolicyName `
        --type "encryption" `
        --policy $securityPolicy `
        --profile $Profile `
        --region $Region
    Write-Host "✓ Security policy created: $securityPolicyName" -ForegroundColor Green
} catch {
    Write-Warning "Security policy creation failed or already exists"
}

# Create network policy for the collection
Write-Host "Creating network policy..." -ForegroundColor Cyan

$networkPolicyName = "$CollectionName-network-policy"
$networkPolicy = @(
    @{
        "Rules" = @(
            @{
                "ResourceType" = "collection"
                "Resource" = @("collection/$CollectionName")
            },
            @{
                "ResourceType" = "dashboard"
                "Resource" = @("collection/$CollectionName")
            }
        )
        "AllowFromPublic" = $true
    }
) | ConvertTo-Json -Depth 10

try {
    aws opensearchserverless create-security-policy `
        --name $networkPolicyName `
        --type "network" `
        --policy $networkPolicy `
        --profile $Profile `
        --region $Region
    Write-Host "✓ Network policy created: $networkPolicyName" -ForegroundColor Green
} catch {
    Write-Warning "Network policy creation failed or already exists"
}

# Get current user ARN for data access policy
$callerIdentity = aws sts get-caller-identity --profile $Profile --region $Region | ConvertFrom-Json
$userArn = $callerIdentity.Arn

# Create data access policy
Write-Host "Creating data access policy..." -ForegroundColor Cyan

$dataAccessPolicyName = "$CollectionName-data-policy"
$dataAccessPolicy = @(
    @{
        "Rules" = @(
            @{
                "ResourceType" = "collection"
                "Resource" = @("collection/$CollectionName")
                "Permission" = @(
                    "aoss:CreateCollectionItems",
                    "aoss:DeleteCollectionItems", 
                    "aoss:UpdateCollectionItems",
                    "aoss:DescribeCollectionItems"
                )
            },
            @{
                "ResourceType" = "index"
                "Resource" = @("index/$CollectionName/*")
                "Permission" = @(
                    "aoss:CreateIndex",
                    "aoss:DeleteIndex",
                    "aoss:UpdateIndex",
                    "aoss:DescribeIndex",
                    "aoss:ReadDocument",
                    "aoss:WriteDocument"
                )
            }
        )
        "Principal" = @($userArn)
    }
) | ConvertTo-Json -Depth 10

try {
    aws opensearchserverless create-access-policy `
        --name $dataAccessPolicyName `
        --type "data" `
        --policy $dataAccessPolicy `
        --profile $Profile `
        --region $Region
    Write-Host "✓ Data access policy created: $dataAccessPolicyName" -ForegroundColor Green
} catch {
    Write-Warning "Data access policy creation failed or already exists"
}

# Create the collection
Write-Host "Creating OpenSearch Serverless collection..." -ForegroundColor Cyan

try {
    $collection = aws opensearchserverless create-collection `
        --name $CollectionName `
        --type "VECTORSEARCH" `
        --description "Vector search collection for SOP Q&A Tool" `
        --profile $Profile `
        --region $Region | ConvertFrom-Json
    
    Write-Host "✓ Collection creation initiated: $CollectionName" -ForegroundColor Green
    Write-Host "Collection ID: $($collection.createCollectionDetail.id)" -ForegroundColor Yellow
} catch {
    Write-Error "Failed to create collection. It may already exist."
    exit 1
}

# Wait for collection to be active
Write-Host "Waiting for collection to become active..." -ForegroundColor Cyan
$maxWaitTime = 300 # 5 minutes
$waitTime = 0
$sleepInterval = 10

do {
    Start-Sleep -Seconds $sleepInterval
    $waitTime += $sleepInterval
    
    try {
        $collectionStatus = aws opensearchserverless batch-get-collection `
            --names $CollectionName `
            --profile $Profile `
            --region $Region | ConvertFrom-Json
        
        $status = $collectionStatus.collectionDetails[0].status
        Write-Host "Collection status: $status" -ForegroundColor Yellow
        
        if ($status -eq "ACTIVE") {
            $endpoint = $collectionStatus.collectionDetails[0].collectionEndpoint
            Write-Host "✓ Collection is active!" -ForegroundColor Green
            Write-Host "Collection endpoint: $endpoint" -ForegroundColor Green
            break
        }
    } catch {
        Write-Warning "Error checking collection status"
    }
    
    if ($waitTime -ge $maxWaitTime) {
        Write-Error "Timeout waiting for collection to become active"
        exit 1
    }
} while ($true)

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Collection Name: $CollectionName" -ForegroundColor White
Write-Host "Collection Endpoint: $endpoint" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Update your .env file with the collection endpoint" -ForegroundColor White
Write-Host "2. Create vector indices using the Python application" -ForegroundColor White
Write-Host "3. Test the connection with your application" -ForegroundColor White