# S3 Buckets Setup Script
# This script creates S3 buckets with appropriate policies and lifecycle rules
# for the SOP Q&A Tool

param(
    [Parameter(Mandatory=$false)]
    [string]$RawDocsBucket = "sop-qa-raw-docs",
    
    [Parameter(Mandatory=$false)]
    [string]$ChunksBucket = "sop-qa-processed-chunks",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1",
    
    [Parameter(Mandatory=$false)]
    [string]$Profile = "default",
    
    [Parameter(Mandatory=$false)]
    [string]$BucketSuffix = ""
)

# Set AWS CLI profile and region
$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

# Add random suffix if not provided to ensure unique bucket names
if ([string]::IsNullOrEmpty($BucketSuffix)) {
    $BucketSuffix = Get-Random -Minimum 1000 -Maximum 9999
}

$RawDocsBucket = "$RawDocsBucket-$BucketSuffix"
$ChunksBucket = "$ChunksBucket-$BucketSuffix"

Write-Host "Setting up S3 buckets for SOP Q&A Tool" -ForegroundColor Green
Write-Host "Raw Documents Bucket: $RawDocsBucket" -ForegroundColor Yellow
Write-Host "Processed Chunks Bucket: $ChunksBucket" -ForegroundColor Yellow
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

# Function to create bucket with error handling
function New-S3BucketWithPolicy {
    param(
        [string]$BucketName,
        [string]$Purpose,
        [hashtable]$LifecycleRules = @{}
    )
    
    Write-Host "Creating $Purpose bucket: $BucketName..." -ForegroundColor Cyan
    
    try {
        # Create bucket
        if ($Region -eq "us-east-1") {
            aws s3api create-bucket --bucket $BucketName --profile $Profile --region $Region
        } else {
            aws s3api create-bucket --bucket $BucketName --profile $Profile --region $Region --create-bucket-configuration LocationConstraint=$Region
        }
        Write-Host "✓ Bucket created: $BucketName" -ForegroundColor Green
    } catch {
        Write-Warning "Bucket creation failed or bucket already exists: $BucketName"
    }
    
    # Enable versioning
    Write-Host "Enabling versioning for $BucketName..." -ForegroundColor Cyan
    try {
        aws s3api put-bucket-versioning --bucket $BucketName --versioning-configuration Status=Enabled --profile $Profile --region $Region
        Write-Host "✓ Versioning enabled for $BucketName" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to enable versioning for $BucketName"
    }
    
    # Enable server-side encryption
    Write-Host "Enabling server-side encryption for $BucketName..." -ForegroundColor Cyan
    $encryptionConfig = @{
        "Rules" = @(
            @{
                "ApplyServerSideEncryptionByDefault" = @{
                    "SSEAlgorithm" = "AES256"
                }
                "BucketKeyEnabled" = $true
            }
        )
    } | ConvertTo-Json -Depth 10 -Compress
    
    try {
        aws s3api put-bucket-encryption --bucket $BucketName --server-side-encryption-configuration $encryptionConfig --profile $Profile --region $Region
        Write-Host "✓ Server-side encryption enabled for $BucketName" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to enable encryption for $BucketName"
    }
    
    # Block public access
    Write-Host "Blocking public access for $BucketName..." -ForegroundColor Cyan
    try {
        aws s3api put-public-access-block --bucket $BucketName --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" --profile $Profile --region $Region
        Write-Host "✓ Public access blocked for $BucketName" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to block public access for $BucketName"
    }
    
    # Apply lifecycle rules if provided
    if ($LifecycleRules.Count -gt 0) {
        Write-Host "Applying lifecycle rules for $BucketName..." -ForegroundColor Cyan
        $lifecycleJson = $LifecycleRules | ConvertTo-Json -Depth 10 -Compress
        
        try {
            aws s3api put-bucket-lifecycle-configuration --bucket $BucketName --lifecycle-configuration $lifecycleJson --profile $Profile --region $Region
            Write-Host "✓ Lifecycle rules applied for $BucketName" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to apply lifecycle rules for $BucketName"
        }
    }
}

# Create Raw Documents Bucket with lifecycle rules
$rawDocsLifecycle = @{
    "Rules" = @(
        @{
            "ID" = "DeleteOldVersions"
            "Status" = "Enabled"
            "Filter" = @{
                "Prefix" = ""
            }
            "NoncurrentVersionExpiration" = @{
                "NoncurrentDays" = 90
            }
        },
        @{
            "ID" = "TransitionToIA"
            "Status" = "Enabled"
            "Filter" = @{
                "Prefix" = ""
            }
            "Transitions" = @(
                @{
                    "Days" = 30
                    "StorageClass" = "STANDARD_IA"
                },
                @{
                    "Days" = 90
                    "StorageClass" = "GLACIER"
                }
            )
        }
    )
}

New-S3BucketWithPolicy -BucketName $RawDocsBucket -Purpose "Raw Documents" -LifecycleRules $rawDocsLifecycle

# Create Processed Chunks Bucket with lifecycle rules
$chunksLifecycle = @{
    "Rules" = @(
        @{
            "ID" = "DeleteOldVersions"
            "Status" = "Enabled"
            "Filter" = @{
                "Prefix" = ""
            }
            "NoncurrentVersionExpiration" = @{
                "NoncurrentDays" = 30
            }
        },
        @{
            "ID" = "TransitionToIA"
            "Status" = "Enabled"
            "Filter" = @{
                "Prefix" = ""
            }
            "Transitions" = @(
                @{
                    "Days" = 7
                    "StorageClass" = "STANDARD_IA"
                },
                @{
                    "Days" = 30
                    "StorageClass" = "GLACIER"
                }
            )
        }
    )
}

New-S3BucketWithPolicy -BucketName $ChunksBucket -Purpose "Processed Chunks" -LifecycleRules $chunksLifecycle

# Create bucket policy for application access
Write-Host "Creating bucket policies..." -ForegroundColor Cyan

# Get current user ARN
$callerIdentity = aws sts get-caller-identity --profile $Profile --region $Region | ConvertFrom-Json
$userArn = $callerIdentity.Arn
$accountId = $callerIdentity.Account

# Create policy for raw documents bucket
$rawDocsBucketPolicy = @{
    "Version" = "2012-10-17"
    "Statement" = @(
        @{
            "Sid" = "AllowApplicationAccess"
            "Effect" = "Allow"
            "Principal" = @{
                "AWS" = $userArn
            }
            "Action" = @(
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            )
            "Resource" = @(
                "arn:aws:s3:::$RawDocsBucket",
                "arn:aws:s3:::$RawDocsBucket/*"
            )
        }
    )
} | ConvertTo-Json -Depth 10 -Compress

try {
    aws s3api put-bucket-policy --bucket $RawDocsBucket --policy $rawDocsBucketPolicy --profile $Profile --region $Region
    Write-Host "✓ Bucket policy applied for $RawDocsBucket" -ForegroundColor Green
} catch {
    Write-Warning "Failed to apply bucket policy for $RawDocsBucket"
}

# Create policy for chunks bucket
$chunksBucketPolicy = @{
    "Version" = "2012-10-17"
    "Statement" = @(
        @{
            "Sid" = "AllowApplicationAccess"
            "Effect" = "Allow"
            "Principal" = @{
                "AWS" = $userArn
            }
            "Action" = @(
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            )
            "Resource" = @(
                "arn:aws:s3:::$ChunksBucket",
                "arn:aws:s3:::$ChunksBucket/*"
            )
        }
    )
} | ConvertTo-Json -Depth 10 -Compress

try {
    aws s3api put-bucket-policy --bucket $ChunksBucket --policy $chunksBucketPolicy --profile $Profile --region $Region
    Write-Host "✓ Bucket policy applied for $ChunksBucket" -ForegroundColor Green
} catch {
    Write-Warning "Failed to apply bucket policy for $ChunksBucket"
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Raw Documents Bucket: $RawDocsBucket" -ForegroundColor White
Write-Host "Processed Chunks Bucket: $ChunksBucket" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "`nBucket Features Enabled:" -ForegroundColor Cyan
Write-Host "✓ Versioning" -ForegroundColor White
Write-Host "✓ Server-side encryption (AES256)" -ForegroundColor White
Write-Host "✓ Public access blocked" -ForegroundColor White
Write-Host "✓ Lifecycle rules for cost optimization" -ForegroundColor White
Write-Host "✓ Bucket policies for application access" -ForegroundColor White
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Update your .env file with the bucket names:" -ForegroundColor White
Write-Host "   S3_RAW_BUCKET=$RawDocsBucket" -ForegroundColor Gray
Write-Host "   S3_CHUNKS_BUCKET=$ChunksBucket" -ForegroundColor Gray
Write-Host "2. Test bucket access with your application" -ForegroundColor White