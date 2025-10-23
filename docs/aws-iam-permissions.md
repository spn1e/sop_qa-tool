# AWS IAM Permissions for SOP Q&A Tool

This document outlines the minimum required IAM permissions for the SOP Q&A Tool to operate in AWS mode. The permissions are organized by service and include both setup permissions (for infrastructure creation) and runtime permissions (for application operation).

## Overview

The SOP Q&A Tool requires access to the following AWS services:
- **Amazon Bedrock** - For LLM inference (Claude) and embeddings (Titan)
- **Amazon S3** - For document storage (raw documents and processed chunks)
- **Amazon OpenSearch Serverless** - For vector search and indexing
- **AWS Textract** - For OCR processing of scanned documents
- **AWS STS** - For identity verification and temporary credentials

## Setup Permissions (Infrastructure Creation)

These permissions are required to run the setup scripts and create the necessary AWS infrastructure. They can be removed after initial setup.

### OpenSearch Serverless Setup Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "OpenSearchServerlessSetup",
            "Effect": "Allow",
            "Action": [
                "aoss:CreateCollection",
                "aoss:CreateSecurityPolicy",
                "aoss:CreateAccessPolicy",
                "aoss:BatchGetCollection",
                "aoss:ListCollections",
                "aoss:ListSecurityPolicies",
                "aoss:ListAccessPolicies"
            ],
            "Resource": "*"
        }
    ]
}
```

### S3 Setup Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3BucketSetup",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:PutBucketVersioning",
                "s3:PutBucketEncryption",
                "s3:PutBucketPublicAccessBlock",
                "s3:PutBucketLifecycleConfiguration",
                "s3:PutBucketPolicy",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::sop-qa-raw-docs-*",
                "arn:aws:s3:::sop-qa-processed-chunks-*"
            ]
        }
    ]
}
```

## Runtime Permissions (Application Operation)

These permissions are required for the application to operate normally and should be maintained throughout the application lifecycle.

### Complete Runtime Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockAccess",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
            ]
        },
        {
            "Sid": "S3DocumentStorage",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetObjectVersion",
                "s3:DeleteObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::sop-qa-raw-docs-*",
                "arn:aws:s3:::sop-qa-raw-docs-*/*",
                "arn:aws:s3:::sop-qa-processed-chunks-*",
                "arn:aws:s3:::sop-qa-processed-chunks-*/*"
            ]
        },
        {
            "Sid": "OpenSearchServerlessAccess",
            "Effect": "Allow",
            "Action": [
                "aoss:APIAccessAll"
            ],
            "Resource": [
                "arn:aws:aoss:*:*:collection/sop-qa-collection"
            ]
        },
        {
            "Sid": "TextractOCR",
            "Effect": "Allow",
            "Action": [
                "textract:DetectDocumentText",
                "textract:AnalyzeDocument"
            ],
            "Resource": "*"
        },
        {
            "Sid": "STSIdentityVerification",
            "Effect": "Allow",
            "Action": [
                "sts:GetCallerIdentity"
            ],
            "Resource": "*"
        }
    ]
}
```

## Service-Specific Permission Details

### Amazon Bedrock Permissions

**Required Actions:**
- `bedrock:InvokeModel` - For synchronous model inference
- `bedrock:InvokeModelWithResponseStream` - For streaming responses (optional)

**Required Resources:**
- Claude 3 Sonnet: `arn:aws:bedrock:*::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0`
- Claude 3 Haiku: `arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0` (fallback)
- Titan Embeddings v2: `arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0`

**Cost Considerations:**
- Claude 3 Sonnet: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Titan Embeddings v2: ~$0.0001 per 1K tokens
- Consider using Claude 3 Haiku for cost optimization (~$0.25/$1.25 per 1M tokens)

### Amazon S3 Permissions

**Required Actions:**
- `s3:GetObject` - Read documents and processed chunks
- `s3:PutObject` - Store new documents and chunks
- `s3:DeleteObject` - Remove outdated or invalid content
- `s3:ListBucket` - List bucket contents for management
- `s3:GetObjectVersion`/`s3:DeleteObjectVersion` - Version management

**Bucket Structure:**
```
sop-qa-raw-docs-{suffix}/
├── {doc_id}/
│   ├── original.{ext}
│   ├── metadata.json
│   └── extracted_text.txt

sop-qa-processed-chunks-{suffix}/
├── {doc_id}/
│   ├── chunks.jsonl
│   ├── summary.json
│   └── embeddings.npy
```

### OpenSearch Serverless Permissions

**Required Actions:**
- `aoss:APIAccessAll` - Full API access to the collection

**Data Access Policy (Applied via OpenSearch Console or API):**
```json
[
  {
    "Rules": [
      {
        "ResourceType": "collection",
        "Resource": ["collection/sop-qa-collection"],
        "Permission": [
          "aoss:CreateCollectionItems",
          "aoss:DeleteCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems"
        ]
      },
      {
        "ResourceType": "index",
        "Resource": ["index/sop-qa-collection/*"],
        "Permission": [
          "aoss:CreateIndex",
          "aoss:DeleteIndex",
          "aoss:UpdateIndex",
          "aoss:DescribeIndex",
          "aoss:ReadDocument",
          "aoss:WriteDocument"
        ]
      }
    ],
    "Principal": ["arn:aws:iam::{account-id}:user/{username}"]
  }
]
```

### AWS Textract Permissions

**Required Actions:**
- `textract:DetectDocumentText` - Extract text from images and PDFs
- `textract:AnalyzeDocument` - Advanced document analysis (optional)

**Usage Patterns:**
- Used as fallback when standard text extraction fails
- Processes scanned PDFs and image-based documents
- Cost: ~$1.50 per 1,000 pages for DetectDocumentText

## IAM Role vs User Permissions

### For EC2/ECS Deployment (Recommended)

Create an IAM role with the runtime permissions and attach it to your compute instance:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

### For Local Development

Create an IAM user with programmatic access and attach the runtime policy. Use AWS CLI profiles for credential management:

```bash
aws configure --profile sop-qa-tool
```

## Security Best Practices

### 1. Principle of Least Privilege
- Only grant permissions required for specific operations
- Use resource-specific ARNs where possible
- Regularly audit and remove unused permissions

### 2. Resource Restrictions
```json
{
    "Condition": {
        "StringLike": {
            "s3:prefix": [
                "documents/*",
                "chunks/*"
            ]
        }
    }
}
```

### 3. IP Restrictions (Optional)
```json
{
    "Condition": {
        "IpAddress": {
            "aws:SourceIp": [
                "203.0.113.0/24",
                "198.51.100.0/24"
            ]
        }
    }
}
```

### 4. Time-Based Access (Optional)
```json
{
    "Condition": {
        "DateGreaterThan": {
            "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "DateLessThan": {
            "aws:CurrentTime": "2024-12-31T23:59:59Z"
        }
    }
}
```

## Cost Optimization

### 1. Bedrock Model Selection
- Use Claude 3 Haiku for simple queries (5x cheaper than Sonnet)
- Implement caching for repeated queries
- Set token limits to prevent runaway costs

### 2. S3 Storage Classes
- Lifecycle policies automatically transition to cheaper storage
- Raw documents → Standard-IA (30 days) → Glacier (90 days)
- Processed chunks → Standard-IA (7 days) → Glacier (30 days)

### 3. OpenSearch Serverless
- Use appropriate OCU (OpenSearch Compute Units) sizing
- Monitor search patterns and optimize index configuration
- Consider data retention policies

## Troubleshooting Common Permission Issues

### 1. Bedrock Access Denied
```
Error: AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```
**Solution:** Ensure the Bedrock foundation models are available in your region and you have the correct model ARNs.

### 2. S3 Access Denied
```
Error: AccessDenied: Access Denied
```
**Solution:** Check bucket policies, IAM permissions, and ensure bucket names match your configuration.

### 3. OpenSearch Connection Failed
```
Error: ConnectionError: Connection to OpenSearch failed
```
**Solution:** Verify network policies allow access and data access policies include your principal ARN.

### 4. Textract Limits Exceeded
```
Error: ProvisionedThroughputExceededException
```
**Solution:** Implement exponential backoff retry logic and consider request batching.

## Monitoring and Logging

### CloudTrail Events to Monitor
- `bedrock:InvokeModel` - Track LLM usage and costs
- `s3:GetObject`/`s3:PutObject` - Monitor document access patterns
- `aoss:*` - Track OpenSearch operations
- `textract:*` - Monitor OCR usage

### CloudWatch Metrics
- Bedrock token usage and latency
- S3 request counts and data transfer
- OpenSearch query performance
- Application error rates

### Cost Alerts
Set up billing alerts for:
- Bedrock usage > $50/month
- S3 storage > $20/month  
- OpenSearch compute > $100/month
- Textract processing > $30/month

## Example IAM Policy Creation

### Using AWS CLI
```bash
# Create the policy
aws iam create-policy \
    --policy-name SOPQAToolRuntimePolicy \
    --policy-document file://sop-qa-runtime-policy.json

# Attach to user
aws iam attach-user-policy \
    --user-name sop-qa-user \
    --policy-arn arn:aws:iam::123456789012:policy/SOPQAToolRuntimePolicy

# Attach to role
aws iam attach-role-policy \
    --role-name sop-qa-role \
    --policy-arn arn:aws:iam::123456789012:policy/SOPQAToolRuntimePolicy
```

### Using Terraform
```hcl
resource "aws_iam_policy" "sop_qa_runtime" {
  name        = "SOPQAToolRuntimePolicy"
  description = "Runtime permissions for SOP Q&A Tool"
  policy      = file("${path.module}/sop-qa-runtime-policy.json")
}

resource "aws_iam_role_policy_attachment" "sop_qa_runtime" {
  role       = aws_iam_role.sop_qa_role.name
  policy_arn = aws_iam_policy.sop_qa_runtime.arn
}
```

This permissions documentation provides the foundation for secure and cost-effective operation of the SOP Q&A Tool in AWS environments.