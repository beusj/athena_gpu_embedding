locals {
  bucket_arn    = "arn:aws:s3:::${var.bucket_name}"
  prefix_scope  = var.prefix_root
  resolved_kms  = var.kms_key_arn
}

# S3 bucket for artifact storage (embeddings, logs, manifests)
resource "aws_s3_bucket" "artifacts" {
  bucket = var.bucket_name
  tags   = merge(var.tags, { Name = "gpu-embed-artifacts-${var.environment}" })
}

# Enable versioning for immutability and audit trail
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Block all public access (security baseline)
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption with S3-managed keys (SSE-S3)
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle policy: expire old versions after retention period
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id = "expire-noncurrent-versions"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = var.log_retention_days
    }

    status = "Enabled"
  }
}
