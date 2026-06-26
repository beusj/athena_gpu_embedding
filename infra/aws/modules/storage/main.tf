locals {
  # When create_bucket = false the bucket is pre-provisioned; we only derive its
  # ARN from the name and apply no managing resources.
  bucket_arn   = var.create_bucket ? aws_s3_bucket.this[0].arn : "arn:aws:s3:::${var.bucket_name}"
  bucket_id    = var.create_bucket ? aws_s3_bucket.this[0].id : var.bucket_name
  prefix_scope = var.prefix_root
  resolved_kms = var.kms_key_arn

  input_prefix  = "${var.prefix_root}/input"
  output_prefix = "${var.prefix_root}/output"
  logs_prefix   = "${var.prefix_root}/logs"

  sse_algorithm = var.kms_key_arn == null ? "AES256" : "aws:kms"
}

resource "aws_s3_bucket" "this" {
  count  = var.create_bucket ? 1 : 0
  bucket = var.bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_public_access_block" "this" {
  count                   = var.create_bucket ? 1 : 0
  bucket                  = aws_s3_bucket.this[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "this" {
  count  = var.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.this[0].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "this" {
  count  = var.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.this[0].id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  count  = var.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.this[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = local.sse_algorithm
      kms_master_key_id = var.kms_key_arn
    }
    bucket_key_enabled = var.kms_key_arn != null
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  count  = var.create_bucket ? 1 : 0
  bucket = aws_s3_bucket.this[0].id

  # Always abort dangling multipart uploads (cost hygiene under Spot churn).
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = var.abort_multipart_days
    }
  }

  dynamic "rule" {
    for_each = var.input_expiration_days > 0 ? [1] : []
    content {
      id     = "expire-inputs"
      status = "Enabled"
      filter {
        prefix = "${local.input_prefix}/"
      }
      expiration {
        days = var.input_expiration_days
      }
    }
  }

  dynamic "rule" {
    for_each = var.output_expiration_days > 0 ? [1] : []
    content {
      id     = "expire-outputs"
      status = "Enabled"
      filter {
        prefix = "${local.output_prefix}/"
      }
      expiration {
        days = var.output_expiration_days
      }
    }
  }

  dynamic "rule" {
    for_each = var.log_retention_days > 0 ? [1] : []
    content {
      id     = "expire-logs"
      status = "Enabled"
      filter {
        prefix = "${local.logs_prefix}/"
      }
      expiration {
        days = var.log_retention_days
      }
    }
  }
}
