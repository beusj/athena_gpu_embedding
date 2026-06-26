locals {
  bucket_arn    = "arn:aws:s3:::${var.bucket_name}"
  prefix_scope  = var.prefix_root
  resolved_kms  = var.kms_key_arn
}

# Placeholder module: no AWS resources yet.
# Add aws_s3_bucket, public access block, encryption defaults, and lifecycle policy
# in a future implementation PR.
