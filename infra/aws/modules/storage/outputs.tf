output "bucket_name" {
  description = "Artifact bucket name"
  value       = aws_s3_bucket.artifacts.id
}

output "bucket_arn" {
  description = "Artifact bucket ARN"
  value       = aws_s3_bucket.artifacts.arn
}

output "prefix_scope" {
  description = "Environment-scoped prefix root"
  value       = local.prefix_scope
}

output "kms_key_arn" {
  description = "KMS key ARN (or null if SSE-S3)"
  value       = local.resolved_kms
}
