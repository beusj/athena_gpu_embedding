output "bucket_name" {
  description = "Artifact bucket name"
  value       = local.bucket_id
}

output "bucket_arn" {
  description = "Artifact bucket ARN"
  value       = local.bucket_arn
}

output "prefix_scope" {
  description = "Environment-scoped prefix root (e.g. gpu-embed/academic-dev)"
  value       = local.prefix_scope
}

output "input_prefix" {
  description = "S3 prefix for input shards"
  value       = local.input_prefix
}

output "output_prefix" {
  description = "S3 prefix for output embeddings"
  value       = local.output_prefix
}

output "kms_key_arn" {
  description = "KMS key ARN used for bucket encryption (null = SSE-S3/AES256)"
  value       = local.resolved_kms
}
