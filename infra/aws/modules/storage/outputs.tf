output "bucket_name" {
  description = "Artifact bucket name"
  value       = var.bucket_name
}

output "bucket_arn" {
  description = "Artifact bucket ARN (derived placeholder)"
  value       = local.bucket_arn
}

output "prefix_scope" {
  description = "Environment-scoped prefix root"
  value       = local.prefix_scope
}

output "kms_key_arn" {
  description = "KMS key ARN (placeholder passthrough)"
  value       = local.resolved_kms
}
