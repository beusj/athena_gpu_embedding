output "bucket_name" {
  description = "Artifact bucket name"
  value       = local.resolved_bucket_name
}

output "bucket_arn" {
  description = "Artifact bucket ARN"
  value       = local.resolved_bucket_arn
}

output "prefix_scope" {
  description = "Environment-scoped prefix root"
  value       = local.resolved_prefix_scope
}

output "batch_job_queue_arn" {
  description = "Batch job queue ARN"
  value       = module.batch_gpu.job_queue_arn
}

output "batch_job_definition_arn" {
  description = "Batch job definition ARN"
  value       = module.batch_gpu.job_definition_arn
}
