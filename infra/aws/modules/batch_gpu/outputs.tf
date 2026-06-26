output "job_queue_arn" {
  description = "Batch job queue ARN"
  value       = local.create ? aws_batch_job_queue.this[0].arn : null
}

output "job_queue_name" {
  description = "Batch job queue name (passed to gpu-embed aws-submit --job-queue)"
  value       = local.create ? aws_batch_job_queue.this[0].name : null
}

output "job_definition_arn" {
  description = "Batch job definition ARN"
  value       = local.create ? aws_batch_job_definition.this[0].arn : null
}

output "job_definition_name" {
  description = "Batch job definition name (passed to gpu-embed aws-submit --job-definition)"
  value       = local.create ? aws_batch_job_definition.this[0].name : null
}

output "compute_environment_arn" {
  description = "Batch compute environment ARN"
  value       = local.create ? aws_batch_compute_environment.this[0].arn : null
}
