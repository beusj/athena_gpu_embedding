output "job_queue_arn" {
  description = "Batch job queue ARN"
  value       = aws_batch_job_queue.gpu_embed.arn
}

output "job_definition_arn" {
  description = "Batch job definition ARN"
  value       = aws_batch_job_definition.gpu_embed_worker.arn
}

output "compute_environment_arn" {
  description = "Batch compute environment ARN"
  value       = aws_batch_compute_environment.gpu_embed.arn
}
