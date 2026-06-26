output "batch_job_role_arn" {
  description = "Batch job role ARN (for container task role)"
  value       = aws_iam_role.batch_job.arn
}

output "batch_task_execution_role_arn" {
  description = "Batch task execution role ARN (for ECS agent)"
  value       = aws_iam_role.batch_task_execution.arn
}
