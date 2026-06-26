output "batch_job_role_arn" {
  description = "Batch job role ARN (created or passed-through)"
  value       = local.job_role_arn
}

output "batch_task_execution_role_arn" {
  description = "ECS task execution role ARN (created or passed-through)"
  value       = local.task_execution_role_arn
}

output "instance_profile_arn" {
  description = "EC2 instance profile ARN for the Batch compute environment"
  value       = local.instance_profile_arn
}
