output "bucket_name" {
  description = "Artifact bucket name (-> GPU_EMBED_AWS_S3_BUCKET)"
  value       = module.storage.bucket_name
}

output "bucket_arn" {
  description = "Artifact bucket ARN"
  value       = module.storage.bucket_arn
}

output "prefix_scope" {
  description = "Environment-scoped prefix root (matches GPU_EMBED_AWS_S3_PREFIX_ROOT/<env>)"
  value       = module.storage.prefix_scope
}

output "batch_job_queue_arn" {
  description = "Batch job queue ARN"
  value       = module.batch_gpu.job_queue_arn
}

output "batch_job_queue_name" {
  description = "Batch job queue name (-> gpu-embed aws-submit --job-queue)"
  value       = module.batch_gpu.job_queue_name
}

output "batch_job_definition_arn" {
  description = "Batch job definition ARN"
  value       = module.batch_gpu.job_definition_arn
}

output "batch_job_definition_name" {
  description = "Batch job definition name (-> gpu-embed aws-submit --job-definition)"
  value       = module.batch_gpu.job_definition_name
}

output "cli_env" {
  description = "Ready-to-export environment variables for the gpu-embed CLI"
  value = {
    GPU_EMBED_AWS_REGION         = var.aws_region
    GPU_EMBED_AWS_ENVIRONMENT    = var.environment
    GPU_EMBED_AWS_S3_BUCKET      = module.storage.bucket_name
    GPU_EMBED_AWS_S3_PREFIX_ROOT = var.s3_prefix_root
    GPU_EMBED_AWS_JOB_QUEUE      = module.batch_gpu.job_queue_name
    GPU_EMBED_AWS_JOB_DEFINITION = module.batch_gpu.job_definition_name
  }
}
