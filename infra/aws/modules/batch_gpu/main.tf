locals {
  queue_name          = "gpu-embed-${var.environment}"
  job_definition_name = "gpu-embed-worker-${var.environment}"
}

# Placeholder module: no AWS resources yet.
# Add aws_batch_job_queue, aws_batch_compute_environment, and aws_batch_job_definition
# in a future implementation PR.
