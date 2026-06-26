locals {
  job_role_name            = "gpu-embed-batch-job-${var.environment}"
  task_execution_role_name = "gpu-embed-batch-task-${var.environment}"
}

# Placeholder module: no AWS resources yet.
# Add aws_iam_role / aws_iam_policy resources for Batch job and task execution
# in a future implementation PR.
