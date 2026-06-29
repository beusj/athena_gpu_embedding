locals {
  job_role_name            = "gpu-embed-batch-job-${var.environment}"
  task_execution_role_name = "gpu-embed-batch-task-${var.environment}"
}

# ============================================================================
# Batch Job Execution Role
# Grant job containers S3 read/write + optional KMS decrypt permissions
# ============================================================================

resource "aws_iam_role" "batch_job" {
  name               = local.job_role_name
  assume_role_policy = data.aws_iam_policy_document.batch_job_assume.json
  tags               = merge(var.tags, { Name = local.job_role_name })
}

data "aws_iam_policy_document" "batch_job_assume" {
  statement {
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

# Job role inline policy: S3 artifact access (read/write prefix-scoped)
resource "aws_iam_role_policy" "batch_job_s3" {
  name   = "s3-artifact-access"
  role   = aws_iam_role.batch_job.id
  policy = data.aws_iam_policy_document.batch_job_s3.json
}

data "aws_iam_policy_document" "batch_job_s3" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = ["${var.bucket_arn}/${var.prefix_scope}/*"]
  }

  statement {
    actions = [
      "s3:ListBucket",
      "s3:GetBucketVersioning"
    ]
    resources = [var.bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.prefix_scope}/*", "${var.prefix_scope}"]
    }
  }
}

# Conditional: Job role KMS permissions if key is provided
resource "aws_iam_role_policy" "batch_job_kms" {
  count  = var.kms_key_arn != null ? 1 : 0
  name   = "kms-decrypt"
  role   = aws_iam_role.batch_job.id
  policy = data.aws_iam_policy_document.batch_job_kms[0].json
}

data "aws_iam_policy_document" "batch_job_kms" {
  count = var.kms_key_arn != null ? 1 : 0

  statement {
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:GenerateDataKey"
    ]
    resources = [var.kms_key_arn]
  }
}

# Job role CloudWatch Logs permissions for job logging
resource "aws_iam_role_policy" "batch_job_logs" {
  name   = "cloudwatch-logs"
  role   = aws_iam_role.batch_job.id
  policy = data.aws_iam_policy_document.batch_job_logs.json
}

data "aws_iam_policy_document" "batch_job_logs" {
  statement {
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = ["arn:aws:logs:*:*:log-group:/aws/batch/gpu-embed-*"]
  }
}

# ============================================================================
# Batch Task Execution Role
# ECS task agent permissions (ECR pull, task role pass, etc.)
# ============================================================================

resource "aws_iam_role" "batch_task_execution" {
  name               = local.task_execution_role_name
  assume_role_policy = data.aws_iam_policy_document.batch_task_execution_assume.json
  tags               = merge(var.tags, { Name = local.task_execution_role_name })
}

data "aws_iam_policy_document" "batch_task_execution_assume" {
  statement {
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

# Attach AWS managed policy for ECS task execution (ECR pull, CloudWatch logs)
resource "aws_iam_role_policy_attachment" "batch_task_execution_managed" {
  role       = aws_iam_role.batch_task_execution.id
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task execution role inline policy: allow passing job role to task
resource "aws_iam_role_policy" "batch_task_execution_pass_role" {
  name   = "pass-job-role"
  role   = aws_iam_role.batch_task_execution.id
  policy = data.aws_iam_policy_document.batch_task_execution_pass_role.json
}

data "aws_iam_policy_document" "batch_task_execution_pass_role" {
  statement {
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.batch_job.arn,
      aws_iam_role.batch_task_execution.arn
    ]
  }
}
