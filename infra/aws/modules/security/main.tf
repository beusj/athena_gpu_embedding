locals {
  job_role_name            = "gpu-embed-batch-job-${var.environment}"
  task_execution_role_name = "gpu-embed-batch-task-${var.environment}"
  instance_role_name       = "gpu-embed-batch-instance-${var.environment}"

  create = var.create_roles

  # Resolved outputs: created role ARNs when managing, else the passthrough ARNs.
  job_role_arn            = local.create ? aws_iam_role.job[0].arn : var.existing_job_role_arn
  task_execution_role_arn = local.create ? aws_iam_role.task_execution[0].arn : var.existing_task_execution_role_arn
  instance_profile_arn    = local.create ? aws_iam_instance_profile.instance[0].arn : var.existing_instance_profile_arn
}

data "aws_partition" "current" {}

# ---------------------------------------------------------------------------
# Assume-role trust policies
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# Scoped artifact-access policy (S3 + optional KMS), attached to the job role
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "artifact_access" {
  statement {
    sid       = "ListScopedPrefix"
    actions   = ["s3:ListBucket"]
    resources = [var.bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.prefix_scope}/*"]
    }
  }

  statement {
    sid = "ReadWriteScopedObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${var.bucket_arn}/${var.prefix_scope}/*"]
  }

  dynamic "statement" {
    for_each = var.kms_key_arn == null ? [] : [var.kms_key_arn]
    content {
      sid = "KmsForArtifacts"
      actions = [
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:GenerateDataKey",
      ]
      resources = [statement.value]
    }
  }
}

# ---------------------------------------------------------------------------
# Batch job role (assumed by the running container; carries artifact access)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "job" {
  count                = local.create ? 1 : 0
  name                 = local.job_role_name
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_assume.json
  permissions_boundary = var.permissions_boundary_arn
  tags                 = var.tags
}

resource "aws_iam_role_policy" "job_artifacts" {
  count  = local.create ? 1 : 0
  name   = "artifact-access"
  role   = aws_iam_role.job[0].id
  policy = data.aws_iam_policy_document.artifact_access.json
}

# ---------------------------------------------------------------------------
# ECS task execution role (pulls the image, writes logs)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task_execution" {
  count                = local.create ? 1 : 0
  name                 = local.task_execution_role_name
  assume_role_policy   = data.aws_iam_policy_document.ecs_tasks_assume.json
  permissions_boundary = var.permissions_boundary_arn
  tags                 = var.tags
}

resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  count      = local.create ? 1 : 0
  role       = aws_iam_role.task_execution[0].name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# EC2 instance role + profile for the Batch managed compute environment
# ---------------------------------------------------------------------------

resource "aws_iam_role" "instance" {
  count                = local.create ? 1 : 0
  name                 = local.instance_role_name
  assume_role_policy   = data.aws_iam_policy_document.ec2_assume.json
  permissions_boundary = var.permissions_boundary_arn
  tags                 = var.tags
}

resource "aws_iam_role_policy_attachment" "instance_ecs" {
  count      = local.create ? 1 : 0
  role       = aws_iam_role.instance[0].name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "instance" {
  count = local.create ? 1 : 0
  name  = "${local.instance_role_name}-profile"
  role  = aws_iam_role.instance[0].name
  tags  = var.tags
}
