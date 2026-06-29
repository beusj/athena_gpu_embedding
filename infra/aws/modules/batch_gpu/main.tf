locals {
  queue_name          = "gpu-embed-${var.environment}"
  job_definition_name = "gpu-embed-worker-${var.environment}"
  compute_env_name    = "gpu-embed-compute-${var.environment}"
  resolved_batch_service_role_arn    = var.manage_iam_resources ? aws_iam_role.batch_service[0].arn : var.batch_service_role_arn
  resolved_batch_instance_profile_arn = var.manage_iam_resources ? aws_iam_instance_profile.batch_instance[0].arn : var.batch_instance_profile_arn
}

# ============================================================================
# Batch Compute Environment
# EC2 type with Spot + on-demand mixed strategy for cost optimization
# ============================================================================

resource "aws_batch_compute_environment" "gpu_embed" {
  name            = local.compute_env_name
  type            = "MANAGED"
  state           = "ENABLED"
  service_role    = local.resolved_batch_service_role_arn
  tags            = merge(var.tags, { Name = local.compute_env_name })

  compute_resources {
    type          = "EC2"
    min_vcpus     = 0
    max_vcpus     = 256
    desired_vcpus = 0
    instance_type = var.instance_families
    subnets       = var.subnet_ids
    security_group_ids = [aws_security_group.batch_job.id]
    instance_role = local.resolved_batch_instance_profile_arn
  }
}

# Service role for Batch (allows Batch to launch EC2/ECS resources)
resource "aws_iam_role" "batch_service" {
  count              = var.manage_iam_resources ? 1 : 0
  name               = "gpu-embed-batch-service-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.batch_service_assume[0].json
  tags               = merge(var.tags, { Name = "gpu-embed-batch-service-${var.environment}" })
}

data "aws_iam_policy_document" "batch_service_assume" {
  count = var.manage_iam_resources ? 1 : 0
  statement {
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role_policy_attachment" "batch_service_policy" {
  count      = var.manage_iam_resources ? 1 : 0
  role       = aws_iam_role.batch_service[0].id
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# EC2 instance role (allows EC2 instances to call AWS APIs)
resource "aws_iam_role" "batch_instance" {
  count              = var.manage_iam_resources ? 1 : 0
  name               = "gpu-embed-batch-instance-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.batch_instance_assume[0].json
  tags               = merge(var.tags, { Name = "gpu-embed-batch-instance-${var.environment}" })
}

data "aws_iam_policy_document" "batch_instance_assume" {
  count = var.manage_iam_resources ? 1 : 0
  statement {
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_instance_profile" "batch_instance" {
  count = var.manage_iam_resources ? 1 : 0
  name = "gpu-embed-batch-instance-${var.environment}"
  role = aws_iam_role.batch_instance[0].name
}

# Attach AmazonEC2ContainerServiceforEC2Role to instance role
resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  count      = var.manage_iam_resources ? 1 : 0
  role       = aws_iam_role.batch_instance[0].id
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# EC2 Spot Fleet role (required when using Spot instances)
resource "aws_iam_role" "ec2_spot_fleet" {
  count              = var.manage_iam_resources && var.spot_enabled ? 1 : 0
  name               = "gpu-embed-ec2-spot-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.ec2_spot_fleet_assume[0].json
  tags               = merge(var.tags, { Name = "gpu-embed-ec2-spot-${var.environment}" })
}

data "aws_iam_policy_document" "ec2_spot_fleet_assume" {
  count = var.manage_iam_resources && var.spot_enabled ? 1 : 0
  statement {
    principals {
      type        = "Service"
      identifiers = ["spotfleet.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role_policy_attachment" "ec2_spot_fleet" {
  count      = var.manage_iam_resources && var.spot_enabled ? 1 : 0
  role       = aws_iam_role.ec2_spot_fleet[0].id
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetRole"
}

# Security group for Batch EC2 instances (allow ECS agent comms)
resource "aws_security_group" "batch_job" {
  name        = "gpu-embed-batch-${var.environment}"
  description = "Security group for GPU embedding Batch jobs"
  vpc_id      = var.vpc_id
  tags        = merge(var.tags, { Name = "gpu-embed-batch-${var.environment}" })

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ============================================================================
# Batch Job Queue
# ============================================================================

resource "aws_batch_job_queue" "gpu_embed" {
  name                 = local.queue_name
  state                = "ENABLED"
  priority             = 1
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_embed.arn
  }
  tags = merge(var.tags, { Name = local.queue_name })
}

# ============================================================================
# Batch Job Definition
# Container definition for gpu-embedder CLI
# ============================================================================

resource "aws_batch_job_definition" "gpu_embed_worker" {
  name                  = local.job_definition_name
  type                  = "container"
  container_properties  = jsonencode(local.container_properties)
  retry_strategy {
    attempts = 3
    evaluate_on_exit {
      on_exit_code = "0" # Success
      action       = "EXIT"
    }
    evaluate_on_exit {
      on_reason = "Task failed*"
      action    = "EXIT"
    }
  }
  tags = merge(var.tags, { Name = local.job_definition_name })
}

locals {
  container_properties = merge(
    {
      image      = var.ecr_image_uri
      vcpus      = var.vcpus
      memory     = var.memory
      gpu_count  = 1
      jobRoleArn = var.job_role_arn
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/aws/batch/gpu-embed-${var.environment}"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "gpu-embed-worker"
        }
      }
      environment = [
        {
          name  = "GPU_EMBED_MODE"
          value = "batch"
        }
      ]
      # Command template: override at job submission time
      # command = ["gpu-embed", "embed", "--csv-path", "Ref::CSVPath"]
    },
    var.task_execution_role_arn != null ? { executionRoleArn = var.task_execution_role_arn } : {}
  )
}
