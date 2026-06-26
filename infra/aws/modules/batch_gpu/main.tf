locals {
  create              = var.create
  compute_env_name    = "gpu-embed-${var.environment}"
  queue_name          = "gpu-embed-${var.environment}"
  job_definition_name = "gpu-embed-worker-${var.environment}"

  # SPOT_CAPACITY_OPTIMIZED does not require a Spot fleet IAM role; on-demand
  # uses BEST_FIT_PROGRESSIVE to spread across the requested families.
  allocation_strategy = var.spot_enabled ? "SPOT_CAPACITY_OPTIMIZED" : "BEST_FIT_PROGRESSIVE"
}

resource "aws_batch_compute_environment" "this" {
  count                    = local.create ? 1 : 0
  compute_environment_name = local.compute_env_name
  type                     = "MANAGED"
  state                    = "ENABLED"
  tags                     = var.tags

  compute_resources {
    type                = var.spot_enabled ? "SPOT" : "EC2"
    allocation_strategy = local.allocation_strategy
    bid_percentage      = var.spot_enabled ? var.spot_bid_percentage : null

    max_vcpus = var.max_vcpus
    min_vcpus = var.min_vcpus

    instance_type      = var.instance_families
    instance_role      = var.instance_profile_arn
    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids

    tags = var.tags
  }
}

resource "aws_batch_job_queue" "this" {
  count    = local.create ? 1 : 0
  name     = local.queue_name
  state    = "ENABLED"
  priority = 1
  tags     = var.tags

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.this[0].arn
  }
}

resource "aws_batch_job_definition" "this" {
  count                 = local.create ? 1 : 0
  name                  = local.job_definition_name
  type                  = "container"
  platform_capabilities = ["EC2"]
  tags                  = var.tags

  # Spot interruptions are expected; let Batch retry the array element.
  retry_strategy {
    attempts = 2
  }

  container_properties = jsonencode({
    image            = var.container_image
    jobRoleArn       = var.job_role_arn
    executionRoleArn = var.task_execution_role_arn
    # Command is overridden per-run by the submitter (gpu-embed aws-submit) so
    # the run id is injected; this default keeps the definition self-describing.
    command = ["gpu-embed", "aws-run-shard"]
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory_mib) },
      { type = "GPU", value = tostring(var.job_gpus) },
    ]
  })
}
