provider "aws" {
  region = var.aws_region
}

locals {
  common_tags = merge(
    {
      Project     = "gpu-embedder"
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )

  s3_prefix_env = "${var.s3_prefix_root}/${var.environment}"
}

module "storage" {
  source = "../../modules/storage"

  environment    = var.environment
  tags           = local.common_tags
  bucket_name    = var.s3_bucket_name
  prefix_root    = local.s3_prefix_env
  kms_key_arn    = var.kms_key_arn
}

module "security" {
  source = "../../modules/security"

  environment  = var.environment
  tags         = local.common_tags
  bucket_arn   = module.storage.bucket_arn
  prefix_scope = module.storage.prefix_scope
  kms_key_arn  = coalesce(var.kms_key_arn, module.storage.kms_key_arn)
}

module "batch_gpu" {
  source = "../../modules/batch_gpu"

  environment              = var.environment
  tags                     = local.common_tags
  instance_families        = var.batch_instance_families
  spot_enabled             = var.batch_spot_enabled
  on_demand_base_capacity  = var.batch_on_demand_base_capacity
  job_role_arn             = module.security.batch_job_role_arn
  task_execution_role_arn  = module.security.batch_task_execution_role_arn
}
