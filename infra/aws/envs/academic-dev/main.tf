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
  resolved_bucket_name = var.manage_storage_resources ? module.storage[0].bucket_name : coalesce(var.precreated_bucket_name, var.s3_bucket_name)
  resolved_bucket_arn = var.manage_storage_resources ? module.storage[0].bucket_arn : coalesce(var.precreated_bucket_arn, "arn:aws:s3:::${var.s3_bucket_name}")
  resolved_prefix_scope = var.manage_storage_resources ? module.storage[0].prefix_scope : coalesce(var.precreated_prefix_scope, local.s3_prefix_env)
}

module "storage" {
  source = "../../modules/storage"
  count  = var.manage_storage_resources ? 1 : 0

  environment    = var.environment
  tags           = local.common_tags
  bucket_name    = var.s3_bucket_name
  prefix_root    = local.s3_prefix_env
  kms_key_arn    = var.kms_key_arn
}

module "security" {
  source = "../../modules/security"
  count  = var.use_precreated_iam_roles ? 0 : (var.manage_storage_resources ? 1 : 0)

  environment  = var.environment
  tags         = local.common_tags
  bucket_arn   = module.storage[0].bucket_arn
  prefix_scope = module.storage[0].prefix_scope
  kms_key_arn  = var.kms_key_arn != null ? var.kms_key_arn : module.storage[0].kms_key_arn
}

module "batch_gpu" {
  source = "../../modules/batch_gpu"

  aws_region               = var.aws_region
  environment              = var.environment
  manage_iam_resources     = !var.use_precreated_iam_roles
  batch_service_role_arn   = var.precreated_batch_service_role_arn
  batch_instance_profile_arn = var.precreated_batch_instance_profile_arn
  batch_spot_fleet_role_arn = var.precreated_batch_spot_fleet_role_arn
  tags                     = local.common_tags
  instance_families        = var.batch_instance_families
  spot_enabled             = var.batch_spot_enabled
  on_demand_base_capacity  = var.batch_on_demand_base_capacity
  job_role_arn             = var.use_precreated_iam_roles ? var.precreated_batch_job_role_arn : module.security[0].batch_job_role_arn
  task_execution_role_arn  = var.use_precreated_iam_roles ? var.precreated_batch_task_execution_role_arn : module.security[0].batch_task_execution_role_arn
  ecr_image_uri            = var.batch_ecr_image_uri
  vcpus                    = var.batch_vcpus
  memory                   = var.batch_memory
  subnet_ids               = var.batch_subnet_ids
  vpc_id                   = var.batch_vpc_id
}
