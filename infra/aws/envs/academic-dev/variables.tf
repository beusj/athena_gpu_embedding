variable "aws_region" {
  description = "AWS region for this environment"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "academic-dev"
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default     = {}
}

variable "s3_bucket_name" {
  description = "Name of the embeddings artifact bucket"
  type        = string
}

variable "manage_storage_resources" {
  description = "When true, Terraform manages S3 bucket resources in this environment; when false, bucket is treated as pre-created."
  type        = bool
  default     = true

  validation {
    condition     = var.manage_storage_resources || var.use_precreated_iam_roles
    error_message = "manage_storage_resources=false requires use_precreated_iam_roles=true."
  }
}

variable "precreated_bucket_name" {
  description = "Pre-created artifact bucket name when manage_storage_resources=false (optional, defaults to s3_bucket_name)."
  type        = string
  default     = null
}

variable "precreated_bucket_arn" {
  description = "Pre-created artifact bucket ARN when manage_storage_resources=false (optional, defaults to arn:aws:s3:::<s3_bucket_name>)."
  type        = string
  default     = null
}

variable "precreated_prefix_scope" {
  description = "Pre-created environment prefix scope when manage_storage_resources=false (optional, defaults to <s3_prefix_root>/<environment>)."
  type        = string
  default     = null
}

variable "s3_prefix_root" {
  description = "Top-level S3 prefix for artifacts"
  type        = string
  default     = "gpu-embed"
}

variable "kms_key_arn" {
  description = "Optional existing KMS key ARN; null means module-managed in future"
  type        = string
  default     = null
}

variable "batch_instance_families" {
  description = "Preferred GPU families for Batch compute"
  type        = list(string)
  default     = ["g5", "g6e"]
}

variable "batch_spot_enabled" {
  description = "Whether Spot-first strategy is enabled"
  type        = bool
  default     = true
}

variable "batch_on_demand_base_capacity" {
  description = "On-demand base capacity for stability"
  type        = number
  default     = 0
}

variable "batch_vcpus" {
  description = "vCPUs per Batch job"
  type        = number
  default     = 4
}

variable "batch_memory" {
  description = "Memory (MB) per Batch job"
  type        = number
  default     = 32768
}

variable "batch_ecr_image_uri" {
  description = "ECR image URI for gpu-embedder Docker image"
  type        = string
}

variable "batch_subnet_ids" {
  description = "VPC subnet IDs for Batch EC2 instances"
  type        = list(string)
}

variable "batch_vpc_id" {
  description = "VPC ID for Batch resources (security group, etc.)"
  type        = string
}

variable "use_precreated_iam_roles" {
  description = "When true, skip IAM management in Terraform and use pre-created role ARNs"
  type        = bool
  default     = false
}

variable "precreated_batch_job_role_arn" {
  description = "Pre-created Batch job role ARN (required when use_precreated_iam_roles=true)"
  type        = string
  default     = null
}

variable "precreated_batch_task_execution_role_arn" {
  description = "Pre-created Batch task execution role ARN (required when use_precreated_iam_roles=true)"
  type        = string
  default     = null
}

variable "precreated_batch_service_role_arn" {
  description = "Pre-created AWS Batch service role ARN (required when use_precreated_iam_roles=true)"
  type        = string
  default     = null
}

variable "precreated_batch_instance_profile_arn" {
  description = "Pre-created EC2 instance profile ARN (required when use_precreated_iam_roles=true)"
  type        = string
  default     = null
}

variable "precreated_batch_spot_fleet_role_arn" {
  description = "Pre-created EC2 Spot Fleet role ARN (optional when spot enabled)"
  type        = string
  default     = null
}
