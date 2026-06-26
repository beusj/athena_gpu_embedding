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

variable "s3_prefix_root" {
  description = "Top-level S3 prefix for artifacts"
  type        = string
  default     = "gpu-embed"
}

variable "kms_key_arn" {
  description = "Optional existing KMS key ARN; null falls back to SSE-S3 (AES256)"
  type        = string
  default     = null
}

# --- create-vs-reuse toggles (locked-down academic accounts) -----------------

variable "create_bucket" {
  description = "Let Terraform create/manage the S3 bucket (false = pre-provisioned)"
  type        = bool
  default     = true
}

variable "create_roles" {
  description = "Let Terraform create the IAM roles (false = supply existing_* ARNs)"
  type        = bool
  default     = true
}

variable "create_batch" {
  description = "Create the Batch compute env/queue/job-def (false until quotas confirmed)"
  type        = bool
  default     = true
}

variable "permissions_boundary_arn" {
  description = "Optional IAM permissions boundary required by the account for new roles"
  type        = string
  default     = null
}

variable "existing_job_role_arn" {
  description = "Pre-created Batch job role ARN (when create_roles = false)"
  type        = string
  default     = null
}

variable "existing_task_execution_role_arn" {
  description = "Pre-created ECS task execution role ARN (when create_roles = false)"
  type        = string
  default     = null
}

variable "existing_instance_profile_arn" {
  description = "Pre-created EC2 instance profile ARN (when create_roles = false)"
  type        = string
  default     = null
}

# --- networking (bring-your-own VPC) -----------------------------------------

variable "subnet_ids" {
  description = "Existing subnet IDs for Batch compute instances"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Existing security group IDs for Batch compute instances"
  type        = list(string)
  default     = []
}

# --- compute / image ---------------------------------------------------------

variable "container_image" {
  description = "ECR image URI for the worker"
  type        = string
  default     = ""
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

variable "batch_max_vcpus" {
  description = "Maximum vCPUs for the compute environment"
  type        = number
  default     = 64
}
