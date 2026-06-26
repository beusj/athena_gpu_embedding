variable "environment" {
  description = "Environment name"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

variable "bucket_arn" {
  description = "Artifact bucket ARN"
  type        = string
}

variable "prefix_scope" {
  description = "Allowed prefix scope for read/write (e.g. gpu-embed/academic-dev)"
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key ARN for encrypt/decrypt permissions (null = SSE-S3 only)"
  type        = string
  default     = null
}

variable "create_roles" {
  description = <<-EOT
    Whether Terraform should create the IAM roles/policies. Many academic
    accounts forbid IAM creation; set this to false and supply the pre-created
    role ARNs below, which the module then simply passes through.
  EOT
  type        = bool
  default     = true
}

variable "existing_job_role_arn" {
  description = "Pre-created Batch job role ARN (used when create_roles = false)"
  type        = string
  default     = null
}

variable "existing_task_execution_role_arn" {
  description = "Pre-created ECS task execution role ARN (used when create_roles = false)"
  type        = string
  default     = null
}

variable "existing_instance_profile_arn" {
  description = "Pre-created EC2 instance profile ARN (used when create_roles = false)"
  type        = string
  default     = null
}

variable "permissions_boundary_arn" {
  description = "Optional IAM permissions boundary to attach to created roles (often mandatory in locked-down accounts)"
  type        = string
  default     = null
}
