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
