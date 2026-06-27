variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "environment" {
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

variable "instance_families" {
  description = "Preferred GPU instance families"
  type        = list(string)
  default     = ["g5", "g6e"]
}

variable "spot_enabled" {
  description = "Whether Spot-first placement is enabled"
  type        = bool
  default     = true
}

variable "on_demand_base_capacity" {
  description = "On-demand base capacity"
  type        = number
  default     = 0
}

variable "job_role_arn" {
  description = "Batch job IAM role ARN"
  type        = string
}

variable "task_execution_role_arn" {
  description = "Batch task execution IAM role ARN"
  type        = string
  default     = null
}

variable "ecr_image_uri" {
  description = "ECR image URI for gpu-embedder Docker image"
  type        = string
}

variable "vcpus" {
  description = "vCPUs per job"
  type        = number
  default     = 4
}

variable "memory" {
  description = "Memory (MB) per job"
  type        = number
  default     = 32768 # 32 GB for GPU workload
}

variable "subnet_ids" {
  description = "VPC subnet IDs for EC2 instances"
  type        = list(string)
}
variable "vpc_id" {
  description = "VPC ID for security group"
  type        = string
}

variable "manage_iam_resources" {
  description = "Whether this module should create/manage IAM roles and instance profile"
  type        = bool
  default     = true
}

variable "batch_service_role_arn" {
  description = "Pre-created AWS Batch service role ARN when manage_iam_resources is false"
  type        = string
  default     = null
}

variable "batch_instance_role_name" {
  description = "Pre-created EC2 instance role name when manage_iam_resources is false"
  type        = string
  default     = null
}

variable "batch_instance_profile_arn" {
  description = "Pre-created instance profile ARN when manage_iam_resources is false"
  type        = string
  default     = null
}

variable "batch_spot_fleet_role_arn" {
  description = "Pre-created EC2 Spot Fleet role ARN when manage_iam_resources is false"
  type        = string
  default     = null
}