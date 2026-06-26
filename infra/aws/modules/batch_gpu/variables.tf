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
