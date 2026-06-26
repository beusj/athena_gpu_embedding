variable "environment" {
  description = "Environment name"
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
  default     = null
}

variable "task_execution_role_arn" {
  description = "Batch task execution IAM role ARN"
  type        = string
  default     = null
}
