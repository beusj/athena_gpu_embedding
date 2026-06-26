variable "environment" {
  description = "Environment name"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

variable "bucket_name" {
  description = "Artifact bucket name"
  type        = string
}

variable "prefix_root" {
  description = "Prefix root for this environment"
  type        = string
}

variable "kms_key_arn" {
  description = "Optional pre-existing KMS key ARN"
  type        = string
  default     = null
}

variable "log_retention_days" {
  description = "Retention period for logs/artifacts, by lifecycle rules"
  type        = number
  default     = 90
}
