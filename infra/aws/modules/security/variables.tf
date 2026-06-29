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
  description = "Allowed prefix scope for read/write"
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key ARN for encrypt/decrypt permissions"
  type        = string
  default     = null
}
