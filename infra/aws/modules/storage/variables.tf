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
  description = "Env-scoped prefix root for artifacts (e.g. gpu-embed/academic-dev)"
  type        = string
}

variable "kms_key_arn" {
  description = "Optional pre-existing KMS key ARN. null falls back to SSE-S3 (AES256)."
  type        = string
  default     = null
}

variable "create_bucket" {
  description = <<-EOT
    Whether Terraform should create and manage the bucket (encryption, public
    access block, versioning, lifecycle). Set to false in locked-down accounts
    where the bucket is pre-provisioned by central IT; the module then only
    derives ARNs/prefixes and applies no changes to the existing bucket.
  EOT
  type        = bool
  default     = true
}

variable "enable_versioning" {
  description = "Enable S3 object versioning (only when create_bucket = true)"
  type        = bool
  default     = true
}

variable "input_expiration_days" {
  description = "Expire input shard objects after N days (0 disables the rule)"
  type        = number
  default     = 30
}

variable "output_expiration_days" {
  description = "Expire output embedding objects after N days (0 disables the rule)"
  type        = number
  default     = 90
}

variable "log_retention_days" {
  description = "Retention period for the logs/ prefix (0 disables the rule)"
  type        = number
  default     = 90
}

variable "abort_multipart_days" {
  description = "Abort incomplete multipart uploads after N days"
  type        = number
  default     = 7
}
