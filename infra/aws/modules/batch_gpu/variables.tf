variable "environment" {
  description = "Environment name"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

variable "create" {
  description = <<-EOT
    Whether to create the Batch compute environment, queue, and job definition.
    Set to false until GPU service quotas and networking are confirmed in the
    account; outputs are then null and nothing is provisioned.
  EOT
  type        = bool
  default     = true
}

variable "instance_families" {
  description = "Preferred GPU instance families (Batch accepts family names, e.g. g5/g6e)"
  type        = list(string)
  default     = ["g5", "g6e"]
}

variable "spot_enabled" {
  description = "Use Spot capacity (SPOT_CAPACITY_OPTIMIZED). false = on-demand EC2."
  type        = bool
  default     = true
}

variable "spot_bid_percentage" {
  description = "Max Spot price as % of on-demand (null = AWS default of 100)"
  type        = number
  default     = null
}

variable "on_demand_base_capacity" {
  description = "Reserved for future mixed-capacity wiring; not used by a single managed CE"
  type        = number
  default     = 0
}

# --- networking (bring-your-own; academic accounts rarely allow VPC creation) -

variable "subnet_ids" {
  description = "Existing subnet IDs the compute environment launches instances into"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Existing security group IDs for compute instances (empty = VPC default SG)"
  type        = list(string)
  default     = []
}

variable "instance_profile_arn" {
  description = "EC2 instance profile ARN for the compute environment"
  type        = string
}

# --- capacity ----------------------------------------------------------------

variable "max_vcpus" {
  description = "Maximum vCPUs for the compute environment"
  type        = number
  default     = 64
}

variable "min_vcpus" {
  description = "Minimum vCPUs (0 lets the environment scale to zero when idle)"
  type        = number
  default     = 0
}

# --- per-job container resources --------------------------------------------

variable "container_image" {
  description = "ECR image URI for the worker (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/gpu-embedder:tag)"
  type        = string
}

variable "job_vcpus" {
  description = "vCPUs requested per array task"
  type        = number
  default     = 4
}

variable "job_memory_mib" {
  description = "Memory (MiB) requested per array task"
  type        = number
  default     = 16384
}

variable "job_gpus" {
  description = "GPUs requested per array task"
  type        = number
  default     = 1
}

variable "job_role_arn" {
  description = "Batch job IAM role ARN (carries S3/KMS artifact access)"
  type        = string
  default     = null
}

variable "task_execution_role_arn" {
  description = "ECS task execution IAM role ARN (image pull + logs)"
  type        = string
  default     = null
}
