terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # Pinned to 5.x: the Batch compute-environment resource uses v5 attribute
      # names (compute_environment_name) that were renamed in provider v6.
      version = "~> 5.0"
    }
  }
}
