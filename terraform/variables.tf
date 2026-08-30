variable "aws_region" {
  description = "AWS Region for Amazon Bedrock and Amazon S3 Vectors."
  type        = string
  default     = "us-east-1"
}
variable "project_name" {
  description = "Prefix used to create deterministic, project-scoped resource names."
  type        = string
  default     = "semantic-layer-benchmark"

  validation {
    condition     = can(regex("^[a-z0-9-]{3,30}$", var.project_name))
    error_message = "project_name must contain 3-30 lowercase letters, numbers, or hyphens."
  }
}

variable "embedding_dimension" {
  description = "Vector dimension returned by the configured embedding model."
  type        = number
  default     = 256

  validation {
    condition     = contains([256, 512, 1024], var.embedding_dimension)
    error_message = "embedding_dimension must be 256, 512, or 1024 for Titan Text Embeddings V2."
  }
}
