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

variable "embedding_model_id" {
  description = "Bedrock embedding model used by both managed Knowledge Bases."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "graph_construction_model_id" {
  description = "Bedrock foundation model used to extract GraphRAG entities and relationships."
  type        = string
  default     = "amazon.nova-micro-v1:0"
}

variable "neptune_provisioned_memory" {
  description = "Provisioned Neptune Analytics memory in m-NCUs. The service minimum is 16."
  type        = number
  default     = 16

  validation {
    condition     = var.neptune_provisioned_memory >= 16 && var.neptune_provisioned_memory <= 24576
    error_message = "neptune_provisioned_memory must be between 16 and 24576 m-NCUs."
  }
}
