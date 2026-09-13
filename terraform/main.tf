provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = var.project_name
      ManagedBy = "Terraform"
      Purpose   = "Public semantic retrieval benchmark"
    }
  }
}

provider "awscc" {
  region = var.aws_region
}
data "aws_caller_identity" "current" {}

locals {
  vector_bucket_name = substr(
    "${var.project_name}-${data.aws_caller_identity.current.account_id}-${var.aws_region}",
    0,
    63,
  )
}

resource "aws_s3vectors_vector_bucket" "catalog" {
  vector_bucket_name = local.vector_bucket_name

  # This benchmark is disposable. Destroy must remove indexes and vectors even
  # when a previous run stopped after indexing.
  force_destroy = true
}

resource "aws_s3vectors_index" "tables" {
  vector_bucket_name = aws_s3vectors_vector_bucket.catalog.vector_bucket_name
  index_name         = "tables"
  data_type          = "float32"
  dimension          = var.embedding_dimension
  distance_metric    = "cosine"

  metadata_configuration {
    # Context is returned to the agent but never used as a structured filter.
    non_filterable_metadata_keys = ["content"]
  }
}
