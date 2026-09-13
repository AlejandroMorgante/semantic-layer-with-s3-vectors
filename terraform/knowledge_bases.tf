data "aws_partition" "current" {}

locals {
  source_bucket_name = substr(
    "${var.project_name}-source-${data.aws_caller_identity.current.account_id}-${var.aws_region}",
    0,
    63,
  )
  kb_vector_bucket_name = substr(
    "${var.project_name}-kb-${data.aws_caller_identity.current.account_id}-${var.aws_region}",
    0,
    63,
  )
  embedding_model_arn = join("", [
    "arn:", data.aws_partition.current.partition, ":bedrock:", var.aws_region,
    "::foundation-model/", var.embedding_model_id,
  ])
  graph_construction_model_arn = join("", [
    "arn:", data.aws_partition.current.partition, ":bedrock:", var.aws_region,
    "::foundation-model/", var.graph_construction_model_id,
  ])
  s3_vectors_kb_name = "${var.project_name}-s3-vectors-kb"
  neptune_kb_name    = "${var.project_name}-neptune-kb"
  graph_name         = "${var.project_name}-graphrag"
  corpus_prefix      = "catalog/"
}

resource "aws_s3_bucket" "knowledge_source" {
  bucket        = local.source_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "knowledge_source" {
  bucket = aws_s3_bucket.knowledge_source.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "knowledge_source" {
  bucket = aws_s3_bucket.knowledge_source.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3vectors_vector_bucket" "knowledge_base" {
  vector_bucket_name = local.kb_vector_bucket_name
  force_destroy      = true
}

resource "aws_s3vectors_index" "knowledge_base" {
  vector_bucket_name = aws_s3vectors_vector_bucket.knowledge_base.vector_bucket_name
  index_name         = "bedrock-kb-tables"
  data_type          = "float32"
  dimension          = var.embedding_dimension
  distance_metric    = "euclidean"

  metadata_configuration {
    non_filterable_metadata_keys = [
      "AMAZON_BEDROCK_TEXT",
      "AMAZON_BEDROCK_METADATA",
    ]
  }
}

resource "aws_neptunegraph_graph" "catalog" {
  graph_name          = local.graph_name
  provisioned_memory  = var.neptune_provisioned_memory
  public_connectivity = false
  replica_count       = 0
  deletion_protection = false

  vector_search_configuration {
    vector_search_dimension = var.embedding_dimension
  }
}

resource "aws_bedrockagent_knowledge_base" "s3_vectors" {
  name     = local.s3_vectors_kb_name
  role_arn = aws_iam_role.s3_vectors_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_model_arn

      embedding_model_configuration {
        bedrock_embedding_model_configuration {
          dimensions          = var.embedding_dimension
          embedding_data_type = "FLOAT32"
        }
      }
    }
  }

  storage_configuration {
    type = "S3_VECTORS"

    s3_vectors_configuration {
      index_arn = aws_s3vectors_index.knowledge_base.index_arn
    }
  }

  depends_on = [aws_iam_role_policy.s3_vectors_kb]
}

resource "aws_bedrockagent_knowledge_base" "neptune" {
  name     = local.neptune_kb_name
  role_arn = aws_iam_role.neptune_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_model_arn

      embedding_model_configuration {
        bedrock_embedding_model_configuration {
          dimensions          = var.embedding_dimension
          embedding_data_type = "FLOAT32"
        }
      }
    }
  }

  storage_configuration {
    type = "NEPTUNE_ANALYTICS"

    neptune_analytics_configuration {
      graph_arn = aws_neptunegraph_graph.catalog.arn

      field_mapping {
        metadata_field = "metadata"
        text_field     = "text"
      }
    }
  }

  depends_on = [aws_iam_role_policy.neptune_kb]
}

resource "aws_bedrockagent_data_source" "s3_vectors" {
  knowledge_base_id    = aws_bedrockagent_knowledge_base.s3_vectors.id
  name                 = "${var.project_name}-s3-vectors-source"
  data_deletion_policy = "DELETE"

  data_source_configuration {
    type = "S3"

    s3_configuration {
      bucket_arn              = aws_s3_bucket.knowledge_source.arn
      bucket_owner_account_id = data.aws_caller_identity.current.account_id
      inclusion_prefixes      = [local.corpus_prefix]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "NONE"
    }
  }
}

# The native AWS provider does not yet expose context_enrichment_configuration.
# AWS Cloud Control manages the same AWS::Bedrock::DataSource lifecycle and keeps
# the graph-construction model declarative instead of hiding it in an AWS CLI call.
resource "awscc_bedrock_data_source" "neptune" {
  knowledge_base_id    = aws_bedrockagent_knowledge_base.neptune.id
  name                 = "${var.project_name}-neptune-source"
  data_deletion_policy = "DELETE"

  data_source_configuration = {
    type = "S3"
    s3_configuration = {
      bucket_arn              = aws_s3_bucket.knowledge_source.arn
      bucket_owner_account_id = data.aws_caller_identity.current.account_id
      inclusion_prefixes      = [local.corpus_prefix]
    }
  }

  vector_ingestion_configuration = {
    chunking_configuration = {
      chunking_strategy = "NONE"
    }
    context_enrichment_configuration = {
      type = "BEDROCK_FOUNDATION_MODEL"
      bedrock_foundation_model_configuration = {
        model_arn = local.graph_construction_model_arn
        enrichment_strategy_configuration = {
          method = "CHUNK_ENTITY_EXTRACTION"
        }
      }
    }
  }
}
