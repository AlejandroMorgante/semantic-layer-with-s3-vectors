locals {
  bedrock_assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "bedrock.amazonaws.com"
      }
      Action = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "aws:SourceAccount" = data.aws_caller_identity.current.account_id
        }
        ArnLike = {
          "aws:SourceArn" = "arn:${data.aws_partition.current.partition}:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:knowledge-base/*"
        }
      }
    }]
  })
}

resource "aws_iam_role" "s3_vectors_kb" {
  name = "AmazonBedrockExecutionRoleForKnowledgeBase_${substr(replace(var.project_name, "-", ""), 0, 18)}S3"

  assume_role_policy = local.bedrock_assume_role_policy
}

resource "aws_iam_role_policy" "s3_vectors_kb" {
  name = "${var.project_name}-s3-vectors-kb"
  role = aws_iam_role.s3_vectors_kb.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeEmbeddingModel"
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = [local.embedding_model_arn]
      },
      {
        Sid      = "ListCorpusBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.knowledge_source.arn]
        Condition = {
          StringLike = {
            "s3:prefix" = ["${local.corpus_prefix}*"]
          }
        }
      },
      {
        Sid      = "ReadCorpus"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.knowledge_source.arn}/${local.corpus_prefix}*"]
      },
      {
        Sid    = "UseS3VectorIndex"
        Effect = "Allow"
        Action = [
          "s3vectors:PutVectors",
          "s3vectors:GetVectors",
          "s3vectors:DeleteVectors",
          "s3vectors:QueryVectors",
          "s3vectors:GetIndex",
        ]
        Resource = [aws_s3vectors_index.knowledge_base.index_arn]
      },
    ]
  })
}

resource "aws_iam_role" "neptune_kb" {
  name = "AmazonBedrockExecutionRoleForKnowledgeBase_${substr(replace(var.project_name, "-", ""), 0, 15)}Graph"

  assume_role_policy = local.bedrock_assume_role_policy
}

resource "aws_iam_role_policy" "neptune_kb" {
  name = "${var.project_name}-neptune-kb"
  role = aws_iam_role.neptune_kb.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeIngestionModels"
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = [local.embedding_model_arn, local.graph_construction_model_arn]
      },
      {
        Sid      = "ListCorpusBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.knowledge_source.arn]
        Condition = {
          StringLike = {
            "s3:prefix" = ["${local.corpus_prefix}*"]
          }
        }
      },
      {
        Sid      = "ReadCorpus"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.knowledge_source.arn}/${local.corpus_prefix}*"]
      },
      {
        Sid    = "UseNeptuneGraph"
        Effect = "Allow"
        Action = [
          "neptune-graph:GetGraph",
          "neptune-graph:ReadDataViaQuery",
          "neptune-graph:WriteDataViaQuery",
          "neptune-graph:DeleteDataViaQuery",
        ]
        Resource = [aws_neptunegraph_graph.catalog.arn]
      },
    ]
  })
}
