output "vector_bucket_name" {
  description = "Dedicated Amazon S3 Vectors bucket used by the benchmark."
  value       = aws_s3vectors_vector_bucket.catalog.vector_bucket_name
}
output "vector_index_name" {
  description = "Vector index containing one vector per synthetic table."
  value       = aws_s3vectors_index.tables.index_name
}

output "vector_index_arn" {
  description = "ARN of the vector index."
  value       = aws_s3vectors_index.tables.index_arn
}

output "knowledge_source_bucket_name" {
  description = "S3 bucket containing the identical corpus ingested by both Knowledge Bases."
  value       = aws_s3_bucket.knowledge_source.id
}

output "knowledge_source_prefix" {
  description = "S3 prefix containing the generated per-table Markdown documents."
  value       = local.corpus_prefix
}

output "s3_vectors_kb_id" {
  description = "Bedrock Knowledge Base ID backed by Amazon S3 Vectors."
  value       = aws_bedrockagent_knowledge_base.s3_vectors.id
}

output "s3_vectors_data_source_id" {
  description = "Data source ID for the S3 Vectors-backed Knowledge Base."
  value       = aws_bedrockagent_data_source.s3_vectors.data_source_id
}

output "neptune_kb_id" {
  description = "Bedrock Knowledge Base ID backed by Neptune Analytics GraphRAG."
  value       = aws_bedrockagent_knowledge_base.neptune.id
}

output "neptune_data_source_id" {
  description = "Data source ID for the Neptune Analytics-backed Knowledge Base."
  value       = awscc_bedrock_data_source.neptune.data_source_id
}

output "neptune_graph_id" {
  description = "Neptune Analytics graph ID used by GraphRAG."
  value       = aws_neptunegraph_graph.catalog.id
}

output "neptune_graph_name" {
  description = "Deterministic Neptune Analytics graph name used by GraphRAG."
  value       = aws_neptunegraph_graph.catalog.graph_name
}

output "knowledge_base_vector_bucket_name" {
  description = "S3 vector bucket used by the managed S3 Vectors Knowledge Base."
  value       = aws_s3vectors_vector_bucket.knowledge_base.vector_bucket_name
}

output "embedding_model_id" {
  description = "Embedding model shared by both managed Knowledge Bases."
  value       = var.embedding_model_id
}

output "embedding_dimension" {
  description = "Embedding dimension shared by both managed Knowledge Bases."
  value       = var.embedding_dimension
}

output "graph_construction_model_id" {
  description = "Foundation model used for Neptune GraphRAG entity extraction."
  value       = var.graph_construction_model_id
}

output "neptune_provisioned_memory" {
  description = "Provisioned Neptune Analytics capacity in m-NCUs."
  value       = var.neptune_provisioned_memory
}
