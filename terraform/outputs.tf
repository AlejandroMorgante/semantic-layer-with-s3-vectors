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
