#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

cleanup_after_failure() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "UP failed. Running project-scoped cleanup before exiting." >&2
    "${ROOT_DIR}/scripts/destroy.sh" || true
  fi
  exit "${status}"
}
trap cleanup_after_failure EXIT

preflight_tools
mkdir -p "${BENCHMARK_DIR}"
uv sync --project "${ROOT_DIR}" --frozen

benchmark_cli generate \
  --tables-per-country "${TABLES_PER_COUNTRY}" \
  --output-dir "${BENCHMARK_DIR}"

tf init -input=false
tf fmt -check
tf validate
tf apply -auto-approve -input=false
tf output -json >"${BENCHMARK_DIR}/infrastructure.json"

bucket_name="$(terraform_output vector_bucket_name)"
index_name="$(terraform_output vector_index_name)"
source_bucket_name="$(terraform_output knowledge_source_bucket_name)"
s3_vectors_kb_id="$(terraform_output s3_vectors_kb_id)"
s3_vectors_data_source_id="$(terraform_output s3_vectors_data_source_id)"
neptune_kb_id="$(terraform_output neptune_kb_id)"
neptune_data_source_id="$(terraform_output neptune_data_source_id)"

aws_cli s3 sync \
  "${BENCHMARK_DIR}/knowledge-base-documents/" \
  "s3://${source_bucket_name}/catalog/" \
  --delete

benchmark_cli preflight \
  --bucket "${bucket_name}" \
  --index "${index_name}" \
  --embedding-model-id "${EMBEDDING_MODEL_ID}" \
  --model-id "${MODEL_ID}"

benchmark_cli sync-knowledge-base \
  --knowledge-base-id "${s3_vectors_kb_id}" \
  --data-source-id "${s3_vectors_data_source_id}"

benchmark_cli sync-knowledge-base \
  --knowledge-base-id "${neptune_kb_id}" \
  --data-source-id "${neptune_data_source_id}"

benchmark_cli index \
  --bucket "${bucket_name}" \
  --index "${index_name}" \
  --catalog "${BENCHMARK_DIR}/catalog.json" \
  --embedding-model-id "${EMBEDDING_MODEL_ID}"

trap - EXIT
echo "UP complete. Resources remain active for benchmarks."
echo "S3 Vectors Knowledge Base: ${s3_vectors_kb_id}"
echo "Neptune GraphRAG Knowledge Base: ${neptune_kb_id}"
echo "Run 'make destroy AWS_REGION=${AWS_REGION}' with the same AWS credentials when finished."
