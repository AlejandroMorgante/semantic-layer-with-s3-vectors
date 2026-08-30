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

tf init -input=false
tf fmt -check
tf validate
tf apply -auto-approve -input=false

bucket_name="$(terraform_output vector_bucket_name)"
index_name="$(terraform_output vector_index_name)"

benchmark_cli preflight \
  --bucket "${bucket_name}" \
  --index "${index_name}" \
  --embedding-model-id "${EMBEDDING_MODEL_ID}" \
  --model-id "${MODEL_ID}"

benchmark_cli generate \
  --tables-per-country "${TABLES_PER_COUNTRY}" \
  --output-dir "${BENCHMARK_DIR}"

benchmark_cli index \
  --bucket "${bucket_name}" \
  --index "${index_name}" \
  --catalog "${BENCHMARK_DIR}/catalog.json" \
  --embedding-model-id "${EMBEDDING_MODEL_ID}"

trap - EXIT
echo "UP complete. Resources remain active for benchmarks."
echo "Run 'make destroy AWS_PROFILE=${AWS_PROFILE} AWS_REGION=${AWS_REGION}' when finished."
