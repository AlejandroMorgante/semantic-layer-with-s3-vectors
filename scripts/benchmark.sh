#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

preflight_tools
bucket_name="$(terraform_output vector_bucket_name)"
index_name="$(terraform_output vector_index_name)"

benchmark_cli benchmark \
  --bucket "${bucket_name}" \
  --index "${index_name}" \
  --catalog "${BENCHMARK_DIR}/catalog.json" \
  --questions "${ROOT_DIR}/data/questions.json" \
  --output-dir "${BENCHMARK_DIR}/results" \
  --model-id "${MODEL_ID}" \
  --embedding-model-id "${EMBEDDING_MODEL_ID}" \
  --top-k "${TOP_K}" \
  --repetitions "${REPETITIONS}"
