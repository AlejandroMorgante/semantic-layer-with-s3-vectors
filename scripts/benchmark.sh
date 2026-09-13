#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_command aws
require_command uv
aws sts get-caller-identity --profile "${AWS_PROFILE}" --region "${AWS_REGION}" >/dev/null

if command -v terraform >/dev/null 2>&1; then
  if [[ -z "${S3_VECTORS_KB_ID}" ]]; then
    S3_VECTORS_KB_ID="$(tf output -raw s3_vectors_kb_id 2>/dev/null || true)"
  fi
  if [[ -z "${NEPTUNE_KB_ID}" ]]; then
    NEPTUNE_KB_ID="$(tf output -raw neptune_kb_id 2>/dev/null || true)"
  fi
fi

benchmark_args=(
  benchmark
  --catalog "${BENCHMARK_DIR}/catalog.json"
  --questions "${ROOT_DIR}/data/questions.json"
  --output-dir "${BENCHMARK_DIR}/results"
  --model-id "${MODEL_ID}"
  --top-k "${TOP_K}"
  --repetitions "${REPETITIONS}"
)

needs_direct_s3=false
if [[ -z "${BENCHMARK_STRATEGIES}" && ( -z "${S3_VECTORS_KB_ID}" || -z "${NEPTUNE_KB_ID}" ) ]]; then
  needs_direct_s3=true
elif [[ ",${BENCHMARK_STRATEGIES}," == *",direct_s3_vectors,"* ]]; then
  needs_direct_s3=true
fi

if [[ "${needs_direct_s3}" == true ]]; then
  require_command terraform
  bucket_name="$(terraform_output vector_bucket_name)"
  index_name="$(terraform_output vector_index_name)"
  benchmark_args+=(
    --bucket "${bucket_name}"
    --index "${index_name}"
    --embedding-model-id "${EMBEDDING_MODEL_ID}"
  )
fi

if [[ -n "${S3_VECTORS_KB_ID}" ]]; then
  benchmark_args+=(--s3-vectors-kb-id "${S3_VECTORS_KB_ID}")
fi
if [[ -n "${NEPTUNE_KB_ID}" ]]; then
  benchmark_args+=(--neptune-kb-id "${NEPTUNE_KB_ID}")
fi
if [[ -n "${BENCHMARK_STRATEGIES}" ]]; then
  benchmark_args+=(--strategies "${BENCHMARK_STRATEGIES}")
fi

benchmark_cli "${benchmark_args[@]}"
