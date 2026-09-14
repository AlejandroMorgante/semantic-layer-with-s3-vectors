#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

preflight_tools
uv sync --project "${ROOT_DIR}" --frozen
tf init -input=false

account_id="$(aws_cli sts get-caller-identity --query Account --output text)"
fallback_bucket="${PROJECT_NAME}-${account_id}-${AWS_REGION}"
fallback_bucket="${fallback_bucket:0:63}"
fallback_kb_bucket="${PROJECT_NAME}-kb-${account_id}-${AWS_REGION}"
fallback_kb_bucket="${fallback_kb_bucket:0:63}"
fallback_source_bucket="${PROJECT_NAME}-source-${account_id}-${AWS_REGION}"
fallback_source_bucket="${fallback_source_bucket:0:63}"
bucket_name="$(tf output -raw vector_bucket_name 2>/dev/null || echo "${fallback_bucket}")"
index_name="$(tf output -raw vector_index_name 2>/dev/null || echo "tables")"
kb_bucket_name="$(tf output -raw knowledge_base_vector_bucket_name 2>/dev/null || echo "${fallback_kb_bucket}")"
source_bucket_name="$(tf output -raw knowledge_source_bucket_name 2>/dev/null || echo "${fallback_source_bucket}")"

# Empty the exact project index first. Terraform force_destroy is the second
# cleanup layer and handles interrupted indexing runs.
benchmark_cli purge --bucket "${bucket_name}" --index "${index_name}" || true

if ! tf destroy -auto-approve -input=false; then
  echo "Terraform destroy failed. Attempting exact-name fallback cleanup." >&2
  benchmark_cli force-delete --bucket "${bucket_name}" --index "${index_name}"
  tf destroy -auto-approve -input=false
fi

benchmark_cli verify-destroyed \
  --bucket "${bucket_name}" \
  --knowledge-base-vector-bucket "${kb_bucket_name}" \
  --source-bucket "${source_bucket_name}" \
  --s3-knowledge-base-name "${PROJECT_NAME}-s3-vectors-kb" \
  --neptune-knowledge-base-name "${PROJECT_NAME}-neptune-kb" \
  --graph-name "${PROJECT_NAME}-graphrag"

echo "DESTROY complete. Verified all benchmark buckets, Knowledge Bases, and graph are absent."
