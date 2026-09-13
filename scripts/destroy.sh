#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

preflight_tools
uv sync --project "${ROOT_DIR}" --frozen
tf init -input=false

account_id="$(aws sts get-caller-identity --profile "${AWS_PROFILE}" --query Account --output text)"
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

benchmark_cli verify-destroyed --bucket "${bucket_name}"
if aws s3vectors get-vector-bucket \
  --vector-bucket-name "${kb_bucket_name}" \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "Knowledge Base vector bucket still exists: ${kb_bucket_name}" >&2
  exit 1
fi
if aws s3api head-bucket \
  --bucket "${source_bucket_name}" \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "Knowledge source bucket still exists: ${source_bucket_name}" >&2
  exit 1
fi

s3_kb_count="$(aws bedrock-agent list-knowledge-bases \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" \
  --max-results 100 \
  --query "length(knowledgeBaseSummaries[?name=='${PROJECT_NAME}-s3-vectors-kb'])" \
  --output text)"
neptune_kb_count="$(aws bedrock-agent list-knowledge-bases \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" \
  --max-results 100 \
  --query "length(knowledgeBaseSummaries[?name=='${PROJECT_NAME}-neptune-kb'])" \
  --output text)"
graph_count="$(aws neptune-graph list-graphs \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}" \
  --max-results 100 \
  --query "length(graphs[?name=='${PROJECT_NAME}-graphrag'])" \
  --output text)"
if [[ "${s3_kb_count}" != "0" || "${neptune_kb_count}" != "0" || "${graph_count}" != "0" ]]; then
  echo "Managed benchmark resources remain after destroy." >&2
  exit 1
fi

echo "DESTROY complete. Verified all benchmark buckets, Knowledge Bases, and graph are absent."
