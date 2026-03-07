#!/usr/bin/env bash
set -euo pipefail

# Re-run the same tuning sweep but with routing enabled.
# Uses derived-cards index (built already) as stage-1 routing.

PY="/Users/tg/Developer/RAG-1/venv/bin/python"
PERSIST_DIR="db/chroma_db"
if [[ -d "db/chroma_db_meta" ]]; then
  # Prefer metadata-enabled index for routing filters + more stable episode_id behavior.
  PERSIST_DIR="db/chroma_db_meta"
fi
DERIVED_PERSIST_DIR="db/chroma_db_derived_cards"
DERIVED_COLLECTION_NAME="derived_cards"
# LLM used for the answer generation step in run_eval.py
GEN_LLM_MODEL="gpt-4.1-mini"
GEN_LLM_TIMEOUT="90"
GEN_LLM_MAX_RETRIES="2"
# Tag used to prefix run-name entries written into experiments/runs.
# This sweep runs against the unified derived-cards index which now includes
# episode_card + season_card + topic_card docs.
RUN_TAG="topiccards_blended_gen_$(echo "$GEN_LLM_MODEL" | tr '.' '_' | tr '-' '_')"

run() {
  echo
  echo "=== $* ==="
  $PY experiments/run_eval.py \
    --persist-dir "$PERSIST_DIR" \
    --retrieval-policy blended \
    --derived-persist-dir "$DERIVED_PERSIST_DIR" \
    --derived-collection-name "$DERIVED_COLLECTION_NAME" \
    --llm-model "$GEN_LLM_MODEL" \
    --timeout "$GEN_LLM_TIMEOUT" \
    --max-retries "$GEN_LLM_MAX_RETRIES" \
    "$@"
}

# Baseline
run --run-name ${RUN_TAG}_baseline_similarity_k3 --search-type similarity --k 3

# Increase k recall
run --run-name ${RUN_TAG}_t1_similarity_k12 --search-type similarity --k 12 --notes "Routing: blended (base + intent-conditioned derived-guided add-on) + similarity k=12"

# MMR
run --run-name ${RUN_TAG}_t2_mmr_k12_fetch40_l07_fixed --search-type mmr --k 12 --fetch-k 40 --lambda-mult 0.7 --notes "Routing: blended (base + intent-conditioned derived-guided add-on) + MMR k=12 fetch_k=40 lambda=0.7"

# Query Expansion + RRF fusion
run --run-name ${RUN_TAG}_qe_rrf_similarity_k12 \
  --search-type similarity \
  --k 12 \
  --query-expansion \
  --expand-n 5 \
  --expand-model gpt-4.1-nano \
  --k-per-query 6 \
  --fusion rrf \
  --rrf-k0 60 \
  --notes "Routing (blended) + query expansion + RRF fusion"

# Query Expansion + RRF + MMR
run --run-name ${RUN_TAG}_qe_rrf_mmr_k12 \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --query-expansion \
  --expand-n 5 \
  --expand-model gpt-4.1-nano \
  --k-per-query 6 \
  --fusion rrf \
  --rrf-k0 60 \
  --notes "Routing (blended) + query expansion + RRF fusion + MMR"


# If metadata index exists but wasn't selected as default (e.g., user forced PERSIST_DIR),
# run one explicit meta-index configuration for comparison.
if [[ -d "db/chroma_db_meta" && "$PERSIST_DIR" != "db/chroma_db_meta" ]]; then
  run --persist-dir db/chroma_db_meta \
    --run-name ${RUN_TAG}_meta_qe_rrf_mmr_k12 \
    --search-type mmr \
    --k 12 \
    --fetch-k 40 \
    --lambda-mult 0.7 \
    --query-expansion \
    --expand-n 5 \
    --expand-model gpt-4.1-nano \
    --k-per-query 6 \
    --fusion rrf \
    --rrf-k0 60 \
    --notes "Routing (blended) + metadata index + query expansion + RRF + MMR"
fi

echo
echo "Done. New run logs are in experiments/runs (run_name prefixed with ${RUN_TAG}_)."
