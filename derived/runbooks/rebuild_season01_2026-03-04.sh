#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source venv/bin/activate

SEASON=1
DOCS_DIR="ingestion/normalized_docs_txt"
OUT_ROOT="derived/artifacts"
SEGMENTS_ROOT="derived/artifacts/segments"
LLM_MODEL="gpt-4.1-nano"

SEG_BUILD_PREFIX="derived_segsummary_season01_nano_2026-03-04"
EP_BUILD_PREFIX="derived_episodecard_season01_nano_2026-03-04"
SEASON_CARD_BUILD_PREFIX="derived_seasoncard_season01_nano_2026-03-04"

# 1) Segments (idempotent)
python -m derived.summarize_season \
  --season "$SEASON" \
  --docs-dir "$DOCS_DIR" \
  --segments-root "$SEGMENTS_ROOT" \
  --out-root "$OUT_ROOT" \
  --build-prefix "$SEG_BUILD_PREFIX" \
  --llm-model "$LLM_MODEL"

# 2) Episode cards (reduce)
python -m derived.reduce_season \
  --season "$SEASON" \
  --docs-dir "$DOCS_DIR" \
  --segments-root "$SEGMENTS_ROOT" \
  --segment-summaries-root "$OUT_ROOT" \
  --out-root "$OUT_ROOT" \
  --segments-build-prefix "$SEG_BUILD_PREFIX" \
  --build-prefix "$EP_BUILD_PREFIX" \
  --llm-model "$LLM_MODEL"

# 3) Season card (reduce-of-reduce)
python -m derived.reduce_season_card \
  --season "$SEASON" \
  --docs-dir "$DOCS_DIR" \
  --out-root "$OUT_ROOT" \
  --episode-cards-build-prefix "$EP_BUILD_PREFIX" \
  --build-prefix "$SEASON_CARD_BUILD_PREFIX" \
  --llm-model "$LLM_MODEL"

echo "OK: rebuilt season $SEASON artifacts."