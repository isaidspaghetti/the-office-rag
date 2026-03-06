#!/usr/bin/env bash
set -euo pipefail

# Parameterized derived-corpus rebuild (segments -> map summaries -> episode reduce -> season reduce)
# Writes NEW artifact build folders (no overwrites) and optionally builds a NEW derived-cards Chroma DB.
#
# Examples:
#   bash derived/runbooks/rebuild_seasons_param.sh --llm-model gpt-4.1-mini --seasons 1,2,3,4 --tag mini_2026-03-05 --build-derived-db
#   bash derived/runbooks/rebuild_seasons_param.sh --llm-model gpt-4.1-nano --seasons 1 --tag nano_test --force

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi

LLM_MODEL="gpt-4.1-mini"
SEASONS="1,2,3,4"
TAG=""
DOCS_DIR="ingestion/normalized_docs_txt"
OUT_ROOT="derived/artifacts"
SEGMENTS_ROOT="derived/artifacts/segments"
FORCE="false"
BUILD_DERIVED_DB="false"
INDEX_ONLY="false"
DERIVED_DB_PERSIST_DIR=""
DERIVED_COLLECTION_NAME="derived_cards"
EMBED_MODEL="text-embedding-3-small"

usage() {
  cat <<EOF
Usage: $0 [args]

Args:
  --llm-model MODEL            LLM model for map/reduce steps (default: $LLM_MODEL)
  --seasons CSV                Seasons to rebuild, e.g. 1,2,3,4 (default: $SEASONS)
  --tag TAG                    Tag used in build prefixes (default: derived from model + UTC date)
  --docs-dir PATH              Normalized docs dir (default: $DOCS_DIR)
  --out-root PATH              Artifacts output root (default: $OUT_ROOT)
  --segments-root PATH         Segments root (default: $SEGMENTS_ROOT)
  --force                      Force re-summarize and re-reduce within the NEW build folders

  --index-only                 Skip ALL LLM map/reduce steps; only build the derived-cards DB from existing artifacts

  --build-derived-db           Build a new combined derived-cards Chroma DB from the builds
  --derived-db-persist-dir DIR Persist dir for derived cards DB (default: auto from tag)
  --derived-collection-name N  Chroma collection name (default: $DERIVED_COLLECTION_NAME)
  --embed-model MODEL          Embedding model for derived DB (default: $EMBED_MODEL)

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-model) LLM_MODEL="$2"; shift 2;;
    --seasons) SEASONS="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --docs-dir) DOCS_DIR="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --segments-root) SEGMENTS_ROOT="$2"; shift 2;;
    --force) FORCE="true"; shift 1;;
    --index-only) INDEX_ONLY="true"; shift 1;;
    --build-derived-db) BUILD_DERIVED_DB="true"; shift 1;;
    --derived-db-persist-dir) DERIVED_DB_PERSIST_DIR="$2"; shift 2;;
    --derived-collection-name) DERIVED_COLLECTION_NAME="$2"; shift 2;;
    --embed-model) EMBED_MODEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

# Default tag: <model_slug>_<YYYY-MM-DD>
if [[ -z "${TAG}" ]]; then
  UTC_DATE="$(date -u +%Y-%m-%d)"
  MODEL_SLUG="${LLM_MODEL//./_}"
  TAG="${MODEL_SLUG}_${UTC_DATE}"
fi

# Season CSV -> array
IFS=',' read -r -a SEASON_ARR <<< "$SEASONS"

if [[ "$INDEX_ONLY" == "true" ]]; then
  # Index-only implies we build a derived DB; we do NOT run summarize/reduce steps.
  BUILD_DERIVED_DB="true"
  echo "=== Index-only mode ==="
  echo "Skipping LLM map/reduce. Will build derived-cards DB from existing artifacts under: $OUT_ROOT"
fi

EP_PREFIXES=()
SEASON_PREFIXES=()

if [[ "$INDEX_ONLY" != "true" ]]; then
for S in "${SEASON_ARR[@]}"; do
  S_TRIM="$(echo "$S" | xargs)"
  if [[ -z "$S_TRIM" ]]; then
    continue
  fi
  if ! [[ "$S_TRIM" =~ ^[0-9]+$ ]]; then
    echo "ERROR: season must be an int, got: '$S_TRIM'" >&2
    exit 2
  fi

  SEASON_PAD=$(printf "%02d" "$S_TRIM")

  SEG_BUILD_PREFIX="derived_segsummary_season${SEASON_PAD}_${TAG}"
  EP_BUILD_PREFIX="derived_episodecard_season${SEASON_PAD}_${TAG}"
  SEASON_CARD_BUILD_PREFIX="derived_seasoncard_season${SEASON_PAD}_${TAG}"

  echo "=== Season $S_TRIM | model=$LLM_MODEL | tag=$TAG ==="

  "$PY" -m derived.summarize_season \
    --season "$S_TRIM" \
    --docs-dir "$DOCS_DIR" \
    --segments-root "$SEGMENTS_ROOT" \
    --out-root "$OUT_ROOT" \
    --build-prefix "$SEG_BUILD_PREFIX" \
    --llm-model "$LLM_MODEL" \
    $( [[ "$FORCE" == "true" ]] && echo "--force-summaries" )

  "$PY" -m derived.reduce_season \
    --season "$S_TRIM" \
    --docs-dir "$DOCS_DIR" \
    --segments-root "$SEGMENTS_ROOT" \
    --segment-summaries-root "$OUT_ROOT" \
    --out-root "$OUT_ROOT" \
    --segments-build-prefix "$SEG_BUILD_PREFIX" \
    --build-prefix "$EP_BUILD_PREFIX" \
    --llm-model "$LLM_MODEL" \
    $( [[ "$FORCE" == "true" ]] && echo "--force" )

  "$PY" -m derived.reduce_season_card \
    --season "$S_TRIM" \
    --docs-dir "$DOCS_DIR" \
    --out-root "$OUT_ROOT" \
    --episode-cards-build-prefix "$EP_BUILD_PREFIX" \
    --build-prefix "$SEASON_CARD_BUILD_PREFIX" \
    --llm-model "$LLM_MODEL" \
    $( [[ "$FORCE" == "true" ]] && echo "--force" )

  EP_PREFIXES+=("$EP_BUILD_PREFIX")
  SEASON_PREFIXES+=("$SEASON_CARD_BUILD_PREFIX")
done

echo "OK: built episode prefixes: ${EP_PREFIXES[*]}"
echo "OK: built season prefixes: ${SEASON_PREFIXES[*]}"
fi

if [[ "$BUILD_DERIVED_DB" == "true" ]]; then
  if [[ -z "$DERIVED_DB_PERSIST_DIR" ]]; then
    DERIVED_DB_PERSIST_DIR="db/chroma_db_derived_cards_${TAG}"
  fi

  EP_PREFIXES_CSV=$(IFS=','; echo "${EP_PREFIXES[*]}")
  SEASON_PREFIXES_CSV=$(IFS=','; echo "${SEASON_PREFIXES[*]}")

  echo "=== Building derived cards DB: $DERIVED_DB_PERSIST_DIR (collection=$DERIVED_COLLECTION_NAME) ==="

  if [[ "$INDEX_ONLY" == "true" ]]; then
    # Let the indexer auto-detect build prefixes under out-root.
    # Keep season filtering consistent with requested seasons.
    "$PY" derived/build_derived_cards_index.py \
      --out-root "$OUT_ROOT" \
      --persist-dir "$DERIVED_DB_PERSIST_DIR" \
      --collection-name "$DERIVED_COLLECTION_NAME" \
      --embed-model "$EMBED_MODEL" \
      --seasons "$SEASONS" \
      --reset
  else
    "$PY" derived/build_derived_cards_index.py \
      --out-root "$OUT_ROOT" \
      --persist-dir "$DERIVED_DB_PERSIST_DIR" \
      --collection-name "$DERIVED_COLLECTION_NAME" \
      --embed-model "$EMBED_MODEL" \
      --episode-cards-build-prefixes "$EP_PREFIXES_CSV" \
      --season-cards-build-prefixes "$SEASON_PREFIXES_CSV" \
      --no-topic-cards \
      --reset
  fi

  echo "OK: derived cards DB built at $DERIVED_DB_PERSIST_DIR"
fi
