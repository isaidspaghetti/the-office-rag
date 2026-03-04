# Resume here (context for a new ChatGPT thread)

If you start a new thread, link this file plus the most recent `experiments/run_metrics.csv` (or a few run JSONs) and you’ll have everything needed to continue.

## What this repo is

A RAG engineering sandbox focused on building a trustworthy “Project Copilot” mindset:
- provenance-first
- measurable improvements over time
- robustness to “aggregation” questions (lists, timelines, season summaries)

## Current state (important milestones)

### Ingestion + metadata
- Metadata is parsed at load time from normalized TXT headers.
- Canonical episode identity is stored as `episode_id = SxxExx`.
- Each chunk gets stable chunk identity:
  - `chunk_type` (e.g., `char`)
  - `chunk_index` (monotonic per `source`)

### Indices
- Default index: `db/chroma_db`
- Metadata-enabled index: `db/chroma_db_meta`

### Derived corpus index (separate)

- Derived index: `db/chroma_db_derived`
- Builder: `python derived/build_derived_index.py --persist-dir db/chroma_db_derived --reset`

Most experiments should explicitly use:
- `--persist-dir db/chroma_db_meta`

### Retrieval features
- Similarity search
- MMR (with score backfill)
- Query expansion (LLM-generated alternates) + RRF fusion

### Evaluation & metrics
- Runs are logged as JSON under `experiments/runs/`.
- `experiments/run_eval.py` logs additional diagnostics:
  - context episode/source diversity
  - “citation drift” (cited episode IDs not present in retrieved context)
  - aggregation readiness score (0..100)

- `experiments/summarize_runs.py` converts run logs into flat tables:
  - `experiments/run_metrics.csv|jsonl`
  - `experiments/case_metrics.csv|jsonl`

## Why the system sometimes feels untrustworthy

The primary recurring failure mode is retrieval recall for aggregation questions:
- MMR can diversify away from multiple similar-but-relevant breakup/relationship scenes.
- Query expansion that only paraphrases (without anchors like names/episodes) can amplify mediocre candidates.

This is expected for dialogue-heavy corpora without a derived “knowledge layer.”

## Suggested next build (highest ROI)

Implement index-time enrichment / distillation into derived docs with provenance:
- character bios (per character)
- relationship chronologies (per pair and per character)
- arc summaries (season-by-season)

Store them as `doc_type=derived` and retrieve them first for aggregation questions, then retrieve raw script chunks for quotes.

## Useful commands

Build metadata index:
- `python ingestion/ingestion_pipeline.py --persist-dir db/chroma_db_meta --use-metadata --reset`

Run eval (metadata index):
- `python experiments/run_eval.py --persist-dir db/chroma_db_meta --run-name <name> --search-type similarity --k 12`

Generate summary tables:
- `python experiments/summarize_runs.py`

## Where to look in code

- `ingestion/ingestion_pipeline.py`: chunking + Chroma persistence
- `ingestion/load_documents.py`: metadata extraction (episode_id, doc_type, etc.)
- `experiments/run_eval.py`: eval harness + metrics
- `experiments/summarize_runs.py`: metrics tables for visualization
- `rag/query_expansion.py`, `rag/fusion.py`: multi-query retrieval
