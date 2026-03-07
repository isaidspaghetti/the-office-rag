# Copilot instructions (RAG-1)

## What this repo is
- A RAG sandbox over **The Office** scripts + episode summaries.
- Primary workflow: **ingest/index → run evals (JSON logs) → summarize metrics → score runs (two-pass judge)**.

## Core components (read these first)
- Script index (Chroma): [ingestion/ingestion_pipeline.py](../ingestion/ingestion_pipeline.py) builds `db/chroma_db*` from `ingestion/normalized_docs_txt/`.
- Derived-cards index (Chroma): [derived/build_derived_cards_index.py](../derived/build_derived_cards_index.py) embeds `derived/artifacts/**` into `db/chroma_db_derived_cards*` (metadata `derived_type` distinguishes card kinds).
- Eval harness + run logs: [experiments/run_eval.py](../experiments/run_eval.py) writes one run JSON per invocation to `experiments/runs/`.
- Run summarization: [experiments/summarize_runs.py](../experiments/summarize_runs.py) → `experiments/run_metrics.{csv,jsonl}` + `experiments/case_metrics.{csv,jsonl}`.
- Run scoring (canonical): [experiments/score_runs.py](../experiments/score_runs.py) + `experiments/gold_answers.json` → `experiments/scored_runs_two_pass/`.

## Repo conventions that matter
- **Canonical episode IDs are `SxxEyy`**. Ingestion metadata + eval routing/grouping assume this (see `episode_id` in [ingestion/load_documents.py](../ingestion/load_documents.py)).
- Prefer **metadata-enabled ingestion** (`--use-metadata`) so eval routing can filter by `{"episode_id": {"$in": [...]}}`; otherwise eval falls back to parsing `source` paths.
- Keep Chroma persist dirs **separate per experiment** (don’t mix chunking strategies in one persist dir).
- Derived artifacts are build-prefixed folders under `derived/artifacts/`; the driver is [derived/runbooks/rebuild_seasons_param.sh](../derived/runbooks/rebuild_seasons_param.sh).

## Commands agents should reach for
- Build script index (dev reset): `python ingestion/ingestion_pipeline.py --persist-dir db/chroma_db_meta --use-metadata --reset`
- Build derived-cards index: `python derived/build_derived_cards_index.py --persist-dir db/chroma_db_derived_cards --reset`
- Run eval (current policies: `script_only|derived_only|derived_then_script`): `python experiments/run_eval.py --persist-dir db/chroma_db_meta --retrieval-policy derived_then_script --k 12 --run-name routed_k12`
- Run eval with query expansion + RRF fusion: `python experiments/run_eval.py --query-expansion --expand-n 5 --fusion rrf --rrf-k0 60` (cache: `experiments/cache/query_expansion_cache.json`)
- Summarize runs: `python experiments/summarize_runs.py`
- Score runs (two-pass): `python experiments/score_runs.py --runs-dir experiments/runs --gold experiments/gold_answers.json`
- Debug retrieval quickly: [retrieval_pipeline.py](../retrieval_pipeline.py) and [experiments/inspect_routing_case.py](../experiments/inspect_routing_case.py)

## Gotchas / current repo state
- There is currently no pinned `requirements.txt`; typical deps are `langchain-openai`, `langchain-core`, `langchain-chroma`, `chromadb`, `python-dotenv`.
- Most CLIs call `load_dotenv()`; set `OPENAI_API_KEY` in repo-root `.env`.
- `--reset` deletes the persist dir.
- Some “dashboard/deploy” docs referenced in README are not present, and `app.py` / `qdrant_index.py` are currently empty stubs—treat the CLI + `experiments/` artifacts as the source of truth.
