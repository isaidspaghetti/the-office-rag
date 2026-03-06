# Copilot instructions (RAG-1)

## Big picture
- This repo is a RAG sandbox over The Office scripts/summaries. The core loop is: **ingest → run evals (JSON logs) → summarize metrics → (optionally) LLM-judge score runs**.
- There are two corpora:
  - **Script index** (raw scripts + episode summaries) built by [ingestion/ingestion_pipeline.py](../ingestion/ingestion_pipeline.py) into `db/chroma_db*`.
  - **Derived-cards index** (episode/season/topic cards) built from `derived/artifacts/` by [derived/build_derived_cards_index.py](../derived/build_derived_cards_index.py) into `db/chroma_db_derived_cards*`.

## Key entrypoints
- Ingest/Index: [ingestion/ingestion_pipeline.py](../ingestion/ingestion_pipeline.py)
  - Supports `--chunking character|scene|scene_window` and `--use-metadata`.
  - Writes stable metadata per chunk: `episode_id` like `S02E11`, plus `doc_type`, `chunk_type`, `chunk_index`.
- Eval harness: [experiments/run_eval.py](../experiments/run_eval.py)
  - Writes one JSON file per run under `experiments/runs/` (run_id includes UTC timestamp + slugified run_name).
  - Retrieval policies: `script_only`, `derived_only`, `derived_then_script`, `auto`, `blended`.
  - Query expansion + fusion lives in [rag/query_expansion.py](../rag/query_expansion.py) and [rag/fusion.py](../rag/fusion.py) (cache default: `experiments/cache/query_expansion_cache.json`).
- Debug helpers: [retrieval_pipeline.py](../retrieval_pipeline.py) (single-query retrieval + optional query-expansion/RRF) and [experiments/inspect_routing_case.py](../experiments/inspect_routing_case.py) (print routing shortlist + top sources for one case).
- Run summarization: [experiments/summarize_runs.py](../experiments/summarize_runs.py) → `experiments/run_metrics.*` and `experiments/case_metrics.*`.
- Run scoring (LLM judge): [experiments/score_runs.py](../experiments/score_runs.py) (gold: `experiments/gold_answers.json`, output: `experiments/scored_runs/`).
  - Note: [apps/rag_runs_dashboard.py](../apps/rag_runs_dashboard.py) is currently a scoring script (not a Streamlit UI).

## Derived-corpus workflow (map/reduce → index)
- Pipeline order and artifact layout are documented in [derived/README.md](../derived/README.md).
- Preferred driver script: [derived/runbooks/rebuild_seasons_param.sh](../derived/runbooks/rebuild_seasons_param.sh)
  - Runs `derived.summarize_season` → `derived.reduce_season` → `derived.reduce_season_card`.
  - Build prefixes are important (new folders per build); examples: `derived_episodecard_season02_<tag>`.
  - Can optionally build a NEW derived-cards DB (persist dir auto-named like `db/chroma_db_derived_cards_<tag>`).

## Conventions that matter when editing
- **Episode IDs are canonical**: use `SxxEyy` everywhere (ingestion metadata, routing, eval heuristics).
- Prefer **metadata-enabled** script indexes (`python ingestion/ingestion_pipeline.py --use-metadata ...`) so routing can push down `{"episode_id": {"$in": [...]}}` filters; otherwise run_eval falls back to parsing `source` paths.
- Keep Chroma persist dirs separate per experiment/chunking strategy (see `db/chroma_db_scene*`, `db/chroma_db_meta`, etc.).
- Derived cards rely on `metadata.derived_type` in `{episode_card, season_card, topic_card}`; routing filters derived stage-1 to those.

## Common commands (copy/paste)
- Build script index: `python ingestion/ingestion_pipeline.py --persist-dir db/chroma_db_meta --use-metadata --reset`
- Run a routed eval: `python experiments/run_eval.py --retrieval-policy blended --derived-persist-dir db/chroma_db_derived_cards --derived-collection-name derived_cards --run-name my_run --search-type similarity --k 12`
- Run the preset routing sweep: `bash experiments/run_routing_sweep.sh` (also available as a VS Code task in `.vscode/tasks.json`)
- Summarize runs: `python experiments/summarize_runs.py`
- Score runs: `python experiments/score_runs.py --runs-dir experiments/runs --gold experiments/gold_answers.json`

## Safety / gotchas
- Most CLIs call `load_dotenv()` and expect `OPENAI_API_KEY` in the repo-root `.env` (do not commit secrets).
- `--reset` deletes the target persist dir (dev-only); don’t run it on a persist dir you care about.
- `auto` routing only uses derived routing for aggregation-like questions (regex in run_eval); for pinpoint questions it stays script-only.
