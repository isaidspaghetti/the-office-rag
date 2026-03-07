# Streamlit Community Cloud deploy (OpenAI + Qdrant)

This repo supports a Streamlit Community Cloud deployment with:

- **Summary mode** (offline analytics): reads `experiments/runs/` + `experiments/scored_runs_two_pass/` from the repo.
- **Chat & Debug mode** (live RAG): queries **Qdrant** (two collections) and uses **OpenAI** for embeddings + answering.

App entrypoint: `app.py`.

## Quick checklist

### A) Index Qdrant (run locally)

Streamlit Cloud does not run your local Chroma DBs. You must upload vectors to Qdrant ahead of time.

- Follow the Qdrant indexing guide: `docs/DEPLOY_STREAMLIT_CLOUD_QDRANT.md`
- One-command option (drops + recreates both collections):

```bash
python qdrant_index.py --mode both --recreate --use-metadata-headers
```

Recommended collection names:
- `office_scripts` (scripts + summaries)
- `office_derived_cards` (episode/season/topic cards)

### B) Generate run logs + two-pass scores (run locally, then commit/push)

Summary mode expects:
- run logs: `experiments/runs/*.json`
- strict two-pass scores: `experiments/scored_runs_two_pass/*.scored.json`

Commands:

```bash
python experiments/run_eval.py --run-name demo_run --search-type similarity --k 12

python experiments/score_runs.py \
  --runs-dir experiments/runs \
  --gold experiments/gold_answers.json \
  --judge-mode two_pass \
  --out-dir experiments/scored_runs_two_pass
```

### C) Deploy to Streamlit Community Cloud

1) Push your repo to GitHub.

2) In Streamlit Community Cloud:
- **Main file path**: `app.py`
- Dependencies: `requirements.txt`

3) Set secrets (App → Settings → Secrets).

Required:
- `OPENAI_API_KEY`
- `QDRANT_URL`

Recommended:
- `QDRANT_API_KEY` (required for Qdrant Cloud)

Collection names:
- `QDRANT_SCRIPTS_COLLECTION=office_scripts`
- `QDRANT_DERIVED_COLLECTION=office_derived_cards`

A template exists at `.streamlit/secrets.toml.example`.

## Smoke tests

- Summary: open `/?mode=summary`
  - Should render even if Qdrant secrets are missing.
  - If it shows a setup panel, verify the repo includes `experiments/runs/` and `experiments/scored_runs_two_pass/`.

- Chat & Debug: open `/?mode=chat_debug`
  - Set Retrieval backend to **Qdrant (cloud)**.
  - Ask a question; you should see retrieved chunks + an answer.

## More detail

- Streamlit Cloud deployment details + artifact expectations: `docs/STREAMLIT_CLOUD_DEPLOYMENT.md`
- Qdrant indexing steps and flags: `docs/DEPLOY_STREAMLIT_CLOUD_QDRANT.md`
