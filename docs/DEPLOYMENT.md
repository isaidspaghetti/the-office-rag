# Deployment (Streamlit Community Cloud)

This repo supports a Streamlit Community Cloud deployment with:

- **Reports mode** (offline analytics): reads `experiments/runs/` + `experiments/scored_runs_two_pass/` from the repo.
- **Chat & Debug mode** (live RAG): uses **Qdrant** (two collections) for retrieval and **OpenAI** for embeddings + chat.

App entrypoint: `app.py`.

## What you need

- An OpenAI API key (`OPENAI_API_KEY`).
- A Qdrant instance (Qdrant Cloud or self-hosted) with:
  - `QDRANT_URL`
  - `QDRANT_API_KEY` (if required by your Qdrant)

Recommended collection names:

- Scripts: `office_scripts`
- Derived cards: `office_derived_cards`

## 1) Index Qdrant (run locally)

Streamlit Community Cloud won’t use your local Chroma DBs. Upload vectors to Qdrant ahead of time.

1) Install deps:

```bash
pip install -r requirements.txt
```

2) Provide env vars (or a repo-root `.env`):

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (optional)

3) Index both corpora (drop + recreate both collections):

```bash
python qdrant_index.py --mode both --recreate --use-metadata-headers
```

Optional flags:

- `--scripts-collection office_scripts`
- `--derived-collection office_derived_cards`
- `--chunk-size 1000 --chunk-overlap 150`

Notes:

- Scripts are chunked from `ingestion/normalized_docs_txt/`.
- Derived cards are loaded from `derived/artifacts/` (auto-detects build prefixes).
- Use `--dry-run` to validate discovery/counts without requiring Qdrant/OpenAI creds.

## 2) Ensure Reports artifacts exist (commit/push)

Reports mode expects these repo artifacts to exist in the deployed GitHub repo:

- Run logs: `experiments/runs/*.json`
- Two-pass scores: `experiments/scored_runs_two_pass/*.scored.json`

Do NOT commit local Chroma persist dirs (e.g., `db/`). Those are machine-specific and unnecessary for Streamlit Cloud.

Generate them locally:

```bash
python experiments/run_eval.py --run-name demo_run --search-type similarity --k 12

python experiments/score_runs.py \
  --runs-dir experiments/runs \
  --gold experiments/gold_answers.json \
  --judge-mode two_pass \
  --out-dir experiments/scored_runs_two_pass
```

Then commit/push `experiments/runs/` and `experiments/scored_runs_two_pass/`.

## 3) Deploy to Streamlit Community Cloud

1) Push your repo to GitHub.

2) In Streamlit Community Cloud:

- **Main file path**: `app.py`
- Dependencies: `requirements.txt`

3) Set secrets (App → Settings → Secrets).

Required:

- `OPENAI_API_KEY`
- `QDRANT_URL`

Recommended:

- `QDRANT_API_KEY`
- `QDRANT_SCRIPTS_COLLECTION=office_scripts`
- `QDRANT_DERIVED_COLLECTION=office_derived_cards`

A template exists at `.streamlit/secrets.toml.example`.

## Smoke tests

- Reports: open `/?mode=summary`
  - Should render even if Qdrant secrets are missing.
  - If it shows a setup panel, ensure the repo includes `experiments/runs/` and `experiments/scored_runs_two_pass/`.

- Chat & Debug: open `/?mode=chat_debug`
  - Pick Retrieval backend **Qdrant (cloud)**.
  - Ask a question; you should see retrieved chunks + an answer.

## Troubleshooting

- Dependency errors on deploy: confirm `requirements.txt` is at repo root; then read Streamlit build logs for the missing package.
- Empty retrieval: verify both Qdrant collections exist, contain points, and the collection names match the secrets exactly.
- Missing secrets UI errors (`missing_qdrant_url`, `missing_openai_api_key`): re-check Streamlit Cloud Secrets.
