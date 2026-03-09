# Deploy (Streamlit Community Cloud) + Qdrant + OpenAI

This repo is designed so the hosted Streamlit app is **UI + retrieval + LLM calls**, while indexing is done **offline** and uploaded to Qdrant.

## What you deploy

- Streamlit entrypoint: [app.py](../app.py)
  - “Summary” mode: [rag_experiment_story_dashboard.py](../rag_experiment_story_dashboard.py)
  - “Chat & Debug” mode: [apps/rag_runs_dashboard.py](../apps/rag_runs_dashboard.py)

The **Chat Playground** uses:
- Qdrant (two collections) for retrieval
- OpenAI for embeddings + answering

## 1) Create Qdrant (two collections)

You need two collections:
- scripts collection (script + summary chunks)
- derived collection (episode/season/topic cards)

Recommended names:
- `office_scripts`
- `office_derived_cards`

## 2) Index data into Qdrant (run locally)

This is a one-time (or occasional) batch job. Run it locally where you have `OPENAI_API_KEY`.

1) Install deps:
- `pip install -r requirements.txt`

2) Set env vars (or use `.env`):
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (if required)

3) Run the indexer:
- Index both corpora (drops & recreates both collections):
  - `python qdrant_index.py --mode both --recreate --use-metadata-headers`

Useful flags:
- `--scripts-collection office_scripts`
- `--derived-collection office_derived_cards`
- `--chunk-size 1000 --chunk-overlap 150`
- `--seasons 1,2,3,4` (limits derived cards)

Notes:
- Scripts are chunked from `ingestion/normalized_docs_txt/`.
- Derived cards are read from `derived/artifacts/` (auto-detects build prefixes).

## 3) Deploy to Streamlit Community Cloud

1) Push the repo to GitHub.

2) In Streamlit Community Cloud:
- **New app** → select your repo/branch
- **Main file path**: `app.py`

3) Add secrets in the Streamlit UI (App → Settings → Secrets):

Required:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` (if needed)
- `QDRANT_SCRIPTS_COLLECTION=office_scripts`
- `QDRANT_DERIVED_COLLECTION=office_derived_cards`

You can copy the template from: [.streamlit/secrets.toml.example](../.streamlit/secrets.toml.example)

## 4) Verify in the app

- Open **Chat Playground**
  - Retrieval backend should default to **Qdrant (cloud)**
  - Retrieval policy: `script_only`, `derived_only`, or `hybrid`

If Chat Playground says `missing_qdrant_url`, your secrets are not set.

## Troubleshooting

- If you see dependency errors on deploy:
  - confirm `requirements.txt` is at repo root
  - check Streamlit build logs for the missing package name

- If retrieval returns empty results:
  - verify both Qdrant collections exist and contain points
  - confirm the collection names match secrets exactly

- If OpenAI calls fail:
  - verify `OPENAI_API_KEY` is set in Streamlit secrets
