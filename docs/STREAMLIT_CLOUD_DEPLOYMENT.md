# Streamlit Community Cloud deployment (Qdrant + OpenAI)

This repo is designed to deploy to **Streamlit Community Cloud** with:

- **OpenAI** for embeddings + chat
- **Qdrant** for retrieval
- Two Qdrant collections:
  - `scripts` (script + summary chunks)
  - `derived` (derived cards: episode/season/topic)

The app entrypoint is **`app.py`**.

---

## 1) What gets deployed

Streamlit Cloud runs the Streamlit app from your GitHub repo. It does **not** use your local `venv/` or local Chroma DBs.

Recommended to commit/push:

- `app.py` (single Streamlit entrypoint)
- `rag_experiment_story_dashboard.py` (Summary mode)
- `apps/rag_runs_dashboard.py` (Chat & Debug mode)
- `experiments/runs/` (run logs)
- `experiments/scored_runs_two_pass/` (canonical two-pass judge outputs)
- `experiments/gold_answers.json` (optional, but improves coverage diagnostics)

The Summary dashboard will show an explicit setup panel if these artifacts are missing.

---

## 2) Streamlit Cloud settings

In Streamlit Cloud:

- App file: `app.py`
- Python version: whatever Streamlit Cloud provides (this repo expects a modern 3.x)
- Dependencies: `requirements.txt`

---

## 3) Secrets (Streamlit Cloud)

Set these in **Streamlit Cloud → App → Settings → Secrets**.

### Required

- `OPENAI_API_KEY`
- `QDRANT_URL`

### Recommended

- `QDRANT_API_KEY` (required for Qdrant Cloud)

### Collection names (two collections)

- `QDRANT_SCRIPTS_COLLECTION`
  - Default used by the app: `office_scripts`
- `QDRANT_DERIVED_COLLECTION`
  - Default used by the app: `office_derived_cards`

Example `secrets.toml`:

```toml
OPENAI_API_KEY = "..."
QDRANT_URL = "https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "..."
QDRANT_SCRIPTS_COLLECTION = "office_scripts"
QDRANT_DERIVED_COLLECTION = "office_derived_cards"
```

---

## 4) Expected behavior / smoke test

### Summary mode (no live retrieval)

Open:

- `/?mode=summary`

This mode is **fully offline** once run logs + two-pass scored files are present. It should work even if Qdrant secrets are missing.

If the page shows a “Required experiment artifacts not found” panel:

- Ensure `experiments/runs/` exists in the deployed repo
- Ensure `experiments/scored_runs_two_pass/` exists and contains `*.scored.json` with `scoring_meta.judge_mode == "two_pass"`

### Chat & Debug mode (live Qdrant retrieval)

Open:

- `/?mode=chat_debug`

In the left sidebar (Chat Playground), pick:

- Retrieval backend: **Qdrant (cloud)**
- Collections: scripts + derived

Ask a question; you should see retrieved chunks and an answer.

If you get “missing_qdrant_url” or “missing_openai_api_key”, check the Secrets configuration.

---

## 5) How to build the required artifacts (local)

You usually generate logs locally, then push them for Cloud viewing.

### Generate run logs

```bash
python experiments/run_eval.py --run-name demo_run --search-type similarity --k 12
```

### Score with the strict two-pass judge (canonical outputs)

```bash
python experiments/score_runs.py \
  --runs-dir experiments/runs \
  --gold experiments/gold_answers.json \
  --judge-mode two_pass \
  --out-dir experiments/scored_runs_two_pass
```

Notes:
- The dashboards **only** consume the two-pass outputs in `experiments/scored_runs_two_pass/`.
- If you scored previously in single-pass mode, regenerate with `--judge-mode two_pass`.
