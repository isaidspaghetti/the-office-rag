# RAG-1 (Office RAG sandbox)

A fast-moving RAG engineering sandbox over a dialogue-heavy corpus (The Office scripts + episode summaries). The goal is to practice the engineering moves needed for a trustworthy “Project Copilot”:

- Verifiable answers with source provenance
- Awareness of missing coverage (avoid confident guessing)
- Change-over-time / drift handling ("as-of" reasoning)
- Measurable quality (runs + metrics + trend charts)
- Operational guardrails (privacy, permissions, monitoring)

This repo intentionally keeps the system simple enough to iterate quickly, while still logging the right artifacts to learn from.

## Quickstart

### 1) Environment

- Python + a virtualenv.
- Add an OpenAI key to `.env` at the repo root:
  - `OPENAI_API_KEY=...`

Dependencies are intentionally lightweight and revolve around:
- `langchain-openai`, `langchain-chroma`, `langchain-core`, `chromadb`, `python-dotenv`

### 2) Build an index (Chroma)

The ingestion CLI reads normalized docs from `ingestion/normalized_docs_txt/` and writes a persisted Chroma index.

- Build the metadata-enabled index (recommended):

  `python ingestion/ingestion_pipeline.py --persist-dir db/chroma_db_meta --use-metadata --reset`

Notes:
- `--reset` deletes the target persist dir (dev-only).
- Metadata writes canonical `episode_id` like `S02E11`, plus chunk identity (`chunk_type`, `chunk_index`).

### 2b) Build a separate derived corpus index (summaries + derived cards)

This creates a *separate* Chroma index intended for broad/aggregation questions and routing.

- Build a summaries-only derived index:

  `python derived/build_derived_index.py --persist-dir db/chroma_db_derived --reset`

- Optionally add a few LLM-generated reference cards (character bios + relationship timelines):

  `python derived/build_derived_index.py --persist-dir db/chroma_db_derived --reset --generate-character-bios --generate-relationship-timelines`

### 3) Run evals (writes JSON logs)

Eval reads `experiments/test_queries.json` and writes one JSON file per run to `experiments/runs/`.

- Baseline:

  `python experiments/run_eval.py --run-name baseline_similarity_k3 --search-type similarity --k 3`

- Similarity w/ higher recall:

  `python experiments/run_eval.py --run-name t1_similarity_k12 --search-type similarity --k 12`

- MMR (diverse results) + tuned params:

  `python experiments/run_eval.py --run-name t2_mmr_k12_fetch40_l07_fixed --search-type mmr --k 12 --fetch-k 40 --lambda-mult 0.7`

- Query expansion + RRF fusion:

  `python experiments/run_eval.py --run-name qe_rrf_similarity_k12 --search-type similarity --k 12 --query-expansion --expand-n 5 --expand-model gpt-4.1-nano --k-per-query 6 --fusion rrf --rrf-k0 60`

If you want runs to use the metadata index, pass:
- `--persist-dir db/chroma_db_meta`

### 4) Turn runs into chartable tables

- Generate flat metrics tables (CSV + JSONL):

  `python experiments/summarize_runs.py`

Outputs:
- `experiments/run_metrics.csv` and `experiments/run_metrics.jsonl`
- `experiments/case_metrics.csv` and `experiments/case_metrics.jsonl`

These are designed to be easy inputs for plotting (Excel/Sheets) or for an LLM to generate visuals.

## Repo layout

- `ingestion/`
  - `ingestion_pipeline.py`: builds persisted Chroma DBs
  - `load_documents.py`: loads + enriches metadata from normalized TXT headers
  - `normalize_docs.py`: produces `normalized_docs_txt/` (already present in this repo)
- `rag/`
  - query expansion + fusion utilities
- `experiments/`
  - `run_eval.py`: eval harness that writes run logs with diagnostics + heuristics
  - `summarize_runs.py`: flattens run logs to tables
  - `runs/`: historical run JSONs
- `db/`
  - Chroma persist dirs (e.g., `db/chroma_db_meta`)

## What we measure (high-level)

The eval harness logs:
- Retrieval metrics: latency, score stats, empty-rate
- Grounding heuristics: quote-in-context rate, cited-episode-not-in-context rate
- Context diversity: distinct episodes/sources, concentration (top episode share), normalized entropy
- Aggregation readiness score: heuristic 0..100 indicating whether the retrieved context supports “list/timeline/summary” questions

For more detail, see `experiments/tuning steps.md`.

## Docs

- `docs/RESUME.md`: snapshot for starting a new ChatGPT thread
- `docs/GLOSSARY.md`: terminology handout for lunch-and-learn
- `docs/DERIVED_CHECKLIST_AND_SCHEMAS.md`: derived-corpus build checklist, schemas, provenance contract

## Known gotchas / lessons learned

- MMR can hurt “list all X” aggregation questions by trading recall for diversity.
- Query expansion only helps if expanded queries add discriminative anchors (entities, events); pure paraphrases can amplify mediocre candidates.
- Index-time enrichment (derived docs) becomes important once questions require multi-episode synthesis.

## Next step (recommended)

Start producing derived artifacts (index-time enrichment) with provenance:
- character bios
- relationship timelines
- season/arc summaries
- motif/gag trackers

Then answer-time becomes a two-stage flow:
1) retrieve derived docs to build the "map"
2) retrieve raw script chunks for quotes/citations

See `docs/RESUME.md` for a “where we left off” snapshot.
