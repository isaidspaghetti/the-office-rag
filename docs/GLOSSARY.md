# Glossary (RAG → trustworthy Project Copilot)

This is a quick handout mapping common RAG/LLM knowledge-base terms to practical meaning, failure modes, and where they show up in this repo.

## Core terms

### RAG (Retrieval-Augmented Generation)
**Meaning:** Use a retriever (vector search, keyword search, or both) to pull relevant context, then ask an LLM to answer grounded in that context.

**Common failure modes:**
- Retrieval miss (recall): you didn’t fetch the evidence.
- Context overload: you fetched too much noise.
- Grounding drift: model cites things not in context.

**In this repo:**
- Retrieval + eval harness: `experiments/run_eval.py`
- Chroma vector DB: `db/*`

### Vector store / embeddings
**Meaning:** Convert text chunks into vectors; do approximate nearest neighbor search.

**In this repo:**
- Chroma persistent index: `db/chroma_db_meta`

### Chunking
**Meaning:** Split documents into pieces for retrieval.

**Tradeoffs:**
- Too big → hard to retrieve the right portion, expensive context.
- Too small → loses narrative glue, increases noise and redundancy.

**In this repo:**
- Ingest/chunk: `ingestion/ingestion_pipeline.py`

### Metadata (provenance)
**Meaning:** Structured fields attached to each chunk/doc (episode_id, doc_type, source, chunk_index) so you can filter, group, cite, and debug.

**Why it matters:** Trustworthy systems need stable IDs and traceability.

**In this repo:**
- Canonical `episode_id = SxxExx`, `chunk_index`, `chunk_type` stored at ingestion.
- Parsing/enrichment: `ingestion/load_documents.py`

## Retrieval patterns

### Similarity search
**Meaning:** Top-k by vector similarity.

**When it’s good:** Lookup questions, direct matches.

### MMR (Maximum Marginal Relevance)
**Meaning:** Trade relevance for diversity.

**When it’s good:** Avoids near-duplicate context.

**When it’s bad:** Aggregation questions like “list all X” where many relevant chunks are *similar*; MMR can diversify away from important evidence.

**In this repo:**
- `experiments/run_eval.py` supports `--search-type mmr`.

### Query expansion
**Meaning:** Ask an LLM to rewrite a user question into multiple search queries; retrieve for each; fuse results.

**Failure mode:** If expansions are only paraphrases (no new anchors like names/events), fusion can amplify mediocre candidates.

**In this repo:**
- `rag/query_expansion.py` + `rag/fusion.py` (RRF)

### RRF (Reciprocal Rank Fusion)
**Meaning:** Combine multiple ranked lists into one robust ranking.

**Best use:** Combining diverse retrievers or diverse query formulations.

## Evaluation terms

### Retrieval failure vs grounding failure
**Retrieval failure:** You didn’t retrieve the needed evidence.

**Grounding failure:** The model says things not supported by the retrieved context.

**In this repo:**
- Logged per case in run JSONs: `labels.*` + `diagnostics.*`

### Context diversity metrics
**Meaning:** Whether the retrieved context covers multiple episodes/sources, or is redundant.

**In this repo:**
- `diagnostics.context_diversity.*` and run-level averages in `summary.*`.

### Aggregation readiness score
**Meaning:** A simple 0..100 heuristic estimating whether the retrieved context supports multi-episode “list/timeline/summary” questions.

**In this repo:**
- Per case: `diagnostics.aggregation.readiness_score_0_100`
- Per run: `summary.avg_aggregation_readiness_score*`

### Citation drift
**Meaning:** The answer cites episodes not present in retrieved context.

**In this repo:**
- `diagnostics.cited_episode_ids_not_in_context_count`

## Index-time enrichment (the “knowledge layer”)

### Index-time enrichment / document augmentation
**Meaning:** Run an LLM offline to create derived artifacts (summaries, bios, timelines) from raw chunks, with citations back to source.

**Why it helps:** Aggregation becomes easy: retrieve 2–5 dense derived docs instead of 30 noisy dialogue chunks.

**Gotcha:** Hallucination risk shifts earlier; you must preserve provenance and add spot checks.

### Hierarchical summarization (map-reduce)
**Meaning:** Summarize small units → merge summaries into bigger units (episode→season→series).

## Knowledge base & graph terms

### Structured extraction
**Meaning:** Convert text into validated JSON objects (entities, relations, events), each linked to supporting evidence.

**Example objects:** `Person`, `Relationship`, `Event`, `Quote`.

### Entity resolution
**Meaning:** Deduplicate/merge entities across sources (aliases, spelling, nicknames).

### Knowledge graph (KG)
**Meaning:** Store extracted entities and relations as nodes/edges, often with time and provenance.

**Why graphs help:**
- Better multi-hop reasoning (A→B→C) and “show me the chain of evidence.”
- Better change-over-time (“as of date/episode”) by time-indexing edges.

### Temporal graph / “as-of” reasoning
**Meaning:** Facts evolve; queries should specify time (“as of Q3 2025”) or the system should surface changes/conflicts.

## Production trust topics (for real company data)

### Permissions-aware retrieval
**Meaning:** Retriever must filter by user/project access rights.

### PII / secrets handling
**Meaning:** Redact at ingestion and/or retrieval; avoid logging sensitive payloads.

### Monitoring
**Meaning:** Track retrieval quality, grounding failures, drift, latency/cost; alert on regressions.

## “Where do I get chartable outputs?”

- `python experiments/summarize_runs.py`
  - Writes: `experiments/run_metrics.csv|jsonl` and `experiments/case_metrics.csv|jsonl`

These are good inputs for:
- Google Sheets/Excel charts
- A plotting notebook
- LLM-generated visuals and narrative summaries
