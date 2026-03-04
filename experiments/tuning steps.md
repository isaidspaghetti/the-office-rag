# NOTES
# RAG Goals
- This is a safe sandbox to practice engineering moves and gain familiarity with AI Engineering techniques needed for a "Project Copilot." A Project Copilot is an AI resource that a mid-size company could trust because it (1) cites sources, (2) handles change-over-time, (3) separates facts from synthesis, (4) is measurable, and (5) has operational guardrails (privacy, access control, monitoring).

# This POC demo is also a learning resource
- While only a POC level of polish is present, this project exhibits important AI Engineering practices including Data Modeling & Provenance, testing workflows,  chunking and tuning strategies, 

# Traits Required for a Project Copilot
- Verifiability: every non-trivial claim links to a source chunk
- Coverage awareness: while a 0 temperature is obvious, the system should also 
be aware when they are missing information instead of guessing
- Time & Drift handling: answers are scoped to "as of" time. Conflicts are surfaced, not smoothed over
- Access and Privacy: retrieval aspects user/project permissions and redactions rules
- Measurability: you can track retrieval recall, citation quality, contradiction rate, and "unknown" Correctness over time.


# Metrics we now log (for charts)
## Run-level (in JSON summary)
- avg_context_docs, avg_context_chars
- avg_distinct_episodes_in_context
- avg_distinct_sources_in_context
- avg_top_episode_share_in_context (how concentrated context is in one episode)
- avg_episode_entropy_norm_in_context (episode diversity 0..1)
- episode_citation_rate
- quote_in_context_rate
- grounding_failure_rate (heuristic)
- retrieval_empty_rate
- retrieval_failure_rate (only meaningful for cases with expected episode IDs)
- idk_rate
- avg_cited_episode_ids_not_in_context ("citation drift")
- avg_aggregation_readiness_score (0..100)
- avg_aggregation_readiness_score_agg_questions (0..100, only for aggregation-like questions)

## Case-level (per question)
- diagnostics.context_diversity.* (episode/source counts + dup rates + distributions)
- diagnostics.cited_episode_ids_not_in_context_count
- diagnostics.aggregation.readiness_score_0_100


# Turn runs into tables for visuals
python experiments/summarize_runs.py

### outputs
### - experiments/run_metrics.csv + experiments/run_metrics.jsonl
### - experiments/case_metrics.csv + experiments/case_metrics.jsonl



# Trial prompt steps 
- The followign prompts match run results in the ./runs folder.

# Baseline

python experiments/run_eval.py \
  --run-name baseline_similarity_k3 \
  --search-type similarity \
  --k 3


# Increase k recall 


python experiments/run_eval.py \
  --run-name t1_similarity_k12 \
  --search-type similarity \
  --k 12 \
  --notes "Tuning: increase k to 12 (same similarity)"

# increase MMR (Maximum Marginal Relevance - diversity of results)
## fetch-k is increased at this step to perform MMR on the fetch-k results
## lambda-mult is the 'relevance vs diversity' knob. 1=pure similarity <-> 0 = max diversity

python experiments/run_eval.py \
  --run-name t2_mmr_k12_fetch40_l07_fixed \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --notes "Tuning: MMR k=12 fetch_k=40 lambda=0.7 (fixed mmr impl)"


# Scene level chunking strategy:
## Split doces into scenes (./ingestion/chunk_documents)

python experiments/run_eval.py \
  --persist-dir db/chroma_db_scene \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --run-name scene_chunks_mmr_k12

  ### results: increased precision, decreased narrative cohesion
  ### 72% more vectors increases chance for semantic adjacency but narrative irrelevance.
  ### Common in over-fragmented RAGs


# Scene level but lower MMR - retain more continuity across scenes
python experiments/run_eval.py \
  --persist-dir db/chroma_db_scene \
  --run-name scene_chunks_similarity_k12\
  --search-type similarity \
  --k 12 \
  --notes "Scene Chunks, Removed MMR"


  python experiments/run_eval.py \
  --persist-dir db/chroma_db_scene \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --run-name scene_chunks_mmr_k12


# Back to MMR 
python experiments/run_eval.py \
  --persist-dir db/chroma_db_scene \
  --search-type mmr \
  --k 12 \
  --fetch-k 80 \
  --lambda-mult 0.9 \
  --run-name scene_chunks_mmr_k12_fetch80_lambda09

# Chunk scenes but with a +1 -1 overlap window : fails - hits context token limits
python experiments/run_eval.py \
  --persist-dir db/chroma_db_scene_w1 \
  --search-type mmr \
  --k 12 \
  --fetch-k 80 \
  --lambda-mult 0.9 \
  --run-name scene_chunks_mmr_k12_fetch80_lambda09_w1

# Query Expansion - generate multiple semantically-aligned search queries that cover different phrasings / facets, then retrieve more robustly.
## use Fusion: deduplicate the same chunk, aggregate evidence across chunks retrieved, and rank as RRF (reciprocal rank fusion)

python experiments/run_eval.py \
  --run-name qe_rrf_similarity_k12 \
  --search-type similarity \
  --k 12 \
  --query-expansion \
  --expand-n 5 \
  --expand-model gpt-4.1-nano \
  --k-per-query 6 \
  --fusion rrf \
  --rrf-k0 60 \
  --notes "Query expansion + RRF fusion (retrieval-only)"


# now use with mmr
python experiments/run_eval.py \
  --run-name qe_rrf_mmr_k12 \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --query-expansion \
  --expand-n 5 \
  --expand-model gpt-4.1-nano \
  --k-per-query 6 \
  --fusion rrf \
  --rrf-k0 60 \
  --notes "Query expansion + RRF fusion + MMR"

### best results so far. gets donna ending right and holly ending mostly right, misses carol and helene entirely
### gets teapot right



<!-- Most vector databases use ANN (Approximate Nearest Neighbor) algorithms like:

HNSW

IVF

ScaNN

PQ

This means the search does not necessarily scan the entire corpus.

Instead it quickly finds a high-quality approximation of the nearest vectors.

(caption chunks) don't naturally contain the interpretive glue (arcs, motifs, importance)

 -->

 # Add metadata 
 ## a step I coudl have done first, but this is really in par tto make th enxt step more effective, and is a good practice in general.

 ### Step 1: choose what to metadata:
 #### source, doc_type (script or summary), episode_id, season, episode, title
 #### chunk_type, chunk_index, chunk chars

# Query expansion Hybrid search (vector + keyword) Reranking RRF (if combining multiple retrievers) Agent-based chunking (only if needed)


python experiments/run_eval.py \
  --persist-dir db/chroma_db_meta \
  --run-name qe_rrf_mmr_k12 \
  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --query-expansion \
  --expand-n 5 \
  --expand-model gpt-4.1-nano \
  --k-per-query 6 \
  --fusion rrf \
  --rrf-k0 60 \
  --notes "Query expansion + RRF fusion + MMR + metadata check"


  # Why did we see the girlfriends question fail with mmr?

  ### MMR is a bad match for “list all X” questions. For aggregation questions, the “right” evidence chunks are often semantically similar (relationship/breakup dialogue). MMR will often pick one relevant cluster and  and then “diversify” into unrelated-but-different chunks, tanking recall.


# Add A Derived Corpus:
# Context enrichment and distillation (aka document augmentation)
#

## Step: build a separate derived corpus index (summaries-only to start)

python derived/build_derived_index.py \
  --persist-dir db/chroma_db_derived \
  --reset

### optional: add LLM-generated reference cards (character bios + relationship timelines)
python derived/build_derived_index.py \
  --persist-dir db/chroma_db_derived \
  --reset \
  --generate-character-bios \
  --generate-relationship-timelines
## We 



Episode summary (already have, but can enrich)
Character bios (per character)
Relationship timelines (per pair / per character)
Character arc summaries (season-by-season)
Running gags / motifs / “facts” with citations

Two-stage retrieval at answer time:
Retrieve derived docs first to get the “map” (names, timelines, which episodes matter).
Retrieve raw script chunks second for quotes / grounding and to avoid “LLM-made-up” details.

Hallucination risk shifts earlier: you can accidentally “bake in” wrong facts at index time; you need spot checks + evals.


