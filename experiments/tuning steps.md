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


# Add A Derived Corpus (index-time enrichment)
- Our db is sufficiently answering simple questions, quotes, pinpoint facts, and specific episode questions. It struggles with broader or more complex questions require aggregation (list all, timeline, how does x change), because evidence is scattered and semantically repetitive. 



## Step: build a separate derived corpus index (summaries-only to start)
Used a map-reduce (pyramid) pattern to chunk episodes for summary requets, forcing temp 0, flagging uncertainties to be carried into next reducer level for resolution.
Provenance was difficult here - had to add more indexes to retain evidence


python derived/build_derived_index.py \
  --persist-dir db/chroma_db_derived \
  --reset

Season 5 example (update season/episode counts as needed):

1. Generate *segments*for a season
cd /Users/tg/Developer/RAG-1
source venv/bin/activate
export OPENAI_API_KEY="..."   # if not already in your shell/.env

for i in $(seq -w 1 22); do \
  ep="S09E${i}"; \
  python -m derived.segment_episode \
    --docs-dir ingestion/normalized_docs_txt \
    --episode-id "$ep" \
    --target-tokens 1800 \
    --min-tokens 400 \
    --print-n 0 \
    --out "derived/artifacts/segments/${ep}.json"; \
done

2. Run segment *summaries(map)* of a season (LLM)

python -m derived.summarize_season \
  --season 9 \
  --docs-dir ingestion/normalized_docs_txt \
  --segments-root derived/segments \
  --out-root derived/artifacts \
  --build-prefix derived_segsummary_season09_nano_2026-03-04 \
  --llm-model gpt-4.1-nano

3.  *reduce* season 
python -m derived.reduce_season \
  --season 9 \
  --docs-dir ingestion/normalized_docs_txt \
  --out-root derived/artifacts \
  --segments-root derived/artifacts/segments \
  --segment-summaries-root derived/artifacts \
  --segments-build-prefix derived_segsummary_season09_nano_2026-03-04 \
  --llm-model gpt-4.1-nano \
  --build-prefix derived_episodecard_season09_nano_2026-03-04

4. Create card

python -m derived.reduce_season_card --season 9 --episode-cards-build-prefix derived_episodecard_season09_nano_2026-03-04 --build-prefix derived_seasoncard_season09_nano_2026-03-04 --llm-model gpt-4.1-nano


The total cost to create the derived data was about $.90 using 4.1 nano

## Index the derived data

Episode summary (already have, but can enrich)
Character bios (per character)
Relationship timelines (per pair / per character)
Character arc summaries (season-by-season)
Running gags / motifs / “facts” with citations


# Routing
Two-stage retrieval at answer time:
Retrieve derived docs first to get the “map” (names, timelines, which episodes matter).
Retrieve raw script chunks second for quotes / grounding and to avoid “LLM-made-up” details.

Tier 1: derived data
Tier 2: dialogue level grounding


# Routing sweep (re-run tuning with derived routing)
experiments/run_routing_sweep.sh

This reruns the same tuning configurations you already tested, but with the new `derived_then_script` routing mode:
- Stage 1: query derived cards in `db/chroma_db_derived_cards` (collection `derived_cards`) to shortlist episode IDs
- Stage 2: retrieve from your chosen script index (e.g., `db/chroma_db`, `db/chroma_db_scene`, etc.) constrained to that episode shortlist

Run the sweep:

```bash
chmod +x experiments/run_routing_sweep.sh
./experiments/run_routing_sweep.sh
```

Outputs: new JSON logs under `experiments/runs/` with run names prefixed `topiccards_blended_` (see `RUN_TAG` in `experiments/run_routing_sweep.sh`).

## Routing results
The first attempt used hard filtering for routing with a regex to try and udnersatnd questions, then run everything through distilled -> scripts and we had minimal to worse answers.
A couple new girlfriends showed up, but we lost others in everything. 

Next approach was a blended retreival policy + intent detector + reingesting main script index ith metadata. 

## Next routing steps:
### fix metadata 
Fixed a couple metadata errors lost in ingestion. Mild improvement.
### Blended routing strategy
notable improvmeent [text](runs/2026-03-05T02-33-16Z_meta_ingest_blended_t1_similarity_k12.json)
best so far: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-05T02-36-31Z_meta_ingest_blended_meta_qe_rrf_mmr_k12.json

still misisng carol in this

### More data required.
Here is some analysis:
# More analysis of answwers:
Seasn 2 summary pulled in eps outside s2

Root cause: routing / query expansion produced broad queries (“key events”, “main characters”) and your blended retrieval didn’t enforce season == 2.

solution: for queries that mention a season explicitly, hard-filter by metadata (season=2) before vector search (or as a post-filter with a backfill that still respects the season).

## Jim & dwight conflict:
Why it happened

The retrieval top sources for q4 were mostly late seasons (S07–S09, S08E18, etc.), so the generator can only synthesize from what it sees. It’s not “wrong” in isolation, but it’s not representative.

Root cause: with open-ended “why” questions, MMR + QE tends to bring in diverse but recent or high-salience conflict scenes, not the foundational early-series dynamic.

Next step: treat this as a multi-episode pattern question:

retrieve by time slice (force some early seasons into the evidence set), or

retrieve by reason buckets (generate 3 sub-queries: pranks, authority/roles, rivalry) and then fuse.



# Next, I derived better data cards with different features like relationship cards, timelines, and themes

ran a sweep there. hit issues again with Julie being a girlfriend.


score
python experiments/score_runs.py --runs-dir experiments/runs --gold experiments/gold_answers.json

dashboard
streamlit run apps/rag_runs_dashboard.py


interesting error:
 Julie:\n- Michael went on a date with Julie, an ESL teacher.\n- They hit it off, and Michael liked her
 false
 evidence:
     {
            "rank": 5,
            "source": "/Users/tg/Developer/RAG-1/ingestion/normalized_docs_txt/scripts/season_06/s06e19_happy_hour_script.txt",
            "episode_id": "S06E19",
            "doc_type": "script",
            "chunk_type": "char",
            "chunk_index": 9,
            "first_line": "=== SCENE 024 ===",
            "preview": "=== SCENE 024 ===\n\nMichael: So, what do you do?\nJulie: I am an ESL teacher.\nMichael: Really?\nJulie: Yeah.\nMichael: See, I didn't think you could teach that. I thought that was something you were born with. What am I thinking right now?\nJuli…",
            "score": 0.04713064713064713
          },
 
 This could be:
 summarization bias?
  or in another quote michael says:
        "summary": "Michael shares a childhood story about a mistaken birthday date with a girl named Julie.",

which could confuse the context retrieval, but 

 This looks like a summarization bias, not a retrieval error:


The retrieved chunk shows friendly banter → model infers positive chemistry

But unless your retrieved evidence includes the later part where Julie leaves / Michael pivots to Donna, the model will write the “happy path” version

## REsolution: 
implement three targeted fixes in code: stricter “serious girlfriends” answer rules + a small pre-filter step, include derived summaries/cards directly in aggregation context (not just for routing), and tighten query expansion so it doesn’t broaden “serious” into “dating history”.

Essentially tells the LLM what qualifies as a romantic relationship
/Users/tg/Developer/RAG-1/experiments/runs/2026-03-05T05-19-16Z_topiccards_blended_q5_fix_julie.json
fixes the julie issue, but adds michaels fake girlfriend from new yourk, and misses Holly and Donna both.


Retrieval gives evidence. The model determines how well that evidence is interpreted.


# Frontend:
added ai as judge here : python experiments/score_runs.py --help
usage: score_runs.py [-h] [--runs-dir RUNS_DIR] [--gold GOLD] [--out-dir OUT_DIR]
                     [--judge-model JUDGE_MODEL] [--temperature TEMPERATURE]
                     [--run-name-prefix RUN_NAME_PREFIX]
                     [--run-id-prefix RUN_ID_PREFIX]

Score RAG run logs using gold answers + OpenAI judge.

optional arguments:
  -h, --help            show this help message and exit
  --runs-dir RUNS_DIR   Directory containing run JSON logs
  --gold GOLD           Gold answers JSON file
  --out-dir OUT_DIR     Where to write scored JSON files
  --judge-model JUDGE_MODEL
                        OpenAI model to use as judge
  --temperature TEMPERATURE
  --run-name-prefix RUN_NAME_PREFIX
                        Only score runs whose run.run_name starts with this prefix. Can
                        be provided multiple times. If omitted, scores all runs found.
  --run-id-prefix RUN_ID_PREFIX
                        Only score runs whose run.run_id starts with this prefix. Can
                        be provided multiple times. If omitted, no run_id filtering is



