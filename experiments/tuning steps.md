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

  --search-type mmr \
  --k 12 \
  --fetch-k 40 \
  --lambda-mult 0.7 \
  --notes "Tuning: MMR k=12 fetch_k=40 lambda=0.7 (fixed mmr impl)"


# Query expansion Hybrid search (vector + keyword) Reranking RRF (if combining multiple retrievers) Agent-based chunking (only if needed)