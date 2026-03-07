# Practice progression (two-pass scored)

This report is generated from historical runs and focuses on *RAG practices*, not filenames.
Scoring uses the **two-pass judge**: (1) context-only groundedness, (2) gold-only correctness, then merged.

## Snapshot
- Best run: **meta_ingest_blended_qe_rrf_similarity_k12** (avg_overall=81, Δ vs baseline=+14)
- Runs scanned: 69

## Milestones (representative best per practice)
| practice | representative run_name | policy | search | k | qe | meta_index | derived_index | avg_overall | avg_total_tokens | cases_scored | created_at_utc |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (similarity, small k) | baseline_similarity_k3 | script_only | similarity | 3 | no | no | no | 67 | 811 | 7 | 2026-03-02T19:41:43Z |
| Increase k (more recall) | t1_similarity_k12 | script_only | similarity | 12 | no | no | no | 75 | 2892 | 7 | 2026-03-02T19:45:31Z |
| MMR (diversify context) | t2_mmr_k12_fetch40_l07_fixed | script_only | mmr | 12 | no | no | no | 65 | 2750 | 7 | 2026-03-02T23:27:02Z |
| Query expansion + RRF (recall) | qe_rrf_similarity_k12 | script_only | similarity | 12 | yes | no | no | 67 | 2863 | 7 | 2026-03-03T05:40:45Z |
| Metadata-enabled filtering | evo_4_1_mini_2026-03-05_llm_step02_script_only_mmr_k8 | script_only | mmr | 8 | no | yes | no | 73 | 2028 | 7 | 2026-03-06T06:23:14Z |
| Derived routing (episode shortlisting) | meta_ingest_auto_qe_rrf_similarity_k12 | auto | similarity | 12 | yes | no | yes | 80 | 2762 | 7 | 2026-03-05T02:29:38Z |
| Blended retrieval (baseline + routed) | meta_ingest_blended_qe_rrf_similarity_k12 | blended | similarity | 12 | yes | no | yes | 81 | 2663 | 7 | 2026-03-05T02:34:58Z |
| Topic cards (derived evidence) + blended | topiccards_blended_qe_rrf_similarity_k12 | blended | similarity | 12 | yes | no | yes | 81 | 2666 | 7 | 2026-03-05T04:38:50Z |

## What changed (plain-English)
- **Increase k**: trades cost for recall (more chunks in context).
- **MMR**: diversifies retrieved chunks to reduce redundancy and cover more evidence.
- **Query expansion + RRF**: generates alternate retrieval queries and fuses results for higher recall on ambiguous queries.
- **Metadata-enabled filtering**: allows hard constraints (e.g., season/episode) to prevent drift across irrelevant episodes.
- **Derived routing**: uses higher-level derived cards to shortlist episodes, then retrieves scripts within that shortlist.
- **Blended retrieval**: combines baseline recall with routed expansions, improving robustness across question types.

## Notes / gotchas
- `avg_total_tokens` is pulled from the original run log when available (missing if the run didn’t log usage).
- Some runs match multiple practices; this report assigns a single **primary** practice by priority to keep the story simple.
