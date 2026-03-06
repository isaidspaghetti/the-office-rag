# Evolution report: evo_4_1_mini_2026-03-05_*

Retrieval-only sweep (no LLM answering) using script index db/chroma_db_meta and derived index db/chroma_db_derived_cards_4_1_mini_2026-03-04. Telemetry disabled via ANONYMIZED_TELEMETRY=False.

## Summary table (retrieval-only)

| step | run_name | policy | search | k | avg_ctx_docs | avg_distinct_eps | top_ep_share | ep_entropy | agg_readiness | empty_rate | retrieval_fail_rate | avg_retrieval_ms | errors |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | evo_4_1_mini_2026-03-05_step01_script_only_sim_k8 | script_only | similarity | 8 | 8.00 | 6.00 | 0.304 | 0.949 | 82.1 | 0.000 | 0.333 | 408 | 0 |
| 02 | evo_4_1_mini_2026-03-05_step02_script_only_mmr_k8 | script_only | mmr | 8 | 8.00 | 6.14 | 0.286 | 0.952 | 83.2 | 0.000 | 0.000 | 783 | 0 |
| 03 | evo_4_1_mini_2026-03-05_step03_derived_only_sim_k8 | derived_only | similarity | 8 | 8.00 | 7.71 | 0.125 | 0.981 | 94.1 | 0.000 | 1.000 | 267 | 0 |
| 04 | evo_4_1_mini_2026-03-05_step04_derived_then_script_sim_k10 | derived_then_script | similarity | 10 | 10.00 | 4.86 | 0.386 | 0.901 | 71.8 | 0.000 | 1.000 | 677 | 0 |
| 05 | evo_4_1_mini_2026-03-05_step05_blended_sim_k12 | blended | similarity | 12 | 12.00 | 8.86 | 0.262 | 0.933 | 82.6 | 0.000 | 0.333 | 460 | 0 |
| 06 | evo_4_1_mini_2026-03-05_step06_blended_sim_k16 | blended | similarity | 16 | 16.00 | 12.00 | 0.214 | 0.938 | 84.7 | 0.000 | 0.333 | 417 | 0 |

## Run files

- step 01: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-13-18Z_evo_4_1_mini_2026_03_05_step01_script_only_sim_k8.json
- step 02: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-13-23Z_evo_4_1_mini_2026_03_05_step02_script_only_mmr_k8.json
- step 03: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-13-50Z_evo_4_1_mini_2026_03_05_step03_derived_only_sim_k8.json
- step 04: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-13-57Z_evo_4_1_mini_2026_03_05_step04_derived_then_script_sim_k10.json
- step 05: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-14-17Z_evo_4_1_mini_2026_03_05_step05_blended_sim_k12.json
- step 06: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-14-23Z_evo_4_1_mini_2026_03_05_step06_blended_sim_k16.json

## Next (when OPENAI_API_KEY is set)

- Re-run the same steps with LLM answering enabled (`--llm-model gpt-4.1-mini`) and then judge-score them with `experiments/score_runs.py`. 
- That will populate `avg_overall`, groundedness, and token-cost metrics for a true end-to-end evolution story.
