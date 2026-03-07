# Evolution report: evo_4_1_mini_2026-03-05_llm_step*

## Summary table (scored)

| step | run_name | policy | search | k | avg_overall | judge_cases | det_episode_ok | det_must_include_ok | det_forbidden_hit | avg_total_tokens | grounding_fail | quote_in_ctx |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | evo_4_1_mini_2026-03-05_llm_step01_script_only_sim_k8 | script_only | similarity | 8 | 50 | 7 | 0.714 | 1.000 | 0.000 | 2077 | 0.000 | 1.000 |
| 02 | evo_4_1_mini_2026-03-05_llm_step02_script_only_mmr_k8 | script_only | mmr | 8 | 57 | 7 | 0.857 | 1.000 | 0.000 | 2028 | 0.000 | 1.000 |
| 03 | evo_4_1_mini_2026-03-05_llm_step03_derived_only_sim_k8 | derived_only | similarity | 8 | 34 | 7 | 0.571 | 0.429 | 0.000 | 3681 |  | 0.000 |
| 04 | evo_4_1_mini_2026-03-05_llm_step04_derived_then_script_sim_k10 | derived_then_script | similarity | 10 | 41 | 7 | 0.571 | 0.571 | 0.000 | 2814 | 0.500 | 0.286 |
| 05 | evo_4_1_mini_2026-03-05_llm_step05_blended_sim_k12 | blended | similarity | 12 | 54 | 7 | 0.714 | 0.857 | 0.000 | 2912 | 0.143 | 0.857 |
| 06 | evo_4_1_mini_2026-03-05_llm_step06_blended_sim_k16 | blended | similarity | 16 | 42 | 7 | 0.714 | 1.000 | 0.000 | 3743 | 0.143 | 0.857 |

## Summary table (retrieval)

| step | run_name | policy | search | k | avg_ctx_docs | avg_distinct_eps | top_ep_share | ep_entropy | agg_readiness | empty_rate | retrieval_fail_rate | avg_retrieval_ms | errors |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | evo_4_1_mini_2026-03-05_llm_step01_script_only_sim_k8 | script_only | similarity | 8 | 8.00 | 6.00 | 0.304 | 0.949 | 82.1 | 0.000 | 0.333 | 241 | 0 |
| 02 | evo_4_1_mini_2026-03-05_llm_step02_script_only_mmr_k8 | script_only | mmr | 8 | 8.00 | 6.14 | 0.286 | 0.952 | 83.2 | 0.000 | 0.000 | 537 | 0 |
| 03 | evo_4_1_mini_2026-03-05_llm_step03_derived_only_sim_k8 | derived_only | similarity | 8 | 8.00 | 7.71 | 0.125 | 0.981 | 94.1 | 0.000 | 1.000 | 210 | 0 |
| 04 | evo_4_1_mini_2026-03-05_llm_step04_derived_then_script_sim_k10 | derived_then_script | similarity | 10 | 10.00 | 4.86 | 0.386 | 0.901 | 71.8 | 0.000 | 1.000 | 491 | 0 |
| 05 | evo_4_1_mini_2026-03-05_llm_step05_blended_sim_k12 | blended | similarity | 12 | 12.00 | 8.86 | 0.262 | 0.933 | 82.6 | 0.000 | 0.333 | 444 | 0 |
| 06 | evo_4_1_mini_2026-03-05_llm_step06_blended_sim_k16 | blended | similarity | 16 | 16.00 | 12.00 | 0.214 | 0.938 | 84.7 | 0.000 | 0.333 | 403 | 0 |

## Run files

- step 01: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-22-20Z_evo_4_1_mini_2026_03_05_llm_step01_script_only_sim_k8.json
- step 02: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-23-14Z_evo_4_1_mini_2026_03_05_llm_step02_script_only_mmr_k8.json
- step 03: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-24-31Z_evo_4_1_mini_2026_03_05_llm_step03_derived_only_sim_k8.json
- step 04: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-24-53Z_evo_4_1_mini_2026_03_05_llm_step04_derived_then_script_sim_k10.json
- step 05: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-25-22Z_evo_4_1_mini_2026_03_05_llm_step05_blended_sim_k12.json
- step 06: /Users/tg/Developer/RAG-1/experiments/runs/2026-03-06T06-25-58Z_evo_4_1_mini_2026_03_05_llm_step06_blended_sim_k16.json

## Scored files

- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-22-20Z_evo_4_1_mini_2026_03_05_llm_step01_script_only_sim_k8.scored.json
- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-23-14Z_evo_4_1_mini_2026_03_05_llm_step02_script_only_mmr_k8.scored.json
- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-24-31Z_evo_4_1_mini_2026_03_05_llm_step03_derived_only_sim_k8.scored.json
- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-24-53Z_evo_4_1_mini_2026_03_05_llm_step04_derived_then_script_sim_k10.scored.json
- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-25-22Z_evo_4_1_mini_2026_03_05_llm_step05_blended_sim_k12.scored.json
- /Users/tg/Developer/RAG-1/experiments/scored_runs/2026-03-06T06-25-58Z_evo_4_1_mini_2026_03_05_llm_step06_blended_sim_k16.scored.json

## Notes

- This report already includes judge scoring (avg_overall) and deterministic checks.
- If you re-run new steps, re-run scoring so the scored files stay in sync.
