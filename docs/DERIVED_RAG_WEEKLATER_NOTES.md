# Derived corpus + RAG (week-later notes)

This is a high-level “where are we / why are we doing this / what comes next” memo for the *The Office* RAG sandbox.

## Why the derived corpus exists

The raw script chunk index is great for:
- pinpoint facts
- exact quotes
- single-episode questions

It struggles on aggregation questions (list/timeline/how X changes) because:
- relevant evidence is scattered across many chunks
- relevant chunks are semantically repetitive (MMR often hurts recall)

The derived corpus solves this by creating **higher-level retrieval units** that already contain the “glue” (threads, arcs, changes) **with provenance pointers** back to the scripts.

## The pipeline (artifacts)

All artifacts live under `derived/artifacts/`.

1) **Segments (deterministic)**
- `derived/artifacts/segments/SxxEyy.json`

2) **Map: Segment summaries (LLM)**
- `derived/artifacts/<SEG_SUMMARY_BUILD_PREFIX>_SxxEyy/segment_summaries/SxxEyy/*.json`

3) **Reduce: Episode derived cards (LLM)**
- `derived/artifacts/<EPISODE_CARD_BUILD_PREFIX>_SxxEyy/episode_cards/SxxEyy.json`

4) **Reduce-of-reduce: Season derived card (LLM)**
- `derived/artifacts/<SEASON_CARD_BUILD_PREFIX>/season_cards/seasonXX.json`

5) **Index build (embedding-only)**
- Build vector indexes over episode/season cards so they can be retrieved like any other corpus.

## Indexes (what we do with RAG)

Think in two tiers:

- **Tier A — “Map” / synthesis retrieval**: derived cards
  - What it’s for: timelines, lists, across-season changes, “what episodes matter?”
  - Index: `db/chroma_db_derived_cards` (recommended single index containing both episode+season cards)

- **Tier B — “Grounding” retrieval**: scripts (and optionally episode summaries)
  - What it’s for: quotes and final verification
  - Index: your existing `db/chroma_db*` indexes built by ingestion

The key design principle: **derived cards guide where to look; scripts prove what’s true**.

## Routing plan (derived-first → scripts-second)

At answer time:

1) **Route** the question to a retrieval policy
- If aggregation-like (list/timeline/compare/across) → use derived-first
- Else → script-only (or derived + script blended)

2) **Derived retrieval** (Tier A)
- Retrieve a handful of episode cards + maybe 1 season card.
- Extract episode IDs from derived evidence pointers.

3) **Script retrieval with filters** (Tier B)
- Retrieve from the script index but *filter* to the shortlist of episode IDs.
- This produces quote-capable grounding while keeping context focused.

4) **Answer**
- Require that specific claims cite script chunks (quotes).
- Use derived cards as navigation context, not as the final authority.

## What to run (copy/paste)

Build the combined derived index:
- Dry run:
  - `python derived/build_derived_cards_index.py --dry-run`
- Build:
  - `python derived/build_derived_cards_index.py --persist-dir db/chroma_db_derived_cards --reset`

Audit what’s built:
- `python derived/audit_derived_coverage.py --write-md docs/DERIVED_COVERAGE_AUDIT.md`

## Current coverage status (as of 2026-03-04)

See `docs/DERIVED_COVERAGE_AUDIT.md` for the full table.

High-level:
- Seasons 01–04: complete (segments + seg summaries + episode cards + season cards)
- Gaps to fix:
  - Season 05: missing `segments/S05E26.json`, missing seg summaries for `S05E11`, missing episode cards for `S05E11` + `S05E26`
  - Season 07: missing episode card for `S07E04`
  - Season 09: missing `segments/S09E23.json`, missing episode card for `S09E23`

## Guardrails / preferences to remember

- Prefer `.env` for `OPENAI_API_KEY` (avoid shell history).
- Keep derived and script indexes in separate persist dirs.
- Use deterministic `stable_doc_id` in derived indexing to avoid duplication.
- Use `--dry-run` before a rebuild.

## Next implementation step

Add a new retrieval mode in `experiments/run_eval.py`:
- baseline: script-only
- derived_then_script: derived cards → episode shortlist → script retrieval with episode filters

This makes the derived corpus measurable using your existing metrics (context diversity, aggregation readiness, citation drift).
