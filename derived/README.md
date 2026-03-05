# Derived corpus pipeline (re-runnable)

This folder contains the **derived-corpus** build pipeline.

The guiding contract is in [docs/DERIVED_CHECKLIST_AND_SCHEMAS.md](../docs/DERIVED_CHECKLIST_AND_SCHEMAS.md).

## What artifacts exist?

All derived artifacts are written under:

- `derived/artifacts/`

Key artifact types:

- **Segments** (deterministic): `derived/artifacts/segments/SxxEyy.json`
- **Segment summaries** (map): `derived/artifacts/<SEG_SUMMARY_BUILD_PREFIX>_SxxEyy/segment_summaries/SxxEyy/*.json`
- **Episode cards** (reduce): `derived/artifacts/<EPISODE_CARD_BUILD_PREFIX>_SxxEyy/episode_cards/SxxEyy.json`
- **Season card** (reduce-of-reduce): `derived/artifacts/<SEASON_CARD_BUILD_PREFIX>/season_cards/seasonXX.json`

## Prereqs

- Create and activate a venv: `python -m venv venv && source venv/bin/activate`
- Install deps (this repo doesn’t currently have a pinned `requirements.txt`; use your existing venv, or install the core deps):
  - `pip install -U langchain-openai langchain-core langchain-chroma chromadb python-dotenv`
- Provide `OPENAI_API_KEY` via `.env` at repo root (recommended) or your shell environment.

Important: avoid exporting secrets into shell history. Prefer `.env`.

## The pipeline order

1) **Segments** (fast, deterministic)

- One file per episode: `derived/artifacts/segments/SxxEyy.json`

2) **Map**: segment summaries

- One LLM call per segment.

3) **Reduce**: episode cards

- One LLM call per episode.

4) **Reduce-of-reduce**: season card

- One LLM call per season.

5) **Index build** (optional but recommended)

- Build Chroma indexes over episode cards and/or season cards.

## Re-running what we already did (S01–S04)

Copy/paste commands are in:

- [derived/runbooks/rebuild_season01_2026-03-04.sh](runbooks/rebuild_season01_2026-03-04.sh)
- [derived/runbooks/rebuild_season02_2026-03-04.sh](runbooks/rebuild_season02_2026-03-04.sh)
- [derived/runbooks/rebuild_season03_2026-03-04.sh](runbooks/rebuild_season03_2026-03-04.sh)
- [derived/runbooks/rebuild_season04_2026-03-04.sh](runbooks/rebuild_season04_2026-03-04.sh)

Each script runs:

- segment generation (idempotent)
- `derived.summarize_season` (map)
- `derived.reduce_season` (reduce)
- `derived.reduce_season_card` (season reduce)

## Index build (derived cards)

Once you have episode and/or season cards:

- Build an **episode-card** index:
  - `python derived/build_episode_cards_index.py --persist-dir db/chroma_db_episode_cards --out-root derived/artifacts --episode-cards-build-prefix derived_episodecard_season04_nano_2026-03-04 --reset`

- Build a **season-card** index:
  - `python derived/build_season_cards_index.py --persist-dir db/chroma_db_season_cards --out-root derived/artifacts --season-cards-build-prefix derived_seasoncard_all_nano_YYYY-MM-DD --reset`

Notes:
- These indexers are pure “read artifacts, embed, write Chroma”. They do not call the LLM.
- Use separate `--persist-dir`s so you can A/B indexes.

### One index with both episode + season cards

If you want a single collection that contains both EpisodeDerivedCardV1 and SeasonDerivedCardV1 (distinguished by metadata `derived_type`):

- Dry run (no API calls):
  - `python derived/build_derived_cards_index.py --dry-run`
- Build (reset + rebuild):
  - `python derived/build_derived_cards_index.py --persist-dir db/chroma_db_derived_cards --reset`
