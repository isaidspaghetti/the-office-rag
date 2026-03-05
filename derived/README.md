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

---

## Entity-focused corpora (characters / relationships / plot objects)

These are **cross-episode** derived artifacts built by aggregating EpisodeDerivedCardV1 files.

- Build (auto-pick top characters + relationship pairs; plot objects come from config):
  - `python derived/build_entity_corpora.py --config derived/entity_corpora/config.example.json`

Outputs are written under `derived/artifacts/<build-prefix>/`:
- `character_cards/`
- `relationship_cards/`
- `plot_object_cards/`

See schema notes in `docs/DERIVED_CHECKLIST_AND_SCHEMAS.md`.

---

## Topic cards (2-pass: retrieval → one-call synthesis)

This is a more chatbot-friendly way to build cards at scale:

1) **Pass 1: Candidate episode discovery (cheap)**
   - For each topic, run retrieval against the script index.
   - Log: top episode IDs + top chunks per episode.

2) **Pass 2: Card synthesis (one LLM call per card)**
   - Feed only the selected evidence chunks.
   - Require: bullet facts + citations + 1–3 exact quotes.
   - If missing, say missing.

Commands:
- Discover only:
  - `python derived/build_topic_cards.py --pass discover --topics derived/topic_cards/topics.example.json --build-prefix topiccards_v1_YYYY-MM-DD`
- Synthesize (after discovery):
  - `python derived/build_topic_cards.py --pass synthesize --topics derived/topic_cards/topics.example.json --build-prefix topiccards_v1_YYYY-MM-DD`
- Both in one go:
  - `python derived/build_topic_cards.py --pass both --topics derived/topic_cards/topics.example.json --build-prefix topiccards_v1_YYYY-MM-DD`
