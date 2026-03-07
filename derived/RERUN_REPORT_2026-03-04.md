# Derived pipeline run report (2026-03-04)

This document captures what was run already and the build prefixes/artifact locations so the run is reproducible.

## Artifact root

- Derived artifacts live under `derived/artifacts/`.

## Status by season (as of 2026-03-04)

### Season 1

- Segment summaries (map): `derived_segsummary_season01_nano_2026-03-04` (done)
- Episode cards (reduce): `derived_episodecard_season01_nano_2026-03-04` (done)

### Season 2

- Segment summaries (map): `derived_segsummary_season02_nano_2026-03-04` (done)
- Episode cards (reduce): `derived_episodecard_season02_nano_2026-03-04` (done)

Note: the Season 2 map manifest shows `segments_root` as `derived/segments`.

### Season 3

- Segment summaries (map): `derived_segsummary_season03_nano_2026-03-04` (done)
- Episode cards (reduce): `derived_episodecard_season03_nano_2026-03-04` (done)

Note: the Season 3 map manifest shows `segments_root` as `derived/segments`.

### Season 4

- Segment summaries (map): `derived_segsummary_season04_nano_2026-03-04` (done)
- Episode cards (reduce): `derived_episodecard_season04_nano_2026-03-04` (done)

Note: the Season 4 map manifest shows `segments_root` as `derived/segments`, while reduce used `derived/artifacts/segments`.

### Season 5

- Segment summaries (map): `derived_segsummary_season05_nano_2026-03-04` (manifest status currently `running`)
- Episode cards (reduce): `derived_episodecard_season05_nano_2026-03-04` (done_with_errors; missing segments under `derived/artifacts/segments` at the time)

### Seasons 6–9

- Segment summaries (map):
  - Season 6: `derived_segsummary_season06_nano_2026-03-04` (manifest status currently `running`)
  - Season 7: `derived_segsummary_season07_nano_2026-03-04` (manifest status currently `running`)
  - Season 8: `derived_segsummary_season08_nano_2026-03-04` (manifest status currently `running`)
  - Season 9: `derived_segsummary_season09_nano_2026-03-04` (manifest status currently `running`)

These look like runs that were started and wrote per-episode outputs, but did not finalize the season manifest (or were interrupted). Re-running `derived.summarize_season` with the same build prefix should resume/finish.

## Consistency note (segments_root)

For best reproducibility, use the same `--segments-root` for both map and reduce.

Recommended default:

- `--segments-root derived/artifacts/segments`

That matches:

- `derived/summarize_season.py` default
- `derived/reduce_season.py` default

If you previously wrote segments somewhere else (e.g. `derived/segments`), either:

- re-generate segments into `derived/artifacts/segments`, or
- pass the same `--segments-root` to *both* map and reduce.
