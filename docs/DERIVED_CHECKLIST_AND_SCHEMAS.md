# Derived corpus: checklist + schemas (episode-first)

This doc is the **engineering contract** for building a derived corpus (distilled “knowledge layer”) from raw episode scripts.

It contains:
- A build checklist (what to implement + verify)
- Data schemas (what we store, versioned)
- A provenance contract (how every derived statement points back to source)
- Token-limit notes (why “episode-level” is still the right starting unit)

## Why episode-first (even though scripts are huge)

Episode-first does **not** mean “send the whole episode to the model in one request.” It means:
- The **unit of truth + provenance** is the episode (`episode_id = SxxExx`)
- We build an **episode-level artifact** using *hierarchical / map-reduce summarization* over smaller segments

That keeps the system:
- debuggable (everything can be traced to an episode)
- composable (episode → season → series)
- retrievable (derived docs are compact and high-signal)

## Goals / non-goals

**Goals**
- Produce compact derived docs that are **better for aggregation** (“list / timeline / arc”) than raw dialogue.
- Preserve provenance (episode + evidence spans) so we can debug hallucinations.
- Make artifacts **rebuildable** and **versioned** so metrics can trend.

**Non-goals (for v1)**
- Perfect scene segmentation (we may not have reliable scene markers).
- A full knowledge graph. (We can extract structured events later, but the first win is reliable derived summaries.)

---

## Build checklist (v1)

### 0) Inputs are sane
- [ ] Normalized scripts exist under `ingestion/normalized_docs_txt/scripts/`.
- [ ] Each script doc has stable metadata: `episode_id`, `season`, `episode`, `title`.
- [ ] Each script doc has stable `source` (filename/path) and consistent formatting.

### 1) Segmenting (the anti-token-limit step)
- [ ] Split each episode script into **segments** sized for your summarizer model.
- [ ] Segment boundaries should align to natural breaks when possible (blank lines / speaker blocks).
- [ ] Each segment gets an ID that is stable across rebuilds:
  - `segment_id = "{episode_id}:seg:{segment_index:03d}"`
- [ ] Store segment boundaries as **character offsets** into the original episode text.

**Recommended starting targets**
- Segment size: ~1,200–2,000 tokens of raw dialogue (model-dependent).
- Segment overlap: 0–150 tokens (optional) to avoid boundary loss.

### 2) Map step: segment summaries
- [ ] For each segment, generate a **structured** summary (`SegmentSummaryV1`) using a deterministic prompt.
- [ ] Keep the output small and regular; avoid “creative prose.”
- [ ] Each segment summary includes:
  - key events / beats
  - who appears
  - notable quotes (optional)
  - open questions / ambiguities (explicitly)

### 3) Reduce step: episode summary
- [ ] Combine all segment summaries into one **episode-level** derived doc (`EpisodeDerivedCardV1`).
- [ ] The reducer must be *constrained*:
  - It may only use information present in the segment summaries.
  - It must output explicit **evidence pointers** back to segments.

### 4) Optional second reduce: season summaries
- [ ] Combine episode derived cards into season-level cards (`SeasonDerivedCardV1`).
- [ ] Preserve `episode_ids` for every claim cluster.

### 5) Persist to derived index
- [ ] Write derived docs into a separate Chroma persist dir (default: `db/chroma_db_derived`).
- [ ] Derived docs must have consistent metadata:
  - `doc_type = derived`
  - `derived_type = episode_card | season_card | segment_summary`
  - `episode_id` (when applicable)
  - `episode_ids` list (when applicable)
  - `build_id`, `schema_version`, `source_index` references

### 6) QA: trust & drift checks
- [ ] Spot-check N episodes end-to-end:
  - segment text → segment summary → episode card
- [ ] Verify *no orphan claims*:
  - every high-level claim lists at least one evidence span
- [ ] Run eval on a small query set:
  - aggregation questions should improve retrieval diversity + readiness
- [ ] Track build stats in a manifest JSON (counts, errors, latency, cost).

---

## Provenance contract (minimum viable)

A derived artifact is only “trusted” if we can point to:
1) the episode (`episode_id`)
2) the segment(s) where it came from (`segment_id`)
3) a small snippet (or offsets) inside the segment

### EvidenceSpanV1

Minimal evidence pointer object:

```json
{
  "episode_id": "S02E11",
  "segment_id": "S02E11:seg:004",
  "source": "scripts/season_02/s02e11_booze_cruise_script.txt",
  "char_start": 12345,
  "char_end": 12789,
  "snippet": "MICHAEL: ..."
}
```

Notes:
- `char_start`/`char_end` are offsets **into the full episode text** (not just the segment). This makes evidence stable even if you tweak segment sizes (as long as the underlying episode text is unchanged).
- `snippet` is for human inspection and quick UI display.

---

## Schemas (v1)

These are “JSON Schema–like” contracts (kept human-readable). We can formalize them into actual JSON Schema files later.

### SegmentSummaryV1

```json
{
  "schema": "SegmentSummaryV1",
  "schema_version": 1,
  "build_id": "2026-03-03_derived_v1",

  "episode_id": "S02E11",
  "segment_id": "S02E11:seg:004",
  "segment_index": 4,

  "segment_char_start": 11800,
  "segment_char_end": 14210,

  "people": ["Michael", "Jim", "Pam"],

  "beats": [
    {
      "type": "event",
      "summary": "Michael gives an awkward motivational speech on the boat.",
      "evidence": [{"episode_id":"S02E11","segment_id":"S02E11:seg:004","source":"...","char_start":12345,"char_end":12789,"snippet":"..."}]
    }
  ],

  "notable_quotes": [
    {
      "quote": "...",
      "speaker": "Michael",
      "evidence": [{"episode_id":"S02E11","segment_id":"S02E11:seg:004","source":"...","char_start":12345,"char_end":12789,"snippet":"..."}]
    }
  ],

  "uncertainties": [
    "If the speaker is ambiguous, say so explicitly."
  ]
}
```

### EpisodeDerivedCardV1

```json
{
  "schema": "EpisodeDerivedCardV1",
  "schema_version": 1,
  "build_id": "2026-03-03_derived_v1",

  "doc_type": "derived",
  "derived_type": "episode_card",

  "episode_id": "S02E11",
  "title": "Booze Cruise",

  "one_paragraph_synopsis": "...",

  "main_threads": [
    {
      "thread": "Michael tries to assert leadership and gives bad advice.",
      "evidence": [
        {"episode_id":"S02E11","segment_id":"S02E11:seg:001","source":"...","char_start":1000,"char_end":1400,"snippet":"..."}
      ]
    }
  ],

  "character_highlights": [
    {
      "character": "Jim",
      "what_changes": "...",
      "evidence": [{"episode_id":"S02E11","segment_id":"S02E11:seg:007","source":"...","char_start":22000,"char_end":22700,"snippet":"..."}]
    }
  ],

  "relationships": [
    {
      "pair": ["Jim", "Pam"],
      "status": "...",
      "evidence": [{"episode_id":"S02E11","segment_id":"S02E11:seg:007","source":"...","char_start":22000,"char_end":22700,"snippet":"..."}]
    }
  ],

  "tags": ["boat", "speech", "team-building"],

  "source_segments": ["S02E11:seg:000", "S02E11:seg:001"],
  "build_notes": "Optional free text"
}
```

### SeasonDerivedCardV1 (optional)

```json
{
  "schema": "SeasonDerivedCardV1",
  "schema_version": 1,
  "build_id": "2026-03-03_derived_v1",

  "doc_type": "derived",
  "derived_type": "season_card",

  "season": 2,
  "episode_ids": ["S02E01", "S02E02"],

  "season_synopsis": "...",
  "arcs": [
    {
      "arc": "Jim/Pam tension escalates",
      "episode_ids": ["S02E01", "S02E07"],
      "evidence": [
        {"episode_id":"S02E07","segment_id":"S02E07:seg:005","source":"...","char_start":15000,"char_end":15400,"snippet":"..."}
      ]
    }
  ]
}
```

---

## Entity-focused corpora (v1)

These are **cross-episode** cards built by aggregating EpisodeDerivedCardV1 artifacts.

Goals:
- Make entity-specific questions (relationships, character traits, recurring objects) retrievable without hard episode routing.
- Keep every card grounded via a small set of verbatim supporting quotes (sourced evidence spans).

### CharacterCardV1

```json
{
  "schema": "CharacterCardV1",
  "schema_version": 1,
  "build_id": "entitycards_v1_2026-03-05",
  "created_at_utc": "2026-03-05T00:00:00Z",

  "doc_type": "derived",
  "derived_type": "character_card",

  "entity_names": ["Dwight"],
  "character": {"name": "Dwight", "aliases": ["Dwight Schrute"]},

  "episode_ids": ["S02E10", "S03E01"],
  "key_facts": ["..."],
  "supporting_quotes": [
    {
      "episode_id": "S02E10",
      "segment_id": "S02E10:seg:001",
      "source": "scripts/...",
      "char_start": 123,
      "char_end": 200,
      "snippet": "Dwight: ...",
      "quote": "Dwight: ..."
    }
  ]
}
```

### RelationshipCardV1

```json
{
  "schema": "RelationshipCardV1",
  "schema_version": 1,
  "build_id": "entitycards_v1_2026-03-05",
  "created_at_utc": "2026-03-05T00:00:00Z",

  "doc_type": "derived",
  "derived_type": "relationship_card",

  "entity_names": ["Michael", "Jan"],
  "relationship": {"entities": ["Michael", "Jan"], "label": "Michael–Jan", "aliases": []},

  "episode_ids": ["S03E01"],
  "key_facts": ["..."],
  "supporting_quotes": [{"episode_id": "S03E01", "segment_id": "S03E01:seg:002", "source": "scripts/...", "char_start": 0, "char_end": 10, "snippet": "...", "quote": "..."}]
}
```

### PlotObjectCardV1

```json
{
  "schema": "PlotObjectCardV1",
  "schema_version": 1,
  "build_id": "entitycards_v1_2026-03-05",
  "created_at_utc": "2026-03-05T00:00:00Z",

  "doc_type": "derived",
  "derived_type": "plot_object_card",

  "entity_names": ["Dundie"],
  "plot_object": {"name": "Dundie", "aliases": ["Dundies", "Dundie Award"]},

  "episode_ids": ["S02E01"],
  "key_facts": ["..."],
  "supporting_quotes": [{"episode_id": "S02E01", "segment_id": "S02E01:seg:000", "source": "scripts/...", "char_start": 0, "char_end": 10, "snippet": "...", "quote": "..."}]
}
```

---

## Token limit: practical guidance

### What *won’t* work
- Sending the entire episode script to the LLM in one go.

### What works (and is still episode-first)
- **Segment the episode** to fit your model window.
- **Map**: summarize each segment to a small structured object.
- **Reduce**: merge structured objects into an episode card.

The reducer prompt can be tiny because it consumes *summaries*, not raw dialogue.

### Recommended prompts style
- JSON-only outputs.
- Explicit “don’t guess; mark uncertainty.”
- Require evidence pointers.

---

## Storage: how we write these into Chroma

Each derived doc’s `page_content` should be a compact text form optimized for retrieval (e.g., synopsis + threads + highlights), while the full structured JSON can be:
- stored in `metadata["json"]` (if small enough)
- or written to disk as `derived/artifacts/{build_id}/...json` and referenced by path in metadata

**Minimal metadata for Chroma**
```json
{
  "doc_type": "derived",
  "derived_type": "episode_card",
  "episode_id": "S02E11",
  "season": 2,
  "title": "Booze Cruise",
  "build_id": "2026-03-03_derived_v1",
  "schema": "EpisodeDerivedCardV1",
  "schema_version": 1
}
```

---

## Integration points (what will change later)

Once derived cards exist, answer-time becomes a two-stage flow:
1) Retrieve derived cards for the “map” (aggregation-friendly)
2) Retrieve raw script chunks for quotes/citations (grounding)

This also gives you a clean place to add routing:
- aggregation-like questions → derived index first
- direct quote / "what did X say" → raw scripts first
