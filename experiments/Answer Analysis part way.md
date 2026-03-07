Yep — looking at your latest run (`meta_ingest_blended_meta_qe_rrf_mmr_k12`) the system is *closer*, but the remaining misses are very explainable from a RAG engineer POV: you’re still getting **(a) the wrong evidence**, **(b) evidence that’s too indirect**, and then the LLM **fills gaps**.

Below is what’s missing / incorrect **based on actual Office knowledge**, and **why your retrieved results still don’t support the right answer**.

---

## q1 — Bears / Beets / Battlestar

**Answer quality:** ✅ Good and grounded.
It correctly identifies **S03E20 “Product Recall”** and includes the actual quote + context. This one is basically an “iconic quote → episode” lookup and your corpus is strong at that.

**Why it works:** the retrieval actually surfaced the right scene (and your generator used it).

---

## q2 — Michael burns his foot

**Answer quality:** ✅ Good and grounded.
Correct: **S02E12 “The Injury.”**

**Why it works:** the phrase “burns his foot / George Foreman grill” is uniquely anchored in dialogue.

---

## q3 — Summarize Season 2

**Answer quality:** ⚠️ Incomplete and partially off-target.

### What’s missing (actual show knowledge)

Season 2’s *big* beats include:

* Jim/Pam tension crescendoing and **“Casino Night” confession + kiss**
* Michael/Jan relationship deepening
* Stamford branch looming (end-of-season setup)
* Major tentpole episodes beyond E1–E7 and E13 (The Injury, Booze Cruise, Casino Night, etc.)

### Why your RAG didn’t answer it well

Your retrieval for q3 is **not season-filtered**. In the run log, it pulled episodes outside S2 (ex: **S04E01**, **S06E18**, **S06E15**, **S01E01**, **S07E23** showed up in the top-12 results). That guarantees a weak season summary because the evidence is literally not “Season 2”.

**Root cause:** routing / query expansion produced broad queries (“key events”, “main characters”) and your blended retrieval didn’t enforce `season == 2`.

**What a RAG engineer does next:** for queries that mention a season explicitly, **hard-filter by metadata** (season=2) *before* vector search (or as a post-filter with a backfill that still respects the season).

---

## q4 — “Why does Dwight dislike Jim?”

**Answer quality:** ⚠️ Kinda plausible, but not the best/true “3 reasons”, and it’s skewed.

### What’s missing (actual show knowledge)

The strongest reasons are:

1. Jim constantly pranks him / humiliates him
2. Dwight is hierarchy/order-obsessed and Jim mocks that
3. They’re rivals for office status + (early on) social/romantic friction around Pam (minor, but real)

Your answer instead leans heavily on **late-series Athlead / “jump ship / poaching Darryl”** evidence, which is not the core reason Dwight dislikes Jim across the show.

### Why it happened

The retrieval top sources for q4 were mostly **late seasons (S07–S09, S08E18, etc.)**, so the generator can only synthesize from what it sees. It’s not “wrong” in isolation, but it’s not representative.

**Root cause:** with open-ended “why” questions, MMR + QE tends to bring in diverse but *recent* or *high-salience* conflict scenes, not the foundational early-series dynamic.

**Next step:** treat this as a **multi-episode pattern question**:

* retrieve **by time slice** (force some early seasons into the evidence set), or
* retrieve by **reason buckets** (generate 3 sub-queries: pranks, authority/roles, rivalry) and then fuse.

---

## q5 — “List Michael’s serious girlfriends and how they ended”

**Answer quality:** ❌ Incorrect/incomplete.

### What’s missing / wrong (actual show knowledge)

A decent “serious girlfriends” list usually includes:

* **Jan** (toxic, ends after breakup / “Dinner Party” era fallout)
* **Carol** (breaks up with him because he’s intense / photo-shopped family card / moves too fast)
* **Holly** (true soulmate; Michael leaves for Colorado with her)
* **Donna** (he ends it after learning she’s married)
  …and depending on definition you might include **Helene** (Pam’s mom; he ends it because she’s too old), etc.

Your answer:

* misses Carol and Helene entirely
* includes **Julie** (arguable “not serious”)
* gives a **bad/unsupported Holly ending** (“dad ill in Colorado, Michael says no”) — that’s not how the relationship ultimately ends.

### Why it happened (from your retrieved results)

Your q5 retrieval top chunks are mostly:

* **S04E08, S04E10, S06E23, S06E19**, etc.
  That’s *not* enough to reconstruct a relationship timeline. You need the “relationship endpoints” episodes, and you didn’t retrieve them.

**Root cause:** this question is basically an **entity aggregation + timeline** task. Plain vector retrieval over dialogue chunks is weak at:

* enumerating “all instances of girlfriends”
* finding “how it ended” (often not in one quote; spread across scenes/episodes)

**What a RAG engineer does next (before graphs):**
Create **derived “relationship cards”** offline (one doc per relationship) extracted from scripts/summaries:

* `Michael ↔ Donna: discovered married → ended it (S06E23 The Chump, etc.)`
* `Michael ↔ Holly: reunion → leaves show to be with her (S07E22/23 Goodbye Michael)`
  These “cards” become the high-precision corpus for q5-style questions.

---

## q6 — Teapot letter

**Answer quality:** ❌ Missed the point.

### Correct (actual show knowledge)

The teapot letter originates from **“Christmas Party” (S02E10)**: Jim’s teapot gift for Pam includes a letter; it becomes symbolic of their relationship; later it’s revisited (notably in the final season).

### Why your system missed it

Your retrieval *did* include **S02E10 Christmas Party**, but the retrieved chunk was a **random scene (liquor store)** — not the teapot scene. Meanwhile it also retrieved **S09E23 Finale**, which contains Pam’s “keep it private” line, so the model latched onto that and hallucinated a definition.

**Root causes (two):**

1. **Chunking mismatch:** your 1000-char chunks mean “teapot” might be in a different chunk than the one retrieved for that episode.
2. **Retrieval mode mismatch:** this is a **rare keyword / object lookup**. Pure vector + MMR is not as good as lexical/hybrid here.

**What a RAG engineer does next:**

* Add **hybrid search (BM25 + vector)** or at minimum a keyword fallback when the query contains concrete objects (“teapot”, “letter”).
* Or add a derived “plot object card” corpus (teapot letter doc with episode IDs).

---

## q7 — “Who is Creed and what is his deal?”

**Answer quality:** ⚠️ Mostly fluff; missing the “deal.”

### What’s missing (actual show knowledge)

Creed’s “deal” is the running implication he’s:

* shady / possibly criminal / using a fake identity
* bizarre non sequiturs and unclear job role
* (eventually) gets arrested in the finale

Your answer is mostly “he’s quirky” because retrieval surfaced harmless Creed quotes (hair dye, “I’m thirty”, etc.) rather than the incriminating ones.

**Root cause:** retrieval didn’t pull the “criminal / identity / arrested” evidence, so the generator played it safe.

**Next step:** for character “what’s their deal” questions, use a derived **character card** (one doc per character) built offline from scripts + summaries.

---

# The real pattern across the failures

The misses aren’t because “RAG can’t get there.” They’re because you’re asking **4 different task types** but using one retrieval strategy:

1. **Quote→episode** (q1/q2) ✅ works with dialogue chunks
2. **Object lookup** (q6) needs lexical/hybrid or smaller chunking
3. **Aggregation/timeline** (q5) needs derived structured docs (relationship cards)
4. **Season summary** (q3) needs metadata filters + season-constrained retrieval

---

# What I’d do next (in order)

1. **Enforce metadata filtering for explicit season/episode queries** (fix q3 immediately).
2. Add **hybrid search** (or a lexical fallback) for object-y queries like q6.
3. Add **derived cards** (relationship cards + character cards + plot-object cards) to handle q5/q7 reliably.
4. Tighten the generator prompt further: if the evidence doesn’t contain the key fact, it must say “I don’t know” *and* explain what’s missing (“no retrieved chunk mentions ‘teapot’”).

If you paste the q5/q6 retrieved chunk texts (not just previews) for the latest run, I can point to the exact lines that caused the Holly/teapot failure and tell you whether it’s retrieval, chunking, or prompt-grounding.


Why your Office POC still maps to production

The Office dataset is an unusually hard case for pure RAG because:

lots of meaning is implicit / symbolic

facts are distributed over episodes

“lists and timelines” are common queries

But company data has the same patterns:

decisions span multiple documents

names and entities are inconsistent

you need rollups (“what changed this week?”)


a 5-layer memory architecture

Most systems look like an ESL pipeline:
ETL pipelines
+ vector search
+ metadata stores
+ summarization
+ extraction
+ routing
+ evaluation