Syllabus (8 Modules, With Concrete Milestones)
Module 1 — Foundations: RAG as a System
Concepts: failure modes (retrieval vs grounding vs reasoning), why “good prompts” don’t fix bad recall, what “production RAG” means.
Lab (this repo): run a small baseline and inspect failures in run_eval.py + run logs under experiments/runs/.
Deliverable: a simple scorecard you trust more than vibes (e.g., citation-in-context %, episode coverage %, “IDK” correctness).
Module 2 — Data Modeling & Provenance (The Root of Trust)
Concepts: canonical IDs, versioning, lineage, immutable raw text vs derived artifacts, chunk identity.
Lab: extend the metadata you already added (episode_id/chunk_index) into a general “provenance contract” for every doc type (raw, summary, derived, extracted fact).
Deliverable: a documented metadata schema + “source of truth” policy (what fields are authoritative, what’s derived).
Module 3 — Retrieval That Supports Aggregation (Not Just Lookups)
Concepts: why MMR hurts “list all …” queries, episode-diverse retrieval, query plans (broad → narrow), hybrid retrieval (dense + lexical), reranking.
Lab: implement “aggregation mode” retrieval: retrieve many → group by episode_id → cap per episode → ensure diversity; add a second pass for quote-hunting.
Deliverable: q5-style questions stop degrading, and you can show before/after eval runs.
Module 4 — Index-Time Enrichment (Hierarchical Summaries)
Concepts: map-reduce summarization, multi-resolution summaries (scene→episode→season→series), “derived docs” as first-class citizens.
Lab: create derived docs in a new collection (or same with doc_type=derived) and cite episode/chunk provenance.
Deliverable: “season summary” and “character bio” answers come primarily from derived docs, with raw scripts used for quotes.
Module 5 — LLM-Built Knowledge Base (Structured Extraction)
Concepts: extraction with schemas (JSON), entity resolution (same person, aliases), confidence scores, normalization, error taxonomy.
Lab: build a “facts store” of structured items like:
Person, Relationship, Event, Role, Quote
each fact: {subject, predicate, object, time_scope, episode_ids, chunk_refs, confidence}
Deliverable: a searchable fact table you can diff/rebuild, plus a “fact audit” CLI that prints supporting sources.
Module 6 — Knowledge Graph Construction (And Why Graphs Help)
Concepts: graph schema design, temporal edges, contradictions, “as-of” queries, paths (“show me evidence that X led to Y”), graph + vector hybrid.
Lab: construct a small graph (start simple: characters + relationships + key events) from extracted facts; store in something pragmatic first (SQLite tables or JSONL), then optionally Neo4j later.
Deliverable: answer complex questions by traversing the graph first (to pick relevant episodes), then retrieve supporting text for citations.
Module 7 — Handling Change Over Time (The Slack Problem)
Concepts: event sourcing, “truth is time-indexed,” decisions evolve, thread/channel context, long conversations, multiple sources disagree.
Office analogy: treat episodes as time slices; relationships evolve; later episodes contradict earlier impressions.
Lab: add time metadata to facts (season, episode, maybe airdate) and teach the system to answer “as of S05E10…” vs “overall”.
Deliverable: the assistant can explain changes and cite “when it changed”.
Module 8 — Productionization: Safety, Access, Monitoring, Feedback
Concepts: permissions-aware retrieval, PII redaction, prompt injection defense (retrieval-time filtering), logging, evaluation in CI, human feedback loops.
Lab: add a “trust report” to each answer: coverage, citations, missingness, and a reasoned “I’m not sure” path.
Deliverable: a demo that a skeptical engineer would respect (and a checklist you could reuse on Slack/Jira data).
Capstone Project (What You’ll End Up With)

A dual-store system:
Vector store for retrieval (raw + derived)
Fact/graph store for structured queries and temporal reasoning
An answer pipeline:
plan → 2) graph/fact lookup → 3) targeted retrieval for citations → 4) compose answer + trust report → 5) log metrics
Suggested next step in this repo

Implement “derived docs + provenance” first (it’s the highest ROI bridge from today’s system to “trustworthy” behavior), then build the structured facts layer on top.
If you tell me your preference for the graph backend (keep it simple with SQLite/JSONL vs Neo4j), I’ll tailor the next module’s implementation plan and start wiring the first derived-doc generator into this project.