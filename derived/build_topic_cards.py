from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_SCRIPT_PERSIST_DIR = "db/chroma_db"
DEFAULT_SCRIPT_COLLECTION_NAME = None
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-nano"

DEFAULT_QUERY_TOP_K = 30
DEFAULT_TOP_EPISODES = 8
DEFAULT_TOP_CHUNKS_PER_EPISODE = 5
DEFAULT_MAX_CHUNK_CHARS = 1400

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_json(obj) + "\n", encoding="utf-8")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0] if "\n" in s else ""
    return s.strip()


def _try_parse_json(content: str) -> Dict[str, Any]:
    s = (content or "").strip()
    if not s:
        raise json.JSONDecodeError("Expecting value", s, 0)

    try:
        return json.loads(s)
    except Exception:
        pass

    s2 = _strip_code_fences(s)
    try:
        return json.loads(s2)
    except Exception:
        pass

    repaired = s2.replace('\\"', '"')
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    if "{" in repaired and "}" in repaired:
        repaired = repaired[repaired.find("{") : repaired.rfind("}") + 1]
    return json.loads(repaired)


def _doc_episode_id(doc: Document) -> Optional[str]:
    ep = (doc.metadata or {}).get("episode_id")
    if not ep:
        return None
    return str(ep).strip().upper() or None


def _doc_chunk_index(doc: Document) -> Optional[int]:
    v = (doc.metadata or {}).get("chunk_index")
    try:
        return int(v)
    except Exception:
        return None


def _chunk_ref_id(doc: Document) -> str:
    ep = _doc_episode_id(doc) or "UNKNOWN_EP"
    idx = _doc_chunk_index(doc)
    if idx is not None:
        return f"{ep}:chunk:{idx:06d}"
    src = str((doc.metadata or {}).get("source") or "")
    return f"{ep}:src:{src}"


def _trim_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


@dataclass(frozen=True)
class TopicSpec:
    topic_id: str
    topic_type: str
    entity_names: List[str]
    queries: List[str]


def load_topics(path: Path) -> List[TopicSpec]:
    raw = _read_json(path)
    topics = raw.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics file must contain a non-empty 'topics' list")

    out: List[TopicSpec] = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        topic_id = str(t.get("topic_id") or "").strip()
        topic_type = str(t.get("topic_type") or "").strip()
        entity_names = [str(x).strip() for x in (t.get("entity_names") or []) if str(x).strip()]
        queries = [str(x).strip() for x in (t.get("queries") or []) if str(x).strip()]
        if not topic_id or not topic_type or not queries:
            continue
        out.append(
            TopicSpec(
                topic_id=topic_id,
                topic_type=topic_type,
                entity_names=entity_names,
                queries=queries,
            )
        )

    if not out:
        raise ValueError("No valid topics found")
    return out


def open_script_store(
    *, persist_dir: str, collection_name: Optional[str], embed_model: str
) -> Chroma:
    embeddings = OpenAIEmbeddings(model=embed_model)
    kwargs: Dict[str, Any] = {
        "persist_directory": persist_dir,
        "embedding_function": embeddings,
    }
    if collection_name:
        kwargs["collection_name"] = collection_name
    return Chroma(**kwargs)


def similarity_search_with_scores(vs: Chroma, query: str, k: int) -> List[Tuple[Document, Optional[float]]]:
    # langchain-chroma returns distance by default in some versions; treat it as a score-ish number.
    # We only use it for within-run ranking.
    results = vs.similarity_search_with_score(query, k=k)
    out: List[Tuple[Document, Optional[float]]] = []
    for doc, score in results:
        out.append((doc, float(score) if score is not None else None))
    return out


def discover_topic_evidence(
    *,
    vs: Chroma,
    topic: TopicSpec,
    query_top_k: int,
    top_episodes: int,
    top_chunks_per_episode: int,
    max_chunk_chars: int,
) -> Dict[str, Any]:
    per_episode: Dict[str, List[Tuple[Document, Optional[float], str]]] = {}
    episode_totals: Dict[str, float] = {}

    for q in topic.queries:
        pairs = similarity_search_with_scores(vs, q, k=query_top_k)
        for doc, score in pairs:
            ep = _doc_episode_id(doc)
            if not ep:
                continue
            per_episode.setdefault(ep, []).append((doc, score, q))
            # Aggregate: sum of best (lowest) distances? We don't know direction; keep simple count+rank proxy.
            # Use a stable proxy: add 1 / (rank+1)
        # Use rank-based aggregation per query to avoid score semantics.
        for rank, (doc, _score) in enumerate(pairs):
            ep = _doc_episode_id(doc)
            if not ep:
                continue
            episode_totals[ep] = episode_totals.get(ep, 0.0) + 1.0 / float(rank + 1)

    top_eps = sorted(episode_totals.items(), key=lambda x: x[1], reverse=True)[:top_episodes]
    top_episode_ids = [ep for ep, _ in top_eps]

    evidence_chunks: List[Dict[str, Any]] = []
    top_chunks_by_episode: Dict[str, List[str]] = {}

    for ep in top_episode_ids:
        candidates = per_episode.get(ep, [])
        # Rank candidates within episode: prefer those whose text mentions entities; then by score; then by length.
        prefer_terms = [t.lower() for t in topic.entity_names if t]

        def _preferred(text: str) -> bool:
            if not prefer_terms:
                return True
            # For relationship topics, prefer chunks mentioning *all* entities to avoid pulling in generic
            # "Michael"-only chunks that are off-topic.
            if topic.topic_type == "relationship" and len(prefer_terms) >= 2:
                return all(t in text for t in prefer_terms)
            return any(t in text for t in prefer_terms)

        def _cand_key(item: Tuple[Document, Optional[float], str]) -> Tuple[int, int, int]:
            doc, score, _q = item
            text = (doc.page_content or "").lower()
            preferred = 1 if _preferred(text) else 0
            score_bucket = 0 if score is None else int(score * 1000)
            return (-preferred, score_bucket, len(doc.page_content or ""))

        candidates_sorted = sorted(candidates, key=_cand_key)

        picked_ids: List[str] = []
        seen_refs = set()
        for doc, score, q in candidates_sorted:
            ref = _chunk_ref_id(doc)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            picked_ids.append(ref)
            evidence_chunks.append(
                {
                    "chunk_ref_id": ref,
                    "episode_id": _doc_episode_id(doc),
                    "source": (doc.metadata or {}).get("source"),
                    "chunk_index": _doc_chunk_index(doc),
                    "query": q,
                    "score": score,
                    "text": _trim_text(doc.page_content or "", max_chunk_chars),
                }
            )
            if len(picked_ids) >= top_chunks_per_episode:
                break
        top_chunks_by_episode[ep] = picked_ids

    return {
        "schema": "TopicEvidenceDiscoveryV1",
        "schema_version": 1,
        "topic_id": topic.topic_id,
        "topic_type": topic.topic_type,
        "entity_names": topic.entity_names,
        "queries": topic.queries,
        "top_episode_ids": top_episode_ids,
        "top_chunks_by_episode": top_chunks_by_episode,
        "evidence_chunks": evidence_chunks,
        "params": {
            "query_top_k": query_top_k,
            "top_episodes": top_episodes,
            "top_chunks_per_episode": top_chunks_per_episode,
            "max_chunk_chars": max_chunk_chars,
        },
    }


def _make_synthesis_prompt(*, topic: TopicSpec, evidence: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You are building a factual card for a RAG knowledge layer. "
        "You MUST only use the provided evidence chunks. "
        "Do not guess or fill gaps. If the evidence does not support a detail, mark it missing. "
        "Output MUST be valid JSON only (no markdown, no code fences). "
        "Any quote you include must be an exact substring of some evidence_chunks[*].text. "
        "Citations must reference chunk_ref_id values from the evidence."
    )

    user = {
        "task": "Synthesize a TopicCardV1 from evidence chunks.",
        "schema": {
            "schema": "TopicCardV1",
            "schema_version": 1,
            "topic_id": topic.topic_id,
            "topic_type": topic.topic_type,
            "entity_names": topic.entity_names,
            "episode_ids": ["SxxEyy"],
            "bullet_facts": [
                {
                    "fact": "<one short bullet fact>",
                    "citations": ["<chunk_ref_id>"]
                }
            ],
            "supporting_quotes": [
                {
                    "quote": "<verbatim quote>",
                    "citations": ["<chunk_ref_id>"]
                }
            ],
            "missing": ["<what could not be supported from evidence>"]
        },
        "rules": [
            "Return JSON only.",
            "bullet_facts: 6-12 items max.",
            "Each bullet_facts[*].citations: 1-3 chunk_ref_id values.",
            "Only write a fact if it is explicitly supported by the cited chunk text.",
            "Choose citations that directly contain the information (avoid unrelated citations).",
            "supporting_quotes: 1-3 quotes (prefer 1-2).",
            "Quotes should be short (<= 220 chars) and should not include scene headers unless necessary.",
            "Each quote must be an exact substring of evidence_chunks[*].text (copy/paste).",
            "If you cannot find a good quote, set supporting_quotes=[] and add a missing entry.",
            "If relationship/plot details are not supported, add them to missing.",
        ],
        "topic": {
            "topic_id": topic.topic_id,
            "topic_type": topic.topic_type,
            "entity_names": topic.entity_names,
            "queries": topic.queries,
        },
        "evidence": evidence,
    }

    return system, _safe_json(user)


def synthesize_topic_card(
    *,
    llm: ChatOpenAI,
    topic: TopicSpec,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    system, user = _make_synthesis_prompt(topic=topic, evidence=evidence)
    msg = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    parsed = _try_parse_json(str(getattr(msg, "content", "") or ""))

    # Validate quotes are exact substrings; if not, drop and record missing.
    chunk_text_by_id = {
        str(c.get("chunk_ref_id")): str(c.get("text") or "") for c in (evidence.get("evidence_chunks") or [])
    }

    missing: List[str] = []
    if isinstance(parsed.get("missing"), list):
        missing = [str(x) for x in parsed.get("missing") if str(x).strip()]

    quotes_out: List[Dict[str, Any]] = []
    for q in parsed.get("supporting_quotes") or []:
        if not isinstance(q, dict):
            continue
        quote = str(q.get("quote") or "")
        citations = [str(x) for x in (q.get("citations") or []) if str(x).strip()]
        ok = False
        for cid in citations:
            text = chunk_text_by_id.get(cid, "")
            if quote and quote in text:
                ok = True
                break
        if not ok:
            missing.append(f"Quote not found in evidence: {quote[:60]}")
            continue
        quotes_out.append({"quote": quote, "citations": citations})

    facts_out: List[Dict[str, Any]] = []
    valid_chunk_ids = set(chunk_text_by_id.keys())

    entity_terms = [t.lower() for t in (topic.entity_names or []) if str(t).strip()]

    def _text_mentions_required_entities(text: str) -> bool:
        if not entity_terms:
            return True
        if topic.topic_type == "relationship" and len(entity_terms) >= 2:
            return all(t in text for t in entity_terms)
        return any(t in text for t in entity_terms)

    def _best_chunks_matching(predicate) -> List[str]:
        hits: List[str] = []
        for cid, text in chunk_text_by_id.items():
            try:
                if predicate(text.lower()):
                    hits.append(cid)
            except Exception:
                continue
        return hits

    def _extract_numbers(s: str) -> List[str]:
        return re.findall(r"\b\d{1,4}\b", s or "")

    def _extract_quoted_substrings(s: str) -> List[str]:
        # Pull out "..." segments and require they exist in evidence if used.
        return [q.strip() for q in re.findall(r'"([^"]+)"', s or "") if q.strip()]
    for f in parsed.get("bullet_facts") or []:
        if not isinstance(f, dict):
            continue
        fact = str(f.get("fact") or "").strip()
        citations = [str(x) for x in (f.get("citations") or []) if str(x).strip()]
        citations = [c for c in citations if c in valid_chunk_ids]
        if not fact:
            continue

        # Heuristic repairs/validation for citations.
        # 1) Ensure at least one citation chunk mentions the required entities.
        if citations and not any(_text_mentions_required_entities(chunk_text_by_id.get(c, "").lower()) for c in citations):
            entity_hits = _best_chunks_matching(_text_mentions_required_entities)
            if entity_hits:
                citations = entity_hits[:3]

        # 2) If the fact contains explicit numbers, ensure at least one citation contains them.
        nums = _extract_numbers(fact)
        if nums:
            for n in nums:
                if citations and any(n in (chunk_text_by_id.get(c, "") or "") for c in citations):
                    continue
                num_hits = _best_chunks_matching(lambda t, n=n: n in t)
                if num_hits:
                    # Prefer the chunk that actually contains the number.
                    citations = list(dict.fromkeys((num_hits[:1] + citations)))[:3]

        if not citations:
            missing.append(f"Fact missing citations: {fact[:80]}")
            continue

        # 3) If the fact includes quoted text, ensure it exists verbatim in at least one cited chunk.
        quoted = _extract_quoted_substrings(fact)
        if quoted:
            citation_texts = [chunk_text_by_id.get(c, "") for c in citations]
            ok = True
            for q in quoted:
                # Skip tiny quoted fragments that are often formatting artifacts.
                if len(q) < 8:
                    continue
                if not any(q in (t or "") for t in citation_texts):
                    ok = False
                    break
            if not ok:
                missing.append(f"Fact contains unsupported quote: {fact[:80]}")
                continue

        # Final check: relationship facts should cite at least one chunk mentioning all entities.
        if topic.topic_type == "relationship" and len(entity_terms) >= 2:
            if not any(all(t in (chunk_text_by_id.get(c, "").lower()) for t in entity_terms) for c in citations):
                missing.append(f"Fact citations lack both entities: {fact[:80]}")
                continue

        facts_out.append({"fact": fact, "citations": citations})

    episode_ids = [str(x).strip().upper() for x in (parsed.get("episode_ids") or []) if str(x).strip()]
    episode_ids = [e for e in episode_ids if re.match(r"^S\d{2}E\d{2}$", e)]
    if not episode_ids:
        episode_ids = [str(x) for x in (evidence.get("top_episode_ids") or []) if str(x).strip()]

    return {
        "schema": "TopicCardV1",
        "schema_version": 1,
        "build_id": str(evidence.get("build_id") or ""),
        "created_at_utc": utc_now_iso(),
        "doc_type": "derived",
        "derived_type": "topic_card",
        "topic_id": topic.topic_id,
        "topic_type": topic.topic_type,
        "entity_names": topic.entity_names,
        "queries": topic.queries,
        "episode_ids": episode_ids,
        "bullet_facts": facts_out,
        "supporting_quotes": quotes_out,
        "missing": missing,
        "evidence_chunk_refs": [str(c.get("chunk_ref_id")) for c in (evidence.get("evidence_chunks") or [])],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Two-pass topic card builder: (1) retrieve evidence, (2) synthesize one card per topic.")
    p.add_argument("--topics", default="derived/topic_cards/topics.example.json", help="Path to topics JSON")
    p.add_argument("--out-root", default="derived/artifacts", help="Output root")
    p.add_argument("--build-prefix", default=None, help="Build directory name under out-root")

    p.add_argument("--script-persist-dir", default=DEFAULT_SCRIPT_PERSIST_DIR, help="Chroma persist dir for scripts")
    p.add_argument("--script-collection-name", default=DEFAULT_SCRIPT_COLLECTION_NAME, help="Chroma collection name (optional)")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)

    p.add_argument("--pass", dest="pass_name", choices=["discover", "synthesize", "both"], default="discover")
    p.add_argument("--query-top-k", type=int, default=DEFAULT_QUERY_TOP_K)
    p.add_argument("--top-episodes", type=int, default=DEFAULT_TOP_EPISODES)
    p.add_argument("--top-chunks-per-episode", type=int, default=DEFAULT_TOP_CHUNKS_PER_EPISODE)
    p.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)

    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=2)

    args = p.parse_args()

    load_dotenv()

    topics_path = Path(args.topics)
    topics = load_topics(topics_path)

    out_root = Path(args.out_root)
    build_prefix = args.build_prefix
    if not build_prefix:
        build_prefix = f"topiccards_v1_{datetime.now(timezone.utc).date().isoformat()}"
    build_dir = out_root / build_prefix

    manifest_path = build_dir / "manifest.json"

    script_persist = str(args.script_persist_dir)
    if not Path(script_persist).is_absolute():
        script_persist = str((_REPO_ROOT / script_persist).resolve())

    manifest: Dict[str, Any]
    if manifest_path.exists():
        try:
            loaded = _read_json(manifest_path)
            manifest = loaded if isinstance(loaded, dict) else {}
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.update(
        {
            "schema": "TopicCardsBuildManifestV1",
            "schema_version": 1,
            "build_id": build_prefix,
            "created_at_utc": utc_now_iso(),
            "topics_path": str(topics_path.as_posix()),
            "params": {
                "query_top_k": int(args.query_top_k),
                "top_episodes": int(args.top_episodes),
                "top_chunks_per_episode": int(args.top_chunks_per_episode),
                "max_chunk_chars": int(args.max_chunk_chars),
            },
            "inputs": {
                "script_persist_dir": script_persist,
                "script_collection_name": args.script_collection_name,
                "embed_model": args.embed_model,
            },
            "outputs": {
                "evidence_dir": str((build_dir / "candidate_episode_discovery").as_posix()),
                "cards_dir": str((build_dir / "topic_cards").as_posix()),
            },
        }
    )
    manifest.setdefault("stats", {})
    if not isinstance(manifest.get("stats"), dict):
        manifest["stats"] = {}
    manifest["stats"]["num_topics"] = len(topics)

    if args.pass_name in {"discover", "both"}:
        vs = open_script_store(
            persist_dir=script_persist,
            collection_name=str(args.script_collection_name) if args.script_collection_name else None,
            embed_model=str(args.embed_model),
        )

        evidence_dir = build_dir / "candidate_episode_discovery"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        for topic in topics:
            evidence = discover_topic_evidence(
                vs=vs,
                topic=topic,
                query_top_k=int(args.query_top_k),
                top_episodes=int(args.top_episodes),
                top_chunks_per_episode=int(args.top_chunks_per_episode),
                max_chunk_chars=int(args.max_chunk_chars),
            )
            evidence["build_id"] = build_prefix
            evidence_path = evidence_dir / f"{topic.topic_id}.json"
            _write_json(evidence_path, evidence)

        manifest["stats"]["num_evidence_files"] = len(list(evidence_dir.glob("*.json")))
        _write_json(manifest_path, manifest)

    if args.pass_name in {"synthesize", "both"}:
        llm = ChatOpenAI(
            model=str(args.llm_model),
            temperature=float(args.temperature),
            timeout=int(args.timeout) if args.timeout else None,
            max_retries=int(args.max_retries) if args.max_retries is not None else None,
        )

        evidence_dir = build_dir / "candidate_episode_discovery"
        cards_dir = build_dir / "topic_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)

        # If we didn't run discovery in this invocation, expect evidence files already exist.
        if not evidence_dir.exists():
            raise SystemExit(f"Missing evidence dir: {evidence_dir}. Run with --pass discover first.")

        for topic in topics:
            evidence_path = evidence_dir / f"{topic.topic_id}.json"
            if not evidence_path.exists():
                continue
            evidence = _read_json(evidence_path)
            evidence["build_id"] = build_prefix

            card = synthesize_topic_card(llm=llm, topic=topic, evidence=evidence)
            out_path = cards_dir / f"{topic.topic_id}.json"
            _write_json(out_path, card)

        manifest["stats"]["num_evidence_files"] = len(list(evidence_dir.glob("*.json")))
        manifest["stats"]["num_cards"] = len(list(cards_dir.glob("*.json")))
        _write_json(manifest_path, manifest)

    print(f"Wrote build to: {build_dir}")
    print(_safe_json(manifest["stats"]))


if __name__ == "__main__":
    main()
