from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from derived.segment_episode import EpisodeScript, iter_episode_scripts

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"
DEFAULT_SEGMENTS_ROOT = "derived/artifacts/segments"
DEFAULT_OUT_ROOT = "derived/artifacts"
# Default to nano to keep reduce costs down; callers can override.
DEFAULT_LLM_MODEL = "gpt-4.1-nano"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _coerce_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


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


def _find_episode(*, docs_dir: Path, episode_id: str) -> EpisodeScript:
    target = episode_id.strip().upper()
    for ep in iter_episode_scripts(docs_dir=docs_dir):
        if ep.episode_id == target:
            return ep
    raise FileNotFoundError(f"Episode {target} not found under {docs_dir}/scripts")


def _load_episode_segments(*, segments_root: Path, episode_id: str) -> Dict[str, Any]:
    path = segments_root / f"{episode_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing episode segments file: {path}. Run derived/segment_episode.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segment_summary(*, path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "SegmentSummaryV1":
        raise ValueError(f"Unexpected schema in {path}: {raw.get('schema')}")
    return raw


def _trim_snippet(s: str, max_chars: int = 180) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _make_evidence_pool(segment_summaries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, int, int], str]]:
    """Return (pool_by_id, key_to_id).

    key is (segment_id, char_start, char_end) to dedupe.
    """

    pool: Dict[str, Dict[str, Any]] = {}
    key_to_id: Dict[Tuple[str, int, int], str] = {}

    def _add(span: Dict[str, Any]) -> Optional[str]:
        if not isinstance(span, dict):
            return None
        segment_id = str(span.get("segment_id") or "").strip()
        try:
            char_start = int(span.get("char_start"))
            char_end = int(span.get("char_end"))
        except Exception:
            return None
        if not segment_id or char_end <= char_start:
            return None

        key = (segment_id, char_start, char_end)
        existing = key_to_id.get(key)
        if existing:
            return existing

        eid = f"E{len(pool) + 1:04d}"
        key_to_id[key] = eid
        pool[eid] = {
            "episode_id": str(span.get("episode_id") or "").strip(),
            "segment_id": segment_id,
            "source": str(span.get("source") or "").strip(),
            "char_start": char_start,
            "char_end": char_end,
            "snippet": _trim_snippet(str(span.get("snippet") or "")),
        }
        return eid

    def _take_some(spans: Any, max_items: int) -> List[str]:
        out: List[str] = []
        for sp in _coerce_list(spans)[:max_items]:
            eid = _add(sp)
            if eid:
                out.append(eid)
        return out

    for ss in segment_summaries:
        for beat in _coerce_list(ss.get("beats")):
            if not isinstance(beat, dict):
                continue
            _take_some(beat.get("evidence"), max_items=2)

        for q in _coerce_list(ss.get("notable_quotes")):
            if not isinstance(q, dict):
                continue
            _take_some(q.get("evidence"), max_items=1)

        for u in _coerce_list(ss.get("uncertainty_items")):
            if not isinstance(u, dict):
                continue
            _take_some(u.get("evidence"), max_items=1)

    return pool, key_to_id


def _segment_context_for_llm(*, segment_summaries: List[Dict[str, Any]], key_to_id: Dict[Tuple[str, int, int], str]) -> List[Dict[str, Any]]:
    segments_out: List[Dict[str, Any]] = []

    def _evidence_ids(spans: Any, max_items: int) -> List[str]:
        ids: List[str] = []
        for sp in _coerce_list(spans)[:max_items]:
            if not isinstance(sp, dict):
                continue
            seg = str(sp.get("segment_id") or "").strip()
            try:
                a = int(sp.get("char_start"))
                b = int(sp.get("char_end"))
            except Exception:
                continue
            eid = key_to_id.get((seg, a, b))
            if eid:
                ids.append(eid)
        return ids

    for ss in sorted(segment_summaries, key=lambda x: int(x.get("segment_index", 0))):
        seg_obj: Dict[str, Any] = {
            "segment_id": ss.get("segment_id"),
            "segment_index": ss.get("segment_index"),
            "people": _coerce_list(ss.get("people")),
            "beats": [],
            "uncertainties": [],
        }

        for beat in _coerce_list(ss.get("beats")):
            if not isinstance(beat, dict):
                continue
            summary = str(beat.get("summary") or "").strip()
            if not summary:
                continue
            seg_obj["beats"].append(
                {
                    "type": str(beat.get("type") or "other").strip(),
                    "summary": summary,
                    "evidence_ids": _evidence_ids(beat.get("evidence"), max_items=2),
                }
            )

        for u in _coerce_list(ss.get("uncertainty_items")):
            if not isinstance(u, dict):
                continue
            question = str(u.get("question") or "").strip()
            if not question:
                continue
            seg_obj["uncertainties"].append(
                {
                    "question": question,
                    "why_uncertain": str(u.get("why_uncertain") or "").strip(),
                    "evidence_ids": _evidence_ids(u.get("evidence"), max_items=1),
                }
            )

        segments_out.append(seg_obj)

    return segments_out


def _make_prompt(*, episode: EpisodeScript, episode_segments: Dict[str, Any], segments_ctx: List[Dict[str, Any]], evidence_pool: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "You are a careful reducer creating an episode-level derived card for a RAG system. "
        "You must ONLY use the provided segment summaries and evidence pool. "
        "DO NOT invent facts, events, motivations, relationships, or outcomes. "
        "If something is not explicitly supported, omit it or mark it as uncertain. "
        "Output MUST be valid JSON only (no markdown, no code fences)."
    )

    user = {
        "task": "Create an EpisodeDerivedCardV1 JSON object.",
        "episode": {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "source": episode.source_path,
        },
        "schema": {
            "schema": "EpisodeDerivedCardV1",
            "schema_version": 1,
            "doc_type": "derived",
            "derived_type": "episode_card",
            "episode_id": episode.episode_id,
            "title": episode.title,
            "one_paragraph_synopsis": "<one paragraph>",
            "main_threads": [
                {"thread": "<thread>", "evidence_ids": ["E0001", "E0002"]}
            ],
            "character_highlights": [
                {"character": "<name>", "what_changes": "<what changes>", "evidence_ids": ["E0001"]}
            ],
            "relationships": [
                {"pair": ["A", "B"], "status": "<status>", "evidence_ids": ["E0001"]}
            ],
            "tags": ["<tag>"] ,
            "uncertainties": [
                {"question": "<open question>", "why_uncertain": "<why>", "evidence_ids": ["E0001"]}
            ],
            "build_notes": "<optional>"
        },
        "rules": [
            "Return JSON only.",
            "Use ONLY the provided segment summaries and evidence_pool.",
            "Do NOT include any claim without at least one evidence_id.",
            "All evidence_ids must exist in evidence_pool.",
            "For each item, use a SMALL number of evidence_ids (typically 1-3, max 6).",
            "Prefer fewer, high-signal threads over many tiny ones.",
            "Keep character_highlights/relationships to what is clearly supported.",
            "If uncertain, add an uncertainties item rather than guessing.",
        ],
        "context": {
            "episode_segments": {
                "episode_id": episode_segments.get("episode_id"),
                "title": episode_segments.get("title"),
                "source": episode_segments.get("source"),
                "segment_ids": [s.get("segment_id") for s in _coerce_list(episode_segments.get("segments"))],
            },
            "segments": segments_ctx,
            "evidence_pool": evidence_pool,
        },
    }

    return system, _safe_json(user)


def _expand_evidence_ids(
    evidence_ids: Any,
    evidence_pool: Dict[str, Dict[str, Any]],
    *,
    max_items: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for eid in _coerce_list(evidence_ids)[: int(max_items)]:
        eid = str(eid).strip()
        if not eid:
            continue
        span = evidence_pool.get(eid)
        if not span:
            continue
        out.append(span)
    return out


def _normalize_episode_card(
    *,
    raw: Dict[str, Any],
    build_id: str,
    created_at_utc: str,
    llm_model: str,
    episode: EpisodeScript,
    episode_segments: Dict[str, Any],
    evidence_pool: Dict[str, Dict[str, Any]],
    segments_build_id: str,
) -> Dict[str, Any]:
    one_paragraph = str(raw.get("one_paragraph_synopsis") or "").strip()

    def _norm_threads() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in _coerce_list(raw.get("main_threads")):
            if not isinstance(t, dict):
                continue
            thread = str(t.get("thread") or "").strip()
            if not thread:
                continue
            evidence = _expand_evidence_ids(t.get("evidence_ids"), evidence_pool, max_items=6)
            if not evidence:
                continue
            out.append({"thread": thread, "evidence": evidence})
        return out

    def _norm_char_highlights() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for h in _coerce_list(raw.get("character_highlights")):
            if not isinstance(h, dict):
                continue
            character = str(h.get("character") or "").strip()
            what_changes = str(h.get("what_changes") or "").strip()
            if not character or not what_changes:
                continue
            evidence = _expand_evidence_ids(h.get("evidence_ids"), evidence_pool, max_items=4)
            if not evidence:
                continue
            out.append({"character": character, "what_changes": what_changes, "evidence": evidence})
        return out

    def _norm_relationships() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in _coerce_list(raw.get("relationships")):
            if not isinstance(r, dict):
                continue
            pair = [str(x).strip() for x in _coerce_list(r.get("pair")) if str(x).strip()]
            if len(pair) != 2:
                continue
            status = str(r.get("status") or "").strip()
            if not status:
                continue
            evidence = _expand_evidence_ids(r.get("evidence_ids"), evidence_pool, max_items=4)
            if not evidence:
                continue
            out.append({"pair": pair, "status": status, "evidence": evidence})
        return out

    def _norm_tags() -> List[str]:
        tags: List[str] = []
        for t in _coerce_list(raw.get("tags")):
            s = str(t).strip()
            if s:
                tags.append(s)
        # de-dupe preserving order
        out: List[str] = []
        seen = set()
        for t in tags:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            out.append(t)
        return out

    def _norm_uncertainties() -> Tuple[List[str], List[Dict[str, Any]]]:
        items_out: List[Dict[str, Any]] = []
        text_out: List[str] = []

        for u in _coerce_list(raw.get("uncertainties")):
            if isinstance(u, str):
                q = u.strip()
                if q:
                    text_out.append(q)
                continue

            if not isinstance(u, dict):
                continue
            q = str(u.get("question") or "").strip()
            if not q:
                continue
            why = str(u.get("why_uncertain") or "").strip()
            evidence = _expand_evidence_ids(u.get("evidence_ids"), evidence_pool, max_items=3)
            if not evidence:
                continue
            text_out.append(q)
            items_out.append({"question": q, "why_uncertain": why, "evidence": evidence})

        return text_out, items_out

    source_segments = [s.get("segment_id") for s in _coerce_list(episode_segments.get("segments")) if isinstance(s, dict)]

    uncertainties_text, uncertainty_items = _norm_uncertainties()

    build_notes = str(raw.get("build_notes") or "").strip()
    if segments_build_id:
        prefix = f"segments_build_id={segments_build_id}"
        build_notes = f"{prefix}; {build_notes}".strip("; ")

    return {
        "schema": "EpisodeDerivedCardV1",
        "schema_version": 1,
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "llm_model": llm_model,
        "doc_type": "derived",
        "derived_type": "episode_card",
        "episode_id": episode.episode_id,
        "title": episode.title,
        "one_paragraph_synopsis": one_paragraph,
        "main_threads": _norm_threads(),
        "character_highlights": _norm_char_highlights(),
        "relationships": _norm_relationships(),
        "tags": _norm_tags(),
        "source_segments": source_segments,
        "build_notes": build_notes,
        "uncertainties": uncertainties_text,
        "uncertainty_items": uncertainty_items,
    }


@dataclass(frozen=True)
class ReduceConfig:
    docs_dir: Path
    segments_root: Path
    segment_summaries_root: Path
    out_root: Path
    episode_id: str
    segments_build_id: str
    llm_model: str
    build_id: Optional[str]
    force: bool
    prompt_only: bool


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")


def reduce_episode(cfg: ReduceConfig) -> Path:
    load_dotenv()

    created_at = utc_now_iso()
    build_id = (cfg.build_id or "").strip() or f"derived_episodecard_{created_at.replace(':', '-')}"

    episode = _find_episode(docs_dir=cfg.docs_dir, episode_id=cfg.episode_id)
    episode_segments = _load_episode_segments(segments_root=cfg.segments_root, episode_id=episode.episode_id)

    expected_segment_ids = [
        s.get("segment_id")
        for s in _coerce_list(episode_segments.get("segments"))
        if isinstance(s, dict) and s.get("segment_id")
    ]

    if not expected_segment_ids:
        raise ValueError(f"No segments found in {cfg.segments_root}/{episode.episode_id}.json")

    seg_summ_dir = cfg.segment_summaries_root / cfg.segments_build_id / "segment_summaries" / episode.episode_id
    if not seg_summ_dir.exists():
        raise FileNotFoundError(
            f"Missing segment_summaries directory: {seg_summ_dir}. "
            f"Run derived/summarize_segments.py first."
        )

    segment_summaries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for seg_id in expected_segment_ids:
        path = seg_summ_dir / f"{seg_id}.json"
        if not path.exists():
            missing.append(seg_id)
            continue
        segment_summaries.append(_load_segment_summary(path=path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} segment summaries for {episode.episode_id} build {cfg.segments_build_id}: {missing[:5]}"
        )

    evidence_pool, key_to_id = _make_evidence_pool(segment_summaries)
    segments_ctx = _segment_context_for_llm(segment_summaries=segment_summaries, key_to_id=key_to_id)

    out_dir = cfg.out_root / build_id / "episode_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode.episode_id}.json"

    manifest_path = cfg.out_root / build_id / "manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "EpisodeCardRunManifestV1",
        "schema_version": 1,
        "build_id": build_id,
        "created_at_utc": created_at,
        "episode": {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "source": episode.source_path,
        },
        "inputs": {
            "segments_build_id": cfg.segments_build_id,
            "segment_summaries_dir": str(seg_summ_dir.as_posix()),
            "segment_count": len(segment_summaries),
            "evidence_pool_size": len(evidence_pool),
        },
        "config": {
            "llm_model": cfg.llm_model,
            "force": bool(cfg.force),
            "prompt_only": bool(cfg.prompt_only),
        },
        "plan": {
            "planned_llm_calls": 0 if cfg.prompt_only else 1,
            "out_file": f"episode_cards/{episode.episode_id}.json",
        },
        "execution": {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "llm_calls_made": 0,
            "written": 0,
            "errors": 0,
        },
    }
    _write_manifest(manifest_path, manifest)

    if out_path.exists() and not cfg.force:
        manifest["execution"]["status"] = "skipped_existing"
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        _write_manifest(manifest_path, manifest)
        return out_dir

    system, user = _make_prompt(
        episode=episode,
        episode_segments=episode_segments,
        segments_ctx=segments_ctx,
        evidence_pool=evidence_pool,
    )

    if cfg.prompt_only:
        prompt_path = out_dir / f"{episode.episode_id}.prompt.json"
        prompt_path.write_text(
            json.dumps({"system": system, "user": user}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote prompt for review: {prompt_path}")
        manifest["execution"]["status"] = "prompt_only"
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        _write_manifest(manifest_path, manifest)
        return out_dir

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    llm = ChatOpenAI(model=cfg.llm_model, temperature=0.0)

    t0 = time.time()
    call_started = utc_now_iso()
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    manifest["execution"]["llm_calls_made"] += 1
    content = (getattr(msg, "content", None) or "").strip()

    try:
        raw = _try_parse_json(content)
    except Exception:
        bad_path = out_dir / f"{episode.episode_id}.raw.txt"
        bad_path.write_text(content, encoding="utf-8")
        manifest["execution"]["errors"] += 1
        manifest["execution"]["status"] = "error_invalid_json"
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        _write_manifest(manifest_path, manifest)
        raise ValueError(f"LLM did not return valid JSON. Raw saved to {bad_path}")

    normalized = _normalize_episode_card(
        raw=raw,
        build_id=build_id,
        created_at_utc=created_at,
        llm_model=cfg.llm_model,
        episode=episode,
        episode_segments=episode_segments,
        evidence_pool=evidence_pool,
        segments_build_id=cfg.segments_build_id,
    )

    out_path.write_text(_safe_json(normalized), encoding="utf-8")
    ms = int((time.time() - t0) * 1000)
    print(f"Wrote {out_path.name} ({ms}ms)")

    manifest["execution"]["written"] = 1
    manifest["execution"]["status"] = "done"
    manifest["execution"]["ended_at_utc"] = utc_now_iso()
    manifest["execution"]["call_started_at_utc"] = call_started
    manifest["execution"]["call_duration_ms"] = ms
    _write_manifest(manifest_path, manifest)

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Reduce SegmentSummaryV1 files into an EpisodeDerivedCardV1")
    p.add_argument("--episode-id", required=True, help="Episode ID like S07E24")
    p.add_argument(
        "--segments-build-id",
        required=True,
        help="Build ID folder that contains segment_summaries/<episode_id>/...",
    )
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p.add_argument("--segments-root", default=DEFAULT_SEGMENTS_ROOT)
    p.add_argument("--artifacts-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--build-id", default=None, help="Output build id (folder under derived/artifacts)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--prompt-only", action="store_true")

    args = p.parse_args()

    cfg = ReduceConfig(
        docs_dir=Path(args.docs_dir),
        segments_root=Path(args.segments_root),
        segment_summaries_root=Path(args.artifacts_root),
        out_root=Path(args.artifacts_root),
        episode_id=str(args.episode_id).strip().upper(),
        segments_build_id=str(args.segments_build_id).strip(),
        llm_model=str(args.llm_model).strip(),
        build_id=args.build_id,
        force=bool(args.force),
        prompt_only=bool(args.prompt_only),
    )

    reduce_episode(cfg)


if __name__ == "__main__":
    main()
