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

try:
    # Raised when OpenAI blocks a response for policy reasons.
    from openai import ContentFilterFinishReasonError  # type: ignore
except Exception:  # pragma: no cover
    ContentFilterFinishReasonError = None  # type: ignore

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from derived.segment_episode import EpisodeScript, EpisodeSegment, iter_episode_scripts, segment_episode_text

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"
DEFAULT_OUT_ROOT = "derived/artifacts"
# Default to nano to keep per-segment map costs down.
DEFAULT_LLM_MODEL = "gpt-4.1-nano"

# Prompt/normalization caps to avoid giant outputs (which can truncate or produce invalid JSON).
MAX_BEATS = 6
MAX_SUPPORTING_QUOTES_PER_BEAT = 2
MAX_NOTABLE_QUOTES = 4
MAX_SUPPORTING_QUOTES_PER_NOTABLE = 2
MAX_UNCERTAINTIES = 3
MAX_SUPPORTING_QUOTES_PER_UNCERTAINTY = 1
MAX_TOTAL_SUPPORTING_QUOTES = 14
MAX_QUOTE_CHARS = 220


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _find_episode(*, docs_dir: Path, episode_id: str) -> EpisodeScript:
    target = episode_id.strip().upper()
    for ep in iter_episode_scripts(docs_dir=docs_dir):
        if ep.episode_id == target:
            return ep
    raise FileNotFoundError(f"Episode {target} not found under {docs_dir}/scripts")


def _segment_text(ep: EpisodeScript, seg: EpisodeSegment) -> str:
    return ep.body_text[seg.segment_char_start : seg.segment_char_end]


def _make_prompt(*, episode: EpisodeScript, segment: EpisodeSegment, segment_text: str) -> Tuple[str, str]:
    # We ask the model to provide verbatim snippets it is basing claims on.
    # Then we can locate those snippets and compute char offsets deterministically.
    system = (
        "You are a careful analyst summarizing a segment of a TV episode script for a RAG system. "
        "You must ONLY use the provided segment text as evidence. "
        "DO NOT make up facts, events, motivations, relationships, or outcomes. "
        "If a detail is not explicitly supported by the segment text, omit it or mark it as unknown/uncertain. "
        "Do not guess. If unsure, record it under uncertainties. "
        "Output MUST be valid JSON only (no markdown, no code fences). "
        "Any supporting_quote you output must be an exact substring of the segment text. "
        "Hard limits: beats <= 6; per-beat supporting_quotes <= 2; notable_quotes <= 4; "
        "uncertainties <= 3; keep quotes short. "
        "Never include raw newline characters inside JSON strings (replace with a space)."
    )

    user = {
        "task": "Create a SegmentSummaryV1 JSON object.",
        "schema": {
            "schema": "SegmentSummaryV1",
            "schema_version": 1,
            "episode_id": episode.episode_id,
            "segment_id": segment.segment_id,
            "segment_index": segment.segment_index,
            "people": ["<character names>"] ,
            "beats": [
                {
                    "type": "event|joke|conflict|reveal|relationship|other",
                    "summary": "<one sentence>",
                    "supporting_quotes": ["<verbatim snippet 1>", "<verbatim snippet 2>"]
                }
            ],
            "notable_quotes": [
                {
                    "quote": "<verbatim quote>",
                    "speaker": "<speaker or null>",
                    "supporting_quotes": ["<verbatim snippet that contains the quote>"]
                }
            ],
            "uncertainties": [
                {
                    "question": "<what is unknown / unresolved>",
                    "why_uncertain": "<why this segment cannot answer it>",
                    "supporting_quotes": ["<verbatim snippet that shows the ambiguity>"]
                }
            ]
        },
        "rules": [
            "Return JSON only.",
            "You may ONLY summarize what is explicitly supported by the segment text.",
            "Do NOT invent names, events, intent, or outcomes.",
            "supporting_quotes must be exact substrings from the segment text.",
            "uncertainties.supporting_quotes must also be exact substrings from the segment text.",
            f"Beats: at most {MAX_BEATS}.",
            f"Each beat.supporting_quotes: at most {MAX_SUPPORTING_QUOTES_PER_BEAT} items.",
            f"notable_quotes: at most {MAX_NOTABLE_QUOTES} items.",
            f"Each notable_quotes[*].supporting_quotes: at most {MAX_SUPPORTING_QUOTES_PER_NOTABLE} items.",
            f"uncertainties: at most {MAX_UNCERTAINTIES} items.",
            f"Each uncertainties[*].supporting_quotes: at most {MAX_SUPPORTING_QUOTES_PER_UNCERTAINTY} item.",
            f"Total count of all supporting_quotes across the whole JSON: <= {MAX_TOTAL_SUPPORTING_QUOTES}.",
            f"Keep every quote string <= {MAX_QUOTE_CHARS} characters.",
            "Do not put newline characters inside any JSON string values.",
            "If the segment is mostly filler, keep beats small and say so.",
        ],
        "context": {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "source": episode.source_path,
            "segment_id": segment.segment_id,
            "segment_offsets_in_episode_body": {
                "char_start": segment.segment_char_start,
                "char_end": segment.segment_char_end,
            },
            "segment_text": segment_text,
        },
    }

    return system, _safe_json(user)


def _locate_evidence_spans(
    *,
    episode: EpisodeScript,
    segment: EpisodeSegment,
    segment_text: str,
    supporting_quotes: List[str],
    max_snippet_chars: int = 240,
) -> List[Dict[str, Any]]:
    """Turn verbatim supporting_quotes into EvidenceSpanV1 objects using substring search.

    Offsets are relative to the episode body (header stripped), matching segment_episode.py.
    """
    spans: List[Dict[str, Any]] = []

    for q in supporting_quotes:
        q = (q or "").strip()
        if not q:
            continue

        found_text = q
        rel = segment_text.find(q)
        if rel < 0:
            # Best-effort: tolerate whitespace differences (e.g., model replaces newlines with spaces).
            parts = [p for p in re.split(r"\s+", q) if p]
            if len(parts) >= 2:
                pattern = r"\s+".join(re.escape(p) for p in parts)
                m = re.search(pattern, segment_text, flags=re.DOTALL)
                if m is not None:
                    rel = m.start()
                    found_text = m.group(0)
        if rel < 0:
            continue

        abs_start = int(segment.segment_char_start + rel)
        abs_end = int(abs_start + len(found_text))

        snippet = found_text
        if len(snippet) > max_snippet_chars:
            snippet = snippet[: max_snippet_chars - 3] + "..."

        spans.append(
            {
                "episode_id": episode.episode_id,
                "segment_id": segment.segment_id,
                "source": episode.source_path,
                "char_start": abs_start,
                "char_end": abs_end,
                "snippet": snippet,
            }
        )

    # Fallback: if nothing matched, add a segment-level span so we never lose provenance.
    if not spans:
        fallback = segment_text.strip()[:max_snippet_chars]
        spans.append(
            {
                "episode_id": episode.episode_id,
                "segment_id": segment.segment_id,
                "source": episode.source_path,
                "char_start": int(segment.segment_char_start),
                "char_end": int(segment.segment_char_end),
                "snippet": fallback,
            }
        )

    return spans


def _coerce_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        # Remove leading fence line
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s.rsplit("\n", 1)[0] if "\n" in s else ""
    return s.strip()


def _escape_control_chars_in_json_strings(s: str) -> str:
    """Escape control characters that appear inside JSON string literals.

    Some model outputs contain literal newlines (or other control chars) inside quoted strings,
    which is invalid JSON. This function makes a best-effort pass converting those to escaped
    sequences so that `json.loads` can succeed.
    """

    out: List[str] = []
    in_string = False
    escaping = False

    for ch in s:
        if in_string:
            if escaping:
                out.append(ch)
                escaping = False
                continue

            if ch == "\\":
                out.append(ch)
                escaping = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 32:
                out.append(f"\\u{ord(ch):04x}")
                continue

            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True

    return "".join(out)


def _try_parse_json(content: str) -> Dict[str, Any]:
    """Best-effort JSON parser for LLM output.

    The LLM occasionally returns almost-JSON (e.g. markdown fences, trailing commas,
    or mistakenly using \"...\" as the literal quotes for string values).
    """

    s = (content or "").strip()
    if not s:
        raise json.JSONDecodeError("Expecting value", s, 0)

    # Attempt 1: raw JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Attempt 2: strip fences and try again
    s2 = _strip_code_fences(s)
    try:
        return json.loads(s2)
    except Exception:
        pass

    # Attempt 3: common repairs
    repaired = _escape_control_chars_in_json_strings(s2)
    repaired = repaired.replace('\\"', '"')
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    # Some models prepend text; try extracting first {...} block.
    if "{" in repaired and "}" in repaired:
        repaired = repaired[repaired.find("{") : repaired.rfind("}") + 1]
    return json.loads(repaired)


def _normalize_segment_summary(
    *,
    raw: Dict[str, Any],
    episode: EpisodeScript,
    segment: EpisodeSegment,
    segment_text: str,
    build_id: str,
    created_at_utc: str,
    llm_model: str,
) -> Dict[str, Any]:
    """Normalize the model output into a concrete SegmentSummaryV1 with evidence spans."""

    def _clean_text(x: Any) -> str:
        s = str(x or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > MAX_QUOTE_CHARS:
            s = s[: MAX_QUOTE_CHARS - 1].rstrip() + "…"
        return s

    # De-dupe people while preserving order.
    people: List[str] = []
    seen_people = set()
    for p in _coerce_list(raw.get("people")):
        sp = str(p).strip()
        if not sp or sp in seen_people:
            continue
        seen_people.add(sp)
        people.append(sp)

    remaining_quotes_budget = MAX_TOTAL_SUPPORTING_QUOTES

    beats_out: List[Dict[str, Any]] = []
    for b in _coerce_list(raw.get("beats")):
        if not isinstance(b, dict):
            continue
        summary = str(b.get("summary") or "").strip()
        if not summary:
            continue
        btype = str(b.get("type") or "other").strip()
        supporting_quotes: List[str] = []
        for q in _coerce_list(b.get("supporting_quotes")):
            sq = _clean_text(q)
            if sq:
                supporting_quotes.append(sq)
        supporting_quotes = supporting_quotes[: min(MAX_SUPPORTING_QUOTES_PER_BEAT, remaining_quotes_budget)]
        remaining_quotes_budget -= len(supporting_quotes)
        evidence = _locate_evidence_spans(
            episode=episode,
            segment=segment,
            segment_text=segment_text,
            supporting_quotes=supporting_quotes,
        )
        beats_out.append(
            {
                "type": btype,
                "summary": summary,
                "evidence": evidence,
            }
        )

        if len(beats_out) >= MAX_BEATS:
            break

    quotes_out: List[Dict[str, Any]] = []
    for q in _coerce_list(raw.get("notable_quotes")):
        if not isinstance(q, dict):
            continue
        quote = _clean_text(q.get("quote"))
        if not quote:
            continue
        speaker = q.get("speaker")
        if speaker is None:
            speaker_out: Optional[str] = None
        else:
            speaker_out = str(speaker).strip() or None

        supporting_quotes: List[str] = []
        for x in _coerce_list(q.get("supporting_quotes")):
            sx = _clean_text(x)
            if sx:
                supporting_quotes.append(sx)
        supporting_quotes = supporting_quotes[: min(MAX_SUPPORTING_QUOTES_PER_NOTABLE, remaining_quotes_budget)]
        remaining_quotes_budget -= len(supporting_quotes)
        evidence = _locate_evidence_spans(
            episode=episode,
            segment=segment,
            segment_text=segment_text,
            supporting_quotes=supporting_quotes or [quote],
        )

        quotes_out.append(
            {
                "quote": quote,
                "speaker": speaker_out,
                "evidence": evidence,
            }
        )

        if len(quotes_out) >= MAX_NOTABLE_QUOTES:
            break

    uncertainty_items: List[Dict[str, Any]] = []
    uncertainties_text: List[str] = []

    for u in _coerce_list(raw.get("uncertainties")):
        if len(uncertainty_items) >= MAX_UNCERTAINTIES:
            break
        # Back-compat: allow a simple string uncertainty.
        if isinstance(u, str):
            text = u.strip()
            if not text:
                continue
            uncertainties_text.append(text)
            uncertainty_items.append(
                {
                    "question": text,
                    "why_uncertain": "",
                    "evidence": _locate_evidence_spans(
                        episode=episode,
                        segment=segment,
                        segment_text=segment_text,
                        supporting_quotes=[],
                    ),
                }
            )
            continue

        if not isinstance(u, dict):
            continue

        question = str(u.get("question") or "").strip()
        why_uncertain = str(u.get("why_uncertain") or "").strip()
        supporting_quotes: List[str] = []
        for q in _coerce_list(u.get("supporting_quotes")):
            sq = _clean_text(q)
            if sq:
                supporting_quotes.append(sq)
        supporting_quotes = supporting_quotes[: min(MAX_SUPPORTING_QUOTES_PER_UNCERTAINTY, remaining_quotes_budget)]
        remaining_quotes_budget -= len(supporting_quotes)
        if not question:
            continue

        evidence = _locate_evidence_spans(
            episode=episode,
            segment=segment,
            segment_text=segment_text,
            supporting_quotes=supporting_quotes,
        )

        uncertainties_text.append(question)
        uncertainty_items.append(
            {
                "question": question,
                "why_uncertain": why_uncertain,
                "evidence": evidence,
            }
        )

    return {
        "schema": "SegmentSummaryV1",
        "schema_version": 1,
        "build_id": build_id,
        "created_at_utc": created_at_utc,
        "llm_model": llm_model,
        "episode_id": episode.episode_id,
        "segment_id": segment.segment_id,
        "segment_index": segment.segment_index,
        "segment_char_start": segment.segment_char_start,
        "segment_char_end": segment.segment_char_end,
        "people": people,
        "beats": beats_out,
        "notable_quotes": quotes_out,
        # Keep the legacy list-of-strings for convenience/compatibility.
        "uncertainties": uncertainties_text,
        # Structured uncertainties for episode-level resolution.
        "uncertainty_items": uncertainty_items,
    }


@dataclass(frozen=True)
class SummarizeConfig:
    docs_dir: Path
    out_root: Path
    episode_id: str
    llm_model: str
    target_tokens: int
    min_tokens: int
    max_segments: Optional[int]
    segment_index: Optional[int]
    build_id: Optional[str]
    force: bool
    prompt_only: bool
    timeout: Optional[float] = None
    max_retries: Optional[int] = None


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")


def summarize_episode_segments(cfg: SummarizeConfig) -> Path:
    load_dotenv()

    created_at = utc_now_iso()
    build_id = (cfg.build_id or "").strip() or f"derived_segsummary_{created_at.replace(':', '-')}"

    ep = _find_episode(docs_dir=cfg.docs_dir, episode_id=cfg.episode_id)
    segments = segment_episode_text(
        episode_id=ep.episode_id,
        body_text=ep.body_text,
        target_tokens=int(cfg.target_tokens),
        min_tokens=int(cfg.min_tokens),
    )

    if cfg.segment_index is not None:
        segments = [s for s in segments if s.segment_index == int(cfg.segment_index)]

    if cfg.max_segments is not None:
        segments = segments[: int(cfg.max_segments)]

    out_dir = cfg.out_root / build_id / "segment_summaries" / ep.episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg.out_root / build_id / "manifest.json"
    planned = [
        {
            "segment_id": s.segment_id,
            "segment_index": s.segment_index,
            "segment_char_start": s.segment_char_start,
            "segment_char_end": s.segment_char_end,
            "approx_tokens": s.approx_tokens,
            "out_file": f"segment_summaries/{ep.episode_id}/{s.segment_id}.json",
        }
        for s in segments
    ]
    manifest: Dict[str, Any] = {
        "schema": "SegmentSummaryRunManifestV1",
        "schema_version": 1,
        "build_id": build_id,
        "created_at_utc": created_at,
        "episode": {
            "episode_id": ep.episode_id,
            "title": ep.title,
            "source": ep.source_path,
            "body_char_count": len(ep.body_text),
        },
        "config": {
            "llm_model": cfg.llm_model,
            "timeout": cfg.timeout,
            "max_retries": cfg.max_retries,
            "target_tokens": int(cfg.target_tokens),
            "min_tokens": int(cfg.min_tokens),
            "max_segments": cfg.max_segments,
            "segment_index": cfg.segment_index,
            "force": bool(cfg.force),
            "prompt_only": bool(cfg.prompt_only),
        },
        "plan": {
            "planned_segments": planned,
            "planned_llm_calls": 0 if cfg.prompt_only else len(segments),
        },
        "execution": {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "llm_calls_planned": 0 if cfg.prompt_only else len(segments),
            "llm_calls_made": 0,
            "skipped_existing": 0,
            "written": 0,
            "errors": 0,
            "segments": [],
        },
    }
    _write_manifest(manifest_path, manifest)

    # Prompt-only mode: write a prompt.json file for review and exit.
    if cfg.prompt_only:
        if not segments:
            raise ValueError("No segments to summarize (check --segment-index / --max-segments)")
        s = segments[0]
        s_text = _segment_text(ep, s)
        system, user = _make_prompt(episode=ep, segment=s, segment_text=s_text)
        prompt_path = out_dir / f"{s.segment_id}.prompt.json"
        prompt_path.write_text(
            json.dumps({"system": system, "user": user}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote prompt for review: {prompt_path}")

        manifest["execution"]["status"] = "prompt_only"
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        manifest["execution"]["segments"].append(
            {
                "segment_id": s.segment_id,
                "status": "prompt_written",
                "prompt_file": f"segment_summaries/{ep.episode_id}/{s.segment_id}.prompt.json",
            }
        )
        _write_manifest(manifest_path, manifest)
        return out_dir

    # LLM mode
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    # Prefer JSON-mode to reduce invalid / truncated JSON outputs.
    try:
        llm_kwargs: Dict[str, Any] = {
            "model": cfg.llm_model,
            "temperature": 0.0,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        }
        if cfg.timeout is not None:
            llm_kwargs["timeout"] = float(cfg.timeout)
        if cfg.max_retries is not None:
            llm_kwargs["max_retries"] = int(cfg.max_retries)
        llm = ChatOpenAI(**llm_kwargs)
    except TypeError:
        # Older langchain_openai versions may not accept model_kwargs here.
        llm_kwargs = {
            "model": cfg.llm_model,
            "temperature": 0.0,
        }
        if cfg.timeout is not None:
            llm_kwargs["timeout"] = float(cfg.timeout)
        if cfg.max_retries is not None:
            llm_kwargs["max_retries"] = int(cfg.max_retries)
        llm = ChatOpenAI(**llm_kwargs)

    n_total = len(segments)
    for i, s in enumerate(segments):
        out_path = out_dir / f"{s.segment_id}.json"
        if out_path.exists() and not cfg.force:
            print(f"[{i+1}/{n_total}] Skip (exists): {out_path.name}")
            manifest["execution"]["skipped_existing"] += 1
            manifest["execution"]["segments"].append(
                {
                    "segment_id": s.segment_id,
                    "status": "skipped_existing",
                    "out_file": f"segment_summaries/{ep.episode_id}/{s.segment_id}.json",
                }
            )
            _write_manifest(manifest_path, manifest)
            continue

        s_text = _segment_text(ep, s)
        system, user = _make_prompt(episode=ep, segment=s, segment_text=s_text)

        t0 = time.time()
        call_started = utc_now_iso()
        filtered = False
        try:
            msg = llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            content = (getattr(msg, "content", None) or "").strip()
        except Exception as e:
            if ContentFilterFinishReasonError is not None and isinstance(e, ContentFilterFinishReasonError):
                # Graceful degradation: produce a minimal JSON that avoids quoting
                # any potentially sensitive text, but still preserves segment provenance.
                filtered = True
                content = _safe_json(
                    {
                        "people": [],
                        "beats": [
                            {
                                "type": "other",
                                "summary": "(Omitted: model response blocked by content filter for this segment.)",
                                "supporting_quotes": [],
                            }
                        ],
                        "notable_quotes": [],
                        "uncertainties": [
                            {
                                "question": "Model response blocked by content filter for this segment.",
                                "why_uncertain": "The LLM response was rejected; summary omitted.",
                                "supporting_quotes": [],
                            }
                        ],
                    }
                )
            else:
                raise

        manifest["execution"]["llm_calls_made"] += 1

        try:
            raw = _try_parse_json(content)
        except Exception:
            # Keep the raw response for debugging.
            bad_path = out_dir / f"{s.segment_id}.raw.txt"
            bad_path.write_text(content, encoding="utf-8")
            manifest["execution"]["errors"] += 1
            manifest["execution"]["segments"].append(
                {
                    "segment_id": s.segment_id,
                    "status": "error_invalid_json",
                    "raw_file": f"segment_summaries/{ep.episode_id}/{s.segment_id}.raw.txt",
                    "call_started_at_utc": call_started,
                }
            )
            _write_manifest(manifest_path, manifest)
            raise ValueError(f"LLM did not return valid JSON for {s.segment_id}. Raw saved to {bad_path}")

        normalized = _normalize_segment_summary(
            raw=raw,
            episode=ep,
            segment=s,
            # If the model response was filtered, avoid storing raw segment text
            # in fallback evidence snippets.
            segment_text=("" if filtered else s_text),
            build_id=build_id,
            created_at_utc=created_at,
            llm_model=cfg.llm_model,
        )

        out_path.write_text(_safe_json(normalized), encoding="utf-8")
        ms = int((time.time() - t0) * 1000)
        print(f"[{i+1}/{n_total}] Wrote {out_path.name} ({ms}ms)")

        manifest["execution"]["written"] += 1
        manifest["execution"]["segments"].append(
            {
                "segment_id": s.segment_id,
                "status": "written",
                "out_file": f"segment_summaries/{ep.episode_id}/{s.segment_id}.json",
                "call_started_at_utc": call_started,
                "call_duration_ms": ms,
            }
        )
        _write_manifest(manifest_path, manifest)

    manifest["execution"]["status"] = "done"
    manifest["execution"]["ended_at_utc"] = utc_now_iso()
    _write_manifest(manifest_path, manifest)

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Map step: summarize episode segments into SegmentSummaryV1 JSON.")
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="Normalized docs dir")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Artifact output root")
    p.add_argument("--episode-id", required=True, help="Episode ID like S02E11")
    p.add_argument(
        "--build-id",
        default=None,
        help="Optional build/run ID. If omitted, uses a timestamped ID. Useful for resumable runs.",
    )
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="LLM model")
    p.add_argument("--timeout", type=float, default=None, help="Optional request timeout seconds")
    p.add_argument("--max-retries", type=int, default=None, help="Optional max retries for transient errors")
    p.add_argument("--target-tokens", type=int, default=1800, help="Approx target tokens per segment")
    p.add_argument("--min-tokens", type=int, default=400, help="Approx min tokens before splitting")
    p.add_argument("--max-segments", type=int, default=None, help="Only summarize the first N segments")
    p.add_argument("--segment-index", type=int, default=None, help="Only summarize a specific segment index")
    p.add_argument("--force", action="store_true", help="Overwrite existing output JSON")
    p.add_argument(
        "--prompt-only",
        action="store_true",
        help="Do not call the LLM; write a .prompt.json file for the first selected segment.",
    )

    args = p.parse_args()

    out_dir = summarize_episode_segments(
        SummarizeConfig(
            docs_dir=Path(args.docs_dir),
            out_root=Path(args.out_root),
            episode_id=str(args.episode_id),
            build_id=(str(args.build_id) if args.build_id else None),
            llm_model=str(args.llm_model),
            timeout=(float(args.timeout) if args.timeout is not None else None),
            max_retries=(int(args.max_retries) if args.max_retries is not None else None),
            target_tokens=int(args.target_tokens),
            min_tokens=int(args.min_tokens),
            max_segments=(int(args.max_segments) if args.max_segments is not None else None),
            segment_index=(int(args.segment_index) if args.segment_index is not None else None),
            force=bool(args.force),
            prompt_only=bool(args.prompt_only),
        )
    )
    print(f"Done. Artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
