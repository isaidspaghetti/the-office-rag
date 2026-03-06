"""RAG Evaluation & Observability Dashboard (Streamlit).

Run locally:
  streamlit run apps/rag_runs_dashboard.py

Constraints:
- Streamlit + matplotlib only (no pandas required).
- Defensive to missing keys / schema drift in run logs.
- Lazy: don't parse heavy run contents until user selects.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import streamlit as st


APP_TITLE = "RAG Evaluation & Observability Dashboard"

EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)
EP_IN_SOURCE_RE = re.compile(r"s(\d{2})e(\d{2})", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


# -----------------------------
# Friendly labels (UI)
# -----------------------------
def friendly_retrieval_policy(policy: Any) -> str:
    p = str(policy or "").strip().lower()
    if not p:
        return "(missing)"
    if p == "blended":
        return "hybrid"
    return p


def is_derived_persist_dir(path: Any) -> bool:
    s = str(path or "").lower()
    return "derived" in s or "chroma_db_derived" in s


def _short_path(p: Any) -> str:
    try:
        return str(Path(str(p)).name)
    except Exception:
        return str(p)


def friendly_index_label(persist_dir: str, *, kind: str) -> str:
    """Return a human-friendly label for a Chroma persist dir.

    kind: 'script' | 'derived'
    """
    base = Path(str(persist_dir)).name
    b = base.lower()

    if kind == "script":
        if base == "chroma_db_meta":
            return "Metadata index (recommended) — episode_id filters + routing"
        if base == "chroma_db":
            return "Baseline index — character chunks (no strict metadata guarantees)"
        if b.startswith("chroma_db_scene"):
            return "Advanced index — scene-based chunks (experimental)"
        return f"Index: {base}"

    # kind == 'derived'
    meta = None
    try:
        mp = Path(str(persist_dir)).expanduser() / "_build_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        meta = None

    if isinstance(meta, dict):
        created = str(meta.get("created_at_utc") or "").strip()
        seasons = meta.get("seasons_filter")
        seasons_s = ""
        if isinstance(seasons, list) and seasons:
            try:
                seasons_s = " | seasons: " + ",".join(str(int(x)) for x in seasons)
            except Exception:
                seasons_s = ""

        # Keep title compact; details in suffix.
        suffix = ""
        if created:
            suffix += f" | built: {created}"
        suffix += seasons_s
        return f"Derived cards index — {base}{suffix}".strip()

    # Fall back to path name.
    if base == "chroma_db_derived_cards":
        return "Derived cards index (default)"
    return f"Derived cards index — {base}"


# -----------------------------
# Small utilities (defensive)
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s, 10)
    except Exception:
        return None
    return None


def scored_judge_mode(scored_obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not scored_obj or not isinstance(scored_obj, dict):
        return None
    mode = safe_get(scored_obj, "scoring_meta.judge_mode", None)
    if not isinstance(mode, str) or not mode.strip():
        return None
    return mode.strip().lower()


def is_two_pass_scored(scored_obj: Optional[Dict[str, Any]]) -> bool:
    mode = scored_judge_mode(scored_obj)
    return mode in {"two_pass", "two-pass", "2pass"}


def filter_scored_obj(
    scored_obj: Optional[Dict[str, Any]],
    *,
    require_two_pass: bool,
) -> Optional[Dict[str, Any]]:
    if not scored_obj or not isinstance(scored_obj, dict) or scored_obj.get("_error"):
        return None
    if require_two_pass and (not is_two_pass_scored(scored_obj)):
        return None
    return scored_obj


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        return float(s)
    except Exception:
        return None


_BLENDED_UI_RE = re.compile(r"\bblended\b", re.IGNORECASE)


def _sanitize_for_ui(obj: Any) -> Any:
    """Sanitize objects before rendering so UI never shows forbidden terms.

    Policy naming constraint: never display 'blended' in the UI; use 'Hybrid'.
    """
    if isinstance(obj, str):
        return _BLENDED_UI_RE.sub("Hybrid", obj)
    if isinstance(obj, dict):
        return {
            (_sanitize_for_ui(k) if isinstance(k, str) else k): _sanitize_for_ui(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_for_ui(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_ui(x) for x in obj]
    return obj


def truncate(s: Any, n: int = 140) -> str:
    t = str(s or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    if len(t) <= n:
        return t
    return t[:n] + "…"


def preview_text(text: str, n: int = 240) -> str:
    """Compact preview helper (mirrors experiments/run_eval.py behavior)."""
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(t) <= int(n):
        return t
    return t[: int(n)] + "…"


def normalize_episode_id(s: Any) -> Optional[str]:
    if not s:
        return None
    txt = str(s).strip().upper()
    m = EP_RE.search(txt)
    return m.group(0).upper() if m else None


def extract_episode_id_from_source(source: Any) -> Optional[str]:
    if not source:
        return None
    s = str(source)
    m = EP_IN_SOURCE_RE.search(s)
    if not m:
        return None
    return f"S{m.group(1)}E{m.group(2)}"


def is_episode_lookup_question(question: Any) -> bool:
    q = str(question or "").strip().lower()
    if not q:
        return False
    return bool(re.search(r"\b(which|what)\s+episode\b|\bin\s+which\s+episode\b|\bepisode\s+is\b", q))


def tokenize(text: Any) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or "").lower())]


def jaccard_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    a = set(a_tokens)
    b = set(b_tokens)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


# -----------------------------
# File listing & loading (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_run_files(runs_dir: str) -> List[str]:
    """List run log files, newest-first (by mtime)."""
    p = Path(runs_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    files = [x for x in p.glob("*.json") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(x) for x in files]


@st.cache_data(show_spinner=False)
def list_scored_files(scored_dir: str) -> List[str]:
    p = Path(scored_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    files = [x for x in p.glob("*.scored.json") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(x) for x in files]


@st.cache_data(show_spinner=False)
def list_scored_run_ids(scored_dir: str, *, require_two_pass: bool) -> List[str]:
    """Return run_ids that have a scored file in scored_dir.

    If require_two_pass is True, only include scored files whose scoring_meta.judge_mode
    indicates two-pass judging.
    """
    p = Path(scored_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []

    files = [x for x in p.glob("*.scored.json") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    out: List[str] = []
    for f in files:
        # Filename convention: {run_id}.scored.json
        run_id = f.name[: -len(".scored.json")] if f.name.endswith(".scored.json") else f.stem
        if not run_id:
            continue

        if not require_two_pass:
            out.append(run_id)
            continue

        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_two_pass_scored(obj):
            out.append(run_id)

    # De-dupe while preserving mtime order.
    seen: set[str] = set()
    uniq: List[str] = []
    for rid in out:
        if rid in seen:
            continue
        seen.add(rid)
        uniq.append(rid)
    return uniq


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}", "_path": str(p)}


def load_run(path: str) -> Dict[str, Any]:
    return load_json_file(path)


def load_scored_run(path: str) -> Dict[str, Any]:
    return load_json_file(path)


@st.cache_data(show_spinner=False)
def list_chroma_persist_dirs(db_root: str, *, include_advanced: bool = False) -> List[str]:
    """List Chroma persist directories under db_root.

    Heuristic: directory containing a chroma.sqlite3 file.
    """
    root = Path(db_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "chroma.sqlite3").exists():
            out.append(str(p))
    # Also include nested matches one level deep (some users nest by tag).
    for p in sorted(root.glob("*/*")):
        if p.is_dir() and (p / "chroma.sqlite3").exists():
            out.append(str(p))
    # De-dupe while keeping stable order.
    seen: set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x in seen:
            continue
        base = Path(x).name
        if (not include_advanced) and base.startswith("chroma_db_scene"):
            # Hide experimental indexes unless explicitly requested.
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


@st.cache_data(show_spinner=False)
def list_chroma_collections(persist_dir: str) -> List[str]:
    """List collection names present in a Chroma persist dir."""
    p = Path(persist_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(p))
        cols = client.list_collections()
        names: List[str] = []
        for c in cols:
            name = getattr(c, "name", None)
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return sorted(set(names))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def list_llm_models_from_runs(runs_dir: str, *, limit_files: int = 200) -> List[str]:
    """Best-effort model name discovery from historical run logs."""
    p = Path(runs_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    files = [x for x in p.glob("*.json") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    models: set[str] = set()
    for rf in files[: int(limit_files)]:
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        m = safe_get(obj, "config.llm.model", None)
        if isinstance(m, str) and m.strip():
            models.add(m.strip())
        # Sometimes we only stored query expansion model.
        m2 = safe_get(obj, "config.retrieval.query_expansion.model", None)
        if isinstance(m2, str) and m2.strip():
            models.add(m2.strip())
    return sorted(models)


@st.cache_data(show_spinner=False)
def curated_run_files_for_story(
    run_files: List[str],
    *,
    scored_dir: str,
    require_two_pass: bool,
) -> List[str]:
    """Pick a small, representative set of runs for an exec-friendly story.

    Returns a list of run file paths (subset of run_files), ordered roughly by the
    RAG practice progression:
      baseline -> metadata -> mmr -> query expansion -> hybrid/derived routing -> hybrid+QE
    """

    order = [
        "baseline",
        "metadata_index",
        "mmr",
        "query_expansion",
        "hybrid",
        "hybrid_qe",
    ]

    def _classify(run_obj: Dict[str, Any]) -> str:
        policy = str(safe_get(run_obj, "config.retrieval.policy", "") or "").strip().lower()
        search = str(safe_get(run_obj, "config.retrieval.search_type", "") or "").strip().lower()
        qe = bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))
        script_persist = str(safe_get(run_obj, "config.data_version.script.persist_directory", "") or "")
        derived_persist = str(safe_get(run_obj, "config.data_version.derived.persist_directory", "") or "")

        has_meta = "chroma_db_meta" in script_persist
        has_derived = bool(derived_persist and is_derived_persist_dir(derived_persist))
        is_hybrid = policy in {"blended", "hybrid"}

        if has_derived and is_hybrid:
            return "hybrid_qe" if qe else "hybrid"
        if qe:
            return "query_expansion"
        if search == "mmr":
            return "mmr"
        if has_meta:
            return "metadata_index"
        return "baseline"

    def _dt(run_obj: Dict[str, Any], *, run_file: str) -> datetime:
        dt0 = _parse_utc_iso(safe_get(run_obj, "run.created_at_utc", ""))
        if dt0 is not None:
            return dt0
        try:
            return datetime.fromtimestamp(Path(str(run_file)).stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Gather candidate rows (scored runs only).
    candidates: List[Dict[str, Any]] = []
    for rf in run_files:
        run_obj = load_run(rf)
        if not isinstance(run_obj, dict) or run_obj.get("_error"):
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "").strip()
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        if not scored_path:
            continue
        scored_obj_raw = load_scored_run(scored_path)
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=bool(require_two_pass))
        if not scored_obj:
            continue
        avg_overall = safe_int(safe_get(scored_obj, "score_summary.avg_overall", None))
        cases_scored = safe_int(safe_get(scored_obj, "score_summary.cases_scored", None))
        if avg_overall is None or (cases_scored or 0) <= 0:
            continue

        candidates.append(
            {
                "run_file": rf,
                "bucket": _classify(run_obj),
                "avg_overall": int(avg_overall),
                "cases_scored": int(cases_scored or 0),
                "created_at": _dt(run_obj, run_file=rf),
            }
        )

    # Pick best per bucket.
    best_by_bucket: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        b = str(c.get("bucket") or "baseline")
        prev = best_by_bucket.get(b)
        if prev is None:
            best_by_bucket[b] = c
            continue
        # Prefer higher avg_overall; tie-break newest.
        if int(c.get("avg_overall") or -1) > int(prev.get("avg_overall") or -1):
            best_by_bucket[b] = c
        elif int(c.get("avg_overall") or -1) == int(prev.get("avg_overall") or -1):
            if c.get("created_at") and prev.get("created_at") and c["created_at"] > prev["created_at"]:
                best_by_bucket[b] = c

    out: List[str] = []
    for b in order:
        if b in best_by_bucket:
            out.append(str(best_by_bucket[b]["run_file"]))

    # Fallback: ensure not empty.
    if not out and candidates:
        candidates.sort(key=lambda c: (-int(c.get("avg_overall") or -1), c.get("created_at") or datetime(1970, 1, 1, tzinfo=timezone.utc)), reverse=False)
        out = [str(candidates[0]["run_file"]) ]

    return out


def _select_index(options: List[str], current: Optional[str]) -> int:
    if not options:
        return 0
    if current is None:
        return 0
    try:
        return options.index(current)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def load_gold_answers(gold_path: str) -> Dict[str, Dict[str, Any]]:
    """Return {case_id: gold_row} if file exists; else {}."""
    p = Path(gold_path).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("case_id") or "").strip()
            if cid:
                out[cid] = row
        return out
    except Exception:
        return {}


def find_scored_run_for_run_id(scored_dir: str, run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    p = Path(scored_dir).expanduser()
    candidate = p / f"{run_id}.scored.json"
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return None


def index_scored_cases(scored_run: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not scored_run or not isinstance(scored_run, dict):
        return {}
    rows = scored_run.get("scored_cases")
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("case_id") or "").strip()
        if cid:
            out[cid] = r
    return out


def join_scoring(run_obj: Dict[str, Any], scored_obj: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return {case_id: merged_case_scoring_dict} (judge+deterministic only)."""
    idx = index_scored_cases(scored_obj)
    out: Dict[str, Dict[str, Any]] = {}
    for c in (run_obj.get("cases") or []):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id") or "").strip()
        if not cid:
            continue
        sc = idx.get(cid) or {}
        merged = {
            "judge": (sc.get("judge") if isinstance(sc.get("judge"), dict) else None),
            "deterministic": (sc.get("deterministic") if isinstance(sc.get("deterministic"), dict) else None),
            "judge_error": sc.get("judge_error"),
            "error": sc.get("error"),
        }
        out[cid] = merged
    return out


# -----------------------------
# Flattening (no pandas)
# -----------------------------
def _case_retrieval_results(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = safe_get(case, "retrieval.results", [])
    return res if isinstance(res, list) else []


def _case_max_retrieval_score(case: Dict[str, Any]) -> Optional[float]:
    scores: List[float] = []
    for r in _case_retrieval_results(case):
        if not isinstance(r, dict):
            continue
        s = safe_float(r.get("score"))
        if s is not None:
            scores.append(float(s))
    return max(scores) if scores else None


def _case_mean_topk_retrieval_score(case: Dict[str, Any], k: int = 3) -> Optional[float]:
    scores: List[float] = []
    for r in _case_retrieval_results(case):
        if not isinstance(r, dict):
            continue
        s = safe_float(r.get("score"))
        if s is not None:
            scores.append(float(s))
    if not scores:
        return None
    scores.sort(reverse=True)
    top = scores[: max(1, int(k))]
    return float(sum(top) / len(top)) if top else None


def compute_retrieval_coverage(
    *,
    case: Dict[str, Any],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[float], str]:
    """Return (coverage_value, method)."""
    cid = str(case.get("case_id") or "").strip()
    gold = gold_by_case_id.get(cid) if cid else None
    expected = (gold or {}).get("expected_episode_ids") if isinstance(gold, dict) else None

    if isinstance(expected, list) and expected:
        expected_set = {normalize_episode_id(x) for x in expected}
        expected_set = {x for x in expected_set if x}
        if not expected_set:
            return None, "gold_expected_missing"

        retrieved_eps: set[str] = set()
        for r in _case_retrieval_results(case):
            if not isinstance(r, dict):
                continue
            eid = normalize_episode_id(r.get("episode_id")) or extract_episode_id_from_source(r.get("source"))
            if eid:
                retrieved_eps.add(eid)

        hit = bool(retrieved_eps & expected_set)
        return (1.0 if hit else 0.0), "gold_episode_match"

    # Proxy: use a retrieval score signal.
    proxy = _case_max_retrieval_score(case)
    if proxy is None:
        proxy = _case_mean_topk_retrieval_score(case, k=3)
    return proxy, "score_proxy"


def flatten_rows_for_run(
    *,
    run_obj: Dict[str, Any],
    scoring_by_case_id: Dict[str, Dict[str, Any]],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    run_name = str(safe_get(run_obj, "run.run_name", "") or "")
    created_at = str(safe_get(run_obj, "run.created_at_utc", "") or "")

    vectorstore = safe_get(run_obj, "config.vectorstore", {})
    doc_count = safe_int((vectorstore or {}).get("doc_count"))
    num_vectors = safe_int((vectorstore or {}).get("num_vectors"))

    rows: List[Dict[str, Any]] = []

    cases = run_obj.get("cases")
    if not isinstance(cases, list):
        return rows

    for c in cases:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id") or "").strip()
        question = str(c.get("question") or "")

        ans = safe_get(c, "answer", {})
        usage = safe_get(ans, "usage", {})

        retrieval_latency_ms = safe_int(safe_get(c, "retrieval.latency_ms", None))
        answer_latency_ms = safe_int(safe_get(c, "answer.latency_ms", None))

        context_chars = safe_int(safe_get(c, "answer.stats.context_chars", None))
        context_docs = safe_int(safe_get(c, "answer.stats.context_docs", None))

        prompt_tokens = safe_int((usage or {}).get("prompt_tokens"))
        completion_tokens = safe_int((usage or {}).get("completion_tokens"))
        total_tokens = safe_int((usage or {}).get("total_tokens"))

        heur = safe_get(c, "heuristics", {})
        said_idk = bool((heur or {}).get("said_idk"))
        cited_episode = bool((heur or {}).get("cited_episode"))
        retrieved_any = bool((heur or {}).get("retrieved_any"))

        scoring = scoring_by_case_id.get(cid) or {}
        judge = scoring.get("judge") if isinstance(scoring.get("judge"), dict) else None
        overall = safe_int((judge or {}).get("overall"))
        correctness = safe_int((judge or {}).get("correctness"))
        groundedness = safe_int((judge or {}).get("groundedness"))
        completeness = safe_int((judge or {}).get("completeness"))
        hallucination = safe_int((judge or {}).get("hallucination"))
        verdict = (judge or {}).get("verdict") if isinstance(judge, dict) else None
        context_sufficiency = (judge or {}).get("context_sufficiency") if isinstance(judge, dict) else None
        if isinstance(context_sufficiency, str):
            context_sufficiency = context_sufficiency.strip().lower() or None
        else:
            context_sufficiency = None

        judge_error_present = bool(scoring.get("judge_error"))

        coverage, coverage_method = compute_retrieval_coverage(case=c, gold_by_case_id=gold_by_case_id)
        coverage_f = safe_float(coverage)

        rows.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "created_at_utc": created_at,
                "case_id": cid,
                "question": question,
                "doc_count": doc_count,
                "num_vectors": num_vectors,
                "context_docs": context_docs,
                "context_chars": context_chars,
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_latency_ms": answer_latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "said_idk": said_idk,
                "cited_episode": cited_episode,
                "retrieved_any": retrieved_any,
                "judge_overall": overall,
                "judge_correctness": correctness,
                "judge_groundedness": groundedness,
                "judge_completeness": completeness,
                "judge_hallucination": hallucination,
                "judge_verdict": verdict,
                "judge_context_sufficiency": context_sufficiency,
                "judge_error_present": judge_error_present,
                "retrieval_coverage": coverage_f,
                "coverage_method": coverage_method,
                "max_retrieval_score": _case_max_retrieval_score(c),
            }
        )

    return rows


def aggregate_run_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def _mean(xs: List[float]) -> Optional[float]:
        xs2 = [float(x) for x in xs if x is not None]
        return float(sum(xs2) / len(xs2)) if xs2 else None

    run_id = rows[0].get("run_id")
    run_name = rows[0].get("run_name")
    created_at = rows[0].get("created_at_utc")

    judge_overalls = [safe_float(r.get("judge_overall")) for r in rows if r.get("judge_overall") is not None]
    judge_correctness = [safe_float(r.get("judge_correctness")) for r in rows if r.get("judge_correctness") is not None]
    judge_groundedness = [safe_float(r.get("judge_groundedness")) for r in rows if r.get("judge_groundedness") is not None]
    judge_completeness = [safe_float(r.get("judge_completeness")) for r in rows if r.get("judge_completeness") is not None]
    judge_hallucination = [safe_float(r.get("judge_hallucination")) for r in rows if r.get("judge_hallucination") is not None]

    prompt_tokens = [safe_float(r.get("prompt_tokens")) for r in rows if r.get("prompt_tokens") is not None]
    completion_tokens = [safe_float(r.get("completion_tokens")) for r in rows if r.get("completion_tokens") is not None]
    total_tokens = [safe_float(r.get("total_tokens")) for r in rows if r.get("total_tokens") is not None]
    context_docs = [safe_float(r.get("context_docs")) for r in rows if r.get("context_docs") is not None]
    context_chars = [safe_float(r.get("context_chars")) for r in rows if r.get("context_chars") is not None]

    retrieval_coverage = [safe_float(r.get("retrieval_coverage")) for r in rows if r.get("retrieval_coverage") is not None]

    latency = [
        safe_float((safe_get(r, "retrieval_latency_ms", None) or 0)) + safe_float((safe_get(r, "answer_latency_ms", None) or 0))
        for r in rows
        if (r.get("retrieval_latency_ms") is not None or r.get("answer_latency_ms") is not None)
    ]

    judged_cases = sum(1 for r in rows if r.get("judge_overall") is not None)
    judge_coverage_rate = float(judged_cases / len(rows)) if rows else None

    judge_error_cases = sum(1 for r in rows if bool(r.get("judge_error_present")))
    judge_error_rate = float(judge_error_cases / len(rows)) if rows else None

    cs_vals = [str(r.get("judge_context_sufficiency") or "").strip().lower() for r in rows]
    cs_vals = [x for x in cs_vals if x]
    cs_total = len(cs_vals)
    cs_counts = {
        "sufficient": sum(1 for x in cs_vals if x == "sufficient"),
        "insufficient": sum(1 for x in cs_vals if x == "insufficient"),
        "unclear": sum(1 for x in cs_vals if x == "unclear"),
    }
    cs_rates = {
        "sufficient": (float(cs_counts["sufficient"] / cs_total) if cs_total else None),
        "insufficient": (float(cs_counts["insufficient"] / cs_total) if cs_total else None),
        "unclear": (float(cs_counts["unclear"] / cs_total) if cs_total else None),
    }

    idk_rate = None
    try:
        idk_rate = float(sum(1 for r in rows if r.get("said_idk")) / len(rows)) if rows else None
    except Exception:
        idk_rate = None

    citation_rate = None
    try:
        citation_rate = float(sum(1 for r in rows if r.get("cited_episode")) / len(rows)) if rows else None
    except Exception:
        citation_rate = None

    return {
        "run_id": run_id,
        "run_name": run_name,
        "created_at_utc": created_at,
        "cases": len(rows),
        "cases_judged": judged_cases,
        "judge_coverage_rate": judge_coverage_rate,
        "judge_error_rate": judge_error_rate,
        "avg_overall": _mean([x for x in judge_overalls if x is not None]),
        "avg_correctness": _mean([x for x in judge_correctness if x is not None]),
        "avg_groundedness": _mean([x for x in judge_groundedness if x is not None]),
        "avg_completeness": _mean([x for x in judge_completeness if x is not None]),
        "avg_hallucination": _mean([x for x in judge_hallucination if x is not None]),
        "context_sufficiency_sufficient_rate": cs_rates.get("sufficient"),
        "context_sufficiency_insufficient_rate": cs_rates.get("insufficient"),
        "context_sufficiency_unclear_rate": cs_rates.get("unclear"),
        "idk_rate": idk_rate,
        "citation_rate": citation_rate,
        "avg_prompt_tokens": _mean([x for x in prompt_tokens if x is not None]),
        "avg_completion_tokens": _mean([x for x in completion_tokens if x is not None]),
        "avg_total_tokens": _mean([x for x in total_tokens if x is not None]),
        "avg_latency_ms": _mean([x for x in latency if x is not None]),
        "avg_context_docs": _mean([x for x in context_docs if x is not None]),
        "avg_context_chars": _mean([x for x in context_chars if x is not None]),
        "avg_retrieval_coverage": _mean([x for x in retrieval_coverage if x is not None]),
    }


# -----------------------------
# Source resolution & viewing
# -----------------------------
def resolve_source_path(source: Any, *, docs_root: Path) -> Optional[Path]:
    if not source:
        return None
    src = str(source)
    # Absolute path.
    try:
        p = Path(src)
        if p.is_absolute() and p.exists() and p.is_file():
            return p
    except Exception:
        pass

    # Relative to docs_root.
    for candidate in (
        (docs_root / src),
        (docs_root / "ingestion" / src),
        (docs_root / "ingestion" / "normalized_docs_txt" / src),
    ):
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def read_text_file_limited(path: str, *, limit_chars: int = 12000) -> Dict[str, Any]:
    p = Path(path)
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        truncated_txt = txt[:limit_chars]
        return {
            "path": str(p),
            "chars": len(txt),
            "truncated": len(txt) > limit_chars,
            "text": truncated_txt,
        }
    except Exception as e:
        return {"path": str(p), "error": f"{type(e).__name__}: {e}", "text": ""}


# -----------------------------
# Flight Recorder (reusable)
# -----------------------------
def reconstruct_context_text(case: Dict[str, Any], *, sep: str = "\n\n---\n\n") -> str:
    # If context is stored explicitly, prefer it.
    ctx = safe_get(case, "answer.context_text", None)
    if isinstance(ctx, str) and ctx.strip():
        return ctx

    parts: List[str] = []
    for r in _case_retrieval_results(case):
        if not isinstance(r, dict):
            continue
        prev = str(r.get("preview") or "")
        if prev.strip():
            parts.append(prev)
    return sep.join(parts)


def compute_attention(case: Dict[str, Any], answer_text: str) -> List[Tuple[int, float, str]]:
    ans_toks = tokenize(answer_text)
    out: List[Tuple[int, float, str]] = []
    for r in _case_retrieval_results(case):
        if not isinstance(r, dict):
            continue
        rank = safe_int(r.get("rank")) or 0
        eid = normalize_episode_id(r.get("episode_id")) or extract_episode_id_from_source(r.get("source")) or ""
        label = f"#{rank} {eid}".strip()
        prev = str(r.get("preview") or "")
        score = jaccard_similarity(ans_toks, tokenize(prev))
        out.append((rank, float(score), label))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def render_attention_map(attn: List[Tuple[int, float, str]]) -> None:
    if not attn:
        st.info("No retrieval results available for attention estimate.")
        return

    # Keep top-N for readability.
    top_n = min(len(attn), 12)
    top = attn[:top_n]
    labels = [t[2] for t in top][::-1]
    vals = [t[1] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(7.5, max(2.8, 0.35 * len(top) + 1.4)))
    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Influence (token Jaccard similarity)")
    ax.set_xlim(0.0, max(0.05, max(vals) * 1.05))
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def render_retrieval_table(results: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                "rank": safe_int(r.get("rank")),
                "doc_type": r.get("doc_type"),
                "derived_type": r.get("derived_type"),
                "episode_id": normalize_episode_id(r.get("episode_id")) or extract_episode_id_from_source(r.get("source")),
                "score": safe_float(r.get("score")),
                "source": r.get("source"),
                "preview": truncate(r.get("preview"), 180),
            }
        )
    if not rows:
        st.info("No retrieval results found for this case.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def export_view_snippet(
    *,
    export_dir: Path,
    run_obj: Dict[str, Any],
    case_obj: Optional[Dict[str, Any]],
    scored_case: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        case_id = str((case_obj or {}).get("case_id") or "") if case_obj else ""
        name_bits = [utc_now_iso().replace(":", "-")]
        if run_id:
            name_bits.append(run_id)
        if case_id:
            name_bits.append(case_id)
        out_path = export_dir / ("__".join(name_bits) + ".json")

        payload = {
            "exported_at_utc": utc_now_iso(),
            "run": safe_get(run_obj, "run", {}),
            "config": safe_get(run_obj, "config", {}),
            "case": case_obj,
            "scored_case": scored_case,
            "extra": extra or {},
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out_path
    except Exception:
        return None


def render_flight_recorder(
    run_obj: Dict[str, Any],
    case_obj: Dict[str, Any],
    scored_case: Optional[Dict[str, Any]] = None,
    *,
    docs_root: Path,
    export_dir: Path,
    key_prefix: str = "",
) -> None:
    st.subheader("Flight Recorder")

    question = str(case_obj.get("question") or "")
    st.markdown("**Question**")
    st.write(question or "(missing question)")

    # Optional expanded queries.
    mq = safe_get(case_obj, "retrieval.multi_query", None)
    expanded = mq.get("expanded_queries") if isinstance(mq, dict) else None
    if isinstance(expanded, list) and expanded:
        with st.expander("Expanded queries", expanded=False):
            for q in expanded:
                st.write("- ", str(q))

    results = _case_retrieval_results(case_obj)
    answer_text = str(safe_get(case_obj, "answer.text", "") or "")

    tab_retrieval, tab_context, tab_answer, tab_scores, tab_attention = st.tabs(
        ["Retrieval", "Context", "Answer", "Scores", "Attention Map"]
    )

    with tab_retrieval:
        render_retrieval_table(results)

        if results:
            options = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                rank = safe_int(r.get("rank")) or 0
                eid = normalize_episode_id(r.get("episode_id")) or extract_episode_id_from_source(r.get("source")) or ""
                src = str(r.get("source") or "")
                options.append((rank, f"#{rank} {eid} — {truncate(src, 80)}"))

            selected = st.selectbox(
                "Chunk viewer: select a retrieved row",
                options=options,
                index=0,
                format_func=lambda x: x[1],
                key=f"{key_prefix}chunk_viewer_select",
            )
            selected_rank = selected[0] if isinstance(selected, tuple) else None

            selected_row: Optional[Dict[str, Any]] = None
            for r in results:
                if not isinstance(r, dict):
                    continue
                if (safe_int(r.get("rank")) or 0) == int(selected_rank or 0):
                    selected_row = r
                    break

            if selected_row:
                st.markdown("**Preview**")
                st.code(str(selected_row.get("preview") or ""), language="text")

                src = selected_row.get("source")
                resolved = resolve_source_path(src, docs_root=docs_root)
                st.markdown("**Source file**")
                if resolved is None:
                    st.write(f"Could not resolve source: {src}")
                else:
                    st.write(str(resolved))
                    show_full = st.checkbox(
                        "Show full file (can be large)",
                        value=False,
                        key=f"{key_prefix}show_full_{resolved}",
                    )
                    info = read_text_file_limited(
                        str(resolved),
                        limit_chars=(80000 if show_full else 14000),
                    )
                    if info.get("error"):
                        st.error(info["error"])
                    else:
                        label = f"File text ({info.get('chars')} chars)" + (" (truncated)" if info.get("truncated") else "")
                        with st.expander(label, expanded=not show_full):
                            st.code(info.get("text") or "", language="text")

    with tab_context:
        ctx = reconstruct_context_text(case_obj)
        if not ctx.strip():
            st.info("No context text stored; reconstructed context is empty.")
        with st.expander("Context sent to LLM (stored or reconstructed)", expanded=False):
            st.code(ctx, language="text")

        diag_view = st.checkbox(
            "Show routing/diagnostics JSON",
            value=False,
            key=f"{key_prefix}show_diag_json",
        )
        if diag_view:
            st.json({
                "retrieval": safe_get(case_obj, "retrieval", {}),
                "diagnostics": safe_get(case_obj, "diagnostics", {}),
                "heuristics": safe_get(case_obj, "heuristics", {}),
                "labels": safe_get(case_obj, "labels", {}),
            })

    with tab_answer:
        st.markdown("**Answer**")
        if answer_text:
            st.write(answer_text)
        else:
            st.info("No answer text present (was the run executed with --no-llm?)")

        cols = st.columns(4)
        cols[0].metric("Retrieval ms", safe_int(safe_get(case_obj, "retrieval.latency_ms", None)) or 0)
        cols[1].metric("Answer ms", safe_int(safe_get(case_obj, "answer.latency_ms", None)) or 0)
        cols[2].metric("Context chars", safe_int(safe_get(case_obj, "answer.stats.context_chars", None)) or 0)
        cols[3].metric("Total tokens", safe_int(safe_get(case_obj, "answer.usage.total_tokens", None)) or 0)

        with st.expander("Answer JSON", expanded=False):
            st.json(safe_get(case_obj, "answer", {}))

    with tab_scores:
        if scored_case is None:
            st.info("No scored_case loaded for this run/case.")
        else:
            judge = scored_case.get("judge") if isinstance(scored_case.get("judge"), dict) else None
            det = scored_case.get("deterministic") if isinstance(scored_case.get("deterministic"), dict) else None

            if judge:
                cols = st.columns(5)
                cols[0].metric("Overall", safe_int(judge.get("overall")) or 0)
                cols[1].metric("Correctness", safe_int(judge.get("correctness")) or 0)
                cols[2].metric("Groundedness", safe_int(judge.get("groundedness")) or 0)
                cols[3].metric("Completeness", safe_int(judge.get("completeness")) or 0)
                cols[4].metric("Hallucination", safe_int(judge.get("hallucination")) or 0)
                st.write("Verdict:", judge.get("verdict"))
                with st.expander("Judge JSON", expanded=False):
                    st.json(judge)
            else:
                st.info("Judge scores missing for this case.")

            if det:
                with st.expander("Deterministic checks", expanded=False):
                    st.json(det)

            if scored_case.get("judge_error"):
                st.warning(f"Judge error: {scored_case.get('judge_error')}")
            if scored_case.get("error"):
                st.warning(f"Score file error: {scored_case.get('error')}")

    with tab_attention:
        if not answer_text:
            st.info("No answer text; attention estimate is not meaningful.")
        else:
            st.caption("Heuristic: token overlap between answer and each retrieved preview.")
            attn = compute_attention(case_obj, answer_text)
            render_attention_map(attn)

    st.divider()
    if st.button("Export current view", type="secondary", key=f"{key_prefix}export_view"):
        out_path = export_view_snippet(
            export_dir=export_dir,
            run_obj=run_obj,
            case_obj=case_obj,
            scored_case=scored_case,
            extra={"page": "flight_recorder"},
        )
        if out_path:
            st.success(f"Exported: {out_path}")
        else:
            st.error("Failed to export snippet.")


# -----------------------------
# Live chat (feature flag)
# -----------------------------
def _usage_from_msg(msg: Any) -> Dict[str, Optional[int]]:
    """Best-effort token usage extraction (mirrors experiments/run_eval.py logic)."""
    prompt = completion = total = None

    usage = getattr(msg, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion = usage.get("output_tokens") or usage.get("completion_tokens")
        total = usage.get("total_tokens")

    if prompt is None and completion is None and total is None:
        rm = getattr(msg, "response_metadata", None)
        if isinstance(rm, dict):
            for key in ("token_usage", "usage"):
                u = rm.get(key)
                if isinstance(u, dict):
                    prompt = u.get("prompt_tokens") or u.get("input_tokens")
                    completion = u.get("completion_tokens") or u.get("output_tokens")
                    total = u.get("total_tokens")
                    break

    def _to_int(x: Any) -> Optional[int]:
        try:
            if x is None:
                return None
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return int(x)
            if isinstance(x, float):
                return int(x)
            s = str(x).strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s, 10)
        except Exception:
            return None
        return None

    return {
        "prompt_tokens": _to_int(prompt),
        "completion_tokens": _to_int(completion),
        "total_tokens": _to_int(total),
    }


def _doc_to_result_row(doc: Any, *, rank: int, score: Optional[float]) -> Dict[str, Any]:
    meta = getattr(doc, "metadata", None) or {}
    txt = str(getattr(doc, "page_content", "") or "")
    src = meta.get("source")
    # Keep keys consistent with run logs where possible.
    return {
        "rank": int(rank),
        "source": src,
        "episode_id": meta.get("episode_id"),
        "doc_type": meta.get("doc_type"),
        "derived_type": meta.get("derived_type"),
        "topic_id": meta.get("topic_id"),
        "topic_type": meta.get("topic_type"),
        "chunk_type": meta.get("chunk_type"),
        "chunk_index": meta.get("chunk_index"),
        "first_line": (txt.splitlines()[0].strip() if txt else ""),
        "preview": preview_text(txt, n=280),
        "score": (float(score) if score is not None else None),
    }


@st.cache_resource(show_spinner=False)
def _live_build_chroma(
    *,
    persist_directory: str,
    collection_name: Optional[str],
    embed_model: str,
    is_derived: bool,
) -> Any:
    """Build a Chroma DB client for live chat.

    Lazy-imports langchain deps so the dashboard still runs without them when live chat is off.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        from langchain_chroma import Chroma  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Live chat requires langchain + chroma deps. "
            "Install: pip install -U langchain-openai langchain-core langchain-chroma chromadb python-dotenv"
        ) from e

    load_dotenv()

    embeddings = OpenAIEmbeddings(model=str(embed_model))
    kwargs: Dict[str, Any] = {
        "persist_directory": str(persist_directory),
        "embedding_function": embeddings,
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if collection_name:
        kwargs["collection_name"] = str(collection_name)

    # is_derived is currently only for future routing/tuning; keep signature stable.
    _ = is_derived
    return Chroma(**kwargs)


def _live_retrieve(
    *,
    question: str,
    script_db: Any,
    derived_db: Optional[Any],
    retrieval_policy: str,
    k: int,
    derived_k: int,
) -> Tuple[List[Tuple[Any, Optional[float]]], Dict[str, Any]]:
    """Minimal live retrieval supporting a few policies.

    Returns ([(doc, score)], routing_details)
    """
    policy = str(retrieval_policy or "").strip().lower()
    k = max(1, int(k))

    def _sim(db: Any, q: str, top_k: int) -> List[Tuple[Any, Optional[float]]]:
        try:
            rows = db.similarity_search_with_relevance_scores(q, k=int(top_k))
            return [(d, float(s) if s is not None else None) for d, s in rows]
        except Exception:
            # Fall back to docs without scores.
            docs = db.similarity_search(q, k=int(top_k))
            return [(d, None) for d in docs]

    if policy in {"script_only", "script", "baseline"}:
        pairs = _sim(script_db, question, k)
        return pairs, {"policy": "script_only"}

    if policy in {"derived_only", "derived"}:
        if derived_db is None:
            return [], {"policy": "derived_only", "error": "derived_db_missing"}
        pairs = _sim(derived_db, question, k)
        return pairs, {"policy": "derived_only"}

    if policy in {"blended", "hybrid"}:
        base = _sim(script_db, question, k)
        der: List[Tuple[Any, Optional[float]]] = []
        if derived_db is not None:
            der = _sim(derived_db, question, max(1, int(derived_k)))

        # Deduplicate by (source, page_content) best-effort.
        seen: set[Tuple[Optional[str], str]] = set()
        combined: List[Tuple[Any, Optional[float]]] = []
        for d, s in (der + base):
            meta = getattr(d, "metadata", None) or {}
            key = (meta.get("source"), str(getattr(d, "page_content", "") or ""))
            if key in seen:
                continue
            seen.add(key)
            combined.append((d, s))
            if len(combined) >= k:
                break
        return combined, {"policy": "blended", "base_k": k, "derived_k": int(derived_k)}

    # Default: script_only
    pairs = _sim(script_db, question, k)
    return pairs, {"policy": "script_only", "note": f"unknown_policy:{retrieval_policy}"}


def answer_question(question: str) -> Dict[str, Any]:
    """Live chat implementation (feature-flagged in UI).

    Reads configuration from st.session_state (set in the sidebar).
    Returns a dict consumed by coerce_live_result_to_case().
    """
    q = str(question or "").strip()
    if not q:
        return {"answer": "", "retrieval": {"results": []}, "answer_meta": {"latency_ms": 0, "retrieval_latency_ms": 0, "usage": {}}}

    # Pull config from sidebar (with safe defaults).
    script_persist_dir = str(st.session_state.get("live_script_persist_dir") or "db/chroma_db")
    script_collection_name = st.session_state.get("live_script_collection_name")
    script_collection_name = str(script_collection_name).strip() if script_collection_name else None

    derived_persist_dir = str(st.session_state.get("live_derived_persist_dir") or "db/chroma_db_derived_cards")
    derived_collection_name = str(st.session_state.get("live_derived_collection_name") or "derived_cards").strip() or None

    retrieval_policy = str(st.session_state.get("live_retrieval_policy") or "script_only")
    embed_model = "text-embedding-3-small"
    llm_model = str(st.session_state.get("live_llm_model") or "gpt-4.1-mini")
    k = safe_int(st.session_state.get("live_k")) or 8
    derived_k = safe_int(st.session_state.get("live_derived_k")) or 6
    temperature = safe_float(st.session_state.get("live_temperature"))
    if temperature is None:
        temperature = 0.0

    # Lazy import LLM deps.
    try:
        from dotenv import load_dotenv  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as e:
        return {
            "answer": (
                "Live chat is enabled, but required LLM dependencies are missing. "
                "Install: pip install -U langchain-openai langchain-core python-dotenv"
            ),
            "retrieval": {"results": []},
            "answer_meta": {"latency_ms": 0, "retrieval_latency_ms": 0, "usage": {}},
            "error": f"missing_deps:{type(e).__name__}",
        }

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "answer": "Live chat requires OPENAI_API_KEY (set it in .env or your environment).",
            "retrieval": {"results": []},
            "answer_meta": {"latency_ms": 0, "retrieval_latency_ms": 0, "usage": {}},
            "error": "missing_openai_api_key",
        }

    # Build / reuse vectorstores.
    try:
        script_db = _live_build_chroma(
            persist_directory=script_persist_dir,
            collection_name=script_collection_name,
            embed_model=embed_model,
            is_derived=False,
        )
        derived_db: Optional[Any] = None
        if retrieval_policy.lower() in {"derived_only", "derived", "blended", "hybrid"}:
            derived_db = _live_build_chroma(
                persist_directory=derived_persist_dir,
                collection_name=derived_collection_name,
                embed_model=embed_model,
                is_derived=True,
            )
    except Exception as e:
        return {
            "answer": f"Live chat failed to initialize vectorstores: {type(e).__name__}: {e}",
            "retrieval": {"results": []},
            "answer_meta": {"latency_ms": 0, "retrieval_latency_ms": 0, "usage": {}},
            "error": f"vectorstore_init_error:{type(e).__name__}",
        }

    # Retrieval
    t0 = time.time()
    try:
        pairs, routing = _live_retrieve(
            question=q,
            script_db=script_db,
            derived_db=derived_db,
            retrieval_policy=retrieval_policy,
            k=int(k),
            derived_k=int(derived_k),
        )
    except Exception as e:
        pairs, routing = [], {"policy": retrieval_policy, "error": f"{type(e).__name__}: {e}"}
    retrieval_ms = int((time.time() - t0) * 1000)

    # Build context for LLM.
    docs_texts: List[str] = []
    results: List[Dict[str, Any]] = []
    for i, (doc, score) in enumerate(pairs, start=1):
        results.append(_doc_to_result_row(doc, rank=i, score=score))
        txt = str(getattr(doc, "page_content", "") or "")
        if txt.strip():
            docs_texts.append(txt)
    context_text = "\n\n---\n\n".join(docs_texts)

    system = (
        "You are a QA assistant for questions about the TV show The Office.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context does not contain the answer, say you don't know.\n"
        "When possible, cite episode identifiers present in the context (e.g., 'S02E06').\n"
        "When you make a specific claim, include at least one short exact quote from the context in double quotes."
    )
    user = f"Question: {q}\n\nContext:\n{context_text}"

    llm = ChatOpenAI(model=llm_model, temperature=float(temperature))
    t1 = time.time()
    try:
        msg = llm.invoke([("system", system), ("human", user)])
        answer_text = str(getattr(msg, "content", "") or "")
        usage = _usage_from_msg(msg)
    except Exception as e:
        answer_text = f"Live chat LLM error: {type(e).__name__}: {e}"
        usage = {}
    latency_ms = int((time.time() - t1) * 1000)

    return {
        "answer": answer_text,
        "retrieval": {
            "results": results,
            "routing": routing,
        },
        "answer_meta": {
            "latency_ms": latency_ms,
            "retrieval_latency_ms": retrieval_ms,
            "usage": usage,
        },
    }


def coerce_live_result_to_case(question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    # Build a minimal case-like object that the Flight Recorder can render.
    retrieval = result.get("retrieval") if isinstance(result.get("retrieval"), dict) else {}
    results = retrieval.get("results") if isinstance(retrieval.get("results"), list) else []
    answer_text = str(result.get("answer") or "")
    answer_meta = result.get("answer_meta") if isinstance(result.get("answer_meta"), dict) else {}
    usage = answer_meta.get("usage") if isinstance(answer_meta.get("usage"), dict) else {}

    # Ensure ranks exist.
    norm_results: List[Dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        rr.setdefault("rank", i)
        rr.setdefault("preview", rr.get("preview") or rr.get("text") or "")
        norm_results.append(rr)

    return {
        "case_id": "live",
        "question": question,
        "retrieval": {"latency_ms": answer_meta.get("retrieval_latency_ms"), "results": norm_results},
        "answer": {
            "text": answer_text,
            "latency_ms": answer_meta.get("latency_ms"),
            "usage": usage,
            "stats": {
                "context_docs": len(norm_results),
                "context_chars": len(reconstruct_context_text({"retrieval": {"results": norm_results}})),
                "answer_chars": len(answer_text),
            },
        },
        "heuristics": {
            "said_idk": "i don't know" in answer_text.lower(),
            "cited_episode": bool(EP_RE.search(answer_text)),
            "retrieved_any": len(norm_results) > 0,
        },
    }


# -----------------------------
# Plot helpers (matplotlib only)
# -----------------------------
def plot_matrix(values: List[List[Optional[float]]], x_labels: List[str], y_labels: List[str], *, title: str) -> None:
    if not values or not x_labels or not y_labels:
        st.info("Matrix is empty.")
        return

    # Replace None with NaN-like sentinel.
    data = [[(v if v is not None else float("nan")) for v in row] for row in values]

    fig_w = max(6.0, 0.85 * len(x_labels) + 2.5)
    fig_h = max(4.0, 0.35 * len(y_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Annotate.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            v = values[i][j]
            if v is None:
                txt = ""
            else:
                txt = f"{v:.0f}" if v >= 10 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    st.pyplot(fig, clear_figure=True)


def plot_scatter(points: List[Tuple[float, float, str]], *, title: str, x_label: str, y_label: str) -> None:
    if not points:
        st.info("No points to plot.")
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    labels = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(xs, ys, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    for x, y, lbl in zip(xs, ys, labels):
        if lbl:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.7)
    st.pyplot(fig, clear_figure=True)


def plot_pie(counts: Dict[str, int], *, title: str) -> None:
    labels = [k for k, v in counts.items() if int(v) > 0]
    sizes = [int(counts[k]) for k in labels]
    if not sizes:
        st.info("No failures detected.")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Failure mode classification
# -----------------------------
def classify_failure(
    *,
    row: Dict[str, Any],
    coverage: Optional[float],
) -> str:
    verdict = str(row.get("judge_verdict") or "").strip().lower()
    said_idk = bool(row.get("said_idk"))
    retrieved_any = bool(row.get("retrieved_any"))
    cited_episode = bool(row.get("cited_episode"))
    question = str(row.get("question") or "")

    cov = safe_float(coverage)
    coverage_low = (cov is not None and cov < 0.5)

    if verdict:
        if verdict == "idk_preferred":
            return "missing_context"
        if verdict == "incorrect":
            return "missing_context" if coverage_low else "reasoning_failure"
        if verdict in {"partially_correct", "partial", "partially correct"}:
            return "missing_context" if coverage_low else "grounding_failure"
        if verdict == "correct":
            return "ok"

    # Fallback heuristics.
    if not retrieved_any:
        return "retrieval_failure"
    if said_idk:
        return "missing_context"
    if is_episode_lookup_question(question) and (not cited_episode):
        return "grounding_failure"
    return "ok"


def pareto_frontier(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """Return Pareto-optimal points for (score higher, tokens lower)."""
    # Each point: (run_id, score, tokens)
    pts = [(rid, float(score), float(tokens)) for rid, score, tokens in points]
    out: List[Tuple[str, float, float]] = []
    for i, (rid, s, t) in enumerate(pts):
        dominated = False
        for j, (rid2, s2, t2) in enumerate(pts):
            if i == j:
                continue
            if (s2 >= s and t2 <= t) and (s2 > s or t2 < t):
                dominated = True
                break
        if not dominated:
            out.append((rid, s, t))
    out.sort(key=lambda x: (-x[1], x[2]))
    return out


# -----------------------------
# Pages
# -----------------------------
def page_chat_playground(*, enable_live_chat: bool, docs_root: Path, export_dir: Path) -> None:
    st.header("Chat Playground")
    st.caption("Optional live mode (feature-flagged). Use Run Explorer for historical inspection.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if not enable_live_chat:
        st.info("Live chat disabled; use Run Explorer for inspection.")

    # Render history.
    for i, msg in enumerate(st.session_state.chat_messages):
        role = msg.get("role")
        content = msg.get("content")
        with st.chat_message(role or "assistant"):
            st.write(content)
            if role == "assistant" and msg.get("flight"):
                with st.expander("Inspect Answer", expanded=False):
                    flight = msg["flight"]
                    render_flight_recorder(
                        flight["run"],
                        flight["case"],
                        scored_case=None,
                        docs_root=docs_root,
                        export_dir=export_dir,
                        key_prefix=f"chat_hist_{i}_",
                    )

    prompt = st.chat_input("Ask a question about The Office…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if not enable_live_chat:
            st.write("Live chat is disabled. Enable it in the sidebar to use this page.")
            return

        try:
            result = answer_question(prompt)
            answer_text = str(result.get("answer") or "")
            st.write(answer_text or "(empty answer)")

            case = coerce_live_result_to_case(prompt, result)
            run = {
                "run": {
                    "run_id": f"live_{utc_now_iso()}",
                    "run_name": "live_chat",
                    "created_at_utc": utc_now_iso(),
                    "notes": "Live chat session (synthetic run object)",
                },
                "config": {"llm": {"enabled": True}, "retrieval": {"policy": "live"}},
                "cases": [case],
                "summary": {},
            }

            with st.expander("Inspect Answer", expanded=False):
                render_flight_recorder(
                    run,
                    case,
                    scored_case=None,
                    docs_root=docs_root,
                    export_dir=export_dir,
                    key_prefix="chat_live_",
                )

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "flight": {"run": run, "case": case},
                }
            )
        except Exception as e:
            st.error(f"Live chat error: {type(e).__name__}: {e}")


def page_run_explorer(
    *,
    project_root: Path,
    runs_dir: Path,
    scored_dir: Path,
    docs_root: Path,
    diagnostics_mode: str,
    export_dir: Path,
) -> None:
    st.header("Run Explorer")
    st.caption("Browse historical run logs and inspect individual cases with the Flight Recorder.")

    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    selected = st.selectbox("Select a run file", options=run_files, index=0)
    if not selected:
        return

    with st.spinner("Loading run log…"):
        run_obj = load_run(selected)

    if run_obj.get("_error"):
        st.error(f"Failed to load run: {run_obj.get('_error')}")
        st.write(run_obj.get("_path"))
        return

    run_meta = safe_get(run_obj, "run", {})
    st.subheader("Run")
    cols = st.columns(4)
    cols[0].metric("Run name", str(run_meta.get("run_name") or ""))
    cols[1].metric("Run id", str(run_meta.get("run_id") or ""))
    cols[2].metric("Created", str(run_meta.get("created_at_utc") or ""))
    cols[3].metric("Cases", len(run_obj.get("cases") or []) if isinstance(run_obj.get("cases"), list) else 0)

    with st.expander("Config", expanded=False):
        st.json(safe_get(run_obj, "config", {}))
    with st.expander("Summary", expanded=False):
        st.json(safe_get(run_obj, "summary", {}))

    # Scoring is optional; lazy-load it only if requested.
    scored_obj: Optional[Dict[str, Any]] = None
    scoring_idx: Dict[str, Dict[str, Any]] = {}
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)

    with st.expander("Scoring (optional)", expanded=False):
        st.write("Scored file:", scored_path or "(none found)")
        load_scoring = st.checkbox("Load scored run (if available)", value=bool(scored_path))
        if load_scoring and scored_path:
            scored_obj = load_scored_run(scored_path)
            if scored_obj.get("_error"):
                st.error(f"Failed to load scored run: {scored_obj.get('_error')}")
                scored_obj = None
            else:
                require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
                if require_two_pass and (not is_two_pass_scored(scored_obj)):
                    st.warning("Scored file is not two-pass; ignoring due to dashboard filter.")
                    scored_obj = None
                
            if scored_obj is not None:
                scoring_idx = index_scored_cases(scored_obj)
                st.caption(f"Loaded scores for {len(scoring_idx)} cases")

    cases = run_obj.get("cases")
    if not isinstance(cases, list) or not cases:
        st.warning("Run has no cases[].")
        return

    # Build table.
    table_rows: List[Dict[str, Any]] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id") or "")
        q = str(c.get("question") or "")
        usage = safe_get(c, "answer.usage", {})
        tokens = safe_int((usage or {}).get("total_tokens"))
        r_ms = safe_int(safe_get(c, "retrieval.latency_ms", None))
        a_ms = safe_int(safe_get(c, "answer.latency_ms", None))
        overall = None
        if scoring_idx.get(cid) and isinstance(scoring_idx[cid].get("judge"), dict):
            overall = safe_int(scoring_idx[cid]["judge"].get("overall"))

        table_rows.append(
            {
                "case_id": cid,
                "question": truncate(q, 120),
                "judge_overall": overall,
                "retrieval_ms": r_ms,
                "answer_ms": a_ms,
                "total_tokens": tokens,
                "context_chars": safe_int(safe_get(c, "answer.stats.context_chars", None)),
                "said_idk": bool(safe_get(c, "heuristics.said_idk", False)),
            }
        )

    sort_key = st.selectbox(
        "Sort cases by",
        options=["judge_overall", "total_tokens", "retrieval_ms", "answer_ms", "context_chars", "case_id"],
        index=0,
    )
    desc = st.checkbox("Descending", value=True)

    def _sort_val(r: Dict[str, Any]) -> Any:
        v = r.get(sort_key)
        if v is None:
            return -1 if desc else 10**18
        return v

    table_rows.sort(key=_sort_val, reverse=desc)
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    case_ids = [str(r["case_id"]) for r in table_rows if r.get("case_id")]
    selected_case_id = st.selectbox("Open Flight Recorder for case", options=case_ids, index=0)
    if not selected_case_id:
        return

    case_obj = next((c for c in cases if isinstance(c, dict) and str(c.get("case_id")) == selected_case_id), None)
    if not isinstance(case_obj, dict):
        st.error("Could not find selected case in run.")
        return

    scored_case_obj = scoring_idx.get(selected_case_id) if scoring_idx else None
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    render_flight_recorder(
        run_obj,
        case_obj,
        scored_case_obj,
        docs_root=docs_root,
        export_dir=export_dir,
        key_prefix=f"run_explorer_{run_id}_{selected_case_id}_",
    )


def page_experiment_comparison(
    *,
    runs_dir: Path,
    scored_dir: Path,
    gold_by_case_id: Dict[str, Dict[str, Any]],
    docs_root: Path,
    export_dir: Path,
) -> None:
    st.header("Experiment Comparison")
    st.caption("Compare multiple runs: leaderboard, matrix view, and case-by-case diffs.")

    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    selected_runs = st.multiselect(
        "Select runs to compare",
        options=run_files,
        default=run_files[: min(5, len(run_files))],
    )
    if not selected_runs:
        st.info("Select one or more runs.")
        return

    loaded: List[Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
    for path in selected_runs:
        run_obj = load_run(path)
        if run_obj.get("_error"):
            st.warning(f"Skipping run (load error): {path}")
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        scored_obj_raw = load_scored_run(scored_path) if scored_path else None
        require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
        scoring_idx = index_scored_cases(scored_obj) if scored_obj else {}
        loaded.append((path, run_obj, scoring_idx))

    if not loaded:
        st.warning("No runs could be loaded.")
        return

    # Leaderboard.
    leaderboard: List[Dict[str, Any]] = []
    per_run_rows: Dict[str, List[Dict[str, Any]]] = {}

    for path, run_obj, scoring_idx in loaded:
        scoring_by_case_id = join_scoring(run_obj, {"scored_cases": [{"case_id": k, **v} for k, v in scoring_idx.items()]})
        rows = flatten_rows_for_run(run_obj=run_obj, scoring_by_case_id=scoring_by_case_id, gold_by_case_id=gold_by_case_id)
        per_run_rows[path] = rows
        agg = aggregate_run_rows(rows)
        agg["run_file"] = path
        leaderboard.append(agg)

    st.subheader("Leaderboard")
    leaderboard.sort(key=lambda r: (safe_float(r.get("avg_overall")) or -1.0), reverse=True)
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    # Matrix view.
    st.subheader("Questions × Runs matrix")
    metric = st.selectbox(
        "Cell value",
        options=["judge_overall", "judge_correctness", "judge_groundedness", "retrieval_coverage", "max_retrieval_score"],
        index=0,
    )

    # Build y-axis cases from the first run (stable). Limit for readability.
    max_cases = st.slider("Max questions to display", min_value=5, max_value=80, value=25, step=5)

    first_rows = per_run_rows[loaded[0][0]]
    case_ids = [str(r.get("case_id") or "") for r in first_rows][: int(max_cases)]
    y_labels = [f"{cid}" for cid in case_ids]
    x_labels = [truncate(str(safe_get(run_obj, "run.run_name", Path(path).stem) or ""), 24) for path, run_obj, _ in loaded]

    values: List[List[Optional[float]]] = []
    for cid in case_ids:
        row_vals: List[Optional[float]] = []
        for path, _run_obj, _sc in loaded:
            rows = per_run_rows.get(path) or []
            match = next((r for r in rows if str(r.get("case_id")) == cid), None)
            row_vals.append(safe_float((match or {}).get(metric)))
        values.append(row_vals)

    plot_matrix(values, x_labels=x_labels, y_labels=y_labels, title=f"{metric} (per case)")

    # Compare two runs for one case.
    st.subheader("Compare two runs for one case")
    run_a = st.selectbox("Run A", options=[p for p, _r, _s in loaded], index=0)
    run_b = st.selectbox("Run B", options=[p for p, _r, _s in loaded], index=min(1, len(loaded) - 1))

    rows_a = per_run_rows.get(run_a) or []
    case_opts = [str(r.get("case_id")) for r in rows_a if r.get("case_id")]
    selected_case = st.selectbox("Case id", options=case_opts, index=0)

    def _find_case(path: str, case_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        run_obj = next((ro for p, ro, _ in loaded if p == path), None)
        scoring_idx = next((sc for p, _ro, sc in loaded if p == path), {})
        if not run_obj or not isinstance(run_obj.get("cases"), list):
            return None, None, None
        case_obj = next((c for c in run_obj["cases"] if isinstance(c, dict) and str(c.get("case_id")) == case_id), None)
        scored_case = scoring_idx.get(case_id) if scoring_idx else None
        return run_obj, case_obj, scored_case

    run_obj_a, case_obj_a, scored_a = _find_case(run_a, selected_case)
    run_obj_b, case_obj_b, scored_b = _find_case(run_b, selected_case)

    if not (run_obj_a and case_obj_a and run_obj_b and case_obj_b):
        st.warning("Could not load both cases for comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Run A — retrieved sources (top 8)**")
        res_a = _case_retrieval_results(case_obj_a)
        for r in (res_a[:8] if res_a else []):
            if isinstance(r, dict):
                st.write("- ", r.get("source"))

    with col2:
        st.markdown("**Run B — retrieved sources (top 8)**")
        res_b = _case_retrieval_results(case_obj_b)
        for r in (res_b[:8] if res_b else []):
            if isinstance(r, dict):
                st.write("- ", r.get("source"))

    st.markdown("**Answer diff**")
    a_txt = str(safe_get(case_obj_a, "answer.text", "") or "")
    b_txt = str(safe_get(case_obj_b, "answer.text", "") or "")
    diff = "\n".join(difflib.unified_diff(a_txt.splitlines(), b_txt.splitlines(), fromfile="run_a", tofile="run_b", lineterm=""))
    st.code(diff or "(no diff)", language="diff")

    st.markdown("**Score diff**")
    def _judge_overall(sc: Optional[Dict[str, Any]]) -> Optional[int]:
        j = sc.get("judge") if isinstance(sc, dict) else None
        return safe_int((j or {}).get("overall")) if isinstance(j, dict) else None

    oa = _judge_overall(scored_a)
    ob = _judge_overall(scored_b)
    st.write({"run_a_overall": oa, "run_b_overall": ob, "delta": (ob - oa) if (oa is not None and ob is not None) else None})

    with st.expander("Inspect Run A (Flight Recorder)", expanded=False):
        render_flight_recorder(
            run_obj_a,
            case_obj_a,
            scored_a,
            docs_root=docs_root,
            export_dir=export_dir,
            key_prefix=f"compare_a_{selected_case}_",
        )
    with st.expander("Inspect Run B (Flight Recorder)", expanded=False):
        render_flight_recorder(
            run_obj_b,
            case_obj_b,
            scored_b,
            docs_root=docs_root,
            export_dir=export_dir,
            key_prefix=f"compare_b_{selected_case}_",
        )


def page_diagnostics(
    *,
    runs_dir: Path,
    scored_dir: Path,
    gold_by_case_id: Dict[str, Dict[str, Any]],
    diagnostics_mode: str,
    docs_root: Path,
    export_dir: Path,
) -> None:
    st.header("Diagnostics")
    st.caption("Engineering diagnostics: coverage, score distributions, funnels, and failure modes.")

    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    selected_runs = st.multiselect(
        "Select runs",
        options=run_files,
        default=run_files[: min(6, len(run_files))],
    )
    if not selected_runs:
        st.info("Select one or more runs.")
        return

    # Load chosen runs.
    all_rows: List[Dict[str, Any]] = []
    per_run: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = {}

    for path in selected_runs:
        run_obj = load_run(path)
        if run_obj.get("_error"):
            st.warning(f"Skipping run (load error): {path}")
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        scored_obj_raw = load_scored_run(scored_path) if scored_path else None
        require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
        scoring_idx = index_scored_cases(scored_obj) if scored_obj else {}
        scoring_by_case_id = join_scoring(run_obj, {"scored_cases": [{"case_id": k, **v} for k, v in scoring_idx.items()]})
        rows = flatten_rows_for_run(run_obj=run_obj, scoring_by_case_id=scoring_by_case_id, gold_by_case_id=gold_by_case_id)
        per_run[path] = (run_obj, rows, scoring_idx)
        all_rows.extend(rows)

    if not all_rows:
        st.warning("No case rows loaded.")
        return

    # A) Coverage map.
    st.subheader("A) Retrieval Coverage Map")
    st.caption(
        "Coverage uses gold expected_episode_ids when available; otherwise it uses a retrieval-score proxy. "
        "Y-axis uses judge overall when available; otherwise uses correctness proxy."
    )

    points: List[Tuple[float, float, str]] = []
    if diagnostics_mode == "Per-case":
        for r in all_rows:
            cov = safe_float(r.get("retrieval_coverage"))
            if cov is None:
                continue
            y = safe_float(r.get("judge_overall"))
            if y is None:
                y = safe_float(r.get("judge_correctness"))
            if y is None:
                continue
            points.append((float(cov), float(y), ""))
        plot_scatter(points, title="Coverage vs quality (per case)", x_label="retrieval_coverage", y_label="judge_overall (or proxy)")
    else:
        for path, (_run_obj, rows, _sc) in per_run.items():
            covs = [safe_float(r.get("retrieval_coverage")) for r in rows if r.get("retrieval_coverage") is not None]
            ys = [safe_float(r.get("judge_overall")) for r in rows if r.get("judge_overall") is not None]
            if not ys:
                ys = [safe_float(r.get("judge_correctness")) for r in rows if r.get("judge_correctness") is not None]
            if not covs or not ys:
                continue
            x = float(sum(covs) / len(covs))
            y = float(sum(ys) / len(ys))
            lbl = truncate(str(safe_get(per_run[path][0], "run.run_name", Path(path).stem) or ""), 18)
            points.append((x, y, lbl))
        plot_scatter(points, title="Coverage vs quality (per run)", x_label="mean coverage", y_label="mean judge overall (or proxy)")

    # B) Score distribution.
    st.subheader("B) Retrieval Score Distribution")
    one_run = st.selectbox("Select a run for per-case plots", options=list(per_run.keys()), index=0)
    run_obj, _rows, scoring_idx = per_run[one_run]
    cases = run_obj.get("cases") if isinstance(run_obj.get("cases"), list) else []
    case_ids = [str(c.get("case_id")) for c in cases if isinstance(c, dict) and c.get("case_id")]
    if case_ids:
        case_id = st.selectbox("Case id", options=case_ids, index=0)
        case_obj = next((c for c in cases if isinstance(c, dict) and str(c.get("case_id")) == case_id), None)
        if isinstance(case_obj, dict):
            res = _case_retrieval_results(case_obj)
            scores = [safe_float(r.get("score")) for r in res if isinstance(r, dict) and safe_float(r.get("score")) is not None]
            ranks = [safe_int(r.get("rank")) for r in res if isinstance(r, dict) and safe_float(r.get("score")) is not None]
            if scores and ranks:
                fig, ax = plt.subplots(figsize=(7.5, 3.8))
                ax.bar([int(x) for x in ranks], [float(s) for s in scores])
                ax.set_xlabel("rank")
                ax.set_ylabel("retrieval score")
                ax.set_title(f"Retrieval scores by rank ({case_id})")
                ax.grid(alpha=0.25)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No retrieval scores present for this case.")

            # C) Context funnel
            st.subheader("C) Context Funnel")
            vectorstore = safe_get(run_obj, "config.vectorstore", {})
            num_vectors = safe_int((vectorstore or {}).get("num_vectors"))
            retrieved_docs = len(res)
            context_chars = safe_int(safe_get(case_obj, "answer.stats.context_chars", None))
            prompt_tokens = safe_int(safe_get(case_obj, "answer.usage.prompt_tokens", None))
            total_tokens = safe_int(safe_get(case_obj, "answer.usage.total_tokens", None))

            cols = st.columns(5)
            cols[0].metric("num_vectors", num_vectors if num_vectors is not None else 0)
            cols[1].metric("retrieved_docs", retrieved_docs)
            cols[2].metric("context_chars", context_chars if context_chars is not None else 0)
            cols[3].metric("prompt_tokens", prompt_tokens if prompt_tokens is not None else 0)
            cols[4].metric("total_tokens", total_tokens if total_tokens is not None else 0)

            funnel_vals = [
                float(num_vectors or 0),
                float(retrieved_docs),
                float(context_chars or 0),
                float(prompt_tokens or 0),
                float(total_tokens or 0),
            ]
            funnel_labels = ["num_vectors", "retrieved_docs", "context_chars", "prompt_tokens", "total_tokens"]
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            ax.bar(range(len(funnel_vals)), funnel_vals)
            ax.set_xticks(range(len(funnel_vals)))
            ax.set_xticklabels(funnel_labels, rotation=20, ha="right")
            ax.set_title("Context funnel")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, clear_figure=True)

    # D) Failure mode pie.
    st.subheader("D) Failure Mode Pie")
    counts: Dict[str, int] = {"ok": 0, "retrieval_failure": 0, "missing_context": 0, "grounding_failure": 0, "reasoning_failure": 0}
    for r in all_rows:
        cov = safe_float(r.get("retrieval_coverage"))
        label = classify_failure(row=r, coverage=cov)
        counts[label] = int(counts.get(label, 0)) + 1
    plot_pie({k: v for k, v in counts.items() if k != "ok"}, title="Failure modes (heuristic)")

    st.dataframe(
        [{"failure_mode": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
        use_container_width=True,
        hide_index=True,
    )


def page_executive_insights(
    *,
    runs_dir: Path,
    scored_dir: Path,
    gold_by_case_id: Dict[str, Dict[str, Any]],
    export_dir: Path,
) -> None:
    st.header("Executive Insights")
    st.caption("Trends, cost vs quality, and Pareto-optimal efficiency frontier.")

    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
    scored_run_ids = set(list_scored_run_ids(str(scored_dir), require_two_pass=require_two_pass))

    # Build a friendly label map and a better default selection.
    # We default to scored runs so the charts don't look "broken" while bulk scoring is still in flight.
    run_labels: Dict[str, str] = {}
    scored_run_files: List[str] = []
    unscored_run_files: List[str] = []

    scan_limit_default = min(250, len(run_files))
    scan_limit_max = min(2000, len(run_files))
    scan_limit = scan_limit_default
    if len(run_files) > scan_limit_default:
        scan_limit = st.slider(
            "Scan newest N runs for scoring coverage",
            min_value=min(50, scan_limit_max),
            max_value=scan_limit_max,
            value=scan_limit_default,
            step=50,
            help="Higher = finds older scored runs, but loads more run logs.",
        )

    for path in run_files[: int(scan_limit)]:
        run_obj = load_run(path)
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        run_name = str(safe_get(run_obj, "run.run_name", "") or "")
        created_at = str(safe_get(run_obj, "run.created_at_utc", "") or "")

        has_score = bool(run_id and run_id in scored_run_ids)
        badge = "✓" if has_score else "…"
        name = run_name or Path(path).stem
        label = f"{badge} {created_at} — {name}".strip()
        run_labels[path] = label

        if has_score:
            scored_run_files.append(path)
        else:
            unscored_run_files.append(path)

    with st.expander("Scoring coverage", expanded=False):
        st.write(
            {
                "runs_found": len(run_files),
                "runs_scanned": int(scan_limit),
                "runs_labeled": len(run_labels),
                "scored_files_found": len(scored_run_ids),
                "require_two_pass": bool(require_two_pass),
                "runs_with_scores_in_scanned_window": len(scored_run_files),
            }
        )
        if require_two_pass and scored_run_ids:
            st.caption("✓ indicates a two-pass scored file found for that run_id.")

    show_only_scored = st.checkbox(
        "Show only scored runs",
        value=True,
        help="Recommended while bulk two-pass rescoring is still running.",
    )
    selectable = scored_run_files if show_only_scored else list(run_labels.keys())

    # Curated story mode: pick a small representative set of runs that
    # tells the RAG-practices progression (baseline -> metadata -> mmr -> qe -> hybrid).
    curated_story_mode = bool(st.session_state.get("curated_story_mode", True))
    default_selection: List[str]
    if curated_story_mode:
        # Use the same selectable window (usually the scanned window) so defaults are consistent
        # with the scoring-coverage scan.
        default_selection = curated_run_files_for_story(
            list(selectable),
            scored_dir=str(scored_dir),
            require_two_pass=bool(require_two_pass),
        )
        if not default_selection:
            default_selection = selectable[: min(10, len(selectable))]
    else:
        default_selection = selectable[: min(10, len(selectable))]

    if not default_selection and run_files:
        # Fallback so the UI is usable even when nothing is scored yet.
        default_selection = run_files[: min(10, len(run_files))]

    selected_runs = st.multiselect(
        "Select runs",
        options=selectable,
        default=default_selection,
        format_func=lambda p: run_labels.get(p, str(p)),
        key="exec_selected_runs",
    )
    if not selected_runs:
        st.info("Select one or more runs.")
        return

    leaderboard: List[Dict[str, Any]] = []
    points_for_frontier: List[Tuple[str, float, float]] = []

    for path in selected_runs:
        run_obj = load_run(path)
        if run_obj.get("_error"):
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        scored_obj_raw = load_scored_run(scored_path) if scored_path else None
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
        scoring_idx = index_scored_cases(scored_obj) if scored_obj else {}
        scoring_by_case_id = join_scoring(run_obj, {"scored_cases": [{"case_id": k, **v} for k, v in scoring_idx.items()]})
        rows = flatten_rows_for_run(run_obj=run_obj, scoring_by_case_id=scoring_by_case_id, gold_by_case_id=gold_by_case_id)
        agg = aggregate_run_rows(rows)
        agg["run_file"] = path
        leaderboard.append(agg)

        score = safe_float(agg.get("avg_overall"))
        tokens = safe_float(agg.get("avg_total_tokens"))
        if run_id and score is not None and tokens is not None:
            points_for_frontier.append((run_id, float(score), float(tokens)))

    if not leaderboard:
        st.warning("No runs could be loaded.")
        return

    leaderboard.sort(key=lambda r: (safe_float(r.get("avg_overall")) or -1.0), reverse=True)

    # Executive scorecard (CTO-friendly)
    st.subheader("Executive scorecard")
    best_row = leaderboard[0] if leaderboard else None
    latest_row = None
    if leaderboard:
        latest_row = max(
            leaderboard,
            key=lambda r: (_parse_utc_iso(r.get("created_at_utc")) or datetime(1970, 1, 1, tzinfo=timezone.utc)),
        )

    def _fmt_rate(x: Any) -> str:
        v = safe_float(x)
        return "—" if v is None else f"{100.0 * float(v):.0f}%"

    def _fmt_ms(x: Any) -> str:
        v = safe_float(x)
        return "—" if v is None else f"{float(v):.0f} ms"

    def _fmt_num(x: Any) -> str:
        v = safe_float(x)
        return "—" if v is None else f"{float(v):.1f}"

    cols = st.columns(6)
    if best_row:
        cols[0].metric("Best avg_overall", "—" if safe_float(best_row.get("avg_overall")) is None else f"{safe_float(best_row.get('avg_overall')):.0f}")
        cols[1].metric("Best groundedness", _fmt_num(best_row.get("avg_groundedness")))
        cols[2].metric("Ctx insufficient", _fmt_rate(best_row.get("context_sufficiency_insufficient_rate")))
        cols[3].metric("IDK rate", _fmt_rate(best_row.get("idk_rate")))
        cols[4].metric("Avg tokens", "—" if safe_float(best_row.get("avg_total_tokens")) is None else f"{safe_float(best_row.get('avg_total_tokens')):.0f}")
        cols[5].metric("Avg latency", _fmt_ms(best_row.get("avg_latency_ms")))

    if latest_row and best_row and latest_row is not best_row:
        with st.expander("Latest run snapshot", expanded=False):
            st.write(
                {
                    "run_name": latest_row.get("run_name"),
                    "created_at_utc": latest_row.get("created_at_utc"),
                    "avg_overall": latest_row.get("avg_overall"),
                    "avg_groundedness": latest_row.get("avg_groundedness"),
                    "context_insufficient_rate": latest_row.get("context_sufficiency_insufficient_rate"),
                    "avg_total_tokens": latest_row.get("avg_total_tokens"),
                    "avg_latency_ms": latest_row.get("avg_latency_ms"),
                    "judge_error_rate": latest_row.get("judge_error_rate"),
                    "judge_coverage_rate": latest_row.get("judge_coverage_rate"),
                }
            )

    # Score over time.
    st.subheader("Score over time")
    times: List[Tuple[datetime, float, str]] = []
    for r in leaderboard:
        t = str(r.get("created_at_utc") or "")
        s = safe_float(r.get("avg_overall"))
        dt = _parse_utc_iso(t)
        if dt is not None and s is not None:
            times.append((dt, float(s), t))
    times.sort(key=lambda x: x[0])
    if times:
        xs = list(range(len(times)))
        ys = [s for _dt, s, _ts in times]
        x_labels = [ts for _dt, _s, ts in times]
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.plot(xs, ys, marker="o")
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("avg_overall")
        ax.set_title("Average overall score over time")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info(
            "No avg_overall values found. Likely cause: selected runs are not scored yet "
            "(or are filtered out by Require two-pass scores)."
        )

    # Cost vs quality scatter.
    st.subheader("Cost vs quality")
    pts: List[Tuple[float, float, str]] = []
    x_label = "avg_total_tokens"
    for r in leaderboard:
        s = safe_float(r.get("avg_overall"))
        if s is None:
            continue
        t = safe_float(r.get("avg_total_tokens"))
        if t is None:
            # Older run logs sometimes lack total_tokens; fall back to prompt_tokens.
            t = safe_float(r.get("avg_prompt_tokens"))
            if t is not None:
                x_label = "avg_prompt_tokens (proxy)"
        if t is None:
            # As a last resort, plot against context size proxy so the chart still works.
            t = safe_float(r.get("avg_context_chars"))
            if t is not None:
                x_label = "avg_context_chars (proxy)"
        if t is None:
            continue
        lbl = truncate(str(r.get("run_name") or ""), 18)
        pts.append((float(t), float(s), lbl))

    if pts:
        plot_scatter(pts, title="Cost proxy vs score", x_label=x_label, y_label="avg_overall")
    else:
        st.info("No cost metrics available yet (missing tokens + context proxies).")

    # Efficiency frontier.
    st.subheader("Efficiency frontier (Pareto optimal)")
    # If tokens are missing, fall back to prompt tokens, then context size.
    frontier_points: List[Tuple[str, float, float]] = []
    x_key = "avg_total_tokens"
    for r in leaderboard:
        rid = str(r.get("run_id") or "").strip()
        s = safe_float(r.get("avg_overall"))
        if not rid or s is None:
            continue
        t = safe_float(r.get("avg_total_tokens"))
        if t is None:
            t = safe_float(r.get("avg_prompt_tokens"))
            if t is not None:
                x_key = "avg_prompt_tokens"
        if t is None:
            t = safe_float(r.get("avg_context_chars"))
            if t is not None:
                x_key = "avg_context_chars"
        if t is None:
            continue
        frontier_points.append((rid, float(s), float(t)))

    frontier = pareto_frontier(frontier_points) if frontier_points else []
    if frontier:
        st.dataframe(
            [{"run_id": rid, "avg_overall": s, x_key: t} for rid, s, t in frontier],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Not enough data for frontier yet (need avg_overall and a cost axis).")

    # Top improvements table.
    st.subheader("Top improvements")
    baseline = st.selectbox(
        "Baseline run",
        options=[r.get("run_file") for r in leaderboard if r.get("run_file")],
        index=min(1, len(leaderboard) - 1),
    )
    best = leaderboard[0].get("run_file")
    base_row = next((r for r in leaderboard if r.get("run_file") == baseline), None)
    best_row = next((r for r in leaderboard if r.get("run_file") == best), None)
    if base_row and best_row:
        diffs = {
            "avg_overall": (safe_float(best_row.get("avg_overall")) or 0) - (safe_float(base_row.get("avg_overall")) or 0),
            "avg_correctness": (safe_float(best_row.get("avg_correctness")) or 0) - (safe_float(base_row.get("avg_correctness")) or 0),
            "avg_groundedness": (safe_float(best_row.get("avg_groundedness")) or 0) - (safe_float(base_row.get("avg_groundedness")) or 0),
            "idk_rate": (safe_float(best_row.get("idk_rate")) or 0) - (safe_float(base_row.get("idk_rate")) or 0),
            "citation_rate": (safe_float(best_row.get("citation_rate")) or 0) - (safe_float(base_row.get("citation_rate")) or 0),
            "avg_total_tokens": (safe_float(best_row.get("avg_total_tokens")) or 0) - (safe_float(base_row.get("avg_total_tokens")) or 0),
            "avg_latency_ms": (safe_float(best_row.get("avg_latency_ms")) or 0) - (safe_float(base_row.get("avg_latency_ms")) or 0),
        }
        st.write({
            "best_run": {"run_name": best_row.get("run_name"), "run_file": best},
            "baseline_run": {"run_name": base_row.get("run_name"), "run_file": baseline},
        })
        st.dataframe(
            [{"metric": k, "delta(best - baseline)": v} for k, v in diffs.items()],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Export executive summary", type="secondary"):
            out_path = export_view_snippet(
                export_dir=export_dir,
                run_obj={"run": {"run_id": "executive"}},
                case_obj=None,
                scored_case=None,
                extra={"leaderboard": leaderboard[:20], "frontier": frontier, "diffs": diffs},
            )
            if out_path:
                st.success(f"Exported: {out_path}")
            else:
                st.error("Failed to export.")

    with st.expander("Leaderboard rows", expanded=False):
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)


# -----------------------------
# History page (CTO demo-friendly)
# -----------------------------
def _parse_utc_iso(ts: Any) -> Optional[datetime]:
    s = str(ts or "").strip()
    if not s:
        return None
    try:
        # Handle trailing Z.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_markdown_file(path: str, *, limit_chars: int = 60000) -> Dict[str, Any]:
    """Read a markdown file and return a small info dict.

    Uses a char limit to keep the UI snappy.
    """
    p = Path(path).expanduser()
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        truncated_txt = txt[: int(limit_chars)]
        return {
            "path": str(p),
            "chars": len(txt),
            "truncated": len(txt) > int(limit_chars),
            "text": truncated_txt,
        }
    except Exception as e:
        return {"path": str(p), "error": f"{type(e).__name__}: {e}", "text": ""}


@st.cache_data(show_spinner=False)
def list_evolution_report_files(evolution_dir: str) -> List[str]:
    p = Path(evolution_dir).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    files = [x for x in p.glob("*.md") if x.is_file()]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(x) for x in files]


def _default_evolution_report_index(files: List[str]) -> int:
    """Prefer scored reports when present; else pick the newest."""
    if not files:
        return 0
    for i, f in enumerate(files):
        name = Path(f).name.lower()
        if "scored" in name or "llm_scored" in name:
            return i
    return 0


def scored_summary(scored_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract a small, stable summary from a scored run file.

    The scorer schema may drift; keep this defensive.
    """
    if not scored_obj or not isinstance(scored_obj, dict) or scored_obj.get("_error"):
        return {
            "present": False,
            "cases_scored": None,
            "avg_overall": None,
            "overall_scores": None,
            "judge_cases": None,
            "judge_error_cases": None,
        }

    cases_scored = safe_int(safe_get(scored_obj, "score_summary.cases_scored", None))
    avg_overall = safe_int(safe_get(scored_obj, "score_summary.avg_overall", None))
    overall_scores = safe_get(scored_obj, "score_summary.overall_scores", None)
    if not isinstance(overall_scores, list):
        overall_scores = None

    judge_cases = 0
    judge_error_cases = 0
    rows = scored_obj.get("scored_cases")
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("judge"), dict):
                judge_cases += 1
            if r.get("judge_error"):
                judge_error_cases += 1

    return {
        "present": True,
        "cases_scored": cases_scored,
        "avg_overall": avg_overall,
        "overall_scores": overall_scores,
        "judge_cases": judge_cases,
        "judge_error_cases": judge_error_cases,
    }


def collapse_cards_by_run_name(
    cards: List[Dict[str, Any]],
    *,
    prefer: str = "best_score",
) -> List[Dict[str, Any]]:
    """Collapse multiple run logs with identical run_name.

    prefer:
      - best_score: keep the card with highest avg_overall; tie-breaker newest
      - newest: keep newest by created_at_utc/mtime
    """

    def _dt(c: Dict[str, Any]) -> datetime:
        dt = _parse_utc_iso(c.get("created_at_utc"))
        if dt is not None:
            return dt
        try:
            return datetime.fromtimestamp(Path(str(c.get("run_file") or "")).stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for c in cards:
        name = str(c.get("run_name") or "").strip() or "(missing)"
        by_name.setdefault(name, []).append(c)

    out: List[Dict[str, Any]] = []
    for name, xs in by_name.items():
        if len(xs) == 1:
            out.append(xs[0])
            continue

        if prefer == "newest":
            best = max(xs, key=_dt)
            out.append(best)
            continue

        # best_score (default): prefer scored runs; fall back to newest
        def _score_key(c: Dict[str, Any]) -> Tuple[float, datetime]:
            s = safe_float(c.get("avg_overall"))
            # Treat missing score as very low.
            s2 = float(s) if s is not None else -1.0
            return (s2, _dt(c))

        best = max(xs, key=_score_key)
        out.append(best)

    # Keep a stable, readable ordering.
    out.sort(key=lambda c: _dt(c))
    return out


def _run_card(run_obj: Dict[str, Any], *, scored_obj: Optional[Dict[str, Any]], run_file: str) -> Dict[str, Any]:
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    run_name = str(safe_get(run_obj, "run.run_name", "") or "")
    created_at = str(safe_get(run_obj, "run.created_at_utc", "") or "")

    retrieval_policy = str(safe_get(run_obj, "config.retrieval.policy", "") or "")
    search_type = str(safe_get(run_obj, "config.retrieval.search_type", "") or "")
    k = safe_int(safe_get(run_obj, "config.retrieval.k", None))
    qe_enabled = bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))

    llm_model = str(safe_get(run_obj, "config.llm.model", "") or "").strip() or None
    qe_model = str(safe_get(run_obj, "config.retrieval.query_expansion.model", "") or "").strip() or None
    embed_model = str(safe_get(run_obj, "config.embedding.model", "") or "").strip() or None

    script_persist = str(safe_get(run_obj, "config.data_version.script.persist_directory", "") or "")
    derived_persist = str(safe_get(run_obj, "config.data_version.derived.persist_directory", "") or "")
    derived_build_tag = str(safe_get(run_obj, "config.data_version.derived.build_tag", "") or "")

    sc = scored_summary(scored_obj if isinstance(scored_obj, dict) else None)
    avg_overall = sc.get("avg_overall")

    summary = safe_get(run_obj, "summary", {})

    return {
        "run_file": run_file,
        "scored_present": sc.get("present"),
        "cases_scored": sc.get("cases_scored"),
        "judge_cases": sc.get("judge_cases"),
        "judge_error_cases": sc.get("judge_error_cases"),
        "run_id": run_id,
        "run_name": run_name,
        "created_at_utc": created_at,
        "retrieval_policy": retrieval_policy,
        "retrieval_policy_display": friendly_retrieval_policy(retrieval_policy),
        "search_type": search_type,
        "k": k,
        "qe_enabled": qe_enabled,
        "llm_model": llm_model,
        "qe_model": qe_model,
        "embed_model": embed_model,
        "avg_overall": avg_overall,
        "quote_in_context_rate": safe_float((summary or {}).get("quote_in_context_rate")),
        "grounding_failure_rate": safe_float((summary or {}).get("grounding_failure_rate")),
        "retrieval_failure_rate": safe_float((summary or {}).get("retrieval_failure_rate")),
        "retrieval_empty_rate": safe_float((summary or {}).get("retrieval_empty_rate")),
        "idk_rate": safe_float((summary or {}).get("idk_rate")),
        "avg_total_tokens": safe_float((summary or {}).get("avg_total_tokens")),
        "avg_context_docs": safe_float((summary or {}).get("avg_context_docs")),
        "avg_context_chars": safe_float((summary or {}).get("avg_context_chars")),
        "avg_aggregation_readiness_score": safe_float((summary or {}).get("avg_aggregation_readiness_score")),
        "script_persist_directory": script_persist,
        "derived_persist_directory": derived_persist,
        "derived_build_tag": derived_build_tag,
    }


def _trend_plot(
    *,
    points: List[Tuple[datetime, float, str]],
    title: str,
    y_label: str,
) -> None:
    if not points:
        st.info("No data points for this metric.")
        return

    points = sorted(points, key=lambda t: t[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(xs, ys, marker="o", linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    # Keep the x-axis readable.
    ax.set_xticks(xs)
    ax.set_xticklabels([x.strftime("%m-%d %H:%M") for x in xs], rotation=25, ha="right", fontsize=8)
    st.pyplot(fig, clear_figure=True)


def page_history(
    *,
    project_root: Path,
    runs_dir: Path,
    scored_dir: Path,
    docs_root: Path,
    export_dir: Path,
) -> None:
    st.header("History")
    st.caption(
        "A demo-friendly view of progress over time: what improved, what regressed, and why — grounded in logged run metrics."
    )
    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    limit = st.slider("Max runs to load", min_value=10, max_value=min(400, len(run_files)), value=min(120, len(run_files)), step=10)
    selected_files = run_files[: int(limit)]

    # Build run cards (lightweight summaries), newest-first input but we will sort by timestamp.
    cards: List[Dict[str, Any]] = []
    for rf in selected_files:
        run_obj = load_run(rf)
        if run_obj.get("_error"):
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        scored_obj_raw = load_scored_run(scored_path) if scored_path else None
        require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
        cards.append(_run_card(run_obj, scored_obj=scored_obj, run_file=rf))

    if not cards:
        st.warning("No run logs could be loaded.")
        return

    # Time ordering.
    def _card_dt(c: Dict[str, Any]) -> datetime:
        dt = _parse_utc_iso(c.get("created_at_utc"))
        if dt is not None:
            return dt
        # Fall back to file mtime.
        try:
            return datetime.fromtimestamp(Path(str(c.get("run_file") or "")).stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    cards_sorted = sorted(cards, key=_card_dt)

    # --- Presentation filters (default: keep charts clean) ---
    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    hide_unscored = c1.checkbox("Hide unscored runs", value=True)
    collapse_dupes = c2.checkbox("Collapse duplicate run_name", value=True)
    collapse_pref = c2.selectbox("When collapsing, keep", options=["best_score", "newest"], index=0)

    name_prefix = c3.text_input("Run name prefix (optional)", value="")
    name_prefix = str(name_prefix or "").strip()

    filtered = list(cards_sorted)
    if name_prefix:
        filtered = [c for c in filtered if str(c.get("run_name") or "").startswith(name_prefix)]

    if hide_unscored:
        filtered = [c for c in filtered if safe_float(c.get("avg_overall")) is not None and (safe_int(c.get("cases_scored")) or 0) > 0]

    if collapse_dupes:
        filtered = collapse_cards_by_run_name(filtered, prefer=str(collapse_pref))

    st.caption(f"Showing {len(filtered)} / {len(cards_sorted)} runs")

    # --- Executive summary (CTO / stakeholder-friendly) ---
    st.subheader("Executive summary")

    def _score(c: Dict[str, Any]) -> Optional[float]:
        return safe_float(c.get("avg_overall"))

    scored_cards = [c for c in filtered if _score(c) is not None]

    if scored_cards:
        scored_by_time = sorted(scored_cards, key=_card_dt)
        best = max(scored_cards, key=lambda c: float(_score(c) or -1.0))
        latest = scored_by_time[-1]
        earliest = scored_by_time[0]
        prev = scored_by_time[-2] if len(scored_by_time) >= 2 else None

        best_s = float(_score(best) or 0.0)
        latest_s = float(_score(latest) or 0.0)
        earliest_s = float(_score(earliest) or 0.0)
        prev_s = float(_score(prev) or 0.0) if prev is not None else None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best avg_overall", f"{best_s:.0f}")
        col2.metric(
            "Latest avg_overall",
            f"{latest_s:.0f}",
            delta=(f"{(latest_s - prev_s):+.0f}" if prev_s is not None else None),
        )
        col3.metric("Earliest→latest", f"{(latest_s - earliest_s):+.0f}")
        col4.metric("Scored runs loaded", str(len(scored_cards)))

        st.markdown(
            "\n".join(
                [
                    f"- Current best run: {best.get('run_name')}",
                    f"- Recommended config (best run): policy={friendly_retrieval_policy(best.get('retrieval_policy'))}, search={best.get('search_type')}, k={best.get('k')}, qe={best.get('qe_enabled')}",
                    f"- Script index: {best.get('script_persist_directory')}",
                    f"- Derived index: {best.get('derived_persist_directory')} (tag: {best.get('derived_build_tag')})",
                ]
            )
        )
    else:
        st.info(
            "No scored runs found (avg_overall is empty). Run the scorer to populate judge metrics: "
            "python experiments/score_runs.py --runs-dir experiments/runs --gold experiments/gold_answers.json"
        )

    # --- Tuning notes (your narrative) ---
    tuning_path = project_root / "experiments" / "tuning steps.md"
    notes = load_markdown_file(str(tuning_path))
    with st.expander("Tuning notes (experiments/tuning steps.md)", expanded=False):
        if notes.get("error"):
            st.error(notes["error"])
        else:
            st.markdown(notes.get("text") or "")
            if notes.get("truncated"):
                st.caption("(Notes truncated for UI performance)")

    st.subheader("All runs")

    metric_options = {
        "Judge avg_overall (scored)": ("avg_overall", "avg_overall", "Score (0..100)", True),
        "Quote-in-context rate": ("quote_in_context_rate", "quote_in_context_rate", "rate", False),
        "Grounding failure rate (heuristic)": ("grounding_failure_rate", "grounding_failure_rate", "rate", False),
        "Retrieval failure rate": ("retrieval_failure_rate", "retrieval_failure_rate", "rate", False),
        "IDK rate": ("idk_rate", "idk_rate", "rate", False),
        "Avg total tokens": ("avg_total_tokens", "avg_total_tokens", "tokens", False),
        "Avg context docs": ("avg_context_docs", "avg_context_docs", "docs", False),
        "Avg context chars": ("avg_context_chars", "avg_context_chars", "chars", False),
        "Avg aggregation readiness score": ("avg_aggregation_readiness_score", "avg_aggregation_readiness_score", "score (0..100)", False),
    }

    metric_label = st.selectbox("Trend metric", options=list(metric_options.keys()), index=0)
    metric_key, _col, y_label, scored_only = metric_options[metric_label]

    points: List[Tuple[datetime, float, str]] = []
    for c in filtered:
        dt = _card_dt(c)
        v = safe_float(c.get(metric_key))
        if v is None:
            continue
        if scored_only and c.get("avg_overall") is None:
            continue
        label = str(c.get("run_name") or "")
        points.append((dt, float(v), label))

    _trend_plot(points=points, title=f"{metric_label} over time", y_label=y_label)

    # Table
    with st.expander("Runs table", expanded=False):
        table_rows = []
        for c in filtered:
            row = {
                "created_at_utc": c.get("created_at_utc"),
                "run_name": c.get("run_name"),
                "retrieval_policy": friendly_retrieval_policy(c.get("retrieval_policy")),
                "search_type": c.get("search_type"),
                "k": c.get("k"),
                "qe_enabled": c.get("qe_enabled"),
                "avg_overall": c.get("avg_overall"),
                "cases_scored": c.get("cases_scored"),
                "quote_in_context_rate": c.get("quote_in_context_rate"),
                "grounding_failure_rate": c.get("grounding_failure_rate"),
                "retrieval_failure_rate": c.get("retrieval_failure_rate"),
                "idk_rate": c.get("idk_rate"),
                "avg_total_tokens": c.get("avg_total_tokens"),
                "derived_build_tag": truncate(c.get("derived_build_tag"), 36),
                "run_file": c.get("run_file"),
            }
            table_rows.append(row)
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Deltas (improvement/regression) in time.
    st.subheader("What changed")
    deltas: List[Dict[str, Any]] = []
    prev: Optional[Tuple[datetime, float, Dict[str, Any]]] = None
    for c in filtered:
        dt = _card_dt(c)
        v = safe_float(c.get(metric_key))
        if v is None:
            continue
        if scored_only and c.get("avg_overall") is None:
            continue
        if prev is not None:
            _dt_prev, v_prev, c_prev = prev
            deltas.append(
                {
                    "created_at_utc": c.get("created_at_utc"),
                    "run_name": c.get("run_name"),
                    "delta": float(v) - float(v_prev),
                    "prev_run": c_prev.get("run_name"),
                    "policy": friendly_retrieval_policy(c.get("retrieval_policy")),
                }
            )
        prev = (dt, float(v), c)

    if deltas:
        best = sorted(deltas, key=lambda r: float(r.get("delta") or 0), reverse=True)[:5]
        worst = sorted(deltas, key=lambda r: float(r.get("delta") or 0))[:5]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Biggest improvements (vs previous run)**")
            st.dataframe(best, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Biggest regressions (vs previous run)**")
            st.dataframe(worst, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough points to compute deltas for this metric.")

    # Configuration impact (simple grouped means)
    st.subheader("What tends to work")
    st.caption("Grouped averages are descriptive, not causal.")

    def _group_mean(key: str) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[float]] = {}
        for c in filtered:
            v = safe_float(c.get(metric_key))
            if v is None:
                continue
            if scored_only and c.get("avg_overall") is None:
                continue
            if key == "retrieval_policy":
                g = friendly_retrieval_policy(c.get(key))
            else:
                g = str(c.get(key) or "(missing)")
            buckets.setdefault(g, []).append(float(v))
        rows: List[Dict[str, Any]] = []
        for g, vals in buckets.items():
            rows.append({"group": g, "runs": len(vals), "mean": float(sum(vals) / len(vals)) if vals else None})
        rows.sort(key=lambda r: (safe_float(r.get("mean")) or -1.0), reverse=True)
        return rows

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**By retrieval policy**")
        st.dataframe(_group_mean("retrieval_policy"), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**By search type**")
        st.dataframe(_group_mean("search_type"), use_container_width=True, hide_index=True)
    with col3:
        st.markdown("**By query expansion**")
        st.dataframe(_group_mean("qe_enabled"), use_container_width=True, hide_index=True)

    st.subheader("Drill down")
    st.caption("Pick a run and a case to open the Flight Recorder.")
    if not filtered:
        st.info("No runs match the current filters.")
        return

    run_labels = [f"{c.get('created_at_utc')} — {c.get('run_name')}" for c in filtered]
    idx = st.selectbox("Select run", options=list(range(len(filtered))), format_func=lambda i: run_labels[int(i)], index=max(0, len(filtered) - 1))
    chosen = filtered[int(idx)]
    run_path = str(chosen.get("run_file") or "")
    run_obj = load_run(run_path)
    if run_obj.get("_error"):
        st.error(f"Failed to load run: {run_obj.get('_error')}")
        return

    # Optional score join for the chosen run only.
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
    scored_obj_raw = load_scored_run(scored_path) if scored_path else None
    require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
    scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
    scoring_idx = index_scored_cases(scored_obj) if scored_obj else {}

    cases = run_obj.get("cases") if isinstance(run_obj.get("cases"), list) else []
    case_ids = [str(c.get("case_id")) for c in cases if isinstance(c, dict) and c.get("case_id")]
    if not case_ids:
        st.info("Run has no cases[].")
        return
    selected_case_id = st.selectbox("Select case", options=case_ids, index=0)
    case_obj = next((c for c in cases if isinstance(c, dict) and str(c.get("case_id")) == selected_case_id), None)
    if not isinstance(case_obj, dict):
        st.error("Could not find selected case.")
        return
    scored_case_obj = scoring_idx.get(selected_case_id) if scoring_idx else None

    render_flight_recorder(
        run_obj,
        case_obj,
        scored_case_obj,
        docs_root=docs_root,
        export_dir=export_dir,
        key_prefix=f"history_{run_id}_{selected_case_id}_",
    )


def page_evolution(
    *,
    project_root: Path,
    runs_dir: Path,
    scored_dir: Path,
    docs_root: Path,
    export_dir: Path,
) -> None:
    st.header("Evolution")
    st.caption("Milestone-based view of how the system improved (curated from run configs + scores).")

    # --- Evolution step reports (markdown) ---
    evo_dir = project_root / "experiments" / "evolution_reports"
    evo_files = list_evolution_report_files(str(evo_dir))
    if evo_files:
        with st.expander("Evolution step report (markdown)", expanded=True):
            idx = _default_evolution_report_index(evo_files)
            selected = st.selectbox(
                "Select evolution report",
                options=evo_files,
                index=idx,
                format_func=lambda x: Path(str(x)).name,
                key="evolution_report_select",
            )
            info = load_markdown_file(str(selected), limit_chars=120000)
            if info.get("error"):
                st.error(info["error"])
            else:
                st.caption(f"Loaded: {info.get('path')}")
                st.markdown(info.get("text") or "")
                if info.get("truncated"):
                    st.caption("(Report truncated for UI performance)")
    else:
        st.info(f"No evolution reports found under: {evo_dir}")

    run_files = list_run_files(str(runs_dir))
    if not run_files:
        st.warning(f"No run logs found under: {runs_dir}")
        return

    # Keep it snappy; user can widen if needed.
    limit = st.slider(
        "Max runs to scan",
        min_value=20,
        max_value=min(800, len(run_files)),
        value=min(250, len(run_files)),
        step=25,
    )
    selected_files = run_files[: int(limit)]

    cards: List[Dict[str, Any]] = []
    for rf in selected_files:
        run_obj = load_run(rf)
        if run_obj.get("_error"):
            continue
        run_id = str(safe_get(run_obj, "run.run_id", "") or "")
        scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
        scored_obj_raw = load_scored_run(scored_path) if scored_path else None
        require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
        scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
        cards.append(_run_card(run_obj, scored_obj=scored_obj, run_file=rf))

    # Default: scored runs only.
    scored = [c for c in cards if safe_float(c.get("avg_overall")) is not None and (safe_int(c.get("cases_scored")) or 0) > 0]
    if not scored:
        st.info("No scored runs found in the scanned window. Run the scorer to populate judge metrics.")
        return

    def _milestone_classify(c: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return (milestone_id, human_label, rationale)."""
        policy = str(c.get("retrieval_policy") or "").strip().lower()
        search = str(c.get("search_type") or "").strip().lower()
        qe = bool(c.get("qe_enabled"))
        k = safe_int(c.get("k"))
        script_persist = str(c.get("script_persist_directory") or "")
        derived_persist = str(c.get("derived_persist_directory") or "")

        has_derived = bool(derived_persist and "derived" in derived_persist.lower())
        has_meta = bool("chroma_db_meta" in script_persist)

        # Order matters: we want the milestone buckets to reflect how we actually evolved.
        if has_derived and policy in {"blended", "derived_then_script", "derived_only", "auto"}:
            if qe:
                return (
                    "derived_routed_qe",
                    "Derived/routed + query expansion",
                    "Uses a derived-cards index for episode shortlisting/routing AND query expansion + fusion for recall.",
                )
            return (
                "derived_routed",
                "Derived/routed retrieval",
                "Uses a derived-cards index for episode shortlisting/routing to improve multi-episode context coverage.",
            )

        if qe:
            return (
                "query_expansion",
                "Query expansion + fusion",
                "Generates alternate retrieval queries and fuses results (RRF) to boost recall.",
            )

        if search == "mmr":
            return (
                "mmr",
                "MMR retrieval",
                "Uses MMR to diversify retrieved chunks and reduce redundancy.",
            )

        if has_meta:
            return (
                "metadata_index",
                "Metadata-enabled script index",
                "Uses a metadata-rich script index (episode_id/etc.) enabling tighter filtering and routing.",
            )

        if k is not None and k >= 10:
            return (
                "high_k",
                f"Higher k (k={k})",
                "Increases top-k context size to improve coverage at the cost of tokens.",
            )

        if policy:
            return (
                "baseline",
                f"Baseline ({policy}, {search or 'similarity'})",
                "Baseline retrieval setup (no derived routing, no query expansion).",
            )

        return ("baseline", "Baseline", "Baseline retrieval setup.")

    scored_enriched: List[Dict[str, Any]] = []
    for c in scored:
        mid, label, why = _milestone_classify(c)
        c2 = dict(c)
        c2["milestone_id"] = mid
        c2["milestone_label"] = label
        c2["milestone_why"] = why
        scored_enriched.append(c2)

    buckets: Dict[str, Dict[str, Any]] = {}
    for c in scored_enriched:
        mid = str(c.get("milestone_id") or "baseline")
        label = str(c.get("milestone_label") or "Baseline")
        b = buckets.setdefault(mid, {"id": mid, "label": label, "why": str(c.get("milestone_why") or ""), "cards": []})
        b["cards"].append(c)

    # Keep a sensible milestone order.
    order = [
        "baseline",
        "high_k",
        "metadata_index",
        "mmr",
        "query_expansion",
        "derived_routed",
        "derived_routed_qe",
    ]
    bucket_list = sorted(buckets.values(), key=lambda b: (order.index(b["id"]) if b["id"] in order else 999, b["label"]))

    def _mean(xs: List[Optional[float]]) -> Optional[float]:
        vals = [float(x) for x in xs if x is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def _median(xs: List[Optional[float]]) -> Optional[float]:
        vals = sorted([float(x) for x in xs if x is not None])
        if not vals:
            return None
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)

    def _best(xs: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
        if not xs:
            return None
        return max(xs, key=lambda c: (safe_float(c.get(key)) or -1.0, safe_int(c.get("cases_scored")) or 0))

    # --- Evolution highlights (automatic: uses what we have in logs) ---
    best_overall = _best(scored_enriched, "avg_overall")
    baseline_bucket = buckets.get("baseline")
    best_baseline = _best(list((baseline_bucket or {}).get("cards") or []), "avg_overall")

    st.subheader("Highlights")
    cols = st.columns(4)
    cols[0].metric("Runs scanned", int(limit))
    cols[1].metric("Scored runs", len(scored_enriched))
    cols[2].metric("Best avg_overall", int(safe_float((best_overall or {}).get("avg_overall")) or 0))
    base_best = safe_float((best_baseline or {}).get("avg_overall"))
    overall_best = safe_float((best_overall or {}).get("avg_overall"))
    delta = (overall_best - base_best) if (overall_best is not None and base_best is not None) else None
    cols[3].metric("Best - baseline (Δ)", (f"{delta:+.0f}" if delta is not None else "—"))

    with st.expander("Why these milestones (how we bucket runs)", expanded=False):
        for b in bucket_list:
            st.markdown(f"- **{b.get('label')}**: {b.get('why')}")

    # --- Milestone aggregates (this is what we plot) ---
    milestone_rows: List[Dict[str, Any]] = []
    for b in bucket_list:
        xs = list(b.get("cards") or [])
        best = _best(xs, "avg_overall")
        milestone_rows.append(
            {
                "milestone": b.get("label"),
                "runs": len(xs),
                "best_avg_overall": safe_float((best or {}).get("avg_overall")),
                "mean_avg_overall": _mean([safe_float(c.get("avg_overall")) for c in xs]),
                "median_avg_overall": _median([safe_float(c.get("avg_overall")) for c in xs]),
                "mean_total_tokens": _mean([safe_float(c.get("avg_total_tokens")) for c in xs]),
                "mean_quote_in_context_rate": _mean([safe_float(c.get("quote_in_context_rate")) for c in xs]),
                "mean_grounding_failure_rate": _mean([safe_float(c.get("grounding_failure_rate")) for c in xs]),
                "best_run_name": (best or {}).get("run_name"),
                "best_llm_model": (best or {}).get("llm_model"),
                "best_qe_model": (best or {}).get("qe_model"),
                "rationale": b.get("why"),
                "best_run_file": (best or {}).get("run_file"),
            }
        )

    st.subheader("Milestones (summary)")
    st.dataframe(milestone_rows, use_container_width=True, hide_index=True)

    st.subheader("Milestone bars")
    st.caption(
        "We show multiple bars because evolution is multi-objective (quality, grounding, and cost). "
        "Note: for failure rates, lower is better."
    )

    metric_choices = {
        "Best avg_overall (higher is better)": ("best_avg_overall", "avg_overall"),
        "Mean avg_overall (higher is better)": ("mean_avg_overall", "avg_overall"),
        "Mean total tokens (lower is better)": ("mean_total_tokens", "tokens"),
        "Mean quote-in-context rate (higher is better)": ("mean_quote_in_context_rate", "rate"),
        "Mean grounding failure rate (lower is better)": ("mean_grounding_failure_rate", "rate"),
        "Runs per milestone": ("runs", "count"),
    }

    default_metrics = [
        "Best avg_overall (higher is better)",
        "Mean total tokens (lower is better)",
        "Mean grounding failure rate (lower is better)",
    ]
    selected_metrics = st.multiselect(
        "Which metrics to plot",
        options=list(metric_choices.keys()),
        default=default_metrics,
    )
    selected_metrics = selected_metrics[:3]

    for mlabel in selected_metrics:
        key, unit = metric_choices[mlabel]
        labels = [str(r.get("milestone") or "") for r in milestone_rows]
        raw_vals: List[float] = []
        for r in milestone_rows:
            v = r.get(key)
            if v is None:
                raw_vals.append(0.0)
            else:
                raw_vals.append(float(v))

        fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.45 * len(labels) + 1.2)))
        ax.barh(range(len(raw_vals)), raw_vals)
        ax.set_yticks(range(len(raw_vals)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(unit)
        ax.set_title(mlabel)
        ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    # Representative run per milestone.
    reps: List[Dict[str, Any]] = []
    for b in bucket_list:
        xs = list(b.get("cards") or [])
        if not xs:
            continue
        rep = max(xs, key=lambda c: (safe_float(c.get("avg_overall")) or -1.0, safe_int(c.get("cases_scored")) or 0))
        reps.append({
            "milestone": b["label"],
            "rep_run_name": rep.get("run_name"),
            "avg_overall": safe_float(rep.get("avg_overall")),
            "cases_scored": safe_int(rep.get("cases_scored")),
            "retrieval_policy": friendly_retrieval_policy(rep.get("retrieval_policy")),
            "search_type": rep.get("search_type"),
            "k": rep.get("k"),
            "qe_enabled": rep.get("qe_enabled"),
            "llm_model": rep.get("llm_model"),
            "qe_model": rep.get("qe_model"),
            "avg_total_tokens": safe_float(rep.get("avg_total_tokens")),
            "quote_in_context_rate": safe_float(rep.get("quote_in_context_rate")),
            "grounding_failure_rate": safe_float(rep.get("grounding_failure_rate")),
            "milestone_rationale": str(b.get("why") or ""),
            "rep_selection": "Picked as representative: highest avg_overall in milestone (tie-breaker: cases_scored).",
            "run_file": rep.get("run_file"),
        })

    st.subheader("Milestones (best representative run)")
    st.dataframe(reps, use_container_width=True, hide_index=True)

    with st.expander("Representative-run reasoning", expanded=False):
        st.caption("This is the narrative layer you can use in a demo.")
        for r in reps:
            st.markdown(
                "\n".join(
                    [
                        f"- **{r.get('milestone')}** → **{r.get('rep_run_name')}** (avg_overall={r.get('avg_overall')})",
                        f"  - Why milestone: {r.get('milestone_rationale')}",
                        f"  - Why this run: {r.get('rep_selection')}",
                    ]
                )
            )

    # Simple bar plot of representative scores.
    try:
        labels = [str(r.get("milestone") or "") for r in reps]
        vals = [safe_float(r.get("avg_overall")) or 0.0 for r in reps]
        fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.45 * len(labels) + 1.2)))
        ax.barh(range(len(vals)), vals)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("avg_overall")
        ax.set_title("Representative score by milestone")
        ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        pass

    st.subheader("Drill down")
    milestone_options = [b["label"] for b in bucket_list]
    chosen_label = st.selectbox("Milestone", options=milestone_options, index=0)
    chosen_bucket = next((b for b in bucket_list if b["label"] == chosen_label), None)
    if not chosen_bucket:
        return

    if chosen_bucket.get("why"):
        st.info(f"Milestone rationale: {chosen_bucket.get('why')}")

    # Runs within milestone.
    xs = list(chosen_bucket.get("cards") or [])
    xs.sort(key=lambda c: (safe_float(c.get("avg_overall")) or -1.0), reverse=True)
    st.caption(f"Runs in milestone: {len(xs)}")
    with st.expander("Runs in this milestone", expanded=False):
        rows = [
            {
                "run_name": c.get("run_name"),
                "created_at_utc": c.get("created_at_utc"),
                "avg_overall": c.get("avg_overall"),
                "cases_scored": c.get("cases_scored"),
                "retrieval_policy": friendly_retrieval_policy(c.get("retrieval_policy")),
                "search_type": c.get("search_type"),
                "k": c.get("k"),
                "qe_enabled": c.get("qe_enabled"),
                "run_file": c.get("run_file"),
            }
            for c in xs
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    def _run_label(c: Dict[str, Any]) -> str:
        score = safe_float(c.get("avg_overall"))
        score_s = f"{score:.0f}" if score is not None else "?"
        model = str(c.get("llm_model") or "").strip() or "(model?)"
        policy = friendly_retrieval_policy(c.get("retrieval_policy")) or "policy?"
        search = str(c.get("search_type") or "").strip() or "search?"
        k0 = c.get("k")
        qe = "qe" if bool(c.get("qe_enabled")) else "no-qe"
        return f"{score_s} — {c.get('run_name')} [{model} | {policy}/{search} k={k0} | {qe}]"

    run_labels = [_run_label(c) for c in xs]
    ridx = st.selectbox("Run", options=list(range(len(xs))), format_func=lambda i: run_labels[int(i)], index=0)
    chosen_run = xs[int(ridx)]

    with st.expander("Why this run is interesting", expanded=False):
        st.markdown("**Milestone**")
        st.write(str(chosen_run.get("milestone_label") or chosen_bucket.get("label") or ""))
        st.markdown("**Rationale**")
        st.write(str(chosen_run.get("milestone_why") or chosen_bucket.get("why") or ""))
        st.markdown("**Key config (from run log)**")
        st.json(
            {
                "llm_model": chosen_run.get("llm_model"),
                "qe_model": chosen_run.get("qe_model"),
                "embed_model": chosen_run.get("embed_model"),
                "retrieval_policy": friendly_retrieval_policy(chosen_run.get("retrieval_policy")),
                "search_type": chosen_run.get("search_type"),
                "k": chosen_run.get("k"),
                "qe_enabled": chosen_run.get("qe_enabled"),
                "script_persist_directory": chosen_run.get("script_persist_directory"),
                "derived_persist_directory": chosen_run.get("derived_persist_directory"),
                "derived_build_tag": chosen_run.get("derived_build_tag"),
            }
        )

    run_path = str(chosen_run.get("run_file") or "")
    run_obj = load_run(run_path)
    if run_obj.get("_error"):
        st.error(f"Failed to load run: {run_obj.get('_error')}")
        return

    run_id = str(safe_get(run_obj, "run.run_id", "") or "")
    scored_path = find_scored_run_for_run_id(str(scored_dir), run_id)
    scored_obj_raw = load_scored_run(scored_path) if scored_path else None
    require_two_pass = bool(st.session_state.get("require_two_pass_scoring", True))
    scored_obj = filter_scored_obj(scored_obj_raw, require_two_pass=require_two_pass)
    scoring_idx = index_scored_cases(scored_obj) if scored_obj else {}

    cases = run_obj.get("cases") if isinstance(run_obj.get("cases"), list) else []
    case_ids = [str(c.get("case_id")) for c in cases if isinstance(c, dict) and c.get("case_id")]
    if not case_ids:
        st.info("Run has no cases[].")
        return
    selected_case_id = st.selectbox("Case", options=case_ids, index=0)
    case_obj = next((c for c in cases if isinstance(c, dict) and str(c.get("case_id")) == selected_case_id), None)
    if not isinstance(case_obj, dict):
        st.error("Could not find selected case.")
        return

    scored_case_obj = scoring_idx.get(selected_case_id) if scoring_idx else None
    render_flight_recorder(
        run_obj,
        case_obj,
        scored_case_obj,
        docs_root=docs_root,
        export_dir=export_dir,
        key_prefix=f"evolution_{run_id}_{selected_case_id}_",
    )


# -----------------------------
# App scaffold
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Make the top page selector sticky (exec-friendly).
    st.markdown(
        """
<style>
    .sticky-nav {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--background-color, white);
        padding: 0.4rem 0 0.2rem 0;
        margin: 0;
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        backdrop-filter: blur(6px);
    }

    /* Keep widget spacing tight inside the sticky bar */
    .sticky-nav [data-testid="stSegmentedControl"],
    .sticky-nav [data-testid="stRadio"] {
        margin-top: 0.2rem;
        margin-bottom: 0.0rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )

    # Ensure debug views (st.json/st.write) don't leak "blended".
    _orig_json = st.json

    def _json_sanitized(obj: Any, *args: Any, **kwargs: Any) -> Any:
        return _orig_json(_sanitize_for_ui(obj), *args, **kwargs)

    st.json = _json_sanitized  # type: ignore[assignment]

    _orig_write = st.write

    def _write_sanitized(*args: Any, **kwargs: Any) -> Any:
        args2 = tuple(_sanitize_for_ui(a) for a in args)
        return _orig_write(*args2, **kwargs)

    st.write = _write_sanitized  # type: ignore[assignment]

    PAGES = [
        "Chat Playground",
        "Run Explorer",
        "Evolution",
        "History",
        "Experiment Comparison",
        "Diagnostics",
        "Executive Insights",
    ]

    # Top navigation: segmented control feels more like a navbar than radio buttons.
    default_page = st.session_state.get("nav_page") or PAGES[1]
    st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        page = st.segmented_control("Page", options=PAGES, default=default_page)
    else:  # Back-compat for older Streamlit
        page = st.radio(
            "Page",
            options=PAGES,
            index=max(0, PAGES.index(default_page)) if default_page in PAGES else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)
    if not page:
        page = PAGES[1]
    st.session_state["nav_page"] = page

    # Sidebar configuration.
    st.sidebar.header("Config")
    default_root = Path.cwd()

    with st.sidebar.expander("Paths", expanded=False):
        project_root = Path(
            st.text_input("Project root", value=str(default_root))
        ).expanduser()

        runs_dir = Path(st.text_input("Runs dir", value=str(project_root / "experiments" / "runs"))).expanduser()
        scored_dir = Path(
            st.text_input("Scored runs dir", value=str(project_root / "experiments" / "scored_runs_two_pass"))
        ).expanduser()
        docs_root = Path(
            st.text_input("Normalized docs root", value=str(project_root))
        ).expanduser()

    # Live chat is always enabled; controls are shown only on the Chat page.
    enable_live_chat = True

    diagnostics_mode = "Per-case"
    if page == "Diagnostics":
        diagnostics_mode = st.sidebar.radio("Diagnostics view", options=["Per-run", "Per-case"], index=1)

    require_two_pass_scoring = st.sidebar.toggle(
        "Require two-pass scores",
        value=True,
        help="When enabled, the dashboard ignores scored files unless scoring_meta.judge_mode == 'two_pass'.",
    )
    st.session_state["require_two_pass_scoring"] = bool(require_two_pass_scoring)

    curated_story_mode = st.sidebar.toggle(
        "Curated story mode",
        value=True,
        help="Exec-friendly defaults: auto-pick a small representative set of scored runs.",
    )
    st.session_state["curated_story_mode"] = bool(curated_story_mode)

    if page == "Chat Playground":
        st.sidebar.subheader("Live Chat")
        st.sidebar.caption("Uses your local Chroma indexes + OpenAI; requires OPENAI_API_KEY.")

        show_advanced_indexes = st.sidebar.checkbox(
            "Show advanced indexes",
            value=False,
            help="Includes experimental scene-based chunking indexes (not recommended for new demos).",
        )

        # Discover options at runtime.
        db_root = str(project_root / "db")
        persist_dir_options = list_chroma_persist_dirs(db_root, include_advanced=bool(show_advanced_indexes))
        # Prefer common defaults when present.
        script_default = str(project_root / "db" / "chroma_db")
        derived_default = str(project_root / "db" / "chroma_db_derived_cards")

        # Models: discovered from historical run logs (plus a couple sane defaults).
        discovered_models = list_llm_models_from_runs(str(project_root / "experiments" / "runs"))
        model_options = []
        for m in (discovered_models + ["gpt-4.1-mini", "gpt-4.1-nano"]):
            if m and m not in model_options:
                model_options.append(m)

        retrieval_policy = st.sidebar.selectbox(
            "Retrieval policy",
            options=["script_only", "derived_only", "blended"],
            index=_select_index(["script_only", "derived_only", "blended"], st.session_state.get("live_retrieval_policy")),
            format_func=lambda p: {
                "script_only": "Script-only",
                "derived_only": "Derived-only",
                "blended": "Hybrid (scripts + derived routing)",
            }.get(str(p), str(p)),
        )
        st.session_state["live_retrieval_policy"] = retrieval_policy

        uses_script = retrieval_policy in {"script_only", "blended", "hybrid"}
        uses_derived = retrieval_policy in {"derived_only", "blended", "hybrid"}

        if uses_script:
            # Script persist dir dropdown.
            script_persist_dir_options = (persist_dir_options if persist_dir_options else [script_default])
            script_persist_current = st.session_state.get("live_script_persist_dir") or script_default
            script_persist_dir = st.sidebar.selectbox(
                "Index",
                options=script_persist_dir_options,
                index=_select_index(script_persist_dir_options, script_persist_current),
                format_func=lambda p: friendly_index_label(str(p), kind="script"),
                help=(
                    "Select which script index to use for retrieval. "
                    "The metadata index enables stronger filtering/routing (episode_id)."
                ),
            )
            st.session_state["live_script_persist_dir"] = script_persist_dir
            # Remove script collection selection: default collection only.
            st.session_state["live_script_collection_name"] = None
        else:
            st.sidebar.caption("Script index controls hidden (derived_only).")

        if uses_derived:
            # Derived persist dir dropdown.
            derived_persist_dir_options = (persist_dir_options if persist_dir_options else [derived_default])
            derived_persist_current = st.session_state.get("live_derived_persist_dir") or derived_default
            derived_persist_dir = st.sidebar.selectbox(
                "Derived index version",
                options=derived_persist_dir_options,
                index=_select_index(derived_persist_dir_options, derived_persist_current),
                format_func=lambda p: friendly_index_label(str(p), kind="derived"),
                help="Select which derived-cards index to use for routing/rollups.",
            )
            st.session_state["live_derived_persist_dir"] = derived_persist_dir

            # Derived collection dropdown.
            derived_collections = list_chroma_collections(str(derived_persist_dir))
            derived_collection_options = ["derived_cards"]
            for c in derived_collections:
                if c not in derived_collection_options:
                    derived_collection_options.append(c)
            derived_collection_current = st.session_state.get("live_derived_collection_name") or "derived_cards"
            derived_collection_choice = st.sidebar.selectbox(
                "Derived collection",
                options=derived_collection_options,
                index=_select_index(derived_collection_options, str(derived_collection_current)),
            )
            st.session_state["live_derived_collection_name"] = derived_collection_choice
        else:
            st.sidebar.caption("Derived index controls hidden (script_only).")

        llm_model_choice = st.sidebar.selectbox(
            "LLM model",
            options=(model_options if model_options else ["gpt-4.1-mini"]),
            index=0,
        )
        st.session_state["live_llm_model"] = llm_model_choice

        st.session_state["live_temperature"] = st.sidebar.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
        )
        st.session_state["live_k"] = st.sidebar.number_input(
            "Top-k (final context)",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
        )
        if retrieval_policy in {"blended", "hybrid"}:
            st.session_state["live_derived_k"] = st.sidebar.number_input(
                "Top-k derived (hybrid)",
                min_value=1,
                max_value=40,
                value=int(st.session_state.get("live_derived_k") or 6),
                step=1,
            )

    exports_dir = project_root / "experiments" / "exports"
    st.sidebar.caption(f"Exports: {exports_dir}")

    # Optional gold answers.
    gold_path = str(project_root / "experiments" / "gold_answers.json")
    gold_by_case_id = load_gold_answers(gold_path)
    if not gold_by_case_id:
        st.sidebar.caption("Gold answers: not found or empty (coverage uses score proxy)")
    else:
        st.sidebar.caption(f"Gold answers loaded: {len(gold_by_case_id)} cases")

    # Routing: lazy page load.
    if page == "Chat Playground":
        page_chat_playground(enable_live_chat=enable_live_chat, docs_root=docs_root, export_dir=exports_dir)
    elif page == "Run Explorer":
        page_run_explorer(
            project_root=project_root,
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            docs_root=docs_root,
            diagnostics_mode=diagnostics_mode,
            export_dir=exports_dir,
        )
    elif page == "Evolution":
        page_evolution(
            project_root=project_root,
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            docs_root=docs_root,
            export_dir=exports_dir,
        )
    elif page == "History":
        page_history(
            project_root=project_root,
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            docs_root=docs_root,
            export_dir=exports_dir,
        )
    elif page == "Experiment Comparison":
        page_experiment_comparison(
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            gold_by_case_id=gold_by_case_id,
            docs_root=docs_root,
            export_dir=exports_dir,
        )
    elif page == "Diagnostics":
        page_diagnostics(
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            gold_by_case_id=gold_by_case_id,
            diagnostics_mode=diagnostics_mode,
            docs_root=docs_root,
            export_dir=exports_dir,
        )
    else:
        page_executive_insights(
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            gold_by_case_id=gold_by_case_id,
            export_dir=exports_dir,
        )


if __name__ == "__main__":
    main()