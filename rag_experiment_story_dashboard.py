"""RAG Experiment Story Dashboard (Streamlit).

Goal
- Tell the story of a RAG project evolution in <2 minutes.
- Curate runs into stages, pick representative runs, and summarize learnings.
- Keep raw run exploration in a Deep Debug page only.

Run logs live under:
- experiments/runs/*.json (schema: run, config, cases[], summary)

Scored logs live under (supported):
- experiments/scored_runs_two_pass/*.scored.json (preferred if present)
- experiments/scored_runs/*.scored.json

This file intentionally avoids non-standard deps beyond Streamlit + Plotly.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

try:  # Plotly preferred
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    px = None  # type: ignore
    go = None  # type: ignore
    _HAS_PLOTLY = False


APP_TITLE = "RAG Evaluation & Observability Dashboard"

# -----------------------------------------------------------------------------
# Regex / parsing helpers
# -----------------------------------------------------------------------------
EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)
EP_IN_SOURCE_RE = re.compile(r"s(\d{2})e(\d{2})", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc_iso(ts: Any) -> Optional[datetime]:
    s = str(ts or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        # Ensure tz-aware.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


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


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
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
    return t if len(t) <= n else (t[:n] + "…")


def normalize_episode_id(s: Any) -> Optional[str]:
    if not s:
        return None
    m = EP_RE.search(str(s).strip().upper())
    return m.group(0).upper() if m else None


def extract_episode_id_from_source(source: Any) -> Optional[str]:
    if not source:
        return None
    m = EP_IN_SOURCE_RE.search(str(source))
    if not m:
        return None
    return f"S{m.group(1)}E{m.group(2)}"


def extract_episode_ids_from_case(case: Dict[str, Any]) -> List[str]:
    # Prefer explicit diagnostics when present (run_eval v4+).
    eps = safe_get(case, "diagnostics.context_episode_ids", None)
    if isinstance(eps, list) and eps:
        out = [normalize_episode_id(x) for x in eps]
        return sorted({x for x in out if x})

    # Otherwise infer from retrieval results.
    results = safe_get(case, "retrieval.results", [])
    if not isinstance(results, list):
        return []

    out: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        eid = normalize_episode_id(r.get("episode_id")) or extract_episode_id_from_source(r.get("source"))
        if eid:
            out.add(eid)
    return sorted(out)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunFile:
    path: str
    obj: Dict[str, Any]


@dataclass(frozen=True)
class ScoredFile:
    run_id: str
    path: str
    obj: Dict[str, Any]


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return
    yield from sorted(root.rglob("*.json"))


@st.cache_data(show_spinner=False)
def load_runs(runs_dir: str) -> List[RunFile]:
    p = Path(runs_dir).expanduser()
    out: List[RunFile] = []
    if not p.exists() or not p.is_dir():
        return out

    for rf in sorted(p.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(obj, dict) and "run" in obj and "cases" in obj):
            continue
        out.append(RunFile(path=str(rf), obj=obj))
    return out


def _run_id_from_scored_filename(path: Path) -> Optional[str]:
    name = path.name
    if name.endswith(".scored.json"):
        return name[: -len(".scored.json")]
    return None


@st.cache_data(show_spinner=False)
def load_scored_runs(*, scored_dirs: Sequence[str], prefer_two_pass: bool = True) -> Dict[str, ScoredFile]:
    """Load scored runs into {run_id: ScoredFile}.

    If multiple scored dirs provide the same run_id, prefer earlier dirs.
    Callers should pass dirs in preference order.
    """
    out: Dict[str, ScoredFile] = {}

    dirs = [str(x) for x in scored_dirs if str(x).strip()]
    if not dirs:
        return out

    for d in dirs:
        p = Path(d).expanduser()
        if not p.exists() or not p.is_dir():
            continue
        for sf in sorted(p.glob("*.scored.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            run_id = _run_id_from_scored_filename(sf)
            if not run_id or run_id in out:
                continue
            try:
                obj = json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict) or "scored_cases" not in obj:
                continue

            # Optional: enforce two-pass scoring when available/desired.
            if prefer_two_pass:
                mode = safe_get(obj, "scoring_meta.judge_mode", None)
                if isinstance(mode, str) and mode.strip():
                    if mode.strip().lower() != "two_pass":
                        # Skip single-pass scored file when we are in two-pass mode.
                        continue

            out[run_id] = ScoredFile(run_id=run_id, path=str(sf), obj=obj)

    return out


@st.cache_data(show_spinner=False)
def load_gold_answers(gold_path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(gold_path).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
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


# -----------------------------------------------------------------------------
# Stage classification / technique flags
# -----------------------------------------------------------------------------
STAGE_ORDER = [
    "Baseline",
    "Increased Recall",
    "MMR / Diversity",
    "Scene Chunking",
    "Query Expansion + Fusion",
    "Metadata / Routing",
    "Derived Summaries / Cards",
    "Model Upgrades",
    "Other / Uncategorized",
]


def _run_created_at(run_obj: Dict[str, Any], *, fallback_path: Optional[str] = None) -> datetime:
    dt = _parse_utc_iso(safe_get(run_obj, "run.created_at_utc", ""))
    if dt is not None:
        return dt
    if fallback_path:
        try:
            return datetime.fromtimestamp(Path(fallback_path).stat().st_mtime, tz=timezone.utc)
        except Exception:
            pass
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def run_name(run_obj: Dict[str, Any], *, fallback_path: Optional[str] = None) -> str:
    name = str(safe_get(run_obj, "run.run_name", "") or "").strip()
    if name:
        return name
    if fallback_path:
        return Path(fallback_path).stem
    return "(missing)"


def _friendly_policy(policy: str) -> str:
    p = (policy or "").strip().lower()
    if not p or p in {"script_only", "script", "baseline"}:
        return "Script-only"
    if p in {"blended", "hybrid"}:
        return "Hybrid"
    if p == "derived_only":
        return "Derived-only"
    if p == "derived_then_script":
        return "Derived→Script"
    if p == "auto":
        return "Auto-route"
    return p.replace("_", " ").title()


def _friendly_search(run_obj: Dict[str, Any]) -> Optional[str]:
    search = _search_type(run_obj)
    if not search:
        return None
    if search == "similarity":
        return "sim"
    if search == "mmr":
        fetch_k = safe_int(safe_get(run_obj, "config.retrieval.fetch_k", None))
        lam = safe_float(safe_get(run_obj, "config.retrieval.lambda_mult", None))
        bits: List[str] = ["mmr"]
        if lam is not None:
            bits.append(f"λ={lam:.2f}")
        if fetch_k is not None:
            bits.append(f"fetch={fetch_k}")
        if len(bits) == 1:
            return "mmr"
        return "mmr(" + ",".join(bits[1:]) + ")"
    return search


def build_run_display_name(run_obj: Dict[str, Any], *, fallback_path: Optional[str] = None) -> str:
    """CEO-readable label summarizing key config knobs.

    Avoids opaque nicknames; falls back to run_name when config is missing.
    """

    policy = _retrieval_policy(run_obj)
    k = safe_int(safe_get(run_obj, "config.retrieval.k", None))
    qe = bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))

    parts: List[str] = []
    if policy:
        parts.append(f"Policy={_friendly_policy(policy)}")

    search = _friendly_search(run_obj)
    if search:
        parts.append(f"Search={search}")

    if k is not None:
        parts.append(f"k={k}")

    if qe:
        n = safe_int(safe_get(run_obj, "config.retrieval.query_expansion.n", None))
        model = str(safe_get(run_obj, "config.retrieval.query_expansion.model", "") or "").strip() or None
        kq = safe_int(safe_get(run_obj, "config.retrieval.query_expansion.k_per_query", None))
        fusion = str(safe_get(run_obj, "config.retrieval.query_expansion.fusion", "") or "").strip().lower() or None
        rrf_k0 = safe_int(safe_get(run_obj, "config.retrieval.query_expansion.rrf_k0", None))

        qe_bits: List[str] = ["on"]
        if n is not None:
            qe_bits.append(f"n={n}")
        if kq is not None:
            qe_bits.append(f"kq={kq}")
        if fusion:
            fusion_label = "RRF" if fusion == "rrf" else fusion.upper()
            if rrf_k0 is not None and fusion == "rrf":
                qe_bits.append(f"{fusion_label}(k0={rrf_k0})")
            else:
                qe_bits.append(fusion_label)
        if model:
            qe_bits.append(f"model={model}")
        parts.append("QE=" + ",".join(qe_bits))
    else:
        parts.append("QE=off")

    chunk_splitter = str(safe_get(run_obj, "config.chunking.splitter", "") or "").strip() or None
    chunk_size = safe_int(safe_get(run_obj, "config.chunking.chunk_size", None))
    chunk_overlap = safe_int(safe_get(run_obj, "config.chunking.chunk_overlap", None))
    if chunk_splitter and (chunk_size is not None or chunk_overlap is not None):
        bits = [chunk_splitter]
        if chunk_size is not None:
            bits.append(f"size={chunk_size}")
        if chunk_overlap is not None:
            bits.append(f"overlap={chunk_overlap}")
        parts.append("Chunks=" + ",".join(bits))

    script_idx = _script_persist_dir(run_obj)
    if script_idx:
        parts.append(f"ScriptIdx={Path(script_idx).name}")

    derived_idx = _derived_persist_dir(run_obj)
    if derived_idx and ("derived" in derived_idx.lower() or policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}):
        parts.append(f"DerivedIdx={Path(derived_idx).name}")

    model = llm_model(run_obj)
    if model:
        parts.append(f"LLM={model}")

    if parts:
        return " | ".join(parts)

    # Fall back gracefully for legacy/minimal logs.
    return run_name(run_obj, fallback_path=fallback_path)


def run_id(run_obj: Dict[str, Any], *, fallback_path: Optional[str] = None) -> str:
    rid = str(safe_get(run_obj, "run.run_id", "") or "").strip()
    if rid:
        return rid
    if fallback_path:
        return Path(fallback_path).stem
    return ""


def _script_persist_dir(run_obj: Dict[str, Any]) -> str:
    # Prefer provenance block (v4+).
    dv = safe_get(run_obj, "config.data_version.script.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(safe_get(run_obj, "config.vectorstore.persist_directory", "") or "")


def _derived_persist_dir(run_obj: Dict[str, Any]) -> str:
    dv = safe_get(run_obj, "config.data_version.derived.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(safe_get(run_obj, "config.derived_vectorstore.persist_directory", "") or "")


def _retrieval_policy(run_obj: Dict[str, Any]) -> str:
    return str(safe_get(run_obj, "config.retrieval.policy", "") or "").strip().lower()


def _search_type(run_obj: Dict[str, Any]) -> str:
    return str(safe_get(run_obj, "config.retrieval.search_type", "") or "").strip().lower()


def _query_expansion_enabled(run_obj: Dict[str, Any]) -> bool:
    return bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))


def _chunking_signature(run_obj: Dict[str, Any]) -> str:
    splitter = str(safe_get(run_obj, "config.chunking.splitter", "") or "")
    # Also infer from persist dir name when chunking isn't logged.
    persist = _script_persist_dir(run_obj)
    return f"{splitter}|{persist}".lower()


def llm_model(run_obj: Dict[str, Any]) -> Optional[str]:
    m = safe_get(run_obj, "config.llm.model", None)
    if isinstance(m, str) and m.strip():
        return m.strip()
    return None


def classify_stage(
    run_obj: Dict[str, Any],
    *,
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> str:
    """Heuristically assign a run to a stage bucket.

    This is intentionally opinionated and ordered. Earlier matches win.
    """
    name = run_name(run_obj).lower()
    policy = _retrieval_policy(run_obj)
    search = _search_type(run_obj)
    qe = _query_expansion_enabled(run_obj)
    persist = _script_persist_dir(run_obj).lower()
    derived_persist = _derived_persist_dir(run_obj).lower()
    chunk_sig = _chunking_signature(run_obj)

    k = safe_int(safe_get(run_obj, "config.retrieval.k", None))

    # Derived/cards routing.
    if policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"} or "derived" in derived_persist:
        return "Derived Summaries / Cards"

    # Metadata/routing (script index).
    if "chroma_db_meta" in persist or "metadata" in name or "routing" in name:
        return "Metadata / Routing"

    # Query expansion.
    if qe or "qe" in name or "query_expansion" in name or "rrf" in name or "fusion" in name:
        return "Query Expansion + Fusion"

    # Scene chunking.
    if "scene" in chunk_sig or "scene" in name:
        return "Scene Chunking"

    # MMR.
    if search == "mmr" or "mmr" in name:
        return "MMR / Diversity"

    # Increased recall (k).
    if baseline_k is not None and k is not None and k > baseline_k:
        return "Increased Recall"
    if k is not None and k >= 10:
        return "Increased Recall"

    # Model upgrades.
    m = llm_model(run_obj)
    if baseline_llm_model and m and m != baseline_llm_model:
        return "Model Upgrades"
    if "gpt-" in name and baseline_llm_model and baseline_llm_model.replace(".", "_") not in name:
        return "Model Upgrades"

    # Baseline.
    if "baseline" in name or policy in {"script_only", "script", "baseline", ""}:
        return "Baseline"

    return "Other / Uncategorized"


def technique_flags(run_obj: Dict[str, Any]) -> Dict[str, bool]:
    policy = _retrieval_policy(run_obj)
    search = _search_type(run_obj)
    qe = _query_expansion_enabled(run_obj)
    persist = _script_persist_dir(run_obj).lower()
    chunk_sig = _chunking_signature(run_obj)

    return {
        "qe": bool(qe),
        "mmr": bool(search == "mmr"),
        "scene_chunking": bool("scene" in chunk_sig),
        "metadata_index": bool("chroma_db_meta" in persist),
        "derived_or_routed": bool(policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}),
        "high_k": bool((safe_int(safe_get(run_obj, "config.retrieval.k", None)) or 0) >= 10),
    }


# -----------------------------------------------------------------------------
# Scoring / aggregation
# -----------------------------------------------------------------------------
@dataclass
class RunMetrics:
    run_id: str
    run_name: str
    display_name: str
    created_at: datetime
    run_file: str
    stage: str

    # score sources (prefer scored file)
    avg_overall: Optional[float]
    avg_correctness: Optional[float]
    avg_groundedness: Optional[float]
    avg_completeness: Optional[float]
    avg_hallucination: Optional[float]

    avg_total_tokens: Optional[float]

    # failure rates (heuristic)
    retrieval_failure_rate: Optional[float]
    grounding_failure_rate: Optional[float]
    aggregation_failure_rate: Optional[float]
    reasoning_failure_rate: Optional[float]

    # helpful pointers
    flags: Dict[str, bool]


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _case_judge_from_scored(scored_obj: Optional[Dict[str, Any]], case_id: str) -> Optional[Dict[str, Any]]:
    if not scored_obj or not isinstance(scored_obj, dict):
        return None
    rows = scored_obj.get("scored_cases")
    if not isinstance(rows, list):
        return None
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("case_id") or "").strip() != str(case_id or "").strip():
            continue
        j = r.get("judge")
        return j if isinstance(j, dict) else None
    return None


def _run_level_scores_from_scored(scored_obj: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not scored_obj or not isinstance(scored_obj, dict):
        return {
            "avg_overall": None,
            "avg_correctness": None,
            "avg_groundedness": None,
            "avg_completeness": None,
            "avg_hallucination": None,
        }

    rows = scored_obj.get("scored_cases")
    if not isinstance(rows, list):
        rows = []

    def _avg(key: str) -> Optional[float]:
        vals: List[Optional[float]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            j = r.get("judge")
            if not isinstance(j, dict):
                continue
            v = safe_float(j.get(key))
            if v is not None:
                vals.append(float(v))
        return _mean(vals)

    # overall is 0..100; others are 0..5.
    avg_overall = safe_float(safe_get(scored_obj, "score_summary.avg_overall", None))
    if avg_overall is None:
        avg_overall = _avg("overall")

    return {
        "avg_overall": avg_overall,
        "avg_correctness": _avg("correctness"),
        "avg_groundedness": _avg("groundedness"),
        "avg_completeness": _avg("completeness"),
        "avg_hallucination": _avg("hallucination"),
    }


def _avg_total_tokens(run_obj: Dict[str, Any]) -> Optional[float]:
    # Prefer run summary.
    v = safe_float(safe_get(run_obj, "summary.avg_total_tokens", None))
    if v is not None:
        return float(v)

    # Fall back to per-case usage.
    cases = run_obj.get("cases")
    if not isinstance(cases, list) or not cases:
        return None

    vals: List[Optional[float]] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        usage = safe_get(c, "answer.usage", {})
        if not isinstance(usage, dict):
            continue
        t = safe_float(usage.get("total_tokens"))
        if t is not None:
            vals.append(float(t))
    return _mean(vals)


def question_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if not q:
        return "(missing)"

    if re.search(r"\b(which|what)\s+episode\b|\bin\s+which\s+episode\b|\bepisode\s+is\b", q):
        return "Episode lookup"
    if re.search(r"\bquote\b|\bexact\s+quote\b|\bwhat\s+did\b.+\bsay\b", q):
        return "Quote / line"
    if re.search(r"\b(list|summari[sz]e|overview|timeline|chronolog|across|throughout|all\b|compare)\b", q):
        return "Aggregation"
    if re.search(r"\bwhy\b|\bhow\b", q):
        return "Explanation"
    return "Factual"


def compute_retrieval_coverage(
    *,
    case: Dict[str, Any],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[float], str]:
    """Return (coverage, method).

    coverage:
    - 1.0 if expected episode appears in retrieved context
    - 0.0 if expected exists but not retrieved
    - else: a proxy score (max retrieval score if present)
    """
    cid = str(case.get("case_id") or "").strip()
    gold = gold_by_case_id.get(cid) if cid else None
    expected = (gold or {}).get("expected_episode_ids") if isinstance(gold, dict) else None

    if isinstance(expected, list) and expected:
        expected_set = {normalize_episode_id(x) for x in expected}
        expected_set = {x for x in expected_set if x}
        if not expected_set:
            return None, "gold_expected_missing"

        retrieved_eps = set(extract_episode_ids_from_case(case))
        hit = bool(retrieved_eps & expected_set)
        return (1.0 if hit else 0.0), "gold_episode_match"

    # Proxy: max retrieval score.
    results = safe_get(case, "retrieval.results", [])
    if isinstance(results, list):
        scores: List[float] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            s = safe_float(r.get("score"))
            if s is not None:
                scores.append(float(s))
        if scores:
            return max(scores), "score_proxy"

    return None, "missing"


def classify_failure_mode(
    *,
    case: Dict[str, Any],
    judge: Optional[Dict[str, Any]],
    coverage: Optional[float],
) -> str:
    """Return one of: retrieval_miss | grounding | aggregation | reasoning | ok | unknown."""

    # Prefer explicit labels if present.
    labels = safe_get(case, "labels", {})
    if isinstance(labels, dict):
        if labels.get("retrieval_failure") is True:
            return "retrieval_miss"
        if labels.get("grounding_failure") is True:
            return "grounding"

    qtype = question_type(str(case.get("question") or ""))
    cov = safe_float(coverage)

    if cov is not None and cov < 0.5:
        return "retrieval_miss"

    # Two-pass judge fields: correctness/groundedness/completeness/hallucination.
    if isinstance(judge, dict):
        corr = safe_int(judge.get("correctness"))
        g = safe_int(judge.get("groundedness"))
        comp = safe_int(judge.get("completeness"))

        if qtype == "Aggregation" and comp is not None and comp <= 2:
            return "aggregation"
        if g is not None and g <= 2:
            return "grounding"
        if corr is not None and corr <= 2:
            return "reasoning"

        overall = safe_int(judge.get("overall"))
        if overall is not None and overall >= 80:
            return "ok"
        if overall is not None and overall <= 40:
            # Fall back to a likely bucket.
            if qtype == "Aggregation":
                return "aggregation"
            return "reasoning"

    return "unknown"


def compute_run_metrics(
    *,
    run: RunFile,
    scored: Optional[ScoredFile],
    stage: str,
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> RunMetrics:
    run_obj = run.obj
    rid = run_id(run_obj, fallback_path=run.path)
    rname = run_name(run_obj, fallback_path=run.path)
    display_name = build_run_display_name(run_obj, fallback_path=run.path)
    created = _run_created_at(run_obj, fallback_path=run.path)

    scored_obj = scored.obj if scored else None

    scores = _run_level_scores_from_scored(scored_obj)

    # Failure rates (case-level)
    cases = run_obj.get("cases")
    if not isinstance(cases, list):
        cases = []

    retrieval_fail = grounding_fail = aggregation_fail = reasoning_fail = 0
    n = 0

    for c in cases:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id") or "").strip()
        if not cid:
            continue
        n += 1
        cov, _method = compute_retrieval_coverage(case=c, gold_by_case_id=gold_by_case_id)
        judge = _case_judge_from_scored(scored_obj, cid)
        mode = classify_failure_mode(case=c, judge=judge, coverage=cov)
        if mode == "retrieval_miss":
            retrieval_fail += 1
        elif mode == "grounding":
            grounding_fail += 1
        elif mode == "aggregation":
            aggregation_fail += 1
        elif mode == "reasoning":
            reasoning_fail += 1

    def _rate(x: int) -> Optional[float]:
        if n <= 0:
            return None
        return float(x / n)

    return RunMetrics(
        run_id=rid,
        run_name=rname,
        display_name=display_name,
        created_at=created,
        run_file=run.path,
        stage=stage,
        avg_overall=scores["avg_overall"],
        avg_correctness=scores["avg_correctness"],
        avg_groundedness=scores["avg_groundedness"],
        avg_completeness=scores["avg_completeness"],
        avg_hallucination=scores["avg_hallucination"],
        avg_total_tokens=_avg_total_tokens(run_obj),
        retrieval_failure_rate=_rate(retrieval_fail),
        grounding_failure_rate=_rate(grounding_fail),
        aggregation_failure_rate=_rate(aggregation_fail),
        reasoning_failure_rate=_rate(reasoning_fail),
        flags=technique_flags(run_obj),
    )


def choose_representative_runs(runs: List[RunMetrics]) -> Dict[str, Optional[RunMetrics]]:
    """Return {best, worst, representative} for a stage."""
    if not runs:
        return {"best": None, "worst": None, "representative": None}

    # Prefer scored runs; if score missing, treat as -1.
    def _score(r: RunMetrics) -> float:
        return float(r.avg_overall) if r.avg_overall is not None else -1.0

    best = max(runs, key=_score)
    worst = min(runs, key=_score)

    scored = [r for r in runs if r.avg_overall is not None]
    if scored:
        med = float(median([float(r.avg_overall) for r in scored if r.avg_overall is not None]))
        representative = min(scored, key=lambda r: abs(float(r.avg_overall or 0.0) - med))
    else:
        representative = min(runs, key=lambda r: r.created_at)

    return {"best": best, "worst": worst, "representative": representative}


def compute_stage_metrics(stage: str, stage_runs: List[RunMetrics]) -> Dict[str, Any]:
    reps = choose_representative_runs(stage_runs)
    mean_score = _mean([r.avg_overall for r in stage_runs])
    mean_tokens = _mean([r.avg_total_tokens for r in stage_runs])
    return {
        "stage": stage,
        "runs": len(stage_runs),
        "mean_score": mean_score,
        "mean_tokens": mean_tokens,
        "best": reps["best"],
        "worst": reps["worst"],
        "representative": reps["representative"],
    }


# -----------------------------------------------------------------------------
# Curation / insights generation
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class StoryStage:
    key: str
    title: str
    what_changed: str
    why_tried: str


STORY_STAGE_CARDS: List[StoryStage] = [
    StoryStage(
        key="Baseline",
        title="Baseline",
        what_changed="Script-only similarity retrieval, small k, no special tricks.",
        why_tried="Establish a defensible baseline and failure map.",
    ),
    StoryStage(
        key="Increased Recall",
        title="Increase Recall (k)",
        what_changed="Pulled more context by increasing top-k.",
        why_tried="Test whether misses are primarily recall failures.",
    ),
    StoryStage(
        key="MMR / Diversity",
        title="MMR / Diversity",
        what_changed="Diversified retrieved chunks via MMR to reduce redundancy.",
        why_tried="Reduce repeated evidence and broaden coverage across episodes/scenes.",
    ),
    StoryStage(
        key="Scene Chunking",
        title="Scene Chunking",
        what_changed="Changed chunking strategy away from plain character splits.",
        why_tried="Preserve narrative coherence and improve multi-turn/long-form grounding.",
    ),
    StoryStage(
        key="Query Expansion + Fusion",
        title="Query Expansion + Fusion",
        what_changed="Generated alternate queries and fused results (RRF).",
        why_tried="Boost recall for paraphrases and aggregation-style questions.",
    ),
    StoryStage(
        key="Metadata / Routing",
        title="Metadata / Routing",
        what_changed="Used metadata-rich indexing and tighter filters/routing.",
        why_tried="Make retrieval controllable and auditable, not just best-effort similarity.",
    ),
    StoryStage(
        key="Derived Summaries / Cards",
        title="Derived Summaries / Cards",
        what_changed="Used derived cards/rollups to route and support aggregation questions.",
        why_tried="Improve multi-episode aggregation and reduce context fragmentation.",
    ),
    StoryStage(
        key="Model Upgrades",
        title="Model Upgrades",
        what_changed="Upgraded answer/judge models.",
        why_tried="Separate ‘retrieval is good enough’ from ‘reasoning/formatting is limiting’.",
    ),
    StoryStage(
        key="Other / Uncategorized",
        title="Other",
        what_changed="Runs that don’t fit the main story buckets.",
        why_tried="Keep everything accessible without polluting the narrative.",
    ),
]


def _stage_sort_key(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except Exception:
        return 999


def biggest_lessons(stage_rows: List[Dict[str, Any]]) -> List[str]:
    # Extract simple delta narratives between stages.
    lessons: List[str] = []

    # Build ordered points.
    pts: List[Tuple[str, Optional[float], Optional[float]]] = []
    for row in sorted(stage_rows, key=lambda r: _stage_sort_key(str(r.get("stage")))):
        pts.append((str(row.get("stage")), safe_float(row.get("mean_score")), safe_float(row.get("mean_tokens"))))

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(b - a)

    # Baseline comparison.
    baseline = next((p for p in pts if p[0] == "Baseline"), None)
    if baseline and baseline[1] is not None:
        base_s = baseline[1]
        best = max([p for p in pts if p[1] is not None], key=lambda p: float(p[1] or -1.0), default=None)
        if best and best[0] != "Baseline":
            d = _delta(base_s, best[1])
            if d is not None:
                lessons.append(f"Best stage (‘{best[0]}’) improved avg score by {d:+.0f} vs baseline.")

    # Biggest positive jump between consecutive stages.
    best_jump = None
    worst_jump = None
    for i in range(1, len(pts)):
        prev = pts[i - 1]
        cur = pts[i]
        d = _delta(prev[1], cur[1])
        if d is None:
            continue
        if best_jump is None or d > best_jump[0]:
            best_jump = (d, prev[0], cur[0])
        if worst_jump is None or d < worst_jump[0]:
            worst_jump = (d, prev[0], cur[0])

    if best_jump and best_jump[0] >= 2:
        lessons.append(f"Biggest jump: {best_jump[1]} → {best_jump[2]} ({best_jump[0]:+.0f} avg score).")
    if worst_jump and worst_jump[0] <= -2:
        lessons.append(f"Biggest regression: {worst_jump[1]} → {worst_jump[2]} ({worst_jump[0]:+.0f} avg score).")

    # Token trade-off.
    token_pts = [(s, sc, tk) for (s, sc, tk) in pts if sc is not None and tk is not None]
    if token_pts:
        hi_cost = max(token_pts, key=lambda p: float(p[2] or 0.0))
        lo_cost = min(token_pts, key=lambda p: float(p[2] or 0.0))
        lessons.append(f"Cost varied widely: ‘{lo_cost[0]}’ is lowest tokens; ‘{hi_cost[0]}’ is highest tokens.")

    # Keep tight.
    uniq: List[str] = []
    seen: set[str] = set()
    for l in lessons:
        if l in seen:
            continue
        seen.add(l)
        uniq.append(l)
        if len(uniq) >= 5:
            break
    return uniq


def recommended_next_steps(
    *,
    best_run: Optional[RunMetrics],
    stage_rows: List[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []

    # If best run already uses QE/MMR/derived, recommend remaining levers.
    flags = (best_run.flags if best_run else {})

    if not flags.get("metadata_index"):
        out.append("Standardize on a metadata-rich script index (episode_id filters + auditability).")
    if not flags.get("qe"):
        out.append("Try query expansion + RRF for broad recall improvements.")
    if not flags.get("mmr"):
        out.append("Try MMR for redundancy reduction and broader evidence coverage.")

    out.append("Add more gold + question taxonomy to make failure modes measurable.")
    out.append("Improve citation + quote extraction to harden groundedness.")

    # If derived stage exists but isn’t best, recommend hybrid tuning.
    derived = next((r for r in stage_rows if str(r.get("stage")) == "Derived Summaries / Cards"), None)
    if derived and safe_float(derived.get("mean_score")) is not None:
        out.append("Tune hybrid routing (derived→script) to balance rollups with exact quotes.")

    # Keep concise.
    return out[:6]


# -----------------------------------------------------------------------------
# Case study / before-after selection
# -----------------------------------------------------------------------------
@dataclass
class CaseView:
    run: RunMetrics
    stage: str
    case_id: str
    question: str
    answer_text: str
    judge: Optional[Dict[str, Any]]
    coverage: Optional[float]
    coverage_method: str
    episode_ids: List[str]
    sources: List[str]
    previews: List[str]


def _case_from_run(run_obj: Dict[str, Any], case_id: str) -> Optional[Dict[str, Any]]:
    cases = run_obj.get("cases")
    if not isinstance(cases, list):
        return None
    for c in cases:
        if not isinstance(c, dict):
            continue
        if str(c.get("case_id") or "").strip() == str(case_id).strip():
            return c
    return None


def _case_sources_and_previews(case_obj: Dict[str, Any], *, limit: int = 6) -> Tuple[List[str], List[str]]:
    results = safe_get(case_obj, "retrieval.results", [])
    if not isinstance(results, list):
        return [], []

    sources: List[str] = []
    previews: List[str] = []
    for r in results[: int(limit)]:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source") or "")
        prev = str(r.get("preview") or "")
        if src:
            sources.append(src)
        if prev:
            previews.append(prev)
    return sources, previews


def build_case_view(
    *,
    run: RunFile,
    metrics: RunMetrics,
    scored: Optional[ScoredFile],
    case_id: str,
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> Optional[CaseView]:
    case_obj = _case_from_run(run.obj, case_id)
    if not isinstance(case_obj, dict):
        return None

    q = str(case_obj.get("question") or "")
    ans = str(safe_get(case_obj, "answer.text", "") or "")

    judge = _case_judge_from_scored(scored.obj if scored else None, case_id)
    cov, cov_method = compute_retrieval_coverage(case=case_obj, gold_by_case_id=gold_by_case_id)

    episode_ids = extract_episode_ids_from_case(case_obj)
    sources, previews = _case_sources_and_previews(case_obj)

    return CaseView(
        run=metrics,
        stage=metrics.stage,
        case_id=case_id,
        question=q,
        answer_text=ans,
        judge=judge,
        coverage=cov,
        coverage_method=cov_method,
        episode_ids=episode_ids,
        sources=sources,
        previews=previews,
    )


def _select_baseline_run(metrics: List[RunMetrics]) -> Optional[RunMetrics]:
    baselines = [m for m in metrics if m.stage == "Baseline" and m.avg_overall is not None]
    if baselines:
        # Representative baseline: median baseline.
        med = float(median([float(b.avg_overall) for b in baselines if b.avg_overall is not None]))
        return min(baselines, key=lambda b: abs(float(b.avg_overall or 0.0) - med))

    # Fall back: earliest baseline (even if unscored).
    baselines2 = [m for m in metrics if m.stage == "Baseline"]
    if baselines2:
        return min(baselines2, key=lambda m: m.created_at)

    # Otherwise earliest run.
    if metrics:
        return min(metrics, key=lambda m: m.created_at)
    return None


def _select_best_run(metrics: List[RunMetrics]) -> Optional[RunMetrics]:
    scored = [m for m in metrics if m.avg_overall is not None]
    if scored:
        return max(scored, key=lambda m: float(m.avg_overall or -1.0))
    return None


def _select_regression_run(metrics: List[RunMetrics]) -> Optional[RunMetrics]:
    # Pick a scored run with notably low correctness but decent coverage proxies.
    scored = [m for m in metrics if m.avg_overall is not None]
    if not scored:
        return None

    # Heuristic: choose low avg_overall among high token cost (overstuffed context) → likely regression.
    def _key(m: RunMetrics) -> Tuple[float, float]:
        tokens = float(m.avg_total_tokens or 0.0)
        score = float(m.avg_overall or 0.0)
        return (score, -tokens)

    return min(scored, key=_key)


def _select_intermediate_run(metrics: List[RunMetrics]) -> Optional[RunMetrics]:
    scored = [m for m in metrics if m.avg_overall is not None]
    if len(scored) < 3:
        return None
    scored_sorted = sorted(scored, key=lambda m: float(m.avg_overall or -1.0))
    return scored_sorted[len(scored_sorted) // 2]


# -----------------------------------------------------------------------------
# Plotly helpers
# -----------------------------------------------------------------------------
def _plotly_or_warning() -> bool:
    if _HAS_PLOTLY:
        return True
    st.warning("Plotly is not installed; some charts are disabled.")
    return False


def plot_stage_score_line(stage_rows: List[Dict[str, Any]]) -> None:
    rows = [
        {
            "stage": r["stage"],
            "mean_score": safe_float(r.get("mean_score")),
        }
        for r in stage_rows
    ]
    rows = [r for r in rows if r["mean_score"] is not None]
    rows.sort(key=lambda r: _stage_sort_key(str(r["stage"])))
    if not rows:
        st.info("No stage scores available yet (missing scored runs).")
        return

    if not _plotly_or_warning():
        st.write(rows)
        return

    fig = px.line(
        rows,
        x="stage",
        y="mean_score",
        markers=True,
        title="Avg judge score by stage",
    )
    fig.update_layout(xaxis_title="Stage", yaxis_title="Avg overall score (0..100)")
    st.plotly_chart(fig, use_container_width=True)


def plot_failure_pie(failure_counts: Dict[str, int]) -> None:
    rows = [{"failure_mode": k, "count": int(v)} for k, v in failure_counts.items() if int(v) > 0]
    if not rows:
        st.info("No failure labels available.")
        return

    if not _plotly_or_warning():
        st.write(rows)
        return

    fig = px.pie(rows, names="failure_mode", values="count", title="Failure mode mix")
    st.plotly_chart(fig, use_container_width=True)


def plot_cost_vs_quality(points: List[Dict[str, Any]]) -> None:
    pts = [p for p in points if p.get("avg_overall") is not None and p.get("avg_total_tokens") is not None]
    if not pts:
        st.info("No cost-vs-quality points available (missing tokens or scores).")
        return

    if not _plotly_or_warning():
        st.write(pts[:20])
        return

    fig = px.scatter(
        pts,
        x="avg_total_tokens",
        y="avg_overall",
        color="stage",
        hover_data=["display_name"],
        title="Cost vs quality (runs)",
    )
    fig.update_layout(xaxis_title="Avg total tokens (proxy)", yaxis_title="Avg overall score")
    st.plotly_chart(fig, use_container_width=True)


def plot_impact_bars(rows: List[Dict[str, Any]]) -> None:
    rows = [r for r in rows if r.get("delta") is not None]
    if not rows:
        st.info("Not enough scored runs to compute impact.")
        return

    rows.sort(key=lambda r: float(r.get("delta") or 0.0), reverse=True)

    if not _plotly_or_warning():
        st.write(rows)
        return

    fig = px.bar(
        rows,
        x="delta",
        y="technique",
        orientation="h",
        title="Technique impact (enabled vs disabled)",
        hover_data=["enabled_mean", "disabled_mean", "enabled_n", "disabled_n"],
    )
    fig.update_layout(xaxis_title="Δ avg overall (enabled - disabled)", yaxis_title="Technique")
    st.plotly_chart(fig, use_container_width=True)


def plot_failure_by_qtype_stacked(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No per-case failures to plot.")
        return

    if not _plotly_or_warning():
        st.write(rows[:20])
        return

    fig = px.bar(
        rows,
        x="question_type",
        y="count",
        color="failure_mode",
        title="Failure modes by question type",
    )
    fig.update_layout(xaxis_title="Question type", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def plot_coverage_scatter(rows: List[Dict[str, Any]]) -> None:
    pts = [r for r in rows if r.get("coverage") is not None and r.get("overall") is not None]
    if not pts:
        st.info("Coverage scatter unavailable (need coverage and scores).")
        return

    if not _plotly_or_warning():
        st.write(pts[:20])
        return

    fig = px.scatter(
        pts,
        x="coverage",
        y="overall",
        color="failure_mode",
        hover_data=["case_id", "display_name", "stage"],
        title="Retrieval coverage vs answer score",
    )
    fig.update_layout(xaxis_title="Retrieval coverage", yaxis_title="Judge overall (0..100)")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
def badge_from_delta(delta: Optional[float]) -> Tuple[str, str]:
    if delta is None:
        return "⚠ Mixed", "No comparable score"
    if delta >= 2.0:
        return "✅ Win", f"Improved by {delta:+.0f}"
    if delta <= -2.0:
        return "❌ Regression", f"Dropped by {delta:+.0f}"
    return "⚠ Mixed", f"Changed by {delta:+.0f}"


def format_config_knobs(run_obj: Dict[str, Any]) -> str:
    pol_raw = _retrieval_policy(run_obj)
    pol = _friendly_policy(pol_raw) if pol_raw else ""
    search = _friendly_search(run_obj)
    k = safe_int(safe_get(run_obj, "config.retrieval.k", None))
    qe = bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))
    chunk = str(safe_get(run_obj, "config.chunking.splitter", "") or "")
    persist = str(_script_persist_dir(run_obj) or "")
    derived_persist = str(_derived_persist_dir(run_obj) or "")

    bits = []
    if pol:
        bits.append(f"policy={pol}")
    if search:
        bits.append(f"search={search}")
    if k is not None:
        bits.append(f"k={k}")
    bits.append(f"qe={'on' if qe else 'off'}")
    if chunk:
        bits.append(f"chunk={chunk}")
    if persist:
        bits.append(f"index={Path(persist).name}")
    if derived_persist:
        bits.append(f"derived={Path(derived_persist).name}")
    return ", ".join(bits)


# -----------------------------------------------------------------------------
# Page implementations
# -----------------------------------------------------------------------------
def render_summary_page(
    *,
    runs_by_id: Dict[str, RunFile],
    metrics: List[RunMetrics],
    stage_rows: List[Dict[str, Any]],
    failure_counts: Dict[str, int],
) -> None:
    st.title(APP_TITLE)

    st.markdown(
        """
This project used RAG as a sandbox to practice real AI engineering:
- evaluation
- monitoring
- verification
- failure analysis
- measurable iteration
""".strip()
    )

    best = _select_best_run(metrics)
    baseline = _select_baseline_run(metrics)

    st.subheader("Best result so far")
    cols = st.columns(4)
    if best is None:
        cols[0].metric("Best run", "(none)")
        cols[1].metric("Avg overall", "—")
    else:
        cols[0].metric("Best run", best.display_name)
        cols[1].metric("Avg overall", f"{float(best.avg_overall or 0.0):.0f}")
        cols[2].metric("Stage", best.stage)
        cols[3].metric("Avg tokens", "—" if best.avg_total_tokens is None else f"{best.avg_total_tokens:.0f}")

        with st.expander("Key config knobs", expanded=False):
            ro = runs_by_id.get(best.run_id)
            if ro is not None:
                st.code(format_config_knobs(ro.obj), language="text")

            st.caption("Why it worked (auto): high-level signal from stage + technique flags")
            flags = best.flags
            reasons: List[str] = []
            if flags.get("qe"):
                reasons.append("Query expansion likely improved recall for paraphrases.")
            if flags.get("mmr"):
                reasons.append("MMR likely reduced redundant chunks.")
            if flags.get("metadata_index"):
                reasons.append("Metadata index enabled tighter filtering and auditability.")
            if flags.get("derived_or_routed"):
                reasons.append("Derived routing likely improved multi-episode aggregation context.")
            if not reasons:
                reasons.append("Likely benefited from better model / configuration compared to baseline.")
            for r in reasons[:5]:
                st.write("-", r)

    st.subheader("Biggest lessons")
    lessons = biggest_lessons(stage_rows)
    if lessons:
        for l in lessons:
            st.write("-", l)
    else:
        st.info("Not enough scored runs to generate lessons yet.")

    st.subheader("High-signal charts")
    st.caption("Only three charts on purpose: progression, failures, cost tradeoffs.")

    plot_stage_score_line(stage_rows)
    st.caption("Interpretation: stage means show which engineering moves mattered most.")

    plot_failure_pie(failure_counts)
    st.caption("Interpretation: the largest slice is the highest leverage failure mode to tackle next.")

    plot_cost_vs_quality(
        [
            {
                "display_name": m.display_name,
                "stage": m.stage,
                "avg_overall": m.avg_overall,
                "avg_total_tokens": m.avg_total_tokens,
            }
            for m in metrics
        ]
    )
    st.caption("Interpretation: Pareto runs sit top-left (high score, low cost).")

    st.subheader("Recommended next steps")
    for s in recommended_next_steps(best_run=best, stage_rows=stage_rows):
        st.write("-", s)


def render_timeline_page(
    *,
    stage_rows: List[Dict[str, Any]],
) -> None:
    st.title("Experiment timeline")
    st.caption("A curated, step-by-step narrative of what changed, why, and what happened.")

    stage_map = {str(r.get("stage")): r for r in stage_rows}

    prev_score: Optional[float] = None
    for card in STORY_STAGE_CARDS:
        row = stage_map.get(card.key)
        if row is None:
            # Hide empty stages by default; keep discoverable.
            continue

        mean_score = safe_float(row.get("mean_score"))
        delta = (mean_score - prev_score) if (mean_score is not None and prev_score is not None) else None
        verdict, verdict_note = badge_from_delta(delta)

        st.subheader(f"{card.title} — {verdict}")
        st.write(card.what_changed)
        st.write("Why:", card.why_tried)

        cols = st.columns(4)
        cols[0].metric("Runs", int(row.get("runs") or 0))
        cols[1].metric("Avg score", "—" if mean_score is None else f"{mean_score:.0f}")
        cols[2].metric("Δ vs prev", "—" if delta is None else f"{delta:+.0f}")
        cols[3].metric("Takeaway", verdict_note)

        # Representative runs.
        rep = row.get("representative")
        best = row.get("best")
        worst = row.get("worst")

        with st.expander("Representative runs", expanded=False):
            def _show(label: str, m: Optional[RunMetrics]) -> None:
                if not m:
                    return
                st.write(f"- {label}: {m.display_name} | score={('—' if m.avg_overall is None else f'{m.avg_overall:.0f}')} | tokens={('—' if m.avg_total_tokens is None else f'{m.avg_total_tokens:.0f}')}")

            _show("Representative", rep)
            _show("Best", best)
            _show("Worst", worst)

        # Compact visualization: bar with best/rep/worst.
        rows = []
        for label, m in [("Worst", worst), ("Rep", rep), ("Best", best)]:
            if m and m.avg_overall is not None:
                rows.append({"kind": label, "score": float(m.avg_overall)})
        if rows and _HAS_PLOTLY:
            fig = px.bar(rows, x="kind", y="score", title="Representative scores")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        elif rows:
            st.write(rows)

        st.caption("One sentence takeaway: " + ("Win" if verdict.startswith("✅") else "Regression" if verdict.startswith("❌") else "Mixed") + ".")
        st.divider()

        if mean_score is not None:
            prev_score = mean_score


def render_helped_hurt_page(
    *,
    metrics: List[RunMetrics],
) -> None:
    st.title("What helped / what hurt")
    st.caption("Which engineering moves moved quality, and what traded off against cost.")

    # 1) Impact chart: technique on/off.
    st.subheader("Impact of techniques")

    techniques = [
        ("qe", "Query expansion + fusion"),
        ("mmr", "MMR retrieval"),
        ("scene_chunking", "Scene chunking"),
        ("metadata_index", "Metadata index"),
        ("derived_or_routed", "Derived routing/cards"),
        ("high_k", "High k (>=10)"),
    ]

    scored = [m for m in metrics if m.avg_overall is not None]
    impact_rows: List[Dict[str, Any]] = []
    for key, label in techniques:
        enabled = [m for m in scored if m.flags.get(key)]
        disabled = [m for m in scored if not m.flags.get(key)]
        em = _mean([m.avg_overall for m in enabled])
        dm = _mean([m.avg_overall for m in disabled])
        delta = (em - dm) if (em is not None and dm is not None) else None
        impact_rows.append(
            {
                "technique": label,
                "enabled_mean": em,
                "disabled_mean": dm,
                "delta": delta,
                "enabled_n": len(enabled),
                "disabled_n": len(disabled),
            }
        )

    plot_impact_bars(impact_rows)
    st.caption("Interpretation: this is observational (not causal), but highlights which levers correlate with gains.")

    # 2) ROI chart.
    st.subheader("ROI: cost vs quality")
    plot_cost_vs_quality(
        [
            {
                "display_name": m.display_name,
                "stage": m.stage,
                "avg_overall": m.avg_overall,
                "avg_total_tokens": m.avg_total_tokens,
            }
            for m in scored
        ]
    )
    st.caption("Interpretation: use this to pick an exec-demo run (high score, low token cost).")

    # 3) Regression chart.
    st.subheader("Notable regressions")
    # Find bottom-N scored runs.
    worst = sorted(scored, key=lambda m: float(m.avg_overall or 0.0))[:5]
    if not worst:
        st.info("No scored runs yet.")
        return

    for m in worst:
        st.write(f"- {m.display_name} ({m.stage}) score={m.avg_overall:.0f} tokens={('—' if m.avg_total_tokens is None else f'{m.avg_total_tokens:.0f}')}")
    st.caption("Interpretation: these runs are good targets for postmortems; inspect them in Deep Debug.")


def render_before_after_page(
    *,
    runs_by_id: Dict[str, RunFile],
    scored_by_id: Dict[str, ScoredFile],
    metrics: List[RunMetrics],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> None:
    st.title("Before vs after answers")
    st.caption("Pick a case/question and see baseline → intermediate → regression → best.")

    # Build case_id options from gold (preferred) else from any run.
    case_ids = sorted(gold_by_case_id.keys())
    if not case_ids:
        # Fallback: scan run cases (bounded).
        seen: set[str] = set()
        for m in sorted(metrics, key=lambda x: x.created_at, reverse=True)[:40]:
            r = runs_by_id.get(m.run_id)
            if not r:
                continue
            for c in (r.obj.get("cases") or []):
                if isinstance(c, dict):
                    cid = str(c.get("case_id") or "").strip()
                    if cid:
                        seen.add(cid)
        case_ids = sorted(seen)

    if not case_ids:
        st.warning("No cases found.")
        return

    selected_case = st.selectbox("Case", options=case_ids, index=0)

    base = _select_baseline_run(metrics)
    best = _select_best_run(metrics)
    reg = _select_regression_run(metrics)
    mid = _select_intermediate_run(metrics)

    picks: List[Tuple[str, Optional[RunMetrics]]] = [
        ("Baseline", base),
        ("Intermediate", mid),
        ("Regression", reg),
        ("Best", best),
    ]

    # Materialize views (skip missing).
    views: List[Tuple[str, CaseView]] = []
    question_text = ""

    for label, m in picks:
        if not m:
            continue
        run_obj = runs_by_id.get(m.run_id)
        if not run_obj:
            continue
        scored_obj = scored_by_id.get(m.run_id)
        cv = build_case_view(
            run=run_obj,
            metrics=m,
            scored=scored_obj,
            case_id=selected_case,
            gold_by_case_id=gold_by_case_id,
        )
        if cv is None:
            continue
        views.append((label, cv))
        if cv.question:
            question_text = cv.question

    if not views:
        st.warning("Selected case not found in the chosen representative runs. Try another case.")
        return

    if question_text:
        st.markdown("**Question**")
        st.write(question_text)

    st.subheader("Progression")

    # Render cards.
    for label, cv in views:
        j = cv.judge or {}
        overall = safe_int(j.get("overall"))
        corr = safe_int(j.get("correctness"))
        g = safe_int(j.get("groundedness"))
        comp = safe_int(j.get("completeness"))

        st.markdown(f"### {label}: {cv.run.display_name}")
        cols = st.columns(6)
        cols[0].metric("Stage", cv.stage)
        cols[1].metric("Overall", "—" if overall is None else str(overall))
        cols[2].metric("Corr", "—" if corr is None else str(corr))
        cols[3].metric("Ground", "—" if g is None else str(g))
        cols[4].metric("Compl", "—" if comp is None else str(comp))
        cols[5].metric("Coverage", "—" if cv.coverage is None else f"{float(cv.coverage):.2f}")

        st.markdown("**Answer**")
        st.write(cv.answer_text or "(no answer text logged)")

        with st.expander("Retrieved evidence (top)", expanded=False):
            if cv.episode_ids:
                st.write("Episode IDs:", ", ".join(cv.episode_ids[:20]))
            if cv.sources:
                st.write("Sources:")
                for s in cv.sources[:8]:
                    st.write("-", truncate(s, 140))
            if cv.previews:
                st.write("Previews:")
                for p in cv.previews[:4]:
                    st.code(truncate(p, 600), language="text")

        st.divider()

    st.subheader("Answer diffs")
    # Diff baseline -> best when present.
    base_view = next((v for (lbl, v) in views if lbl == "Baseline"), None)
    best_view = next((v for (lbl, v) in views if lbl == "Best"), None)
    if base_view and best_view:
        diff = "\n".join(
            difflib.unified_diff(
                (base_view.answer_text or "").splitlines(),
                (best_view.answer_text or "").splitlines(),
                fromfile="baseline",
                tofile="best",
                lineterm="",
            )
        )
        st.code(diff or "(no diff)", language="diff")

        st.caption("Why it improved (auto heuristic):")
        reasons: List[str] = []
        bj = base_view.judge or {}
        sj = best_view.judge or {}
        if safe_int(sj.get("groundedness")) is not None and safe_int(bj.get("groundedness")) is not None:
            dg = int(safe_int(sj.get("groundedness")) or 0) - int(safe_int(bj.get("groundedness")) or 0)
            if dg > 0:
                reasons.append(f"Groundedness improved by {dg:+d}.")
        if best_view.coverage is not None and base_view.coverage is not None:
            if float(best_view.coverage) > float(base_view.coverage):
                reasons.append("Retrieval coverage improved (more likely correct episode evidence).")
        if (best_view.answer_text or "").strip() and len(best_view.answer_text) > len(base_view.answer_text or ""):
            reasons.append("Best answer is more complete/verbose (may reflect more context).")
        if not reasons:
            reasons.append("Improvement likely driven by retrieval + model changes, but signal is weak in logs.")
        for r in reasons[:5]:
            st.write("-", r)


def render_failure_page(
    *,
    runs_by_id: Dict[str, RunFile],
    scored_by_id: Dict[str, ScoredFile],
    metrics: List[RunMetrics],
    gold_by_case_id: Dict[str, Dict[str, Any]],
    representative_only: bool,
) -> None:
    st.title("Failure analysis")
    st.caption("Where the system fails and what patterns remain.")

    # Choose which runs to include.
    if representative_only:
        # One representative per stage.
        reps: Dict[str, RunMetrics] = {}
        for m in metrics:
            if m.avg_overall is None:
                continue
            prev = reps.get(m.stage)
            if prev is None:
                reps[m.stage] = m
            else:
                # Keep median-ish representative by score (approx: closer to stage mean).
                pass
        chosen = list(reps.values())
    else:
        chosen = [m for m in metrics if m.avg_overall is not None]

    # Build per-case rows.
    failure_counts: Dict[str, int] = {"retrieval_miss": 0, "grounding": 0, "aggregation": 0, "reasoning": 0, "unknown": 0}
    by_qtype: Dict[Tuple[str, str], int] = {}
    coverage_points: List[Dict[str, Any]] = []

    for m in chosen:
        run_obj = runs_by_id.get(m.run_id)
        if not run_obj:
            continue
        scored_obj = scored_by_id.get(m.run_id)

        cases = run_obj.obj.get("cases")
        if not isinstance(cases, list):
            continue
        for c in cases:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("case_id") or "").strip()
            if not cid:
                continue
            q = str(c.get("question") or "")
            qtype = question_type(q)

            judge = _case_judge_from_scored(scored_obj.obj if scored_obj else None, cid)
            overall = safe_int((judge or {}).get("overall"))
            cov, _method = compute_retrieval_coverage(case=c, gold_by_case_id=gold_by_case_id)
            mode = classify_failure_mode(case=c, judge=judge, coverage=cov)

            failure_counts[mode] = int(failure_counts.get(mode, 0)) + 1
            by_qtype[(qtype, mode)] = int(by_qtype.get((qtype, mode), 0)) + 1

            coverage_points.append(
                {
                    "coverage": safe_float(cov),
                    "overall": safe_float(overall),
                    "failure_mode": mode,
                    "case_id": cid,
                    "display_name": m.display_name,
                    "stage": m.stage,
                }
            )

    st.subheader("Failure mode distribution")
    plot_failure_pie({k: v for k, v in failure_counts.items() if k != "ok"})
    st.caption("Interpretation: treat the largest slice as the next systematic investment area.")

    st.subheader("Failure modes by question type")
    rows: List[Dict[str, Any]] = []
    for (qt, fm), cnt in by_qtype.items():
        rows.append({"question_type": qt, "failure_mode": fm, "count": int(cnt)})
    plot_failure_by_qtype_stacked(rows)
    st.caption("Interpretation: some techniques help specific question families (e.g., QE helps aggregation).")

    st.subheader("Retrieval coverage map")
    plot_coverage_scatter(coverage_points)
    st.caption("Interpretation: top-left failures indicate retrieval is fine but answering is weak (grounding/reasoning).")

    with st.expander("Examples (high-signal failures)", expanded=False):
        # Show a few examples where coverage high but score low.
        bad = [p for p in coverage_points if (p.get("coverage") is not None and float(p["coverage"]) >= 0.5) and (p.get("overall") is not None and float(p["overall"]) <= 50)]
        bad = sorted(bad, key=lambda x: (float(x.get("overall") or 0.0), -float(x.get("coverage") or 0.0)))[:8]
        for b in bad:
            st.write(f"- {b.get('case_id')} | {b.get('stage')} | overall={truncate(b.get('overall'), 6)} | coverage={truncate(b.get('coverage'), 6)} | {b.get('failure_mode')}")


def render_debug_page(
    *,
    runs: List[RunFile],
    scored_by_id: Dict[str, ScoredFile],
    metrics: List[RunMetrics],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> None:
    st.title("Deep debug")
    st.caption("Raw run explorer + per-case inspection. This is intentionally dense.")

    # Filters
    stage = st.selectbox("Stage", options=["(all)"] + STAGE_ORDER, index=0)
    only_scored = st.checkbox("Only scored runs", value=True)

    metrics_by_id = {m.run_id: m for m in metrics}

    # List runs.
    run_choices: List[Tuple[str, str]] = []
    for r in runs:
        rid = run_id(r.obj, fallback_path=r.path)
        m = metrics_by_id.get(rid)
        if stage != "(all)" and m and m.stage != stage:
            continue
        if only_scored and (not m or m.avg_overall is None):
            continue
        label = m.display_name if m is not None else build_run_display_name(r.obj, fallback_path=r.path)
        if m and m.avg_overall is not None:
            label += f" | score={m.avg_overall:.0f}"
        if m:
            label += f" | stage={m.stage}"
        run_choices.append((rid, label))

    if not run_choices:
        st.info("No runs match filters.")
        return

    selected = st.selectbox("Run", options=run_choices, format_func=lambda x: x[1], index=0)
    run_id_selected = selected[0] if isinstance(selected, tuple) else None

    run_obj = next((r for r in runs if run_id(r.obj, fallback_path=r.path) == run_id_selected), None)
    if not run_obj:
        st.warning("Run not found.")
        return

    m = metrics_by_id.get(run_id_selected or "")
    scored = scored_by_id.get(run_id_selected or "")

    st.subheader("Run")
    cols = st.columns(5)
    cols[0].metric(
        "Run",
        (m.display_name if m is not None else build_run_display_name(run_obj.obj, fallback_path=run_obj.path)),
    )
    cols[1].metric("Stage", m.stage if m else "—")
    cols[2].metric("Avg overall", "—" if (not m or m.avg_overall is None) else f"{m.avg_overall:.0f}")
    cols[3].metric("Avg tokens", "—" if (not m or m.avg_total_tokens is None) else f"{m.avg_total_tokens:.0f}")
    cols[4].metric("Cases", len(run_obj.obj.get("cases") or []) if isinstance(run_obj.obj.get("cases"), list) else 0)

    with st.expander("Config JSON", expanded=False):
        st.json(safe_get(run_obj.obj, "config", {}))

    with st.expander("Summary JSON", expanded=False):
        st.json(safe_get(run_obj.obj, "summary", {}))

    if scored is not None:
        with st.expander("Scored JSON (header)", expanded=False):
            st.json({
                "score_summary": safe_get(scored.obj, "score_summary", {}),
                "deterministic_summary": safe_get(scored.obj, "deterministic_summary", {}),
                "scoring_meta": safe_get(scored.obj, "scoring_meta", {}),
            })

    # Case explorer
    cases = run_obj.obj.get("cases")
    if not isinstance(cases, list) or not cases:
        st.info("No cases.")
        return

    case_ids = [str(c.get("case_id") or "") for c in cases if isinstance(c, dict) and str(c.get("case_id") or "").strip()]
    case_ids = sorted(set(case_ids))
    selected_case = st.selectbox("Case", options=case_ids, index=0)

    cobj = _case_from_run(run_obj.obj, selected_case)
    if not cobj:
        st.warning("Case not found.")
        return

    judge = _case_judge_from_scored(scored.obj if scored else None, selected_case)
    cov, cov_method = compute_retrieval_coverage(case=cobj, gold_by_case_id=gold_by_case_id)
    mode = classify_failure_mode(case=cobj, judge=judge, coverage=cov)

    st.subheader("Case")
    st.markdown("**Question**")
    st.write(str(cobj.get("question") or ""))

    cols = st.columns(6)
    cols[0].metric("Failure mode", mode)
    cols[1].metric("Coverage", "—" if cov is None else f"{float(cov):.2f}")
    cols[2].metric("Coverage method", cov_method)

    if judge:
        cols[3].metric("Overall", safe_int(judge.get("overall")) or 0)
        cols[4].metric("Correctness", safe_int(judge.get("correctness")) or 0)
        cols[5].metric("Groundedness", safe_int(judge.get("groundedness")) or 0)

    st.markdown("**Answer**")
    st.write(str(safe_get(cobj, "answer.text", "") or ""))

    with st.expander("Retrieval results", expanded=False):
        st.json(safe_get(cobj, "retrieval", {}))

    with st.expander("Diagnostics", expanded=False):
        st.json(safe_get(cobj, "diagnostics", {}))

    with st.expander("Raw case JSON", expanded=False):
        st.json(cobj)


# -----------------------------------------------------------------------------
# App scaffold
# -----------------------------------------------------------------------------
def build_story_dataset(
    *,
    runs: List[RunFile],
    scored_by_run_id: Dict[str, ScoredFile],
    gold_by_case_id: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, RunFile], List[RunMetrics], List[Dict[str, Any]]]:
    # Baseline identification: earliest scored baseline-like run.
    runs_sorted = sorted(runs, key=lambda r: _run_created_at(r.obj, fallback_path=r.path))

    baseline_llm = None
    baseline_k = None

    if runs_sorted:
        # Choose earliest run as baseline for model/k reference.
        baseline_llm = llm_model(runs_sorted[0].obj)
        baseline_k = safe_int(safe_get(runs_sorted[0].obj, "config.retrieval.k", None))

    runs_by_id: Dict[str, RunFile] = {}
    metrics: List[RunMetrics] = []

    for r in runs:
        rid = run_id(r.obj, fallback_path=r.path)
        if rid:
            runs_by_id[rid] = r

    # Compute stage for each run.
    for r in runs_sorted:
        rid = run_id(r.obj, fallback_path=r.path)
        if not rid:
            continue
        scored = scored_by_run_id.get(rid)

        stage = classify_stage(r.obj, baseline_llm_model=baseline_llm, baseline_k=baseline_k)
        m = compute_run_metrics(run=r, scored=scored, stage=stage, gold_by_case_id=gold_by_case_id)
        metrics.append(m)

    # Stage aggregates (mean score/tokens + representative runs).
    by_stage: Dict[str, List[RunMetrics]] = {}
    for m in metrics:
        by_stage.setdefault(m.stage, []).append(m)

    stage_rows: List[Dict[str, Any]] = []
    for stage, xs in by_stage.items():
        stage_rows.append(compute_stage_metrics(stage, xs))

    stage_rows.sort(key=lambda r: _stage_sort_key(str(r.get("stage"))))

    return runs_by_id, metrics, stage_rows


def _global_failure_counts(
    *,
    runs_by_id: Dict[str, RunFile],
    scored_by_id: Dict[str, ScoredFile],
    metrics: List[RunMetrics],
    gold_by_case_id: Dict[str, Dict[str, Any]],
    representative_only: bool = True,
) -> Dict[str, int]:
    # For the landing page, restrict to representative runs (noise reduction).
    chosen: List[RunMetrics]
    if representative_only:
        by_stage: Dict[str, List[RunMetrics]] = {}
        for m in metrics:
            by_stage.setdefault(m.stage, []).append(m)
        chosen = []
        for stg, xs in by_stage.items():
            rep = choose_representative_runs(xs).get("representative")
            if rep is not None:
                chosen.append(rep)
    else:
        chosen = [m for m in metrics if m.avg_overall is not None]

    counts: Dict[str, int] = {"retrieval_miss": 0, "grounding": 0, "aggregation": 0, "reasoning": 0, "unknown": 0}

    for m in chosen:
        run_obj = runs_by_id.get(m.run_id)
        if not run_obj:
            continue
        scored = scored_by_id.get(m.run_id)
        cases = run_obj.obj.get("cases")
        if not isinstance(cases, list):
            continue
        for c in cases:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("case_id") or "").strip()
            if not cid:
                continue
            judge = _case_judge_from_scored(scored.obj if scored else None, cid)
            cov, _method = compute_retrieval_coverage(case=c, gold_by_case_id=gold_by_case_id)
            mode = classify_failure_mode(case=c, judge=judge, coverage=cov)
            counts[mode] = int(counts.get(mode, 0)) + 1

    return counts


def main(*, set_page_config: bool = True, show_title: bool | None = None) -> None:
    if set_page_config:
        st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")

    if show_title is None:
        show_title = bool(set_page_config)
    if show_title:
        st.title(APP_TITLE)

    # Make the top page selector sticky (storytelling-friendly).
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

    # Top navigation (story-first ordering)
    pages = [
        "Summary",
        "Experiment Timeline",
        "What Helped / What Hurt",
        "Before vs After Answers",
        "Failure Analysis",
        "Deep Debug",
    ]

    default_page = st.session_state.get("story_page") or pages[0]
    st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        page = st.segmented_control("Page", options=pages, default=default_page)
    else:
        page = st.radio(
            "Page",
            options=pages,
            index=max(0, pages.index(default_page)) if default_page in pages else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not page:
        page = pages[0]
    st.session_state["story_page"] = page

    # Sidebar: keep minimal; progressive disclosure.
    st.sidebar.header("Data")
    project_root = Path(st.sidebar.text_input("Project root", value=str(Path.cwd()))).expanduser()

    runs_dir = Path(st.sidebar.text_input("Runs dir", value=str(project_root / "experiments" / "runs"))).expanduser()

    # Prefer two-pass directory when present.
    scored_dirs_default = [
        str(project_root / "experiments" / "scored_runs_two_pass"),
        str(project_root / "experiments" / "scored_runs"),
    ]
    scored_dirs_raw = st.sidebar.text_area(
        "Scored dirs (one per line; first wins)",
        value="\n".join(scored_dirs_default),
        height=80,
    )
    scored_dirs = [x.strip() for x in scored_dirs_raw.splitlines() if x.strip()]

    gold_path = str(project_root / "experiments" / "gold_answers.json")
    gold_by_case_id = load_gold_answers(gold_path)

    require_two_pass = st.sidebar.toggle(
        "Prefer two-pass scores",
        value=True,
        help="If enabled, ignores scored files unless scoring_meta.judge_mode == two_pass.",
    )

    with st.sidebar.expander("Performance", expanded=False):
        st.caption("If you have many runs, reduce scan size.")
        scan_limit = st.slider("Max newest runs to load", min_value=50, max_value=2000, value=500, step=50)

    # Load data.
    with st.spinner("Loading runs + scored artifacts…"):
        all_runs = load_runs(str(runs_dir))
        runs = all_runs[: int(scan_limit)]
        scored_by_run_id = load_scored_runs(scored_dirs=scored_dirs, prefer_two_pass=bool(require_two_pass))

        runs_by_id, metrics, stage_rows = build_story_dataset(
            runs=runs,
            scored_by_run_id=scored_by_run_id,
            gold_by_case_id=gold_by_case_id,
        )

        failure_counts = _global_failure_counts(
            runs_by_id=runs_by_id,
            scored_by_id=scored_by_run_id,
            metrics=metrics,
            gold_by_case_id=gold_by_case_id,
            representative_only=True,
        )

    # Page routing.
    if page == "Summary":
        render_summary_page(
            runs_by_id=runs_by_id,
            metrics=[m for m in metrics if m.avg_overall is not None],
            stage_rows=stage_rows,
            failure_counts=failure_counts,
        )
        return

    if page == "Experiment Timeline":
        render_timeline_page(stage_rows=stage_rows)
        return

    if page == "What Helped / What Hurt":
        render_helped_hurt_page(metrics=metrics)
        return

    if page == "Before vs After Answers":
        render_before_after_page(
            runs_by_id=runs_by_id,
            scored_by_id=scored_by_run_id,
            metrics=metrics,
            gold_by_case_id=gold_by_case_id,
        )
        return

    if page == "Failure Analysis":
        rep_only = st.checkbox("Use representative runs only", value=True)
        render_failure_page(
            runs_by_id=runs_by_id,
            scored_by_id=scored_by_run_id,
            metrics=metrics,
            gold_by_case_id=gold_by_case_id,
            representative_only=bool(rep_only),
        )
        return

    # Deep Debug
    render_debug_page(
        runs=runs,
        scored_by_id=scored_by_run_id,
        metrics=metrics,
        gold_by_case_id=gold_by_case_id,
    )


if __name__ == "__main__":
    main()
