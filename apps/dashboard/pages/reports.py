from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from turtle import left
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st

try:  # Plotly (optional, but preferred for interactive charts)
    import plotly.express as px  # type: ignore

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    px = None  # type: ignore
    _HAS_PLOTLY = False

from apps.dashboard.shared import (
    DEFAULT_RUNS_DIR,
    EXPERIMENTS_DIR,
    REPO_ROOT,
    PhaseRow,
    RunRow,
    _load_phase_summary,
    _load_run_rows_cached,
    _phase_summary_is_step_grouped,
    _pick_latest_phase_summary_json,
    _read_csv_rows,
    _read_phase_summary_obj,
    _step_num,
    _st_dataframe,
    _st_image,
    _try_parse_step_cfg_from_run_name,
    _plot_score_hist,
    _plot_score_trend,
)


_EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


def _html(md: str) -> str:
    """Normalize HTML-in-Markdown strings for Streamlit.

    Streamlit renders HTML via its Markdown pipeline; if any line begins with a
    tab or 4+ leading spaces, Markdown can treat it as an indented code block
    and display the HTML literally. This helper removes leading indentation on
    every line while keeping line breaks.
    """

    s = textwrap.dedent(md).strip("\n")
    lines = [ln.lstrip(" \t") for ln in s.splitlines()]
    return "\n".join(lines).strip()


def _plotly_or_warning() -> bool:
    if _HAS_PLOTLY:
        return True
    st.warning(
        "Plotly is not installed; charts will fall back to tables. Install with: pip install plotly"
    )
    return False


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _safe_int(x: Any) -> Optional[int]:
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


def _safe_float(x: Any) -> Optional[float]:
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


def _normalize_episode_id(s: Any) -> Optional[str]:
    if not s:
        return None
    m = _EP_RE.search(str(s).strip().upper())
    return m.group(0).upper() if m else None


def _question_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if not q:
        return "(missing)"
    if re.search(r"\b(which|what)\s+episode\b|\bin\s+which\s+episode\b|\bepisode\s+is\b", q):
        return "Episode lookup"
    if re.search(r"\bquote\b|\bexact\s+quote\b|\bwhat\s+did\b.+\bsay\b", q):
        return "Quote / line"
    if re.search(
        r"\b(list|summari[sz]e|overview|timeline|chronolog|across|throughout|all\b|compare)\b",
        q,
    ):
        return "Aggregation"
    if re.search(r"\bwhy\b|\bhow\b", q):
        return "Explanation"
    return "Factual"


def _avg_total_tokens(run_obj: Dict[str, Any]) -> Optional[float]:
    # Prefer run summary.
    v = _safe_float(_safe_get(run_obj, "summary.avg_total_tokens", None))
    if v is not None:
        return float(v)

    # Fall back to per-case usage.
    cases = run_obj.get("cases")
    if not isinstance(cases, list) or not cases:
        return None

    vals: List[float] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        usage = _safe_get(c, "answer.usage", {})
        if not isinstance(usage, dict):
            continue
        t = _safe_float(usage.get("total_tokens"))
        if t is not None:
            vals.append(float(t))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _script_persist_dir(run_obj: Dict[str, Any]) -> str:
    # Prefer provenance block (v4+).
    dv = _safe_get(run_obj, "config.data_version.script.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(_safe_get(run_obj, "config.vectorstore.persist_directory", "") or "")


def _derived_persist_dir(run_obj: Dict[str, Any]) -> str:
    dv = _safe_get(run_obj, "config.data_version.derived.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(_safe_get(run_obj, "config.derived_vectorstore.persist_directory", "") or "")


def _retrieval_policy(run_obj: Dict[str, Any]) -> str:
    return str(_safe_get(run_obj, "config.retrieval.policy", "") or "").strip().lower()


def _search_type(run_obj: Dict[str, Any]) -> str:
    return str(_safe_get(run_obj, "config.retrieval.search_type", "") or "").strip().lower()


def _query_expansion_enabled(run_obj: Dict[str, Any]) -> bool:
    return bool(_safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))


def _chunking_signature(run_obj: Dict[str, Any]) -> str:
    splitter = str(_safe_get(run_obj, "config.chunking.splitter", "") or "")
    persist = _script_persist_dir(run_obj)
    return f"{splitter}|{persist}".lower()


def _llm_model(run_obj: Dict[str, Any]) -> Optional[str]:
    m = _safe_get(run_obj, "config.llm.model", None)
    if isinstance(m, str) and m.strip():
        return m.strip()
    return None


def _classify_stage(
    run_obj: Dict[str, Any],
    *,
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> str:
    """Heuristically bucket a run into a story stage.

    Ordered and intentionally opinionated (ported from the older dashboard).
    """
    name = str(_safe_get(run_obj, "run.run_name", "") or "").lower()
    policy = _retrieval_policy(run_obj)
    search = _search_type(run_obj)
    qe = _query_expansion_enabled(run_obj)
    persist = _script_persist_dir(run_obj).lower()
    derived_persist = _derived_persist_dir(run_obj).lower()
    chunk_sig = _chunking_signature(run_obj)

    k = _safe_int(_safe_get(run_obj, "config.retrieval.k", None))

    if (
        policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}
        or "derived" in derived_persist
    ):
        return "Derived Summaries / Cards"
    if "chroma_db_meta" in persist or "metadata" in name or "routing" in name:
        return "Metadata / Routing"
    if qe or "qe" in name or "query_expansion" in name or "rrf" in name or "fusion" in name:
        return "Query Expansion + Fusion"
    if "scene" in chunk_sig or "scene" in name:
        return "Scene Chunking"
    if search == "mmr" or "mmr" in name:
        return "MMR / Diversity"
    if baseline_k is not None and k is not None and k > baseline_k:
        return "Increased Recall"
    if k is not None and k >= 10:
        return "Increased Recall"

    m = _llm_model(run_obj)
    if baseline_llm_model and m and m != baseline_llm_model:
        return "Model Upgrades"
    if "gpt-" in name and baseline_llm_model and baseline_llm_model.replace(".", "_") not in name:
        return "Model Upgrades"

    if "baseline" in name or policy in {"script_only", "script", "baseline", ""}:
        return "Baseline"
    return "Other / Uncategorized"


def _classify_failure_mode(
    *, case: Dict[str, Any], judge: Optional[Dict[str, Any]], coverage: Optional[float]
) -> str:
    """Return one of: retrieval_miss | grounding | aggregation | reasoning | ok | unknown."""
    labels = case.get("labels")
    if isinstance(labels, dict):
        if labels.get("retrieval_failure") is True:
            return "retrieval_miss"
        if labels.get("grounding_failure") is True:
            return "grounding"

    qtype = _question_type(str(case.get("question") or ""))
    cov = _safe_float(coverage)
    if cov is not None and cov < 0.5:
        return "retrieval_miss"

    if isinstance(judge, dict):
        corr = _safe_int(judge.get("correctness"))
        g = _safe_int(judge.get("groundedness"))
        comp = _safe_int(judge.get("completeness"))

        if qtype == "Aggregation" and comp is not None and comp <= 2:
            return "aggregation"
        if g is not None and g <= 2:
            return "grounding"
        if corr is not None and corr <= 2:
            return "reasoning"

        overall = _safe_int(judge.get("overall"))
        if overall is not None and overall >= 80:
            return "ok"
        if overall is not None and overall <= 40:
            if qtype == "Aggregation":
                return "aggregation"
            return "reasoning"

    return "unknown"


@st.cache_data(show_spinner=False)
def _load_run_objs_cached(runs_dir: str, *, max_files: int = 500) -> List[Dict[str, Any]]:
    p = Path(runs_dir)
    if not p.exists():
        return []
    files = sorted(p.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if max_files and len(files) > int(max_files):
        files = files[: int(max_files)]
    out: List[Dict[str, Any]] = []
    for rf in files:
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "run" in obj and "cases" in obj:
                out.append({"path": str(rf), "obj": obj})
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def _load_scored_summary_cached(scored_dir: str) -> Dict[str, Dict[str, Any]]:
    p = Path(scored_dir)
    if not p.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sf in sorted(p.glob("*.scored.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            obj = json.loads(sf.read_text(encoding="utf-8"))
            run = obj.get("run") if isinstance(obj, dict) else None
            rid = str((run or {}).get("run_id") or "") if isinstance(run, dict) else ""
            rid = rid.strip() or sf.name[: -len(".scored.json")]
            avg_overall = _safe_get(obj, "score_summary.avg_overall", None)
            out[rid] = {
                "path": str(sf),
                "avg_overall": (int(avg_overall) if isinstance(avg_overall, int) else None),
            }
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def _load_scored_obj_cached(scored_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    p = Path(scored_dir) / f"{run_id}.scored.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _run_id_from_run_entry(entry: Dict[str, Any]) -> Optional[str]:
    obj = entry.get("obj") if isinstance(entry, dict) else None
    if not isinstance(obj, dict):
        return None
    run = obj.get("run")
    rid = str((run or {}).get("run_id") or "").strip() if isinstance(run, dict) else ""
    if rid:
        return rid
    path = str(entry.get("path") or "")
    if path:
        try:
            return Path(path).stem
        except Exception:
            return None
    return None


def _index_scored_cases_by_id(scored_obj: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(scored_obj, dict):
        return {}
    cases = scored_obj.get("scored_cases")
    if not isinstance(cases, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in cases:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("case_id") or "").strip()
        judge = row.get("judge")
        if not cid or not isinstance(judge, dict):
            continue
        out[cid] = judge
    return out


def _baseline_signature_from_runs(run_objs: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    llm_models: List[str] = []
    ks: List[int] = []
    for o in run_objs:
        m = _llm_model(o)
        if m:
            llm_models.append(m)
        k = _safe_int(_safe_get(o, "config.retrieval.k", None))
        if k is not None:
            ks.append(int(k))

    baseline_llm_model: Optional[str] = None
    if llm_models:
        # Most common model across selected runs is a decent baseline anchor.
        baseline_llm_model = Counter(llm_models).most_common(1)[0][0]

    baseline_k: Optional[int] = min(ks) if ks else None
    return baseline_llm_model, baseline_k


def _failure_mode_counts_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_dir: str,
) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    total = 0

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        cases = ro.get("cases")
        if not isinstance(cases, list) or not cases:
            continue

        scored_obj = _load_scored_obj_cached(scored_dir, rid)
        judge_by_case_id = _index_scored_cases_by_id(scored_obj)

        for c in cases:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("case_id") or "").strip()
            judge = judge_by_case_id.get(cid)
            mode = _classify_failure_mode(case=c, judge=judge, coverage=None)
            counts[mode] += 1
            total += 1

    rows = []
    for mode, n in counts.most_common():
        rows.append(
            {
                "failure_mode": mode,
                "count": int(n),
                "share": (float(n / total) if total else None),
            }
        )
    return rows


def _failure_modes_by_question_type_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_dir: str,
) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str]] = Counter()
    totals_by_qtype: Counter[str] = Counter()

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        cases = ro.get("cases")
        if not isinstance(cases, list) or not cases:
            continue

        scored_obj = _load_scored_obj_cached(scored_dir, rid)
        judge_by_case_id = _index_scored_cases_by_id(scored_obj)

        for c in cases:
            if not isinstance(c, dict):
                continue
            q = str(c.get("question") or "")
            qtype = _question_type(q)
            cid = str(c.get("case_id") or "").strip()
            judge = judge_by_case_id.get(cid)
            mode = _classify_failure_mode(case=c, judge=judge, coverage=None)
            counts[(qtype, mode)] += 1
            totals_by_qtype[qtype] += 1

    rows: List[Dict[str, Any]] = []
    # Stable ordering for readability.
    mode_order = {
        "ok": 0,
        "retrieval_miss": 1,
        "grounding": 2,
        "aggregation": 3,
        "reasoning": 4,
        "unknown": 999,
    }
    qtype_order = {
        "Factual": 0,
        "Episode lookup": 1,
        "Aggregation": 2,
        "Explanation": 3,
        "Quote / line": 4,
        "(missing)": 999,
    }
    for (qtype, mode), n in sorted(
        counts.items(),
        key=lambda kv: (
            qtype_order.get(kv[0][0], 50),
            mode_order.get(kv[0][1], 500),
            kv[0][0],
            kv[0][1],
        ),
    ):
        total = int(totals_by_qtype.get(qtype, 0))
        rows.append(
            {
                "question_type": qtype,
                "failure_mode": mode,
                "count": int(n),
                "share_within_type": (float(n / total) if total else None),
            }
        )
    return rows


def _cost_quality_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue

        tokens = _avg_total_tokens(ro)
        if tokens is None:
            continue

        run = ro.get("run") if isinstance(ro.get("run"), dict) else {}
        run_name = str(run.get("run_name") or "")

        rows.append(
            {
                "run_id": rid,
                "run_name": run_name,
                "stage": _classify_stage(
                    ro,
                    baseline_llm_model=baseline_llm_model,
                    baseline_k=baseline_k,
                ),
                "avg_overall": int(avg_overall),
                "avg_total_tokens": float(tokens),
                "retrieval_policy": _retrieval_policy(ro),
                "search_type": _search_type(ro),
                "k": _safe_int(_safe_get(ro, "config.retrieval.k", None)),
            }
        )
    return rows


def _stage_timeline_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> List[Dict[str, Any]]:
    stage_scores: Dict[str, List[int]] = {}
    stage_counts: Counter[str] = Counter()

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue
        stage = _classify_stage(
            ro,
            baseline_llm_model=baseline_llm_model,
            baseline_k=baseline_k,
        )
        stage_scores.setdefault(stage, []).append(int(avg_overall))
        stage_counts[stage] += 1

    stage_order: Dict[str, int] = {
        "Baseline": 10,
        "Increased Recall": 20,
        "MMR / Diversity": 30,
        "Scene Chunking": 40,
        "Query Expansion + Fusion": 50,
        "Metadata / Routing": 60,
        "Derived Summaries / Cards": 70,
        "Model Upgrades": 80,
        "Other / Uncategorized": 999,
    }

    rows: List[Dict[str, Any]] = []
    for stage, vals in sorted(
        stage_scores.items(),
        key=lambda kv: (stage_order.get(kv[0], 500), kv[0]),
    ):
        if not vals:
            continue
        avg_score = float(sum(vals) / len(vals))
        rows.append(
            {
                "stage": stage,
                "avg_score": round(avg_score, 1),
                "runs": int(stage_counts.get(stage, 0)),
            }
        )

    # Δ vs previous
    prev: Optional[float] = None
    for r in rows:
        cur = _safe_float(r.get("avg_score"))
        if prev is None or cur is None:
            r["delta_vs_previous"] = None
        else:
            r["delta_vs_previous"] = round(float(cur - prev), 1)
        prev = float(cur) if cur is not None else prev

    return rows


def _technique_impact_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> List[Dict[str, Any]]:
    # For each technique, compare mean(avg_overall) for enabled vs disabled.
    tech_defs: List[Tuple[str, str]] = [
        ("metadata_index", "Metadata index"),
        ("scene_chunking", "Scene chunking"),
        ("derived_routing", "Derived routing/cards"),
        ("qe_fusion", "Query expansion + fusion"),
        ("mmr", "MMR retrieval"),
        ("high_k", "High k (>=10)"),
    ]

    enabled: Dict[str, List[int]] = {k: [] for k, _ in tech_defs}
    disabled: Dict[str, List[int]] = {k: [] for k, _ in tech_defs}

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue

        name = str(_safe_get(ro, "run.run_name", "") or "").lower()
        policy = _retrieval_policy(ro)
        search = _search_type(ro)
        qe = _query_expansion_enabled(ro)
        persist = _script_persist_dir(ro).lower()
        derived_persist = _derived_persist_dir(ro).lower()
        chunk_sig = _chunking_signature(ro)
        k = _safe_int(_safe_get(ro, "config.retrieval.k", None))

        # Technique flags (heuristic but stable).
        has_metadata = ("chroma_db_meta" in persist) or ("metadata" in name)
        has_scene = ("scene" in chunk_sig) or ("scene" in name)
        has_derived = (
            policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}
            or "derived" in derived_persist
            or "topic" in name
        )
        has_qe_fusion = qe or ("rrf" in name) or ("fusion" in name)
        has_mmr = (search == "mmr") or ("mmr" in name)
        has_high_k = bool(k is not None and int(k) >= 10)

        flags: Dict[str, bool] = {
            "metadata_index": bool(has_metadata),
            "scene_chunking": bool(has_scene),
            "derived_routing": bool(has_derived),
            "qe_fusion": bool(has_qe_fusion),
            "mmr": bool(has_mmr),
            "high_k": bool(has_high_k),
        }

        for key, _label in tech_defs:
            if flags.get(key) is True:
                enabled[key].append(int(avg_overall))
            else:
                disabled[key].append(int(avg_overall))

    out: List[Dict[str, Any]] = []
    for key, label in tech_defs:
        en = enabled.get(key) or []
        dis = disabled.get(key) or []
        if not en or not dis:
            continue
        en_avg = float(sum(en) / len(en))
        dis_avg = float(sum(dis) / len(dis))
        out.append(
            {
                "technique": label,
                "delta_avg_overall": round(en_avg - dis_avg, 1),
                "enabled_n": int(len(en)),
                "disabled_n": int(len(dis)),
                "enabled_avg": round(en_avg, 1),
                "disabled_avg": round(dis_avg, 1),
            }
        )

    # Sort by delta descending.
    out.sort(key=lambda r: float(r.get("delta_avg_overall") or 0.0), reverse=True)
    return out


def render_summary() -> None:
    st.header("Reports")

    runs_dir = DEFAULT_RUNS_DIR
    canonical_scored_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass"
    ctxfix_all_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass_ctxfix_all_2026-03-07"
    rescored_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass_rescored_2026-03-07"
    rescored_ctxfix_dir = (
        REPO_ROOT / "experiments" / "scored_runs_two_pass_rescored_ctxfix_2026-03-07"
    )

    # Default to the canonical deploy path when it has scored files.
    # This keeps Streamlit Cloud / docs stable while letting us refresh the contents.
    def _has_scored_files(p: Path) -> bool:
        return p.exists() and any(p.glob("*.scored.json"))

    if _has_scored_files(canonical_scored_dir):
        scored_dir = canonical_scored_dir
    elif _has_scored_files(ctxfix_all_dir):
        scored_dir = ctxfix_all_dir
    elif _has_scored_files(rescored_ctxfix_dir):
        scored_dir = rescored_ctxfix_dir
    elif _has_scored_files(rescored_dir):
        scored_dir = rescored_dir
    else:
        scored_dir = canonical_scored_dir

    # Sub-pages within Reports mode.
    # If an old session had the removed "Groups" page selected, fall back.
    if str(st.session_state.get("reports_subpage") or "") in {"Groups", "Timeline", "Charts"}:
        st.session_state["reports_subpage"] = "Analysis"
    try:
        page = st.radio(
            "Reports subpage",
            options=["Overview", "Structure", "Analysis", "Artifacts"],
            horizontal=True,
            key="reports_subpage",
            label_visibility="collapsed",
        )
    except TypeError:
        # Older Streamlit: label_visibility not supported.
        page = st.radio(
            "",
            options=["Overview", "Structure", "Analysis", "Artifacts"],
            horizontal=True,
            key="reports_subpage",
        )

    # Phase summary selection. On Timeline we render the selector at the bottom
    # (per UX request), but we still need a selection early to render Groups.
    candidates = sorted(EXPERIMENTS_DIR.glob("phase_score_summary_*.json"))
    phase_json_default = _pick_latest_phase_summary_json()
    if phase_json_default is None:
        st.warning(
            "No phase summary JSON found under `experiments/`. Run `experiments/plot_phase_scores.py`."
        )
        return

    name_to_path = {p.name: p for p in candidates}
    options = list(name_to_path.keys())
    default_name = phase_json_default.name

    selected_name = str(st.session_state.get("reports_phase_summary_file") or "").strip()
    if not selected_name or selected_name not in name_to_path:
        selected_name = default_name

    try:
        selected_index = options.index(selected_name)
    except Exception:
        selected_index = 0

    phase_json: Path
    if page in {"Artifacts"}:
        picked_name = st.selectbox(
            "Phase summary file",
            options=options,
            index=selected_index,
            key="reports_phase_summary_file",
            help="Switch between phase-level and by-step summaries, and between different scoring sets.",
        )
        phase_json = name_to_path.get(picked_name, phase_json_default)
    else:
        # Overview + Timeline: use the current session selection (or default) without showing a selector.
        phase_json = name_to_path.get(selected_name, phase_json_default)

    phase_obj = _read_phase_summary_obj(phase_json)
    is_step_grouped = _phase_summary_is_step_grouped(phase_obj)

    rows = _load_phase_summary(phase_json)
    if not rows:
        st.warning("Phase summary JSON exists but has no rows.")
        return

    # --- Group naming + sorting ---
    # We show two different concepts in phase summary artifacts:
    # - high-level phases like "baseline", "mmr", "qe+fusion"...
    # - routing sweep steps like "step01".."step06" (best seen in *_by_step summaries)
    _STEP_FRIENDLY: Dict[str, str] = {
        "step01": "Baseline (script-only; no derived)",
        "step02": "Derived-only retrieval",
        "step03": "Routing: derived → script",
        "step04": "Higher k / MMR tuning",
        "step05": "Query expansion + fusion (RRF)",
        "step06": "Best combined",
    }

    _PHASE_FRIENDLY: Dict[str, str] = {
        "baseline": "Baseline",
        "k12": "Increase retrieval k (k=12)",
        "mmr": "MMR",
        "qe+fusion": "QE + fusion (RRF)",
        "qe+fusion+mmr": "QE + fusion (RRF) + MMR",
        "routing_blended": "Routing blended (derived + script)",
        "routing_derived_then_script": "Routing: derived → script",
        "routing_derived_only": "Derived-only",
        "derived_meta": "Derived v2 (metadata index)",
        "script_only": "Script-only",
        "qe": "Query expansion (no fusion)",
        "other": "Other",
    }

    _PHASE_ORDER: Dict[str, int] = {
        "baseline": 10,
        "k12": 20,
        "mmr": 30,
        "qe+fusion": 40,
        "qe+fusion+mmr": 50,
        "routing_derived_only": 60,
        "routing_derived_then_script": 70,
        "derived_meta": 80,
        "routing_blended": 85,
        "script_only": 90,
        "qe": 95,
        "other": 999,
    }

    def _group_display(g: str) -> str:
        g0 = str(g or "").strip()
        if g0.lower() in _STEP_FRIENDLY:
            return f"{g0.lower()} — {_STEP_FRIENDLY[g0.lower()]}"
        if g0.lower() in _PHASE_FRIENDLY:
            return _PHASE_FRIENDLY[g0.lower()]
        return g0

    def _group_sort_key(g: str) -> Tuple[int, int, str]:
        g0 = str(g or "").strip()
        g_l = g0.lower()
        if g_l in _PHASE_ORDER:
            return (0, int(_PHASE_ORDER[g_l]), g_l)
        sn = _step_num(g0)
        if sn is not None:
            return (1, int(sn), g_l)
        return (2, 10**9, g_l)

    def _render_groups_section(rows_in: List[PhaseRow]) -> None:
        st.subheader("Performance Analysis")
        

        st.markdown(
                _html(
                    """<div class="section-card phase-card">
<div class="section-title">Phase 1 — Baseline Retrieval</div>

<div class="body-text">
<p>The system began with a minimal semantic retrieval architecture:</p>
<ul>
<li>basic chunking</li>
<li>similarity search</li>
<li>GPT-4.1 nano for answer generation</li>
</ul>
<p>
This configuration worked well for <b>direct lookups and quotes</b> but struggled with broader reasoning questions.
Without enough context diversity, the model often hallucinated or produced incomplete answers.
</p>
</div>

<div class="highlight">
Baseline runs established a reference point used to evaluate every later experiment.
</div>
</div>""",
                ),
                unsafe_allow_html=True,
            )
        # -----------------------------
        # Header / hero
        # -----------------------------
        st.markdown(
            _html(
                """<div class="hero-card">
<div class="hero-title">Evaluation Methodology</div>

<div class="hero-subtitle">
    The results on this page summarize performance across <b>70+ RAG experiments</b> and architectural variations.
    Each run answers the same standardized question set and is evaluated using <b>AI-as-a-Judge scoring</b>.
</div>

<div class="hero-subtitle">
    Experiments are grouped into major system evolutions to highlight which retrieval and data strategies
    improved accuracy and which introduced regressions.
</div>

<div class="highlight">
    <b>Scoring Method:</b> AI-as-a-Judge evaluation with both single-pass and two-pass judging pipelines.<br/>
    The most recent evaluations incorporate a <b>gold-standard answer set</b> and a <b>two-phase judge</b>
    to improve reliability and reduce scoring bias.
</div>

<div class="muted">
    Different scoring datasets and judge configurations can be selected using the dropdown at the bottom of the page.
</div>
</div>""",
            ),
            unsafe_allow_html=True,
        )


        # -----------------------------
        # Questions Card
        # -----------------------------
        st.markdown(
            _html(
                """<div class="section-card">
    <div class="hero-title">Test Question Set</div>
    <div class="body-text">
        <ul>
            <li>Find the episode where Dwight says something like 'Bears. Beets. Battlestar Galactica.' What is the context?</li>
            <li>In what episode does Michael burn his foot?</li>
            <li>Summarize season 2 of The Office.</li>
            <li>Why does Dwight dislike Jim? Give 3 reasons with examples.</li>
            <li>List Michael Scott’s serious girlfriends and how the relationships ended.</li>
            <li>What is the teapot letter and why is it important?</li>
            <li>Who is Creed and what is his deal?</li>
        </ul>
    </div>
</div>""",
            ),
            unsafe_allow_html=True,
        )

        rows_local = list(rows_in)

        # If a phase-level summary includes step01..stepNN (from a sweep), hide those by default.
        # The dedicated *_by_step summaries are the clearer place to view those.
        if not is_step_grouped:
            has_steps = any(_step_num(r.group) is not None for r in rows_local)
            if has_steps:
                include_steps = st.checkbox(
                    "Include step01..stepNN sweep groups",
                    value=False,
                    key="reports_include_step_groups",
                    help="Phase summaries can include both high-level phases and sweep steps. The steps are usually clearer in the *_by_step summary.",
                )
                if not include_steps:
                    rows_local = [r for r in rows_local if _step_num(r.group) is None]

        # If the summary is grouped by step, show a compact legend so it’s self-explanatory.
        if is_step_grouped:
            legend_lines: List[str] = []
            for r in sorted(
                rows_local,
                key=lambda rr: (_step_num(rr.group) is None, int(_step_num(rr.group) or 10**9)),
            ):
                cfg = _try_parse_step_cfg_from_run_name(r.best_run_name or r.worst_run_name)
                if not cfg:
                    continue
                friendly = _STEP_FRIENDLY.get(str(r.group or "").strip().lower())
                legend_lines.append(
                    (
                        f"- {r.group} — {friendly}: {cfg['policy']} + {cfg['search']} (k={cfg['k']})"
                        if friendly
                        else f"- {r.group}: {cfg['policy']} + {cfg['search']} (k={cfg['k']})"
                    )
                )
            if legend_lines:
                st.caption("Step legend (parsed from run_name):")
                st.markdown("\n".join(legend_lines))

        groups = sorted([r.group for r in rows_local], key=_group_sort_key)
        selected = st.multiselect(
            "Show groups",
            options=groups,
            default=groups,
            format_func=_group_display,
            key=f"reports_groups_selected_{page.lower()}",
        )
        shown = [r for r in rows_local if r.group in set(selected)]

        baseline_group: Optional[str] = None
        baseline_overall: Optional[int] = None
        if is_step_grouped:
            ordered = sorted(
                [r for r in rows_local if _step_num(r.group) is not None],
                key=lambda rr: int(_step_num(rr.group) or 10**9),
            )
            if ordered:
                baseline_group = ordered[0].group
                baseline_overall = (
                    ordered[0].best_avg_overall
                    if ordered[0].best_avg_overall is not None
                    else ordered[0].worst_avg_overall
                )

        shown_sorted = sorted(shown, key=lambda rr: _group_sort_key(rr.group))

        # Plot dynamically (so naming/sorting/filtering match the table).
        try:
            labels = [_group_display(r.group) for r in shown_sorted]
            best_y = [
                float(r.best_avg_overall) if r.best_avg_overall is not None else float("nan")
                for r in shown_sorted
            ]
            worst_y = [
                float(r.worst_avg_overall) if r.worst_avg_overall is not None else float("nan")
                for r in shown_sorted
            ]
            fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(labels)), 4.4))
            ax.plot(labels, best_y, marker="o", label="best avg_overall")
            ax.plot(labels, worst_y, marker="o", label="worst avg_overall")
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylabel("avg_overall")
            ax.set_title("Best/Worst avg_overall by group")
            ax.legend(loc="lower right")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            st.pyplot(fig, clear_figure=True)
        except Exception:
            pass

        _st_dataframe(
            [
                {
                    "group": r.group,
                    "label": _group_display(r.group),
                    "delta_vs_baseline": (
                        (int(r.best_avg_overall) - int(baseline_overall))
                        if (
                            is_step_grouped
                            and baseline_overall is not None
                            and r.best_avg_overall is not None
                        )
                        else None
                    ),
                    "runs_total": r.runs_total,
                    "runs_with_overall": r.runs_with_overall,
                    "best_avg_overall": r.best_avg_overall,
                    "best_run_name": r.best_run_name,
                    "worst_avg_overall": r.worst_avg_overall,
                    "worst_run_name": r.worst_run_name,
                }
                for r in shown_sorted
            ],
            hide_index=True,
        )
        if is_step_grouped and baseline_group and baseline_overall is not None:
            st.caption(f"Baseline for delta is {baseline_group} (avg_overall={baseline_overall}).")

    if page == "Overview":
        # -----------------------------
        # Simple page styling
        # -----------------------------
        st.markdown(
            _html(
                """
			<style>
				.block-container {
					padding-top: 2rem;
					padding-bottom: 2rem;
					padding-left: 3rem;
					padding-right: 3rem;
					max-width: 1200px;
				}

				.hero-card {
					background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
					padding: 2rem 2rem 1.5rem 2rem;
					border-radius: 18px;
					color: white;
					margin-bottom: 1.5rem;
					border: 1px solid rgba(255,255,255,0.08);
				}

				.hero-title {
					font-size: 2rem;
					font-weight: 700;
					line-height: 1.15;
					margin-bottom: 0.5rem;
				}

				.hero-subtitle {
					font-size: 1.05rem;
					color: #cbd5e1;
					line-height: 1.5;
					margin-bottom: 0.25rem;
				}

				.section-card {
					background: #ffffff;
					padding: 1.25rem 1.25rem 1rem 1.25rem;
					border-radius: 16px;
					border: 1px solid #e2e8f0;
					box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
					height: 100%;
				}

				.section-title {
					font-size: 1.05rem;
					font-weight: 700;
					color: #0f172a;
					margin-bottom: 0.6rem;
				}

				.body-text {
					color: #334155;
					font-size: 0.98rem;
					line-height: 1.6;
				}

				.highlight {
					background: #f8fafc;
					border-left: 4px solid #2563eb;
					padding: 0.9rem 1rem;
					border-radius: 10px;
					color: #1e293b;
					margin-top: 0.75rem;
				}

				.pill {
					display: inline-block;
					padding: 0.35rem 0.7rem;
					border-radius: 999px;
					background: #eff6ff;
					color: #1d4ed8;
					font-size: 0.85rem;
					font-weight: 600;
					margin-right: 0.4rem;
					margin-bottom: 0.4rem;
				}

				.muted {
					color: #64748b;
					font-size: 0.92rem;
				}

                /* Phase cards (gradient + white text) */
                .phase-card {
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    color: #ffffff;
                    border: 1px solid rgba(255,255,255,0.10);
                    box-shadow: none;
                }
                .phase-card .section-title {
                    color: #ffffff;
                }
                .phase-card .body-text {
                    color: #e2e8f0;
                }
                .phase-card .highlight {
                    background: rgba(255,255,255,0.06);
                    border-left: 4px solid #2563eb;
                    color: #ffffff;
                }
			</style>
			"""
            ),
            unsafe_allow_html=True,
        )

        # -----------------------------
        # Header / hero
        # -----------------------------
        st.markdown(
            _html(
                """
            <div class="hero-card">
                <div class="hero-title">Project Overview & Summary</div>
                <div class="hero-subtitle">
                    A controlled AI engineering experiment built to measure, debug, and improve a retrieval-based question-answering system.
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        # -----------------------------
        # Main summary
        # -----------------------------
        # left, right = st.columns([1.4, 1], gap="large")

        # with left:
        st.markdown(
            _html(
                """
            <div class="section-card">
                <div class="section-title">Overview</div>
                <div class="body-text">
                    This application is an <b>Evaluation dashboard + Retrieval Augmented Generation (RAG) Chat Bot</b>.
					
The retrival corpus is generated based strictly on the closed captions of <b>The Office</b> tv show. 
It is a training excercise with the idea of taking an extermely sparse corpus, and building intelligence around it with required citation and a 0.0 temperature.

The evolution flows through basic to advanced AI engineering concepts, with a clear path for iterative improvement. It also requires understanding how to design experiments, analyze results, and how to measure and improve them over time.
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            _html(
                """
            <div class="section-card">
                <div class="section-title">Why This Project Matters</div>
                <div class="body-text">
                    AI makes it easy to build prototypes, but <b>building reliable AI systems requires measurement, evaluation, and iteration</b>.
                    This project demonstrates how to engineer AI systems responsibly by:
                    <ul>
                        <li>designing measurable experiments</li>
                        <li>evaluating answer accuracy and retrieval quality</li>
                        <li>identifying failure modes such as hallucination or missing context</li>
                        <li>improving the system through structured architectural changes</li>
                    </ul>
                    Every change was tested across multiple runs to understand <b>what actually improved results</b>.
                    Changes must be measurable. It takes human judgement to intervene, direct, and orchestrate changes. 
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(
                _html(
                    """
                <div class="section-card">
                    <div class="section-title">What This Demonstrates</div>
                    <div style="margin-top: 0.35rem;">
                        <span class="pill">Evaluation Design</span>
                        <span class="pill">Failure Analysis</span>
                        <span class="pill">Retrieval Tuning</span>
                        <span class="pill">Observability</span>
                        <span class="pill">Cost / Quality Tradeoffs</span>
                    </div>
                    <div class="highlight">
                        The value of this project is not just the chatbot itself — it is the
                        <b>engineering discipline behind building, measuring, and improving a system</b>.
                    </div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                _html(
                    """
                <div class="section-card">
                    <div class="section-title">Key Result</div>
                    <div class="body-text">
                        Through iterative experimentation, the system evolved from a simple prototype into a
                        <b>measurably improved retrieval architecture</b>, with clear visibility into:
                        <ul>
                            <li>which techniques improved accuracy</li>
                            <li>which approaches regressed</li>
                            <li>the cost vs. performance tradeoffs of different configurations</li>
                        </ul>
                    </div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )

        # -----------------------------
        # Relevance section
        # -----------------------------
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            _html(
                """
            <div class="section-card">
                <div class="section-title">Why This Matters for Tonic</div>
                <div class="body-text">
                    The same engineering approach applies directly to real business AI systems, including:
                    <ul>
                        <li>internal knowledge copilots</li>
                        <li>document search and summarization tools</li>
                        <li>support and Slack assistants</li>
                        <li>AI-powered product features</li>
                    </ul>
                    This project demonstrates the ability to <b>lead AI engineering initiatives with measurable results</b>,
                    rather than relying on ad-hoc experimentation.
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        # -----------------------------
        # Experimentation Strategy
        # -----------------------------

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            _html(
                """
        <div class="section-card">
        <div class="section-title">Experimentation Strategy</div>

        <div class="body-text">
        This system was intentionally evolved through <b>measured engineering phases</b>, beginning with a minimal RAG
        baseline and gradually introducing more advanced retrieval and indexing techniques.

        The objective was not simply to improve answers — it was to <b>understand why changes improved or degraded system behavior</b>.
        Each architectural change was evaluated through controlled experiments and tracked across multiple runs.
        </div>

        <div class="highlight">
        The core principle: <b>AI systems should be improved through measurement, not intuition.</b>
        </div>

        </div>
        """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.subheader("Phase Rollup")
        # --- Group rollup chart (same as Timeline -> Groups) ---
        st.caption(
            "How to read this chart: the x-axis is a phase/step group from the selected phase summary. "
            "The two lines show the best and worst `avg_overall` observed among runs in that group. "
            "The Timeline view plots individual runs over time; this chart is the group-level rollup that helps explain "
            "the upper/lower envelope you see in the timeline plots."
        )

        rows_local = list(rows)
        if not is_step_grouped:
            # Match the Timeline page default: hide sweep stepXX groups in phase-level summaries.
            rows_local = [r for r in rows_local if _step_num(r.group) is None]

        baseline_group: Optional[str] = None
        baseline_overall: Optional[int] = None
        if is_step_grouped:
            ordered = sorted(
                [r for r in rows_local if _step_num(r.group) is not None],
                key=lambda rr: int(_step_num(rr.group) or 10**9),
            )
            if ordered:
                baseline_group = ordered[0].group
                baseline_overall = (
                    ordered[0].best_avg_overall
                    if ordered[0].best_avg_overall is not None
                    else ordered[0].worst_avg_overall
                )

        shown_sorted = sorted(rows_local, key=lambda rr: _group_sort_key(rr.group))
        try:
            labels = [_group_display(r.group) for r in shown_sorted]
            best_y = [
                float(r.best_avg_overall) if r.best_avg_overall is not None else float("nan")
                for r in shown_sorted
            ]
            worst_y = [
                float(r.worst_avg_overall) if r.worst_avg_overall is not None else float("nan")
                for r in shown_sorted
            ]
            fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(labels)), 4.4))
            ax.plot(labels, best_y, marker="o", label="best avg_overall")
            ax.plot(labels, worst_y, marker="o", label="worst avg_overall")
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylabel("avg_overall")
            ax.set_title("Best/Worst avg_overall by group")
            ax.legend(loc="lower right")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            st.pyplot(fig, clear_figure=True)
        except Exception:
            st.info("Could not render group rollup chart for this phase summary.")

        if is_step_grouped and baseline_group and baseline_overall is not None:
            st.caption(f"Baseline for delta is {baseline_group} (avg_overall={baseline_overall}).")

        phase1, phase2 = st.columns(2)

        with phase1:

            st.markdown(
                _html(
                    """<div class="section-card phase-card">
<div class="section-title">Phase 1 — Baseline Retrieval</div>

<div class="body-text">
<p>The system began with a minimal semantic retrieval architecture:</p>
<ul>
<li>basic chunking</li>
<li>similarity search</li>
<li>GPT-4.1 nano for answer generation</li>
</ul>
<p>
This configuration worked well for <b>direct lookups and quotes</b> but struggled with broader reasoning questions.
Without enough context diversity, the model often hallucinated or produced incomplete answers.
</p>
</div>

<div class="highlight">
Baseline runs established a reference point used to evaluate every later experiment.
</div>
</div>""",
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            st.markdown(
                _html(
                    """<div class="section-card phase-card">
<div class="section-title">Phase 2 — Retrieval Optimization</div>

<div class="body-text">
<p>The next phase focused on improving retrieval quality and recall. Techniques tested included:</p>
<ul>
<li><b>MMR (Maximum Marginal Relevance)</b> to improve context diversity</li>
<li><b>Higher recall retrieval</b> by increasing the number of documents returned</li>
<li><b>Query Expansion + Rank Fusion</b> to combine results from multiple semantic searches</li>
</ul>
<p>
These experiments significantly improved the system’s ability to locate relevant context across episodes.
</p>
</div>

<div class="highlight">
Query expansion combined with MMR produced the most reliable retrieval improvements.
</div>
</div>""",
                ),
                unsafe_allow_html=True,
            )

        with phase2:

            st.markdown(
                _html(
                    """<div class="section-card phase-card">
<div class="section-title">Phase 3 — Index Enrichment</div>

<div class="body-text">
<p>Improving retrieval alone was not sufficient. The next phase focused on improving the data itself.</p>
<p>New capabilities included:</p>
<ul>
<li>structured <b>metadata</b> (season, episode, characters)</li>
<li>derived narrative summaries generated using <b>map-reduce pipelines</b></li>
<li>multiple indexes containing both <b>dialogue and summarized context</b></li>
</ul>
<p>
These derived knowledge layers allowed the system to reason across episodes instead of relying solely on raw dialogue.
</p>
</div>

<div class="highlight">
Data engineering proved just as important as retrieval tuning.
</div>
</div>""",
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            st.markdown(
                _html(
                    """<div class="section-card phase-card">
<div class="section-title">Phase 4 — Evaluation &amp; Observability</div>

<div class="body-text">
<p>A full evaluation framework was built to compare system configurations objectively. The dashboard tracks:</p>
<ul>
<li>automated scoring of runs</li>
<li>AI-based judging of answer quality</li>
<li>retrieval diagnostics</li>
<li>experiment comparisons over time</li>
</ul>
<p>
This made it possible to identify which architectural changes produced real improvements.
</p>
</div>

<div class="highlight">
Evaluation must separate <b>retrieval quality</b> from <b>answer correctness</b>.
Both must be measured to build reliable AI systems.
</div>
</div>""",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            _html(
                """
<div class="section-card">
    <div class="section-title">Key Takeaways</div>
    <div class="body-text">
        <ul>
            <li>AI systems should be improved through measurement, not intuition.</li>
            <li>Design experiments with clear evaluation criteria and failure modes in mind.</li>
            <li>Start with 2+ phase and prompt engineer individually instead of cramming into one judge</li>
            <li>Preplan and prioritize data first</li>
            <li>While having tons of metrics is nice, I should have also chosen specific data points earlier on to highlight (first and second classes)</li>
            <li>Hand written notes and bookmarks of runs would have been much faster for retrospectives and forensics instead of asking AI to summarize later.</li>
            <li>As the model grew, generations gave better insights over individual tuning. Having the ability to pass different routing strategies, judges, and questions against older models would have helped better express performance gains.</li>
        </ul>
    </div>
</div>
"""
            ),
            unsafe_allow_html=True,
        )

        # # -----------------------------
        # # Optional placeholder area for charts / timeline
        # # -----------------------------
        # st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        # st.subheader("Experiment Story")
        # st.caption(
        # 	"This section is a good place to add your stage timeline chart, best run score, and a concise summary of what worked vs. what failed."
        # )

        # placeholder_col1, placeholder_col2, placeholder_col3 = st.columns(3)

        # with placeholder_col1:
        # 	st.metric("Best Run Score", "75", "+8 vs baseline")

        # with placeholder_col2:
        # 	st.metric("Biggest Win", "Higher Recall", "Increasing k improved coverage")

        # with placeholder_col3:
        # 	st.metric("Biggest Regression", "Scene Chunking", "Narrative cohesion dropped")

        # st.info(
        # 	"Next step: replace these placeholder metrics with your real stage chart, failure mix, and cost-vs-quality visual."
        # )
        # return

    if page == "Structure":
        p1 = REPO_ROOT / "system_arch.jpg"
        p2 = REPO_ROOT / "logos.jpg"

        if p1.exists():
            _st_image(str(p1))
        else:
            st.info("Missing image: system_arch.jpg")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        if p2.exists():
            _st_image(str(p2))
        else:
            st.info("Missing image: logos.jpg")
        return

    if page == "Analysis":
        # Order requested: Groups -> Timeline -> selector.
        _render_groups_section(rows)

        # Use the same run subset for the charts as the timeline plots below.
        require_llm = True
        max_points = int(
            st.slider("Max runs plotted", min_value=10, max_value=200, value=120, step=10)
        )

        trend_rows = _load_run_rows_cached(
            runs_dir=DEFAULT_RUNS_DIR.as_posix(),
            scored_dir=scored_dir.as_posix(),
            require_llm=require_llm,
        )
        if max_points and len(trend_rows) > max_points:
            trend_rows = trend_rows[-max_points:]

        selected_run_ids = [r.run_id for r in trend_rows]

        # Build a run_id -> run_obj map for fast per-run access.
        run_entries = _load_run_objs_cached(DEFAULT_RUNS_DIR.as_posix(), max_files=800)
        run_obj_by_id: Dict[str, Dict[str, Any]] = {}
        for e in run_entries:
            if not isinstance(e, dict):
                continue
            rid = _run_id_from_run_entry(e)
            obj = e.get("obj")
            if isinstance(rid, str) and rid and isinstance(obj, dict):
                run_obj_by_id[rid] = obj

        scored_summary_by_id = _load_scored_summary_cached(scored_dir.as_posix())

        baseline_llm_model, baseline_k = _baseline_signature_from_runs(
            [run_obj_by_id[rid] for rid in selected_run_ids if rid in run_obj_by_id]
        )

        st.subheader("Experiment Timeline")
        st.caption("Stage-level progression using real scored runs (avg judge overall).")

        stage_rows = _stage_timeline_rows_for_runs(
            run_ids=selected_run_ids,
            run_obj_by_id=run_obj_by_id,
            scored_summary_by_id=scored_summary_by_id,
            baseline_llm_model=baseline_llm_model,
            baseline_k=baseline_k,
        )
        if stage_rows and _plotly_or_warning():
            stage_ordered = [r.get("stage") for r in stage_rows if r.get("stage")]
            fig = px.line(
                stage_rows,
                x="stage",
                y="avg_score",
                markers=True,
                title=None,
                category_orders={"stage": stage_ordered},
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                xaxis_title="Stage",
                yaxis_title="Avg overall score (0..100)",
            )
            st.plotly_chart(fig, use_container_width=True)
            _st_dataframe(stage_rows, hide_index=True)
        elif stage_rows:
            _st_dataframe(stage_rows, hide_index=True)
        else:
            st.info("No stage timeline data available for the selected run set.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        st.subheader("Impact of techniques")
        st.caption(
            "Technique impact (enabled vs disabled). Interpretation: observational (not causal), but highlights which levers correlate with gains."
        )
        tech_rows = _technique_impact_rows_for_runs(
            run_ids=selected_run_ids,
            run_obj_by_id=run_obj_by_id,
            scored_summary_by_id=scored_summary_by_id,
            baseline_llm_model=baseline_llm_model,
            baseline_k=baseline_k,
        )
        if tech_rows and _plotly_or_warning():
            tech_ordered = [r.get("technique") for r in tech_rows if r.get("technique")]
            fig = px.bar(
                tech_rows,
                x="delta_avg_overall",
                y="technique",
                orientation="h",
                title=None,
                hover_data=["enabled_n", "disabled_n", "enabled_avg", "disabled_avg"],
                category_orders={"technique": tech_ordered},
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                xaxis_title="Δ avg overall (enabled - disabled)",
                yaxis_title="Technique",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif tech_rows:
            _st_dataframe(tech_rows, hide_index=True)
        else:
            st.info("Not enough data to compute technique impact for the selected run set.")

        st.subheader("Failure mode mix")
        st.caption(
            "Interpretation: the largest slice is the highest-leverage failure mode to tackle next. "
            "Computed from the currently plotted runs using the latest scored artifacts."
        )
        fm_rows = _failure_mode_counts_for_runs(
            run_ids=selected_run_ids,
            run_obj_by_id=run_obj_by_id,
            scored_dir=scored_dir.as_posix(),
        )
        if fm_rows and _plotly_or_warning():
            fig = px.pie(
                fm_rows,
                values="count",
                names="failure_mode",
                title=None,
                hole=0.0,
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        elif fm_rows:
            _st_dataframe(fm_rows, hide_index=True)
        else:
            st.info("No failure mode data available for the selected run set.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        st.subheader("Failure modes by question type")
        st.caption(
            "Interpretation: some retrieval/answering techniques help specific question families. "
            "Computed from the same run set using per-case judges + heuristics."
        )
        fmqt_rows = _failure_modes_by_question_type_for_runs(
            run_ids=selected_run_ids,
            run_obj_by_id=run_obj_by_id,
            scored_dir=scored_dir.as_posix(),
        )
        if fmqt_rows and _plotly_or_warning():
            fig = px.bar(
                fmqt_rows,
                x="question_type",
                y="count",
                color="failure_mode",
                title=None,
                barmode="stack",
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                xaxis_title="Question type",
                yaxis_title="Count",
                legend_title_text="failure_mode",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif fmqt_rows:
            _st_dataframe(fmqt_rows, hide_index=True)
        else:
            st.info("No per-type failure mode data available for the selected run set.")

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        st.subheader("Cost vs quality (runs)")
        st.caption(
            "Each point is one scored run. X-axis is average total tokens per answer; Y-axis is avg_overall. "
            "Color groups runs into the experimentation stage based on config/run_name heuristics."
        )
        cq_rows = _cost_quality_rows_for_runs(
            run_ids=selected_run_ids,
            run_obj_by_id=run_obj_by_id,
            scored_summary_by_id=scored_summary_by_id,
            baseline_llm_model=baseline_llm_model,
            baseline_k=baseline_k,
        )
        if cq_rows and _plotly_or_warning():
            fig = px.scatter(
                cq_rows,
                x="avg_total_tokens",
                y="avg_overall",
                color="stage",
                custom_data=[
                    "run_name",
                    "run_id",
                    "retrieval_policy",
                    "search_type",
                    "k",
                ],
                title=None,
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "stage=%{fullData.name}<br>"
                    "avg_overall=%{y}<br>"
                    "avg_total_tokens=%{x:.0f}<br>"
                    "policy=%{customdata[2]} | search=%{customdata[3]} | k=%{customdata[4]}<br>"
                    "run_id=%{customdata[1]}<extra></extra>"
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                xaxis_title="Avg total tokens (proxy)",
                yaxis_title="Avg overall score",
                legend_title_text="stage",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif cq_rows:
            _st_dataframe(cq_rows, hide_index=True)
        else:
            st.info("No scored token/quality data available for the selected run set.")

        if cq_rows:
            st.caption(
                "Summary table (sortable): use `score_per_1k_tokens` to find strong demo runs (high score, low cost)."
            )
            cq_table: List[Dict[str, Any]] = []
            for r in cq_rows:
                tokens = _safe_float(r.get("avg_total_tokens"))
                score = _safe_float(r.get("avg_overall"))
                score_per_1k = (float(score) * 1000.0 / float(tokens)) if tokens and score else None
                cq_table.append(
                    {
                        "stage": r.get("stage"),
                        "avg_overall": r.get("avg_overall"),
                        "avg_total_tokens": (round(float(tokens), 1) if tokens is not None else None),
                        "score_per_1k_tokens": (round(float(score_per_1k), 2) if score_per_1k is not None else None),
                        "retrieval_policy": r.get("retrieval_policy"),
                        "search_type": r.get("search_type"),
                        "k": r.get("k"),
                        "run_name": r.get("run_name"),
                        "run_id": r.get("run_id"),
                    }
                )
            cq_table_sorted = sorted(
                cq_table,
                key=lambda rr: (
                    -(float(rr.get("score_per_1k_tokens")) if rr.get("score_per_1k_tokens") is not None else -1.0),
                    -(float(rr.get("avg_overall")) if rr.get("avg_overall") is not None else -1.0),
                ),
            )
            _st_dataframe(cq_table_sorted, hide_index=True)

        st.divider()

        st.subheader("Timeline")
        st.caption("Computed from scored, LLM-enabled runs (retrieval-only runs are ignored).")

        fig1 = _plot_score_trend(trend_rows, title="Avg overall over time")
        st.pyplot(fig1, clear_figure=True)
        fig2 = _plot_score_hist(trend_rows, title="Avg overall distribution")
        st.pyplot(fig2, clear_figure=True)

        st.divider()
        st.selectbox(
            "Phase summary file",
            options=options,
            index=selected_index,
            key="reports_phase_summary_file",
            help="Switch between phase-level and by-step summaries, and between different scoring sets.",
        )
        return

    if page == "Artifacts":
        st.subheader("Artifacts")
        st.write(f"Summary JSON: `{phase_json.as_posix()}`")
        with st.expander("Show precomputed PNG"):
            png = phase_json.with_suffix(".png")
            if png.exists():
                _st_image(str(png), caption=png.name)
            else:
                st.info("No PNG found next to the JSON.")
        with st.expander("Show raw CSV rows"):
            csv_path = phase_json.with_suffix(".csv")
            if not csv_path.exists():
                st.info("No CSV found next to the JSON.")
            else:
                _st_dataframe(_read_csv_rows(csv_path), hide_index=True)
        with st.expander("Show raw JSON"):
            _st_dataframe(rows, hide_index=True)
        return

    # (Precomputed PNG/CSV are on the Artifacts sub-page.)
