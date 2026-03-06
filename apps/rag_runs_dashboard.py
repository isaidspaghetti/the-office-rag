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


def truncate(s: Any, n: int = 140) -> str:
    t = str(s or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    if len(t) <= n:
        return t
    return t[:n] + "…"


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

    total_tokens = [safe_float(r.get("total_tokens")) for r in rows if r.get("total_tokens") is not None]
    latency = [
        safe_float((safe_get(r, "retrieval_latency_ms", None) or 0)) + safe_float((safe_get(r, "answer_latency_ms", None) or 0))
        for r in rows
        if (r.get("retrieval_latency_ms") is not None or r.get("answer_latency_ms") is not None)
    ]

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
        "avg_overall": _mean([x for x in judge_overalls if x is not None]),
        "avg_correctness": _mean([x for x in judge_correctness if x is not None]),
        "avg_groundedness": _mean([x for x in judge_groundedness if x is not None]),
        "idk_rate": idk_rate,
        "citation_rate": citation_rate,
        "avg_total_tokens": _mean([x for x in total_tokens if x is not None]),
        "avg_latency_ms": _mean([x for x in latency if x is not None]),
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
                    show_full = st.checkbox("Show full file (can be large)", value=False, key=f"show_full_{resolved}")
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

        diag_view = st.checkbox("Show routing/diagnostics JSON", value=False)
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
    if st.button("Export current view", type="secondary"):
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
def answer_question(question: str) -> Dict[str, Any]:
    """Live chat hook.

    Expected return shape (suggested):
      {
        "answer": "...",
        "retrieval": {"results": [...]},
        "answer_meta": {"usage": {...}, "latency_ms": ...},
      }
    """
    raise NotImplementedError(
        "Live chat is enabled, but answer_question() is not implemented. "
        "Wire this up to your RAG client (e.g., a function that runs retrieval + LLM) and return a dict."
    )


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
    for msg in st.session_state.chat_messages:
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
                render_flight_recorder(run, case, scored_case=None, docs_root=docs_root, export_dir=export_dir)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "flight": {"run": run, "case": case},
                }
            )
        except NotImplementedError as e:
            st.error(str(e))
            st.caption(
                "Implement answer_question(question: str) -> dict in apps/rag_runs_dashboard.py "
                "and return retrieval results + answer text."
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
    render_flight_recorder(run_obj, case_obj, scored_case_obj, docs_root=docs_root, export_dir=export_dir)


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
        scored_obj = load_scored_run(scored_path) if scored_path else None
        scoring_idx = index_scored_cases(scored_obj) if scored_obj and not scored_obj.get("_error") else {}
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
        render_flight_recorder(run_obj_a, case_obj_a, scored_a, docs_root=docs_root, export_dir=export_dir)
    with st.expander("Inspect Run B (Flight Recorder)", expanded=False):
        render_flight_recorder(run_obj_b, case_obj_b, scored_b, docs_root=docs_root, export_dir=export_dir)


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
        scored_obj = load_scored_run(scored_path) if scored_path else None
        scoring_idx = index_scored_cases(scored_obj) if scored_obj and not scored_obj.get("_error") else {}
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

    selected_runs = st.multiselect(
        "Select runs",
        options=run_files,
        default=run_files[: min(10, len(run_files))],
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
        scored_obj = load_scored_run(scored_path) if scored_path else None
        scoring_idx = index_scored_cases(scored_obj) if scored_obj and not scored_obj.get("_error") else {}
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

    # Score over time.
    st.subheader("Score over time")
    times: List[Tuple[str, float]] = []
    for r in leaderboard:
        t = str(r.get("created_at_utc") or "")
        s = safe_float(r.get("avg_overall"))
        if t and s is not None:
            times.append((t, float(s)))
    # Sort by timestamp string (ISO-like).
    times.sort(key=lambda x: x[0])
    if times:
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.plot([t for t, _ in times], [s for _, s in times], marker="o")
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([t for t, _ in times], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("avg_overall")
        ax.set_title("Average overall score over time")
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No avg_overall values found (are scored runs present?)")

    # Cost vs quality scatter.
    st.subheader("Cost vs quality")
    pts: List[Tuple[float, float, str]] = []
    for r in leaderboard:
        s = safe_float(r.get("avg_overall"))
        t = safe_float(r.get("avg_total_tokens"))
        if s is None or t is None:
            continue
        lbl = truncate(str(r.get("run_name") or ""), 18)
        pts.append((float(t), float(s), lbl))
    plot_scatter(pts, title="Tokens vs score", x_label="avg_total_tokens", y_label="avg_overall")

    # Efficiency frontier.
    st.subheader("Efficiency frontier (Pareto optimal)")
    frontier = pareto_frontier(points_for_frontier) if points_for_frontier else []
    if frontier:
        st.dataframe(
            [{"run_id": rid, "avg_overall": s, "avg_total_tokens": t} for rid, s, t in frontier],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Not enough data for frontier (need avg_overall + avg_total_tokens).")

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
# App scaffold
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # Sidebar configuration.
    st.sidebar.header("Config")
    default_root = Path.cwd()
    project_root = Path(
        st.sidebar.text_input("Project root", value=str(default_root))
    ).expanduser()

    runs_dir = Path(st.sidebar.text_input("Runs dir", value=str(project_root / "experiments" / "runs"))).expanduser()
    scored_dir = Path(
        st.sidebar.text_input("Scored runs dir", value=str(project_root / "experiments" / "scored_runs"))
    ).expanduser()
    docs_root = Path(
        st.sidebar.text_input("Normalized docs root", value=str(project_root))
    ).expanduser()

    enable_live_chat = st.sidebar.toggle("Enable Live Chat", value=False)
    diagnostics_mode = st.sidebar.radio("Diagnostics view", options=["Per-run", "Per-case"], index=1)

    exports_dir = project_root / "experiments" / "exports"
    st.sidebar.caption(f"Exports: {exports_dir}")

    page = st.sidebar.radio(
        "Page",
        options=[
            "Chat Playground",
            "Run Explorer",
            "Experiment Comparison",
            "Diagnostics",
            "Executive Insights",
        ],
        index=1,
    )

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