from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from apps.dashboard.data.loaders import load_run_entries, load_scored_obj, run_id_from_entry
from apps.dashboard.styles.css import inject_dashboard_css


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def render_run_explorer() -> None:
    """Lightweight run explorer.

    This existed as a mode in the app shell but the original prototype file was empty.
    Keep it minimal: browse run logs and (optionally) their scored artifacts.
    """

    inject_dashboard_css(layout="default")

    st.header("Run Explorer")

    runs_dir = st.text_input("Runs dir", value="experiments/runs")
    scored_dir = st.text_input("Scored dir", value="experiments/scored_runs_two_pass")

    entries = load_run_entries(runs_dir=runs_dir, max_files=1200)
    if not entries:
        st.info("No run JSON logs found.")
        return

    # Build display labels.
    labels: List[str] = []
    id_to_entry: Dict[str, Any] = {}
    for e in entries:
        rid = run_id_from_entry(e)
        run_name = str(_safe_get(e.obj, "run.run_name", "") or "")
        display = f"{rid} — {run_name}" if run_name else rid
        labels.append(display)
        id_to_entry[display] = e

    picked = st.selectbox("Run", options=labels, index=0)
    entry = id_to_entry.get(picked)
    if not entry:
        return

    rid = run_id_from_entry(entry)
    obj = entry.obj

    run_meta = obj.get("run") if isinstance(obj.get("run"), dict) else {}
    cfg = obj.get("config") if isinstance(obj.get("config"), dict) else {}
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("cases", len(obj.get("cases") or []))
    with c2:
        st.metric("policy", str(_safe_get(cfg, "retrieval.policy", "") or ""))
    with c3:
        st.metric("k", str(_safe_get(cfg, "retrieval.k", "") or ""))

    with st.expander("Run metadata"):
        st.json(run_meta)

    with st.expander("Config"):
        st.json(cfg)

    with st.expander("Summary"):
        st.json(summary)

    scored_obj = load_scored_obj(scored_dir=scored_dir, run_id=rid)
    if scored_obj:
        with st.expander("Scored summary"):
            st.json(scored_obj.get("score_summary") or {})

    cases = obj.get("cases")
    if isinstance(cases, list) and cases:
        rows: List[Dict[str, Any]] = []
        for c in cases:
            if not isinstance(c, dict):
                continue
            rows.append(
                {
                    "case_id": c.get("case_id"),
                    "question": str(c.get("question") or "")[:160],
                    "answer": str(_safe_get(c, "answer.text", "") or "")[:160],
                }
            )
        st.subheader("Cases")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Raw run JSON"):
            st.json(obj)
