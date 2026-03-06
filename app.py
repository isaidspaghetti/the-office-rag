"""Hosted Streamlit entrypoint.

Routes between:
- Summary / Executive dashboards
- Chat + Debug dashboards

Usage:
  ./venv/bin/streamlit run app.py

Deep links:
  /?mode=summary
  /?mode=chat_debug
"""

from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st


def _get_query_mode() -> str:
    # Streamlit query params can contain lists; normalize to a simple string.
    try:
        raw = st.query_params.get("mode")
    except Exception:
        raw = None

    if isinstance(raw, list):
        raw = raw[0] if raw else None

    mode = str(raw or "").strip().lower() or "summary"
    return mode


def _set_query_mode(mode: str) -> None:
    try:
        st.query_params["mode"] = str(mode)
    except Exception:
        # Best-effort; app still works without query param persistence.
        pass


def main() -> None:
    st.set_page_config(page_title="RAG Dashboards", layout="wide", initial_sidebar_state="collapsed")

    modes: Dict[str, Tuple[str, str]] = {
        "summary": ("Summary", "Summary / executive views"),
        "chat_debug": ("Chat & Debug", "Chat playground + run explorer + diagnostics"),
    }

    mode = _get_query_mode()
    if mode not in modes:
        mode = "summary"

    # Sticky mode selector (separate from per-dashboard sticky page selectors).
    st.markdown(
        """
<style>
  .sticky-mode {
      position: sticky;
      top: 0;
      z-index: 1000;
      background: var(--background-color, #141414);
      padding: 0.4rem 0 0.2rem 0;
      margin: 0;
      border-bottom: 1px solid rgba(49, 51, 63, 0.2);
      backdrop-filter: blur(6px);
  }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# RAGE & OD")
    st.caption("Rage Evaluation & Observability Dashboard")
    labels = [modes["summary"][0], modes["chat_debug"][0]]
    default_label = modes[mode][0]

    if hasattr(st, "segmented_control"):
        selected_label = st.segmented_control("MODE", options=labels, default=default_label)
    else:
        selected_label = st.radio("MODE", options=labels, index=labels.index(default_label), horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    selected_mode = "summary" if selected_label == modes["summary"][0] else "chat_debug"
    if selected_mode != mode:
        _set_query_mode(selected_mode)
        st.rerun()

    # Delegate rendering to the chosen dashboard.
    if selected_mode == "summary":
        from rag_experiment_story_dashboard import main as story_main

        story_main(set_page_config=False)
        return

    from apps.rag_runs_dashboard import main as runs_main

    runs_main(set_page_config=False)


if __name__ == "__main__":
    main()
