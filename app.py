from __future__ import annotations

from typing import Dict

import streamlit as st
from dotenv import load_dotenv

from apps.dashboard.pages.chat_debug import render_chat_debug
from apps.dashboard.pages.reports import render_summary
from apps.dashboard.pages.run_explorer import render_run_explorer
from apps.dashboard.components.html import markdown_html
from apps.dashboard.styles.css import inject_dashboard_css
from apps.dashboard.shared import _query_params


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="RAG Evaluation & Observability Dashboard", layout="wide")
    inject_dashboard_css(layout="default")
    st.title("RAG Evaluation Dashboard")

    qp = _query_params()
    default_mode_raw = str(qp.get("mode") or "summary").strip().lower()

    # Stable internal mode keys (URL-friendly) with human labels.
    mode_labels: Dict[str, str] = {
        "summary": "Reports",
        "chat_debug": "Chat & Debug",
        "run_explorer": "Run Explorer",
    }

    # Back-compat for older/accidental query param values.
    legacy_map = {
        "reports": "summary",
        "report": "summary",
        "summary": "summary",
        "chat": "chat_debug",
        "chat_debug": "chat_debug",
        "chat & debug": "chat_debug",
        "chat_and_debug": "chat_debug",
        "run explorer": "run_explorer",
        "run_explorer": "run_explorer",
        "explorer": "run_explorer",
    }
    default_mode = legacy_map.get(default_mode_raw, "summary")
    if default_mode not in mode_labels:
        default_mode = "summary"

    mode = st.sidebar.radio(
        "Mode",
        options=list(mode_labels.keys()),
        index=list(mode_labels.keys()).index(default_mode),
        format_func=lambda k: mode_labels.get(str(k), str(k)),
    )

    # First-visit hint to guide people to the interactive chatbot.
    if not bool(st.session_state.get("first_visit_chat_tip_shown")):
        st.session_state["first_visit_chat_tip_shown"] = True
        if str(mode) != "chat_debug":
            markdown_html(
                """
<style>
    .floating-chat-tip {
        position: sticky;
        top: 0.75rem;
        z-index: 10000;
        max-width: 520px;
        margin-bottom: 0.75rem;
    }

    .floating-chat-tip .section-title {
        margin-bottom: 0.25rem;
    }
</style>

<div class="floating-chat-tip card-base card-accent">
    <div class="section-title">💬 Tip</div>
    <div class="body-text">Open the sidebar and select <b>Chat &amp; Debug</b> to start playing around.</div>
</div>
"""
            )

    # Keep URL query param in sync for easy sharing/reloads.
    try:
        if str(qp.get("mode") or "").strip().lower() != str(mode).strip().lower():
            try:
                st.query_params["mode"] = str(mode)  # type: ignore[attr-defined]
            except Exception:
                st.experimental_set_query_params(mode=str(mode))
    except Exception:
        pass

    try:
        if mode == "summary":
            render_summary()
        elif mode == "run_explorer":
            render_run_explorer()
        else:
            render_chat_debug()
    except Exception as e:
        st.error("Dashboard error")
        st.exception(e)


if __name__ == "__main__":
    main()
