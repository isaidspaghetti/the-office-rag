from __future__ import annotations

from typing import Dict

import streamlit as st
from dotenv import load_dotenv

from apps.dashboard.pages.chat_debug import render_chat_debug
from apps.dashboard.pages.reports import render_summary
from apps.dashboard.pages.run_explorer import render_run_explorer
from apps.dashboard.shared import _query_params


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="RAG Evaluation & Observability Dashboard", layout="wide")
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
