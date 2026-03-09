from __future__ import annotations

import textwrap
from typing import Optional

import streamlit as st


def normalize_html(md: str) -> str:
    """Normalize HTML-in-Markdown strings for Streamlit.

    Streamlit renders HTML via its Markdown pipeline; if any line begins with a
    tab or 4+ leading spaces, Markdown can treat it as an indented code block
    and display the HTML literally. This helper removes leading indentation on
    every line while keeping line breaks.
    """

    s = textwrap.dedent(md).strip("\n")
    lines = [ln.lstrip(" \t") for ln in s.splitlines()]
    return "\n".join(lines).strip()


def markdown_html(md: str) -> None:
    """Render HTML via st.markdown with consistent normalization."""

    st.markdown(normalize_html(md), unsafe_allow_html=True)


def spacer(*, rem: float = 1.0, key: Optional[str] = None) -> None:
    """Insert vertical whitespace matching the legacy inline HTML spacing."""

    # key is unused; it exists to make call sites self-documenting.
    _ = key
    st.markdown(f"<div style='height: {float(rem)}rem;'></div>", unsafe_allow_html=True)
