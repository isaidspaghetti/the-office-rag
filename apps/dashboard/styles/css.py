from __future__ import annotations

from typing import Literal

import streamlit as st

from apps.dashboard.components.html import normalize_html


LayoutMode = Literal["default", "overview"]


_BASE_CSS = normalize_html(
    """
/* Shared dashboard UI primitives.
   NOTE: Colors/typography intentionally match the legacy prototype to preserve UX. */

.hero-card {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  padding: 2rem 2rem 1.5rem 2rem;
  border-radius: 18px;
  color: white;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(255,255,255,0.08);
}

/* Variant used in Analysis methodology hero */
.hero-card--blue {
  background: linear-gradient(135deg, #0f172a 0%, #1e40af 45%, #2563eb 100%);
  color: #ffffff;
  padding: 1.5rem 1.75rem;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 6px 24px rgba(15, 23, 42, 0.12);
  margin-bottom: 1.25rem;
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
"""
)


_OVERVIEW_LAYOUT_CSS = normalize_html(
    """
.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
  padding-left: 3rem;
  padding-right: 3rem;
  max-width: 1200px;
}
"""
)


def inject_dashboard_css(*, layout: LayoutMode = "default") -> None:
    """Inject the dashboard CSS.

    This centralizes all HTML/CSS styling in one module. It is safe to call on
    every rerun; injection is de-duplicated via session_state.
    """

    # Streamlit reruns rebuild the app output; emitting CSS on every run is the
    # simplest reliable approach and still keeps styling centralized.
    st.markdown(f"<style>{_BASE_CSS}</style>", unsafe_allow_html=True)
    if layout == "overview":
      st.markdown(f"<style>{_OVERVIEW_LAYOUT_CSS}</style>", unsafe_allow_html=True)
