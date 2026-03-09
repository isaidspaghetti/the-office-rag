from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.dashboard.shared import REPO_ROOT, _st_image
from apps.dashboard.components.html import spacer
from apps.dashboard.styles.css import inject_dashboard_css


def render_structure_page() -> None:
    inject_dashboard_css(layout="default")

    p1 = Path(REPO_ROOT) / "system_arch.jpg"
    p2 = Path(REPO_ROOT) / "logos.jpg"

    if p1.exists():
        _st_image(str(p1))
    else:
        st.info("Missing image: system_arch.jpg")

    spacer(rem=1.0)

    if p2.exists():
        _st_image(str(p2))
    else:
        st.info("Missing image: logos.jpg")
