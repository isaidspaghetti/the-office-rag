from __future__ import annotations

from pathlib import Path
from typing import Any, List

import streamlit as st

from apps.dashboard.shared import PhaseRow, _read_csv_rows, _st_dataframe, _st_image
from apps.dashboard.styles.css import inject_dashboard_css


def render_artifacts_page(*, phase_json: Path, phase_rows: List[PhaseRow]) -> None:
    inject_dashboard_css(layout="default")

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
        # Streamlit can render dataclasses as-is, but keep consistent with prior view.
        _st_dataframe(phase_rows, hide_index=True)
