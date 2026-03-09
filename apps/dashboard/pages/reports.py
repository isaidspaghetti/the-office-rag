from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from apps.dashboard.pages.analysis import render_analysis_page
from apps.dashboard.pages.artifacts import render_artifacts_page
from apps.dashboard.pages.overview import render_overview_page
from apps.dashboard.pages.structure import render_structure_page
from apps.dashboard.shared import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
    PhaseRow,
    _env_or_secret,
    _resolve_under_repo,
    _load_phase_summary,
    _phase_summary_is_step_grouped,
    _pick_latest_phase_summary_json,
    _read_phase_summary_obj,
    _step_num,
)


def render_summary() -> None:
    """Reports mode (Overview/Structure/Analysis/Artifacts).

    This function intentionally stays as a thin router so the four subpages can
    evolve independently without reintroducing the original monolithic file.
    """

    st.header("Reports")

    def _has_scored_files(p: Path) -> bool:
        return p.exists() and p.is_dir() and any(p.glob("*.scored.json"))

    def _pick_scored_dir() -> Path:
        override = (
            _env_or_secret("REPORTS_SCORED_DIR")
            or _env_or_secret("SCORED_DIR")
            or _env_or_secret("SCORED_RUNS_DIR")
        )
        if override:
            p = _resolve_under_repo(str(override))
            if _has_scored_files(p):
                return p
            st.warning(
                f"Scored dir override is set but contains no '*.scored.json' files: {override}"
            )

        canonical = REPO_ROOT / "experiments" / "scored_runs_two_pass"
        if _has_scored_files(canonical):
            return canonical

        # Auto-discover latest scored directory under experiments/.
        candidates = []
        for p in sorted((REPO_ROOT / "experiments").glob("scored_runs_two_pass*")):
            if _has_scored_files(p):
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda pp: pp.stat().st_mtime)
            return candidates[-1]

        return canonical

    scored_dir = _pick_scored_dir()

    # Back-compat: older sessions had now-removed subpages.
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
        page = st.radio(
            "",
            options=["Overview", "Structure", "Analysis", "Artifacts"],
            horizontal=True,
            key="reports_subpage",
        )

    # Phase summary selection.
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
        phase_json = name_to_path.get(selected_name, phase_json_default)

    phase_obj = _read_phase_summary_obj(phase_json)
    is_step_grouped = _phase_summary_is_step_grouped(phase_obj)

    rows: List[PhaseRow] = _load_phase_summary(phase_json)
    if not rows:
        st.warning("Phase summary JSON exists but has no rows.")
        return

    # --- Group naming + sorting ---
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

    if page == "Overview":
        render_overview_page(
            phase_rows=rows,
            is_step_grouped=is_step_grouped,
            group_display=_group_display,
            group_sort_key=_group_sort_key,
        )
        return

    if page == "Structure":
        render_structure_page()
        return

    if page == "Analysis":
        render_analysis_page(
            phase_rows=rows,
            is_step_grouped=is_step_grouped,
            group_display=_group_display,
            group_sort_key=_group_sort_key,
            scored_dir=scored_dir.as_posix(),
        )

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
        render_artifacts_page(phase_json=phase_json, phase_rows=rows)
        return
