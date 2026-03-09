from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st

try:  # Plotly (optional, but preferred for interactive charts)
    import plotly.express as px  # type: ignore

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    px = None  # type: ignore
    _HAS_PLOTLY = False

from apps.dashboard.components.cards import section_card
from apps.dashboard.components.hero import hero_card
from apps.dashboard.components.html import spacer
from apps.dashboard.data.analysis import (
    cost_quality_rows_for_runs,
    failure_mode_counts_for_runs,
    failure_modes_by_question_type_for_runs,
    stage_timeline_rows_for_runs,
    technique_impact_rows_for_runs,
)
from apps.dashboard.data.stage_classifier import baseline_signature_from_runs
from apps.dashboard.data.loaders import load_run_entries, load_scored_summary, run_id_from_entry
from apps.dashboard.shared import (
    DEFAULT_RUNS_DIR,
    PhaseRow,
    RunRow,
    _load_run_rows_cached,
    _plot_score_hist,
    _plot_score_trend,
    _step_num,
    _st_dataframe,
    _try_parse_step_cfg_from_run_name,
)
from apps.dashboard.styles.css import inject_dashboard_css


def _plotly_or_warning() -> bool:
    if _HAS_PLOTLY:
        return True
    st.warning(
        "Plotly is not installed; charts will fall back to tables. Install with: pip install plotly"
    )
    return False


def render_analysis_page(
    *,
    phase_rows: List[PhaseRow],
    is_step_grouped: bool,
    group_display: Any,
    group_sort_key: Any,
    scored_dir: str,
) -> None:
    """Render the Analysis subpage.

    `group_display` and `group_sort_key` are passed in from the Reports wrapper to keep
    group naming consistent across pages.
    """

    inject_dashboard_css(layout="default")

    # -----------------------------
    # Methodology / groups rollup section
    # -----------------------------
    hero_card(
        title="Evaluation Methodology",
        subtitles_html=[
            "The results on this page summarize performance across <b>70+ RAG experiments</b> and architectural variations.",
            "Each run answers the same standardized question set and is evaluated using <b>AI-as-a-Judge scoring</b>.",
            "Experiments are grouped into major system evolutions to highlight which retrieval and data strategies improved accuracy and which introduced regressions.",
        ],
        highlight_html=(
            "<b>Scoring Method:</b> AI-as-a-Judge evaluation with both single-pass and two-pass judging pipelines.<br/>"
            "The most recent evaluations incorporate a <b>gold-standard answer set</b> and a <b>two-phase judge</b> "
            "to improve reliability and reduce scoring bias."
        ),
        footer_html=(
            "<div class=\"muted\">Different scoring datasets and judge configurations can be selected using the dropdown at the bottom of the page.</div>"
        ),
        tone="blue",
    )

    section_card(
        title="Test Question Set",
        body_html=(
            "<ul>"
            "<li>Find the episode where Dwight says something like 'Bears. Beets. Battlestar Galactica.' What is the context?</li>"
            "<li>In what episode does Michael burn his foot?</li>"
            "<li>Summarize season 2 of The Office.</li>"
            "<li>Why does Dwight dislike Jim? Give 3 reasons with examples.</li>"
            "<li>List Michael Scott’s serious girlfriends and how the relationships ended.</li>"
            "<li>What is the teapot letter and why is it important?</li>"
            "<li>Who is Creed and what is his deal?</li>"
            "</ul>"
        ),
        title_class="hero-title",
    )

    rows_local = list(phase_rows)

    if not is_step_grouped:
        has_steps = any(_step_num(r.group) is not None for r in rows_local)
        if has_steps:
            include_steps = st.checkbox(
                "Include step01..stepNN sweep groups",
                value=False,
                key="reports_include_step_groups",
                help=(
                    "Phase summaries can include both high-level phases and sweep steps. "
                    "The steps are usually clearer in the *_by_step summary."
                ),
            )
            if not include_steps:
                rows_local = [r for r in rows_local if _step_num(r.group) is None]

    if is_step_grouped:
        legend_lines: List[str] = []
        for r in sorted(
            rows_local,
            key=lambda rr: (_step_num(rr.group) is None, int(_step_num(rr.group) or 10**9)),
        ):
            cfg = _try_parse_step_cfg_from_run_name(r.best_run_name or r.worst_run_name)
            if not cfg:
                continue
            legend_lines.append(
                f"- {r.group}: {cfg['policy']} + {cfg['search']} (k={cfg['k']})"
            )
        if legend_lines:
            st.caption("Step legend (parsed from run_name):")
            st.markdown("\n".join(legend_lines))

    groups = sorted([r.group for r in rows_local], key=group_sort_key)
    selected = st.multiselect(
        "Show groups",
        options=groups,
        default=groups,
        format_func=group_display,
        key="reports_groups_selected_analysis",
    )
    shown = [r for r in rows_local if r.group in set(selected)]
    shown_sorted = sorted(shown, key=lambda rr: group_sort_key(rr.group))

    st.subheader("Performance Analysis")
    st.caption(
        "How to read: best/worst `avg_overall` observed among runs in each group from the selected phase summary."
    )

    try:
        labels = [group_display(r.group) for r in shown_sorted]
        best_y = [float(r.best_avg_overall) if r.best_avg_overall is not None else float("nan") for r in shown_sorted]
        worst_y = [float(r.worst_avg_overall) if r.worst_avg_overall is not None else float("nan") for r in shown_sorted]
        fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(labels)), 4.4))
        ax.plot(labels, best_y, marker="o", label="best avg_overall")
        ax.plot(labels, worst_y, marker="o", label="worst avg_overall")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("avg_overall")
        ax.set_title("Best/Worst avg_overall by group")
        ax.legend(loc="lower right")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.info("Could not render group rollup chart for this phase summary.")

    _st_dataframe(
        [
            {
                "group": r.group,
                "label": group_display(r.group),
                "runs_total": r.runs_total,
                "runs_with_overall": r.runs_with_overall,
                "best_avg_overall": r.best_avg_overall,
                "best_run_name": r.best_run_name,
                "worst_avg_overall": r.worst_avg_overall,
                "worst_run_name": r.worst_run_name,
            }
            for r in shown_sorted
        ],
        hide_index=True,
    )

    spacer(rem=1.25)

    # -----------------------------
    # Run-based analysis over the timeline set
    # -----------------------------
    require_llm = True
    max_points = int(st.slider("Max runs plotted", min_value=10, max_value=200, value=120, step=10))

    trend_rows = _load_run_rows_cached(
        runs_dir=DEFAULT_RUNS_DIR.as_posix(),
        scored_dir=str(scored_dir),
        require_llm=require_llm,
    )
    if max_points and len(trend_rows) > max_points:
        trend_rows = trend_rows[-max_points:]

    selected_run_ids = [r.run_id for r in trend_rows]

    run_entries = load_run_entries(runs_dir=DEFAULT_RUNS_DIR.as_posix(), max_files=800)
    run_obj_by_id: Dict[str, Dict[str, Any]] = {}
    for e in run_entries:
        rid = run_id_from_entry(e)
        if rid:
            run_obj_by_id[rid] = e.obj

    scored_summary_by_id = load_scored_summary(scored_dir=str(scored_dir))

    baseline_llm_model, baseline_k = baseline_signature_from_runs(
        [run_obj_by_id[rid] for rid in selected_run_ids if rid in run_obj_by_id]
    )

    st.subheader("Experiment Timeline")
    st.caption("Stage-level progression using real scored runs (avg judge overall).")

    stage_rows = stage_timeline_rows_for_runs(
        run_ids=selected_run_ids,
        run_obj_by_id=run_obj_by_id,
        scored_summary_by_id=scored_summary_by_id,
        baseline_llm_model=baseline_llm_model,
        baseline_k=baseline_k,
    )
    if stage_rows and _plotly_or_warning():
        stage_ordered = [r.get("stage") for r in stage_rows if r.get("stage")]
        fig = px.line(stage_rows, x="stage", y="avg_score", markers=True, title=None, category_orders={"stage": stage_ordered})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_title="Stage", yaxis_title="Avg overall score (0..100)")
        st.plotly_chart(fig, use_container_width=True)
        _st_dataframe(stage_rows, hide_index=True)
    elif stage_rows:
        _st_dataframe(stage_rows, hide_index=True)
    else:
        st.info("No stage timeline data available for the selected run set.")

    spacer(rem=1.25)

    st.subheader("Impact of techniques")
    st.caption("Technique impact (enabled vs disabled). Interpretation: observational (not causal).")
    tech_rows = technique_impact_rows_for_runs(
        run_ids=selected_run_ids,
        run_obj_by_id=run_obj_by_id,
        scored_summary_by_id=scored_summary_by_id,
    )
    if tech_rows and _plotly_or_warning():
        tech_ordered = [r.get("technique") for r in tech_rows if r.get("technique")]
        fig = px.bar(
            tech_rows,
            x="delta_avg_overall",
            y="technique",
            orientation="h",
            title=None,
            hover_data=["enabled_n", "disabled_n", "enabled_avg", "disabled_avg"],
            category_orders={"technique": tech_ordered},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_title="Δ avg overall (enabled - disabled)", yaxis_title="Technique", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    elif tech_rows:
        _st_dataframe(tech_rows, hide_index=True)
    else:
        st.info("Not enough data to compute technique impact for the selected run set.")

    st.subheader("Failure mode mix")
    st.caption("Computed from the currently plotted runs using the latest scored artifacts.")
    fm_rows = failure_mode_counts_for_runs(run_ids=selected_run_ids, run_obj_by_id=run_obj_by_id, scored_dir=str(scored_dir))
    if fm_rows and _plotly_or_warning():
        fig = px.pie(fm_rows, values="count", names="failure_mode", title=None, hole=0.0)
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif fm_rows:
        _st_dataframe(fm_rows, hide_index=True)
    else:
        st.info("No failure mode data available for the selected run set.")

    spacer(rem=1.25)

    st.subheader("Failure modes by question type")
    st.caption("Computed from the same run set using per-case judges + heuristics.")
    fmqt_rows = failure_modes_by_question_type_for_runs(run_ids=selected_run_ids, run_obj_by_id=run_obj_by_id, scored_dir=str(scored_dir))
    if fmqt_rows and _plotly_or_warning():
        fig = px.bar(fmqt_rows, x="question_type", y="count", color="failure_mode", title=None, barmode="stack")
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
            xaxis_title="Question type",
            yaxis_title="Count",
            legend_title_text="failure_mode",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif fmqt_rows:
        _st_dataframe(fmqt_rows, hide_index=True)
    else:
        st.info("No per-type failure mode data available for the selected run set.")

    spacer(rem=1.25)

    st.subheader("Cost vs quality (runs)")
    st.caption(
        "Each point is one scored run. X-axis is average total tokens per answer; Y-axis is avg_overall."
    )

    cq_rows = cost_quality_rows_for_runs(
        run_ids=selected_run_ids,
        run_obj_by_id=run_obj_by_id,
        scored_summary_by_id=scored_summary_by_id,
        baseline_llm_model=baseline_llm_model,
        baseline_k=baseline_k,
    )
    if cq_rows and _plotly_or_warning():
        fig = px.scatter(
            cq_rows,
            x="avg_total_tokens",
            y="avg_overall",
            color="stage",
            custom_data=["run_name", "run_id", "retrieval_policy", "search_type", "k"],
            title=None,
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "stage=%{fullData.name}<br>"
                "avg_overall=%{y}<br>"
                "avg_total_tokens=%{x:.0f}<br>"
                "policy=%{customdata[2]} | search=%{customdata[3]} | k=%{customdata[4]}<br>"
                "run_id=%{customdata[1]}<extra></extra>"
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_dark",
            xaxis_title="Avg total tokens (proxy)",
            yaxis_title="Avg overall score",
            legend_title_text="stage",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif cq_rows:
        _st_dataframe(cq_rows, hide_index=True)
    else:
        st.info("No scored token/quality data available for the selected run set.")

    if cq_rows:
        st.caption("Summary table (sortable): use `score_per_1k_tokens` for demo runs (high score, low cost).")
        cq_table: List[Dict[str, Any]] = []
        for r in cq_rows:
            tokens = r.get("avg_total_tokens")
            score = r.get("avg_overall")
            score_per_1k = (float(score) * 1000.0 / float(tokens)) if tokens and score else None
            cq_table.append(
                {
                    "stage": r.get("stage"),
                    "avg_overall": score,
                    "avg_total_tokens": (round(float(tokens), 1) if tokens is not None else None),
                    "score_per_1k_tokens": (round(float(score_per_1k), 2) if score_per_1k is not None else None),
                    "retrieval_policy": r.get("retrieval_policy"),
                    "search_type": r.get("search_type"),
                    "k": r.get("k"),
                    "run_name": r.get("run_name"),
                    "run_id": r.get("run_id"),
                }
            )
        cq_table_sorted = sorted(
            cq_table,
            key=lambda rr: (
                -(float(rr.get("score_per_1k_tokens")) if rr.get("score_per_1k_tokens") is not None else -1.0),
                -(float(rr.get("avg_overall")) if rr.get("avg_overall") is not None else -1.0),
            ),
        )
        _st_dataframe(cq_table_sorted, hide_index=True)

    st.divider()

    st.subheader("Timeline")
    st.caption("Computed from scored, LLM-enabled runs (retrieval-only runs are ignored).")

    fig1 = _plot_score_trend(trend_rows, title="Avg overall over time")
    st.pyplot(fig1, clear_figure=True)
    fig2 = _plot_score_hist(trend_rows, title="Avg overall distribution")
    st.pyplot(fig2, clear_figure=True)
