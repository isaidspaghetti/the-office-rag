from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import streamlit as st

from apps.dashboard.components.cards import pills, section_card
from apps.dashboard.components.hero import hero_card
from apps.dashboard.components.html import spacer
from apps.dashboard.styles.css import inject_dashboard_css
from apps.dashboard.shared import REPO_ROOT, PhaseRow, _step_num


def render_overview_page(
    *,
    phase_rows: List[PhaseRow],
    is_step_grouped: bool,
    group_display: Any,
    group_sort_key: Any,
) -> None:
    inject_dashboard_css(layout="overview")

    hero_card(
        title="Project Overview & Summary",
        subtitles_html=[
            "A controlled AI engineering experiment built to measure, debug, and improve a retrieval-based question-answering system.",
        ],
        tone="dark",
    )

    section_card(
        title="Overview",
        body_html=(
            "Evaluation dashboard and RAG chatbot built solely on closed captions from <b>The Office</b>."
            "<br/><br/>"
            "The system requires citations, runs at temperature 0.0, and is designed to study how retrieval architectures improve through controlled experiments and evaluation."
            "<br/><br/>"
            "Charts show how different techniques affect accuracy, retrieval quality, and cost."
        ),
        classes="card-base card-neutral",
    )

    spacer(rem=1.0)

  
    section_card(
        title="Engineering Focus",
        body_html=(
            f'<div style="margin-top: 0.35rem;">{pills(labels=["Evaluation Design", "Failure Analysis", "Retrieval Tuning", "Observability", "Cost / Quality Tradeoffs"])}</div>'
            "<div class=\"highlight\">"
            "The value of this project is not just the chatbot itself — it is the "
            "<b>engineering discipline behind building, measuring, and improving a system</b>."
            "</div>"
        ),
        body_class=None,
        classes="card-base card-insight",
    )


    spacer(rem=1.25)

    section_card(
        title="Application",
        body_html=(
            "The same engineering approach applies directly to real-world AI systems, including:"
            "<ul>"
            "<li>internal knowledge copilots</li>"
            "<li>document search and summarization tools</li>"
            "<li>support and ticket assistants</li>"
            "<li>AI-powered product features</li>"
            "</ul>"
            "This project demonstrates how to <b>lead an evaluation-driven iteration loop</b>, "
        ),
        classes="card-base card-accent",
    )


    spacer(rem=1.25)

    st.subheader("Primary Source")
    left_ref, right_ref = st.columns([0.35, 0.65], gap="large")
    with left_ref:
        st.image(
            str(REPO_ROOT / "apps" / "dashboard" / "assets" / "Aibook.jpg"),
            caption="AI Engineering (O'Reilly, 2025)",
        )
    with right_ref:
        section_card(
            title="Reference: AI Engineering",
            body_html=(
                "“This project is heavily inspired by Chip Huyen’s <i>AI Engineering</i> (O’Reilly, 2025), especially the emphasis on evaluation-driven development, iteration loops, and production constraints (cost/latency/reliability). I used it as the backbone for the system design and experimentation methodology.”"
                "<br/><br/>"
                "<b>Chip Huyen</b>. <i>AI Engineering: Building Applications with Foundation Models</i>. "
                "O’Reilly Media, 2025."
            ),
            classes="card-base card-neutral",
        )

    spacer(rem=1.0)

    st.subheader("Phase Rollup")
    st.caption(
        "How to read this chart: the x-axis is a phase/step group from the selected phase summary. "
        "The two lines show the best and worst `avg_overall` observed among runs in that group. "
        "The Timeline view plots individual runs over time; this chart is the group-level rollup that helps explain "
        "the upper/lower envelope you see in the timeline plots."
    )

    rows_local = list(phase_rows)
    if not is_step_grouped:
        rows_local = [r for r in rows_local if _step_num(r.group) is None]

    baseline_group: Optional[str] = None
    baseline_overall: Optional[int] = None
    if is_step_grouped:
        ordered = sorted(
            [r for r in rows_local if _step_num(r.group) is not None],
            key=lambda rr: int(_step_num(rr.group) or 10**9),
        )
        if ordered:
            baseline_group = ordered[0].group
            baseline_overall = (
                ordered[0].best_avg_overall
                if ordered[0].best_avg_overall is not None
                else ordered[0].worst_avg_overall
            )

    shown_sorted = sorted(rows_local, key=lambda rr: group_sort_key(rr.group))
    try:
        labels = [group_display(r.group) for r in shown_sorted]
        best_y = [
            float(r.best_avg_overall) if r.best_avg_overall is not None else float("nan")
            for r in shown_sorted
        ]
        worst_y = [
            float(r.worst_avg_overall) if r.worst_avg_overall is not None else float("nan")
            for r in shown_sorted
        ]
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

    if is_step_grouped and baseline_group and baseline_overall is not None:
        st.caption(f"Baseline for delta is {baseline_group} (avg_overall={baseline_overall}).")

    phase1, phase2 = st.columns(2)

    with phase1:
        section_card(
            title="Phase 1 — Baseline Retrieval",
            body_html=(
                "<div class=\"body-text\">"
                "<p>The system began with a minimal semantic retrieval architecture:</p>"
                "<ul>"
                "<li>basic chunking</li>"
                "<li>similarity search</li>"
                "<li>GPT-4.1 nano for answer generation</li>"
                "</ul>"
                "<p>"
                "This configuration worked well for <b>direct lookups and quotes</b> but struggled with broader reasoning questions. "
                "Without enough context diversity, the model often hallucinated or produced incomplete answers."
                "</p>"
                "</div>"
                "<div class=\"highlight\">Baseline runs established a reference point used to evaluate every later experiment.</div>"
            ),
            variant="phase",
            body_class=None,
        )

        spacer(rem=1.0)

        section_card(
            title="Phase 2 — Retrieval Optimization",
            body_html=(
                "<div class=\"body-text\">"
                "<p>The next phase focused on improving retrieval quality and recall. Techniques tested included:</p>"
                "<ul>"
                "<li><b>MMR (Maximum Marginal Relevance)</b> to improve context diversity</li>"
                "<li><b>Higher recall retrieval</b> by increasing the number of documents returned</li>"
                "<li><b>Query Expansion + Rank Fusion</b> to combine results from multiple semantic searches</li>"
                "</ul>"
                "<p>These experiments significantly improved the system’s ability to locate relevant context across episodes.</p>"
                "</div>"
                "<div class=\"highlight\">Query expansion combined with MMR produced the most reliable retrieval improvements.</div>"
            ),
            variant="phase",
            body_class=None,
        )

    with phase2:
        section_card(
            title="Phase 3 — Index Enrichment",
            body_html=(
                "<div class=\"body-text\">"
                "<p>Improving retrieval alone was not sufficient. The next phase focused on improving the data itself.</p>"
                "<p>New capabilities included:</p>"
                "<ul>"
                "<li>structured <b>metadata</b> (season, episode, characters)</li>"
                "<li>derived narrative summaries generated using <b>map-reduce pipelines</b></li>"
                "<li>multiple indexes containing both <b>dialogue and summarized context</b></li>"
                "</ul>"
                "<p>These derived knowledge layers allowed the system to reason across episodes instead of relying solely on raw dialogue.</p>"
                "</div>"
                "<div class=\"highlight\">Data engineering proved just as important as retrieval tuning.</div>"
            ),
            variant="phase",
            body_class=None,
        )

        spacer(rem=1.0)

        section_card(
            title="Phase 4 — Evaluation &amp; Observability",
            body_html=(
                "<div class=\"body-text\">"
                "<p>A full evaluation framework was built to compare system configurations objectively. The dashboard tracks:</p>"
                "<ul>"
                "<li>automated scoring of runs</li>"
                "<li>AI-based judging of answer quality</li>"
                "<li>retrieval diagnostics</li>"
                "<li>experiment comparisons over time</li>"
                "</ul>"
                "<p>This made it possible to identify which architectural changes produced real improvements.</p>"
                "</div>"
                "<div class=\"highlight\">Evaluation must separate <b>retrieval quality</b> from <b>answer correctness</b>. Both must be measured to build reliable AI systems.</div>"
            ),
            variant="phase",
            body_class=None,
        )

    spacer(rem=1.5)

    section_card(
        title="Key Takeaways",
        body_html=(
            "<ul>"
            "<li>AI systems should be improved through measurement, not intuition.</li>"
            "<li>Design experiments with clear evaluation criteria and failure modes in mind.</li>"
            "<li>Start with 2+ phase and prompt engineer individually instead of cramming into one judge</li>"
            "<li>Preplan and prioritize data first</li>"
            "<li>While having tons of metrics is nice, I should have also chosen specific data points earlier on to highlight (first and second classes)</li>"
            "<li>Hand written notes and bookmarks of runs would have been much faster for retrospectives and forensics instead of asking AI to summarize later.</li>"
            "<li>As the model grew, generations gave better insights over individual tuning. Having the ability to pass different routing strategies, judges, and questions against older models would have helped better express performance gains.</li>"
            "</ul>"
        ),
        classes="card-base card-insight",
    )
