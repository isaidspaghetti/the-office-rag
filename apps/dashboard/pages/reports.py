from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st

try:  # Plotly (optional, but preferred for interactive charts)
	import plotly.express as px  # type: ignore

	_HAS_PLOTLY = True
except Exception:  # pragma: no cover
	px = None  # type: ignore
	_HAS_PLOTLY = False

from apps.dashboard.shared import (
	DEFAULT_RUNS_DIR,
	EXPERIMENTS_DIR,
	REPO_ROOT,
	PhaseRow,
	RunRow,
	_load_phase_summary,
	_load_run_rows_cached,
	_phase_summary_is_step_grouped,
	_pick_latest_phase_summary_json,
	_read_csv_rows,
	_read_phase_summary_obj,
	_step_num,
	_st_dataframe,
	_st_image,
	_try_parse_step_cfg_from_run_name,
	_plot_score_hist,
	_plot_score_trend,
)


_EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


def _plotly_or_warning() -> bool:
	if _HAS_PLOTLY:
		return True
	st.warning("Plotly is not installed; charts will fall back to tables. Install with: pip install plotly")
	return False


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
	cur = d
	for part in (path or "").split("."):
		if not part:
			continue
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur


def _safe_int(x: Any) -> Optional[int]:
	try:
		if x is None:
			return None
		if isinstance(x, bool):
			return int(x)
		if isinstance(x, int):
			return int(x)
		if isinstance(x, float):
			return int(x)
		s = str(x).strip()
		if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
			return int(s, 10)
	except Exception:
		return None
	return None


def _safe_float(x: Any) -> Optional[float]:
	try:
		if x is None:
			return None
		if isinstance(x, bool):
			return float(int(x))
		if isinstance(x, (int, float)):
			return float(x)
		s = str(x).strip()
		if not s:
			return None
		return float(s)
	except Exception:
		return None


def _normalize_episode_id(s: Any) -> Optional[str]:
	if not s:
		return None
	m = _EP_RE.search(str(s).strip().upper())
	return m.group(0).upper() if m else None


def _question_type(question: str) -> str:
	q = str(question or "").strip().lower()
	if not q:
		return "(missing)"
	if re.search(r"\b(which|what)\s+episode\b|\bin\s+which\s+episode\b|\bepisode\s+is\b", q):
		return "Episode lookup"
	if re.search(r"\bquote\b|\bexact\s+quote\b|\bwhat\s+did\b.+\bsay\b", q):
		return "Quote / line"
	if re.search(r"\b(list|summari[sz]e|overview|timeline|chronolog|across|throughout|all\b|compare)\b", q):
		return "Aggregation"
	if re.search(r"\bwhy\b|\bhow\b", q):
		return "Explanation"
	return "Factual"


def _avg_total_tokens(run_obj: Dict[str, Any]) -> Optional[float]:
	# Prefer run summary.
	v = _safe_float(_safe_get(run_obj, "summary.avg_total_tokens", None))
	if v is not None:
		return float(v)

	# Fall back to per-case usage.
	cases = run_obj.get("cases")
	if not isinstance(cases, list) or not cases:
		return None

	vals: List[float] = []
	for c in cases:
		if not isinstance(c, dict):
			continue
		usage = _safe_get(c, "answer.usage", {})
		if not isinstance(usage, dict):
			continue
		t = _safe_float(usage.get("total_tokens"))
		if t is not None:
			vals.append(float(t))
	if not vals:
		return None
	return float(sum(vals) / len(vals))


def _script_persist_dir(run_obj: Dict[str, Any]) -> str:
	# Prefer provenance block (v4+).
	dv = _safe_get(run_obj, "config.data_version.script.persist_directory", None)
	if isinstance(dv, str) and dv.strip():
		return dv
	return str(_safe_get(run_obj, "config.vectorstore.persist_directory", "") or "")


def _derived_persist_dir(run_obj: Dict[str, Any]) -> str:
	dv = _safe_get(run_obj, "config.data_version.derived.persist_directory", None)
	if isinstance(dv, str) and dv.strip():
		return dv
	return str(_safe_get(run_obj, "config.derived_vectorstore.persist_directory", "") or "")


def _retrieval_policy(run_obj: Dict[str, Any]) -> str:
	return str(_safe_get(run_obj, "config.retrieval.policy", "") or "").strip().lower()


def _search_type(run_obj: Dict[str, Any]) -> str:
	return str(_safe_get(run_obj, "config.retrieval.search_type", "") or "").strip().lower()


def _query_expansion_enabled(run_obj: Dict[str, Any]) -> bool:
	return bool(_safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))


def _chunking_signature(run_obj: Dict[str, Any]) -> str:
	splitter = str(_safe_get(run_obj, "config.chunking.splitter", "") or "")
	persist = _script_persist_dir(run_obj)
	return f"{splitter}|{persist}".lower()


def _llm_model(run_obj: Dict[str, Any]) -> Optional[str]:
	m = _safe_get(run_obj, "config.llm.model", None)
	if isinstance(m, str) and m.strip():
		return m.strip()
	return None


def _classify_stage(
	run_obj: Dict[str, Any],
	*,
	baseline_llm_model: Optional[str],
	baseline_k: Optional[int],
) -> str:
	"""Heuristically bucket a run into a story stage.

	Ordered and intentionally opinionated (ported from the older dashboard).
	"""
	name = str(_safe_get(run_obj, "run.run_name", "") or "").lower()
	policy = _retrieval_policy(run_obj)
	search = _search_type(run_obj)
	qe = _query_expansion_enabled(run_obj)
	persist = _script_persist_dir(run_obj).lower()
	derived_persist = _derived_persist_dir(run_obj).lower()
	chunk_sig = _chunking_signature(run_obj)

	k = _safe_int(_safe_get(run_obj, "config.retrieval.k", None))

	if policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"} or "derived" in derived_persist:
		return "Derived Summaries / Cards"
	if "chroma_db_meta" in persist or "metadata" in name or "routing" in name:
		return "Metadata / Routing"
	if qe or "qe" in name or "query_expansion" in name or "rrf" in name or "fusion" in name:
		return "Query Expansion + Fusion"
	if "scene" in chunk_sig or "scene" in name:
		return "Scene Chunking"
	if search == "mmr" or "mmr" in name:
		return "MMR / Diversity"
	if baseline_k is not None and k is not None and k > baseline_k:
		return "Increased Recall"
	if k is not None and k >= 10:
		return "Increased Recall"

	m = _llm_model(run_obj)
	if baseline_llm_model and m and m != baseline_llm_model:
		return "Model Upgrades"
	if "gpt-" in name and baseline_llm_model and baseline_llm_model.replace(".", "_") not in name:
		return "Model Upgrades"

	if "baseline" in name or policy in {"script_only", "script", "baseline", ""}:
		return "Baseline"
	return "Other / Uncategorized"


def _classify_failure_mode(*, case: Dict[str, Any], judge: Optional[Dict[str, Any]], coverage: Optional[float]) -> str:
	"""Return one of: retrieval_miss | grounding | aggregation | reasoning | ok | unknown."""
	labels = case.get("labels")
	if isinstance(labels, dict):
		if labels.get("retrieval_failure") is True:
			return "retrieval_miss"
		if labels.get("grounding_failure") is True:
			return "grounding"

	qtype = _question_type(str(case.get("question") or ""))
	cov = _safe_float(coverage)
	if cov is not None and cov < 0.5:
		return "retrieval_miss"

	if isinstance(judge, dict):
		corr = _safe_int(judge.get("correctness"))
		g = _safe_int(judge.get("groundedness"))
		comp = _safe_int(judge.get("completeness"))

		if qtype == "Aggregation" and comp is not None and comp <= 2:
			return "aggregation"
		if g is not None and g <= 2:
			return "grounding"
		if corr is not None and corr <= 2:
			return "reasoning"

		overall = _safe_int(judge.get("overall"))
		if overall is not None and overall >= 80:
			return "ok"
		if overall is not None and overall <= 40:
			if qtype == "Aggregation":
				return "aggregation"
			return "reasoning"

	return "unknown"


@st.cache_data(show_spinner=False)
def _load_run_objs_cached(runs_dir: str, *, max_files: int = 500) -> List[Dict[str, Any]]:
	p = Path(runs_dir)
	if not p.exists():
		return []
	files = sorted(p.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
	if max_files and len(files) > int(max_files):
		files = files[: int(max_files)]
	out: List[Dict[str, Any]] = []
	for rf in files:
		try:
			obj = json.loads(rf.read_text(encoding="utf-8"))
			if isinstance(obj, dict) and "run" in obj and "cases" in obj:
				out.append({"path": str(rf), "obj": obj})
		except Exception:
			continue
	return out


@st.cache_data(show_spinner=False)
def _load_scored_summary_cached(scored_dir: str) -> Dict[str, Dict[str, Any]]:
	p = Path(scored_dir)
	if not p.exists():
		return {}
	out: Dict[str, Dict[str, Any]] = {}
	for sf in sorted(p.glob("*.scored.json"), key=lambda x: x.stat().st_mtime, reverse=True):
		try:
			obj = json.loads(sf.read_text(encoding="utf-8"))
			run = obj.get("run") if isinstance(obj, dict) else None
			rid = str((run or {}).get("run_id") or "") if isinstance(run, dict) else ""
			rid = rid.strip() or sf.name[: -len(".scored.json")]
			avg_overall = _safe_get(obj, "score_summary.avg_overall", None)
			out[rid] = {"path": str(sf), "avg_overall": (int(avg_overall) if isinstance(avg_overall, int) else None)}
		except Exception:
			continue
	return out


@st.cache_data(show_spinner=False)
def _load_scored_obj_cached(scored_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
	p = Path(scored_dir) / f"{run_id}.scored.json"
	if not p.exists():
		return None
	try:
		obj = json.loads(p.read_text(encoding="utf-8"))
		return obj if isinstance(obj, dict) else None
	except Exception:
		return None


def render_summary() -> None:
	st.header("Reports")

	runs_dir = DEFAULT_RUNS_DIR
	canonical_scored_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass"
	ctxfix_all_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass_ctxfix_all_2026-03-07"
	rescored_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass_rescored_2026-03-07"
	rescored_ctxfix_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass_rescored_ctxfix_2026-03-07"

	# Default to the canonical deploy path when it has scored files.
	# This keeps Streamlit Cloud / docs stable while letting us refresh the contents.
	def _has_scored_files(p: Path) -> bool:
		return p.exists() and any(p.glob("*.scored.json"))

	if _has_scored_files(canonical_scored_dir):
		scored_dir = canonical_scored_dir
	elif _has_scored_files(ctxfix_all_dir):
		scored_dir = ctxfix_all_dir
	elif _has_scored_files(rescored_ctxfix_dir):
		scored_dir = rescored_ctxfix_dir
	elif _has_scored_files(rescored_dir):
		scored_dir = rescored_dir
	else:
		scored_dir = canonical_scored_dir

	# Sub-pages within Reports mode.
	# If an old session had the removed "Groups" page selected, fall back.
	if str(st.session_state.get("reports_subpage") or "") == "Groups":
		st.session_state["reports_subpage"] = "Timeline"
	try:
		page = st.radio(
			"Reports subpage",
			options=["Overview", "Timeline", "Artifacts"],
			horizontal=True,
			key="reports_subpage",
			label_visibility="collapsed",
		)
	except TypeError:
		# Older Streamlit: label_visibility not supported.
		page = st.radio(
			"",
			options=["Overview", "Timeline", "Artifacts"],
			horizontal=True,
			key="reports_subpage",
		)

	# Phase summary selection. On Timeline we render the selector at the bottom
	# (per UX request), but we still need a selection early to render Groups.
	candidates = sorted(EXPERIMENTS_DIR.glob("phase_score_summary_*.json"))
	phase_json_default = _pick_latest_phase_summary_json()
	if phase_json_default is None:
		st.warning("No phase summary JSON found under `experiments/`. Run `experiments/plot_phase_scores.py`.")
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

	phase_json: Path
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
		# Overview + Timeline: use the current session selection (or default) without showing a selector.
		phase_json = name_to_path.get(selected_name, phase_json_default)

	phase_obj = _read_phase_summary_obj(phase_json)
	is_step_grouped = _phase_summary_is_step_grouped(phase_obj)

	rows = _load_phase_summary(phase_json)
	if not rows:
		st.warning("Phase summary JSON exists but has no rows.")
		return

	# --- Group naming + sorting ---
	# We show two different concepts in phase summary artifacts:
	# - high-level phases like "baseline", "mmr", "qe+fusion"...
	# - routing sweep steps like "step01".."step06" (best seen in *_by_step summaries)
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

	def _render_groups_section(rows_in: List[PhaseRow]) -> None:
		st.subheader("Groups")
		st.caption(
			"This is a precomputed roll-up of many eval runs into comparable groups (phases or steps). "
			"For each group it reports the best/worst avg_overall observed."
		)

		rows_local = list(rows_in)

		# If a phase-level summary includes step01..stepNN (from a sweep), hide those by default.
		# The dedicated *_by_step summaries are the clearer place to view those.
		if not is_step_grouped:
			has_steps = any(_step_num(r.group) is not None for r in rows_local)
			if has_steps:
				include_steps = st.checkbox(
					"Include step01..stepNN sweep groups",
					value=False,
					key="reports_include_step_groups",
					help="Phase summaries can include both high-level phases and sweep steps. The steps are usually clearer in the *_by_step summary.",
				)
				if not include_steps:
					rows_local = [r for r in rows_local if _step_num(r.group) is None]

		# If the summary is grouped by step, show a compact legend so it’s self-explanatory.
		if is_step_grouped:
			legend_lines: List[str] = []
			for r in sorted(
				rows_local,
				key=lambda rr: (_step_num(rr.group) is None, int(_step_num(rr.group) or 10**9)),
			):
				cfg = _try_parse_step_cfg_from_run_name(r.best_run_name or r.worst_run_name)
				if not cfg:
					continue
				friendly = _STEP_FRIENDLY.get(str(r.group or "").strip().lower())
				legend_lines.append(
					(
						f"- {r.group} — {friendly}: {cfg['policy']} + {cfg['search']} (k={cfg['k']})"
						if friendly
						else f"- {r.group}: {cfg['policy']} + {cfg['search']} (k={cfg['k']})"
					)
				)
			if legend_lines:
				st.caption("Step legend (parsed from run_name):")
				st.markdown("\n".join(legend_lines))

		groups = sorted([r.group for r in rows_local], key=_group_sort_key)
		selected = st.multiselect(
			"Show groups",
			options=groups,
			default=groups,
			format_func=_group_display,
			key=f"reports_groups_selected_{page.lower()}",
		)
		shown = [r for r in rows_local if r.group in set(selected)]

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

		shown_sorted = sorted(shown, key=lambda rr: _group_sort_key(rr.group))

		# Plot dynamically (so naming/sorting/filtering match the table).
		try:
			labels = [_group_display(r.group) for r in shown_sorted]
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
			pass

		_st_dataframe(
			[
				{
					"group": r.group,
					"label": _group_display(r.group),
					"delta_vs_baseline": (
						(int(r.best_avg_overall) - int(baseline_overall))
						if (
							is_step_grouped
							and baseline_overall is not None
							and r.best_avg_overall is not None
						)
						else None
					),
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
		if is_step_grouped and baseline_group and baseline_overall is not None:
			st.caption(f"Baseline for delta is {baseline_group} (avg_overall={baseline_overall}).")

	if page == "Overview":
		# -----------------------------
		# Simple page styling
		# -----------------------------
		st.markdown(
			"""
			<style>
				.block-container {
					padding-top: 2rem;
					padding-bottom: 2rem;
					padding-left: 3rem;
					padding-right: 3rem;
					max-width: 1200px;
				}

				.hero-card {
					background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
					padding: 2rem 2rem 1.5rem 2rem;
					border-radius: 18px;
					color: white;
					margin-bottom: 1.5rem;
					border: 1px solid rgba(255,255,255,0.08);
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
			</style>
			""",
			unsafe_allow_html=True,
		)

		# -----------------------------
		# Header / hero
		# -----------------------------
		st.markdown(
			"""
			<div class="hero-card">
				<div class="hero-title">RAG Evaluation &amp; Observability Dashboard</div>
				<div class="hero-subtitle">
					A controlled AI engineering experiment built to measure, debug, and improve a retrieval-based question-answering system.
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		# -----------------------------
		# Main summary
		# -----------------------------
		left, right = st.columns([1.4, 1], gap="large")

		with left:
			st.markdown(
				"""
				<div class="section-card">
					<div class="section-title">Overview</div>
					<div class="body-text">
						This application is a <b>Evaluation dashboard + Retrieval Augmented Generation (RAG) Chat Bot</b>.
						It answers questions about <i>The Office</i> using only the show’s closed caption dialog as its corpus (knowledge source).
						<br><br>
						The project was selected as an analogous sandbox to a project copilot -- a potential point of revenue for Tonic -- and a way to understand how retrieval systems behave when
						working with sparse real-world data, and how to measure and improve them over time.
					</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

			st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

			st.markdown(
				"""
				<div class="section-card">
					<div class="section-title">Why This Project Matters</div>
					<div class="body-text">
						AI makes it easy to build prototypes, but <b>building reliable AI systems requires measurement, evaluation, and iteration</b>.
						This project demonstrates how to engineer AI systems responsibly by:
						<ul>
							<li>designing measurable experiments</li>
							<li>evaluating answer accuracy and retrieval quality</li>
							<li>identifying failure modes such as hallucination or missing context</li>
							<li>improving the system through structured architectural changes</li>
						</ul>
						Every change was tested across multiple runs to understand <b>what actually improved results</b>.
					</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		with right:
			st.markdown(
				"""
				<div class="section-card">
					<div class="section-title">What This Demonstrates</div>
					<div style="margin-top: 0.35rem;">
						<span class="pill">Evaluation Design</span>
						<span class="pill">Failure Analysis</span>
						<span class="pill">Retrieval Tuning</span>
						<span class="pill">Observability</span>
						<span class="pill">Cost / Quality Tradeoffs</span>
					</div>
					<div class="highlight">
						The value of this project is not just the chatbot itself — it is the
						<b>engineering discipline behind building, measuring, and improving a system</b>.
					</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

			st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

			st.markdown(
				"""
				<div class="section-card">
					<div class="section-title">Key Result</div>
					<div class="body-text">
						Through iterative experimentation, the system evolved from a simple prototype into a
						<b>measurably improved retrieval architecture</b>, with clear visibility into:
						<ul>
							<li>which techniques improved accuracy</li>
							<li>which approaches regressed</li>
							<li>the cost vs. performance tradeoffs of different configurations</li>
						</ul>
					</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		# -----------------------------
		# Relevance section
		# -----------------------------
		st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

		st.markdown(
			"""
			<div class="section-card">
				<div class="section-title">Why This Matters for Tonic</div>
				<div class="body-text">
					The same engineering approach applies directly to real business AI systems, including:
					<ul>
						<li>internal knowledge copilots</li>
						<li>document search and summarization tools</li>
						<li>support and Slack assistants</li>
						<li>AI-powered product features</li>
					</ul>
					This project demonstrates the ability to <b>lead AI engineering initiatives with measurable results</b>,
					rather than relying on ad-hoc experimentation.
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

		# -----------------------------
		# Optional placeholder area for charts / timeline
		# -----------------------------
		st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

		st.subheader("Experiment Story")
		st.caption(
			"This section is a good place to add your stage timeline chart, best run score, and a concise summary of what worked vs. what failed."
		)

		placeholder_col1, placeholder_col2, placeholder_col3 = st.columns(3)

		with placeholder_col1:
			st.metric("Best Run Score", "75", "+8 vs baseline")

		with placeholder_col2:
			st.metric("Biggest Win", "Higher Recall", "Increasing k improved coverage")

		with placeholder_col3:
			st.metric("Biggest Regression", "Scene Chunking", "Narrative cohesion dropped")

		st.info(
			"Next step: replace these placeholder metrics with your real stage chart, failure mix, and cost-vs-quality visual."
		)
		return

	if page == "Timeline":
		# Order requested: Groups -> Timeline -> selector.
		_render_groups_section(rows)

		st.subheader("Timeline")
		require_llm = True
		st.caption("Computed from scored, LLM-enabled runs (retrieval-only runs are ignored).")
		max_points = int(st.slider("Max runs plotted", min_value=10, max_value=200, value=120, step=10))

		trend_rows = _load_run_rows_cached(
			runs_dir=DEFAULT_RUNS_DIR.as_posix(),
			scored_dir=scored_dir.as_posix(),
			require_llm=require_llm,
		)
		if max_points and len(trend_rows) > max_points:
			trend_rows = trend_rows[-max_points:]

		fig1 = _plot_score_trend(trend_rows, title="Avg overall over time")
		st.pyplot(fig1, clear_figure=True)
		fig2 = _plot_score_hist(trend_rows, title="Avg overall distribution")
		st.pyplot(fig2, clear_figure=True)

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
			_st_dataframe(rows, hide_index=True)
		return

	# (Precomputed PNG/CSV are on the Artifacts sub-page.)
