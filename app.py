from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_RUNS_DIR = EXPERIMENTS_DIR / "runs"


@dataclass(frozen=True)
class PhaseRow:
	group: str
	runs_total: int
	runs_with_overall: int
	best_avg_overall: Optional[int]
	best_run_id: Optional[str]
	best_run_name: Optional[str]
	worst_avg_overall: Optional[int]
	worst_run_id: Optional[str]
	worst_run_name: Optional[str]


def _read_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def _query_params() -> Dict[str, str]:
	# Streamlit changed query param APIs over time; support both.
	try:
		qp = st.query_params  # type: ignore[attr-defined]
		return {str(k): str(v) for k, v in dict(qp).items()}
	except Exception:
		qp = st.experimental_get_query_params()
		return {k: (v[0] if isinstance(v, list) and v else str(v)) for k, v in qp.items()}


def _pick_latest_phase_summary_json() -> Optional[Path]:
	# Stable alias: if present, always prefer it.
	alias = EXPERIMENTS_DIR / "phase_score_summary_latest.json"
	if alias.exists():
		return alias

	# Prefer ctxfix summaries when present.
	candidates = sorted(EXPERIMENTS_DIR.glob("phase_score_summary_*.json"))
	if not candidates:
		return None

	ctxfix = [p for p in candidates if "ctxfix" in p.name.lower()]
	if ctxfix:
		# Prefer phase-level (not by_step) and LLM-filtered summaries when available.
		ctxfix_llm_phase = [p for p in ctxfix if "_llm" in p.name.lower() and "by_step" not in p.name.lower()]
		if ctxfix_llm_phase:
			return ctxfix_llm_phase[-1]
		ctxfix_phase = [p for p in ctxfix if "by_step" not in p.name.lower()]
		if ctxfix_phase:
			return ctxfix_phase[-1]
		return ctxfix[-1]

	# Deterministic: take lexicographically last (timestamps are in filename).
	return candidates[-1]


def _st_dataframe(rows: Any, **kwargs: Any) -> Any:
	"""Compatibility wrapper for Streamlit width API changes."""
	try:
		return st.dataframe(rows, width="stretch", **kwargs)
	except TypeError:
		return st.dataframe(rows, use_container_width=True, **kwargs)


def _st_image(img: Any, **kwargs: Any) -> Any:
	"""Compatibility wrapper for Streamlit width API changes."""
	try:
		return st.image(img, width="stretch", **kwargs)
	except TypeError:
		return st.image(img, use_container_width=True, **kwargs)


def _load_phase_summary(path: Path) -> List[PhaseRow]:
	obj = _read_json(path)
	rows = obj.get("rows") if isinstance(obj, dict) else None
	if not isinstance(rows, list):
		return []

	out: List[PhaseRow] = []
	for r in rows:
		if not isinstance(r, dict):
			continue
		out.append(
			PhaseRow(
				group=str(r.get("group") or ""),
				runs_total=int(r.get("runs_total") or 0),
				runs_with_overall=int(r.get("runs_with_overall") or 0),
				best_avg_overall=(int(r["best_avg_overall"]) if r.get("best_avg_overall") is not None else None),
				best_run_id=(str(r.get("best_run_id") or "") or None),
				best_run_name=(str(r.get("best_run_name") or "") or None),
				worst_avg_overall=(
					int(r["worst_avg_overall"]) if r.get("worst_avg_overall") is not None else None
				),
				worst_run_id=(str(r.get("worst_run_id") or "") or None),
				worst_run_name=(str(r.get("worst_run_name") or "") or None),
			)
		)
	return out


def render_summary() -> None:
	st.header("Run Summary")

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

	left, right = st.columns(2)
	with left:
		st.subheader("Artifacts")
		st.write(f"Runs dir: `{runs_dir.as_posix()}`")
		st.write(f"Scored dir: `{scored_dir.as_posix()}`")
		st.write(f"Run logs: {len(list(runs_dir.glob('*.json'))) if runs_dir.exists() else 0}")
		st.write(f"Scored files: {len(list(scored_dir.glob('*.scored.json'))) if scored_dir.exists() else 0}")

	with right:
		st.subheader("Phase Summary")
		candidates = sorted(EXPERIMENTS_DIR.glob("phase_score_summary_*.json"))
		phase_json = _pick_latest_phase_summary_json()
		if phase_json is None:
			st.warning("No phase summary JSON found under `experiments/`. Run `experiments/plot_phase_scores.py`.")
			return

		# Let you pick among available summaries (phase vs by_step, llm-filtered, etc.)
		name_to_path = {p.name: p for p in candidates}
		options = list(name_to_path.keys())
		default_name = phase_json.name
		try:
			default_index = options.index(default_name)
		except Exception:
			default_index = 0
		picked_name = st.selectbox("Summary file", options=options, index=default_index)
		phase_json = name_to_path.get(picked_name, phase_json)
		st.write(f"Using: `{phase_json.name}`")

	rows = _load_phase_summary(phase_json)
	if not rows:
		st.warning("Phase summary JSON exists but has no rows.")
		return

	groups = [r.group for r in rows]
	selected = st.multiselect("Show groups", options=groups, default=groups)
	shown = [r for r in rows if r.group in set(selected)]

	# Table
	_st_dataframe(
		[
			{
				"group": r.group,
				"runs_total": r.runs_total,
				"runs_with_overall": r.runs_with_overall,
				"best_avg_overall": r.best_avg_overall,
				"best_run_name": r.best_run_name,
				"worst_avg_overall": r.worst_avg_overall,
				"worst_run_name": r.worst_run_name,
			}
			for r in shown
		],
		hide_index=True,
	)

	# Show PNG if present.
	png = phase_json.with_suffix(".png")
	if png.exists():
		_st_image(str(png), caption=png.name)

	# Optional: show CSV
	with st.expander("Show raw CSV rows"):
		csv_path = phase_json.with_suffix(".csv")
		if not csv_path.exists():
			st.info("No CSV found next to the JSON.")
		else:
			_st_dataframe(_read_csv_rows(csv_path), hide_index=True)


def render_chat_debug() -> None:
	st.header("Chat & Debug")
	st.info(
		"This mode is not wired up in this repo snapshot. "
		"Use the eval harness in `experiments/` for now, or tell me what backend you want (Chroma vs Qdrant)."
	)


def main() -> None:
	st.set_page_config(page_title="RAG-1 Dashboard", layout="wide")
	st.title("RAG-1 Dashboard")

	qp = _query_params()
	default_mode = str(qp.get("mode") or "summary").strip().lower()

	mode = st.sidebar.radio(
		"Mode",
		options=["summary", "chat_debug"],
		index=0 if default_mode != "chat_debug" else 1,
	)

	try:
		if mode == "summary":
			render_summary()
		else:
			render_chat_debug()
	except Exception as e:
		st.error("Dashboard error")
		st.exception(e)


if __name__ == "__main__":
	main()
