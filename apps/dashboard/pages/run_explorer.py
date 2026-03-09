from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from apps.dashboard.shared import (
	DEFAULT_RUNS_DIR,
	REPO_ROOT,
	_iter_run_files,
	_load_run_obj,
	_load_scored_obj,
	_run_id_from_run_file,
	_short_path,
	_two_pass_overall,
)


def render_run_explorer() -> None:
	st.header("Run Explorer")

	runs_dir = DEFAULT_RUNS_DIR
	canonical_scored_dir = REPO_ROOT / "experiments" / "scored_runs_two_pass"
	if not canonical_scored_dir.exists():
		st.warning("No scored_runs_two_pass directory found; run scoring or check repo artifacts.")

	left, right = st.columns([0.42, 0.58])

	with left:
		st.subheader("Pick a run")
		name_filter = st.text_input("Filter (substring)", value="")
		run_files = _iter_run_files(runs_dir)
		if not run_files:
			st.warning("No run logs found under experiments/runs")
			return

		# Build display options with minimal parsing.
		options: List[Tuple[str, Path]] = []
		for rf in run_files:
			obj = _load_run_obj(rf)
			if not obj:
				continue
			run = obj.get("run") or {}
			run_name = str(run.get("run_name") or "")
			run_id = str(run.get("run_id") or _run_id_from_run_file(rf))
			label = f"{run_name}  —  {run_id}".strip()
			if name_filter and name_filter.lower() not in label.lower():
				continue
			options.append((label, rf))

		if not options:
			st.info("No runs match the filter.")
			return

		labels = [x[0] for x in options]
		picked = st.selectbox("Run", options=labels, index=0)
		run_file = dict(options)[picked]

		obj = _load_run_obj(run_file)
		if not obj:
			st.error("Failed to load run JSON")
			return

		run_id = str((obj.get("run") or {}).get("run_id") or _run_id_from_run_file(run_file))
		scored_obj = _load_scored_obj(canonical_scored_dir, run_id)

		st.caption(f"Run file: {_short_path(run_file)}")
		st.caption(f"Scored: {'yes' if scored_obj else 'no'}")

		with st.expander("Run config", expanded=False):
			st.json(obj.get("config") or {})

	with right:
		cases = obj.get("cases") or []
		if not isinstance(cases, list) or not cases:
			st.warning("Run has no cases")
			return

		st.subheader("Cases")
		case_ids = [str((c or {}).get("case_id") or "") for c in cases if isinstance(c, dict)]
		case_ids = [c for c in case_ids if c]
		picked_case_id = st.selectbox("Case", options=case_ids, index=0)
		case = None
		for c in cases:
			if isinstance(c, dict) and str(c.get("case_id") or "") == picked_case_id:
				case = c
				break
		if case is None:
			st.error("Case not found")
			return

		overall = _two_pass_overall(scored_obj, picked_case_id)
		if overall is not None:
			st.markdown(f"**Two-pass overall:** {overall}")

		st.markdown("**Question**")
		st.write(str(case.get("question") or ""))

		ans = (((case.get("answer") or {}) if isinstance(case.get("answer"), dict) else {}) or {})
		st.markdown("**Answer**")
		st.write(ans.get("text"))
		if ans.get("error"):
			st.error(str(ans.get("error")))

		retr = (((case.get("retrieval") or {}) if isinstance(case.get("retrieval"), dict) else {}) or {})
		results = retr.get("results") or []
		with st.expander(f"Retrieval results ({len(results) if isinstance(results, list) else 0})", expanded=True):
			if isinstance(results, list):
				for r in results[:20]:
					if not isinstance(r, dict):
						continue
					src = r.get("source")
					eid = r.get("episode_id")
					score = r.get("score")
					st.markdown(f"- score={score} | {eid} | {src}")
					st.caption(r.get("first_line") or "")
					st.text(str(r.get("preview") or "")[:600])

		with st.expander("Diagnostics", expanded=False):
			st.json(case.get("diagnostics") or {})
