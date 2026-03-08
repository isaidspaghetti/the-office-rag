from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import matplotlib.pyplot as plt

try:
	from qdrant_client import QdrantClient  # type: ignore
	from langchain_community.vectorstores import Qdrant as QdrantVS  # type: ignore
	_HAS_QDRANT = True
except Exception:
	QdrantClient = None  # type: ignore
	QdrantVS = None  # type: ignore
	_HAS_QDRANT = False


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_RUNS_DIR = EXPERIMENTS_DIR / "runs"


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"

DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS = [
	"db/chroma_db_meta",
	"db/chroma_db",
	"db/chroma_db_scene",
	"db/chroma_db_scene_w1",
]
DEFAULT_LOCAL_DERIVED_PERSIST_DIRS = [
	"db/chroma_db_derived_cards",
]


def _env_or_secret(key: str, default: Optional[str] = None) -> Optional[str]:
	# Streamlit Cloud secrets are available via st.secrets; locally we use env/.env.
	try:
		v = st.secrets.get(key)  # type: ignore[attr-defined]
		if v is not None:
			s = str(v).strip()
			return s if s else default
	except Exception:
		pass
	val = os.environ.get(key)
	if val is None:
		return default
	s = str(val).strip()
	return s if s else default


def _bool_env_or_secret(key: str, default: bool = False) -> bool:
	v = _env_or_secret(key)
	if v is None:
		return bool(default)
	return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_under_repo(path_str: str) -> Path:
	p = Path(path_str)
	if not p.is_absolute():
		p = (REPO_ROOT / p).resolve()
	return p


def _short_path(p: Path) -> str:
	try:
		return p.relative_to(REPO_ROOT).as_posix()
	except Exception:
		return p.as_posix()


def _openai_ready() -> Tuple[bool, str]:
	if bool(_env_or_secret("OPENAI_API_KEY")):
		return True, ""
	return False, "Missing OPENAI_API_KEY (set in .env or Streamlit secrets)."


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


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
	cur = d
	for part in (path or "").split("."):
		if not part:
			continue
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur


def _parse_utc(ts: Any) -> Optional[datetime]:
	s = str(ts or "").strip()
	if not s:
		return None
	try:
		if s.endswith("Z"):
			s = s[:-1] + "+00:00"
		dt = datetime.fromisoformat(s)
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt
	except Exception:
		return None


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


def _iter_run_files(runs_dir: Path) -> List[Path]:
	if not runs_dir.exists():
		return []
	return sorted(runs_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _run_id_from_run_file(run_file: Path) -> str:
	# experiments/run_eval.py names the file "{run_id}.json".
	name = run_file.name
	if name.endswith(".json"):
		return name[: -len(".json")]
	return run_file.stem


def _load_run_obj(run_file: Path) -> Optional[Dict[str, Any]]:
	try:
		obj = _read_json(run_file)
		if isinstance(obj, dict) and "run" in obj and "cases" in obj:
			return obj
		return None
	except Exception:
		return None


def _scored_file_for_run_id(scored_dir: Path, run_id: str) -> Optional[Path]:
	p = scored_dir / f"{run_id}.scored.json"
	return p if p.exists() else None


def _load_scored_obj(scored_dir: Path, run_id: str) -> Optional[Dict[str, Any]]:
	sf = _scored_file_for_run_id(scored_dir, run_id)
	if not sf:
		return None
	try:
		obj = _read_json(sf)
		return obj if isinstance(obj, dict) else None
	except Exception:
		return None


def _two_pass_overall(scored_obj: Optional[Dict[str, Any]], case_id: str) -> Optional[int]:
	if not scored_obj:
		return None
	cases = scored_obj.get("scored_cases")
	if not isinstance(cases, list):
		return None
	for r in cases:
		if not isinstance(r, dict):
			continue
		if str(r.get("case_id") or "") != str(case_id):
			continue
		judge = r.get("judge")
		if isinstance(judge, dict) and isinstance(judge.get("overall"), int):
			return int(judge["overall"])
	return None


def _build_chroma(persist_dir: Path, *, embed_model: str) -> Chroma:
	embeddings = OpenAIEmbeddings(model=embed_model)
	return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def _build_qdrant(*, url: str, api_key: Optional[str], collection: str, embed_model: str) -> Any:
	if not _HAS_QDRANT:
		raise RuntimeError("Qdrant backend not available (missing qdrant-client/langchain-community)")
	client = QdrantClient(url=str(url), api_key=(str(api_key) if api_key else None))
	embeddings = OpenAIEmbeddings(model=embed_model)
	return QdrantVS(client=client, collection_name=str(collection), embeddings=embeddings)


def _format_doc_line(doc: Any) -> str:
	meta = getattr(doc, "metadata", None) or {}
	src = str(meta.get("source") or meta.get("source_file") or "").strip()
	eid = str(meta.get("episode_id") or "").strip()
	dt = str(meta.get("doc_type") or "").strip()
	if eid and dt:
		return f"{eid} | {dt} | {src}".strip(" |")
	if src:
		return src
	return "(no source)"


def _answer_prompt(*, question: str, context: str) -> Tuple[str, str]:
	system = (
		"You are a QA assistant for questions about the TV show The Office.\n"
		"Answer the user's question using ONLY the provided context.\n"
		"If the context does not contain the answer, say you don't know.\n"
		"When possible, cite episode identifiers present in the context (e.g., 'S02E06').\n"
		"When you make a specific claim, include at least one short exact quote from the context in double quotes."
	)
	user = f"Question: {question}\n\nContext:\n{context}"
	return system, user


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


@dataclass(frozen=True)
class RunRow:
	run_id: str
	run_name: str
	created_at: Optional[datetime]
	retrieval_policy: str
	search_type: str
	k: Optional[int]
	llm_enabled: Optional[bool]
	avg_overall: Optional[int]


def _run_id_from_run_obj(obj: Dict[str, Any], *, fallback: str) -> str:
	run = obj.get("run") if isinstance(obj, dict) else None
	rid = str((run or {}).get("run_id") or "").strip() if isinstance(run, dict) else ""
	return rid or fallback


def _load_run_rows(*, runs_dir: Path, scored_dir: Path, require_llm: bool) -> List[RunRow]:
	rows: List[RunRow] = []
	for rf in _iter_run_files(runs_dir):
		obj = _load_run_obj(rf)
		if not obj:
			continue
		run = obj.get("run") if isinstance(obj, dict) else None
		run_name = str((run or {}).get("run_name") or "") if isinstance(run, dict) else ""
		created_at = _parse_utc((run or {}).get("created_at_utc")) if isinstance(run, dict) else None
		run_id = _run_id_from_run_obj(obj, fallback=_run_id_from_run_file(rf))

		retrieval_policy = str(_safe_get(obj, "config.retrieval.policy", "") or "")
		search_type = str(_safe_get(obj, "config.retrieval.search_type", "") or "")
		k_val = _safe_get(obj, "config.retrieval.k", None)
		k_int = int(k_val) if isinstance(k_val, int) else None

		llm_enabled = _safe_get(obj, "config.llm.enabled", None)
		llm_enabled_bool = bool(llm_enabled) if isinstance(llm_enabled, bool) else None
		if require_llm and llm_enabled_bool is not True:
			continue

		scored_obj = _load_scored_obj(scored_dir, run_id)
		avg_overall = _safe_get(scored_obj, "score_summary.avg_overall", None)
		avg_overall_int = int(avg_overall) if isinstance(avg_overall, int) else None

		rows.append(
			RunRow(
				run_id=run_id,
				run_name=run_name,
				created_at=created_at,
				retrieval_policy=retrieval_policy,
				search_type=search_type,
				k=k_int,
				llm_enabled=llm_enabled_bool,
				avg_overall=avg_overall_int,
			)
		)

	# Sort oldest -> newest when timestamps exist; otherwise stable by run_id.
	rows.sort(key=lambda r: (r.created_at or datetime.min.replace(tzinfo=timezone.utc), r.run_id))
	return rows


@st.cache_data(show_spinner=False)
def _load_run_rows_cached(runs_dir: str, scored_dir: str, require_llm: bool) -> List[RunRow]:
	return _load_run_rows(
		runs_dir=Path(runs_dir),
		scored_dir=Path(scored_dir),
		require_llm=bool(require_llm),
	)


def _plot_score_trend(rows: List[RunRow], *, title: str) -> Any:
	pts = [r for r in rows if r.created_at is not None and r.avg_overall is not None]
	fig, ax = plt.subplots(figsize=(10, 3.6))
	if not pts:
		ax.text(0.5, 0.5, "No scored runs with timestamps.", ha="center", va="center")
		ax.set_axis_off()
		fig.suptitle(title)
		return fig

	x = [r.created_at for r in pts]
	y = [r.avg_overall for r in pts]
	ax.plot(x, y, marker="o", linewidth=1.2, markersize=3)
	ax.set_ylim(0, 100)
	ax.set_ylabel("avg_overall")
	ax.grid(True, alpha=0.25)
	fig.suptitle(title)
	fig.autofmt_xdate()
	return fig


def _plot_score_hist(rows: List[RunRow], *, title: str) -> Any:
	vals = [r.avg_overall for r in rows if r.avg_overall is not None]
	fig, ax = plt.subplots(figsize=(10, 3.2))
	if not vals:
		ax.text(0.5, 0.5, "No scored runs.", ha="center", va="center")
		ax.set_axis_off()
		fig.suptitle(title)
		return fig

	ax.hist(vals, bins=list(range(0, 101, 5)), edgecolor="white")
	ax.set_xlim(0, 100)
	ax.set_xlabel("avg_overall")
	ax.set_ylabel("runs")
	ax.grid(True, axis="y", alpha=0.25)
	fig.suptitle(title)
	return fig


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

	st.divider()
	st.subheader("Graphs")

	col_a, col_b = st.columns(2)
	with col_a:
		require_llm = bool(st.checkbox("Only LLM-enabled runs", value=True))
	with col_b:
		max_points = int(st.slider("Max runs plotted", min_value=10, max_value=200, value=120, step=10))

	# Load run-level trend rows (newer / better data lives in experiments/runs + scored_runs_two_pass).
	trend_rows = _load_run_rows_cached(
		runs_dir=DEFAULT_RUNS_DIR.as_posix(),
		scored_dir=scored_dir.as_posix(),
		require_llm=require_llm,
	)
	if max_points and len(trend_rows) > max_points:
		trend_rows = trend_rows[-max_points:]

	fig1 = _plot_score_trend(
		trend_rows,
		title=("Avg overall over time" + (" (LLM-only)" if require_llm else "")),
	)
	st.pyplot(fig1, clear_figure=True)

	fig2 = _plot_score_hist(
		trend_rows,
		title=("Avg overall distribution" + (" (LLM-only)" if require_llm else "")),
	)
	st.pyplot(fig2, clear_figure=True)

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

	# Also show the by-step companion plot when the stable alias exists.
	by_step_alias = EXPERIMENTS_DIR / "phase_score_summary_latest_by_step.json"
	if by_step_alias.exists():
		by_step_png = by_step_alias.with_suffix(".png")
		if by_step_png.exists():
			_st_image(str(by_step_png), caption=by_step_png.name)

	# Optional: show CSV
	with st.expander("Show raw CSV rows"):
		csv_path = phase_json.with_suffix(".csv")
		if not csv_path.exists():
			st.info("No CSV found next to the JSON.")
		else:
			_st_dataframe(_read_csv_rows(csv_path), hide_index=True)


def render_chat_debug() -> None:
	st.header("Chat & Debug")
	load_dotenv()

	openai_ok, openai_err = _openai_ready()
	qdrant_url = _env_or_secret("QDRANT_URL")
	qdrant_api_key = _env_or_secret("QDRANT_API_KEY")
	qdrant_scripts_collection = _env_or_secret("QDRANT_SCRIPTS_COLLECTION", "office_scripts")
	qdrant_derived_collection = _env_or_secret("QDRANT_DERIVED_COLLECTION", "office_derived_cards")

	default_backend = "qdrant" if qdrant_url else "chroma"
	backend = st.sidebar.selectbox(
		"Retrieval backend",
		options=["auto", "chroma", "qdrant"],
		index=0,
		help="Auto selects Qdrant if QDRANT_URL is set; otherwise uses local Chroma.",
	)
	if backend == "auto":
		backend = default_backend

	st.sidebar.subheader("Retrieval")
	retrieval_policy = st.sidebar.selectbox(
		"Policy",
		options=["script_only", "derived_only", "derived_then_script"],
		index=2,
	)
	k = int(st.sidebar.slider("Top-k", min_value=1, max_value=20, value=12))
	derived_k = int(st.sidebar.slider("Derived k (routing)", min_value=1, max_value=30, value=12))
	episode_shortlist_size = int(st.sidebar.slider("Episode shortlist size", min_value=1, max_value=12, value=6))

	st.sidebar.subheader("Models")
	embed_model = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
	llm_enabled = bool(st.sidebar.checkbox("Generate answer (LLM)", value=True))
	llm_model = st.sidebar.text_input("LLM model", value=DEFAULT_LLM_MODEL)
	temperature = float(st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0))

	if llm_enabled and not openai_ok:
		st.warning(openai_err)
		llm_enabled = False

	# Local Chroma config
	script_persist_default = DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS[0]
	derived_persist_default = DEFAULT_LOCAL_DERIVED_PERSIST_DIRS[0]
	all_local_script = [p for p in DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS if (_resolve_under_repo(p)).exists()]
	all_local_derived = [p for p in DEFAULT_LOCAL_DERIVED_PERSIST_DIRS if (_resolve_under_repo(p)).exists()]
	# Auto-discover other chroma_db* dirs under db/
	try:
		for p in sorted((REPO_ROOT / "db").glob("chroma_db*")):
			rel = _short_path(p)
			if rel.startswith("db/") and rel not in all_local_script:
				all_local_script.append(rel)
			if "derived" in rel and rel not in all_local_derived:
				all_local_derived.append(rel)
	except Exception:
		pass

	if backend == "chroma":
		st.caption("Backend: local Chroma (persisted under db/)")
		script_persist = st.sidebar.selectbox(
			"Script persist dir",
			options=(all_local_script or DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS),
			index=0,
		)
		derived_persist = st.sidebar.selectbox(
			"Derived persist dir",
			options=(all_local_derived or DEFAULT_LOCAL_DERIVED_PERSIST_DIRS),
			index=0,
		)
	else:
		st.caption("Backend: Qdrant (cloud)")
		if not _HAS_QDRANT:
			st.error("Qdrant backend is not available (missing dependencies).")
			return
		if not qdrant_url:
			st.error("Missing QDRANT_URL (set in Streamlit secrets or env).")
			return
		st.sidebar.text_input("Qdrant URL", value=qdrant_url, disabled=True)
		st.sidebar.text_input("Scripts collection", value=str(qdrant_scripts_collection), disabled=True)
		st.sidebar.text_input("Derived collection", value=str(qdrant_derived_collection), disabled=True)
		script_persist = script_persist_default
		derived_persist = derived_persist_default

	if "chat_messages" not in st.session_state:
		st.session_state.chat_messages = []

	# Render history
	for m in st.session_state.chat_messages:
		role = str(m.get("role") or "assistant")
		content = str(m.get("content") or "")
		with st.chat_message(role):
			st.markdown(content)

	question = st.chat_input("Ask about The Office…")
	if not question:
		with st.expander("Setup / expectations"):
			st.write("Local: uses Chroma under `db/`. Deployed: auto-switches to Qdrant when QDRANT_URL is set.")
			st.write("Policy `derived_then_script` uses derived cards to route into script chunks.")
		return

	st.session_state.chat_messages.append({"role": "user", "content": question})
	with st.chat_message("user"):
		st.markdown(question)

	# Build vectorstores lazily per request (simple + robust; can be cached later)
	try:
		if backend == "chroma":
			script_db = _build_chroma(_resolve_under_repo(script_persist), embed_model=embed_model)
			derived_db = _build_chroma(_resolve_under_repo(derived_persist), embed_model=embed_model)
		else:
			script_db = _build_qdrant(
				url=str(qdrant_url),
				api_key=qdrant_api_key,
				collection=str(qdrant_scripts_collection),
				embed_model=embed_model,
			)
			derived_db = _build_qdrant(
				url=str(qdrant_url),
				api_key=qdrant_api_key,
				collection=str(qdrant_derived_collection),
				embed_model=embed_model,
			)
	except Exception as e:
		st.error(f"Failed to init vectorstore: {type(e).__name__}: {e}")
		return

	# Retrieval
	t0 = time.time()
	retrieved: List[Tuple[Any, Optional[float]]] = []
	routing: Optional[Dict[str, Any]] = None

	def _as_pairs(res: Any) -> List[Tuple[Any, Optional[float]]]:
		out: List[Tuple[Any, Optional[float]]] = []
		for doc, score in (res or []):
			try:
				out.append((doc, float(score) if score is not None else None))
			except Exception:
				out.append((doc, None))
		return out

	try:
		if retrieval_policy == "script_only":
			retrieved = _as_pairs(script_db.similarity_search_with_relevance_scores(question, k=int(k)))
		elif retrieval_policy == "derived_only":
			retrieved = _as_pairs(derived_db.similarity_search_with_relevance_scores(question, k=int(k)))
		else:
			# derived -> script routing
			derived_pairs = _as_pairs(derived_db.similarity_search_with_relevance_scores(question, k=int(derived_k)))
			shortlist: List[str] = []
			seen: set[str] = set()
			for d, _s in derived_pairs:
				meta = getattr(d, "metadata", None) or {}
				eid = str(meta.get("episode_id") or "").strip().upper()
				if not eid:
					# derived episode cards also have an identity line like "Episode card: S02E12 — ..."
					txt = getattr(d, "page_content", None) or ""
					for tok in str(txt).split():
						if len(tok) == 6 and tok.upper().startswith("S") and "E" in tok.upper():
							eid = tok.strip().upper().strip(",.;:()[]{}")
							break
				if not eid or eid in seen:
					continue
				seen.add(eid)
				shortlist.append(eid)
				if len(shortlist) >= int(episode_shortlist_size):
					break

			# Best-effort: retrieve a larger pool, then filter by episode_id.
			candidate_k = max(int(k) * 12, 60)
			script_pairs = _as_pairs(script_db.similarity_search_with_relevance_scores(question, k=int(candidate_k)))
			filtered: List[Tuple[Any, Optional[float]]] = []
			if shortlist:
				allowed = {s.strip().upper() for s in shortlist}
				for d, s in script_pairs:
					meta = getattr(d, "metadata", None) or {}
					eid = str(meta.get("episode_id") or "").strip().upper()
					if eid and eid in allowed:
						filtered.append((d, s))
						if len(filtered) >= int(k):
							break

			# If we got nothing, fall back to unfiltered top-k.
			retrieved = filtered if filtered else script_pairs[: int(k)]
			routing = {
				"policy": "derived_then_script",
				"episode_shortlist": shortlist,
				"derived_top": [
					{
						"rank": i + 1,
						"episode_id": str((getattr(d, "metadata", None) or {}).get("episode_id") or "").strip(),
						"score": (float(s) if s is not None else None),
						"source": str((getattr(d, "metadata", None) or {}).get("source_file") or (getattr(d, "metadata", None) or {}).get("source") or ""),
					}
					for i, (d, s) in enumerate(derived_pairs[: min(len(derived_pairs), 12)])
				],
			}
	except Exception as e:
		st.error(f"Retrieval failed: {type(e).__name__}: {e}")
		retrieved = []

	retrieval_ms = int((time.time() - t0) * 1000)

	with st.expander(f"Retrieved context ({len(retrieved)} docs, {retrieval_ms}ms)", expanded=True):
		if routing:
			st.write("Routing:")
			st.json(routing)
		for i, (d, s) in enumerate(retrieved, start=1):
			st.markdown(f"**#{i}**  score={s if s is not None else 'n/a'}")
			st.write(_format_doc_line(d))
			st.text((getattr(d, "page_content", None) or "")[:900])

	context = "\n\n---\n\n".join([str(getattr(d, "page_content", None) or "") for (d, _s) in retrieved])

	answer_text = ""
	if llm_enabled:
		try:
			llm = ChatOpenAI(model=str(llm_model), temperature=float(temperature))
			sys, user = _answer_prompt(question=question, context=context)
			msg = llm.invoke([("system", sys), ("human", user)])
			answer_text = str(getattr(msg, "content", "") or "").strip()
		except Exception as e:
			answer_text = f"(LLM error: {type(e).__name__}: {e})"
	else:
		answer_text = "(LLM disabled)"

	st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})
	with st.chat_message("assistant"):
		st.markdown(answer_text or "(empty)")

	if st.sidebar.button("Clear chat"):
		st.session_state.chat_messages = []
		st.rerun()


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


def main() -> None:
	load_dotenv()
	st.set_page_config(page_title="RAG-1 Dashboard", layout="wide")
	st.title("RAG-1 Dashboard")

	qp = _query_params()
	default_mode = str(qp.get("mode") or "summary").strip().lower()

	options = ["summary", "run_explorer", "chat_debug"]
	default_idx = 0
	if default_mode in options:
		default_idx = options.index(default_mode)
	mode = st.sidebar.radio("Mode", options=options, index=default_idx)

	try:
		if mode == "summary":
			render_summary()
		elif mode == "run_explorer":
			render_run_explorer()
		else:
			render_chat_debug()
	except Exception as e:
		st.error("Dashboard error")
		st.exception(e)


if __name__ == "__main__":
	main()
