from __future__ import annotations

import csv
import json
import os
import re
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

from rag.fusion import rrf_fuse
from rag.query_expansion import QueryExpansionConfig, build_expander_llm, expand_queries

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


def _similarity_pairs(db: Any, question: str, *, k: int) -> List[Tuple[Any, Optional[float]]]:
	"""Similarity retrieval returning (doc, score?) pairs when possible."""
	try:
		res = db.similarity_search_with_relevance_scores(question, k=int(k))
		out: List[Tuple[Any, Optional[float]]] = []
		for doc, score in (res or []):
			try:
				out.append((doc, float(score) if score is not None else None))
			except Exception:
				out.append((doc, None))
		return out
	except Exception:
		# Some vectorstores don't implement relevance scores.
		docs = db.similarity_search(question, k=int(k))
		return [(d, None) for d in (docs or [])]


def _mmr_pairs(
	db: Any,
	question: str,
	*,
	k: int,
	fetch_k: Optional[int],
	lambda_mult: float,
) -> List[Tuple[Any, Optional[float]]]:
	"""MMR retrieval returning (doc, score?) pairs.

	Vectorstores often return docs-only for MMR. We backfill best-effort similarity scores
	from the same candidate pool for debugging/ranking visibility.
	"""
	effective_fetch_k = int(fetch_k) if isinstance(fetch_k, int) and fetch_k > 0 else int(max(int(k) * 4, 20))

	try:
		mmr_docs = db.max_marginal_relevance_search(
			question,
			k=int(k),
			fetch_k=int(effective_fetch_k),
			lambda_mult=float(lambda_mult),
		)
	except Exception:
		# If MMR isn't supported, fall back.
		return _similarity_pairs(db, question, k=int(k))

	# Score backfill via similarity pass.
	score_map: Dict[Tuple[Optional[str], str], float] = {}
	try:
		cands = _similarity_pairs(db, question, k=int(effective_fetch_k))
		for doc, score in cands:
			if score is None:
				continue
			meta = getattr(doc, "metadata", None) or {}
			key = (meta.get("source"), getattr(doc, "page_content", "") or "")
			prev = score_map.get(key)
			s = float(score)
			if prev is None or s > prev:
				score_map[key] = s
	except Exception:
		score_map = {}

	out: List[Tuple[Any, Optional[float]]] = []
	for d in (mmr_docs or []):
		meta = getattr(d, "metadata", None) or {}
		key = (meta.get("source"), getattr(d, "page_content", "") or "")
		out.append((d, score_map.get(key)))
	return out


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


_STEP_GROUP_RE = re.compile(r"^step(?P<num>\d{2,3})$", re.IGNORECASE)
_STEP_CFG_RE = re.compile(
	r"step\d{2,3}_(?P<policy>script_only|derived_only|derived_then_script|blended)_(?P<search>sim|mmr)_k(?P<k>\d+)",
	re.IGNORECASE,
)


def _read_phase_summary_obj(path: Path) -> Dict[str, Any]:
	obj = _read_json(path)
	return obj if isinstance(obj, dict) else {}


def _phase_summary_is_step_grouped(obj: Dict[str, Any]) -> bool:
	gb = str(obj.get("group_by") or "").strip().lower()
	return gb == "step"


def _try_parse_step_cfg_from_run_name(run_name: Optional[str]) -> Optional[Dict[str, Any]]:
	if not run_name:
		return None
	m = _STEP_CFG_RE.search(str(run_name))
	if not m:
		return None
	return {
		"policy": str(m.group("policy")).lower(),
		"search": ("similarity" if str(m.group("search")).lower() == "sim" else "mmr"),
		"k": int(m.group("k")),
	}


def _step_num(group: str) -> Optional[int]:
	m = _STEP_GROUP_RE.fullmatch(str(group or "").strip())
	if not m:
		return None
	try:
		return int(m.group("num"), 10)
	except Exception:
		return None


def _load_phase_summary(path: Path) -> List[PhaseRow]:
	obj = _read_phase_summary_obj(path)
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
		st.subheader("Corpus")
		run_logs_n = len(list(runs_dir.glob("*.json"))) if runs_dir.exists() else 0
		scored_files_n = len(list(scored_dir.glob("*.scored.json"))) if scored_dir.exists() else 0
		m1, m2 = st.columns(2)
		with m1:
			st.metric("Run logs", run_logs_n)
		with m2:
			st.metric("Scored files", scored_files_n)

	with right:
		st.subheader("Phase Summary")
		st.caption(
			"This is a precomputed roll-up of many eval runs into a few comparable groups (" 
			"usually different experiment phases / settings). For each group it reports how many runs exist, "
			"how many were scored, and the best/worst avg_overall observed. It’s meant as a quick health-check "
			"and ‘what worked / what regressed’ view — not a single run’s details."
		)
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

	phase_obj = _read_phase_summary_obj(phase_json)
	is_step_grouped = _phase_summary_is_step_grouped(phase_obj)

	rows = _load_phase_summary(phase_json)
	if not rows:
		st.warning("Phase summary JSON exists but has no rows.")
		return

	# If a phase-level summary includes step01..stepNN (from a sweep), hide those by default.
	# The dedicated *_by_step summaries are the clearer place to view those.
	if not is_step_grouped:
		has_steps = any(_step_num(r.group) is not None for r in rows)
		if has_steps:
			include_steps = st.checkbox(
				"Include step01..stepNN sweep groups",
				value=False,
				help="Phase summaries can include both high-level phases and sweep steps. The steps are usually clearer in the *_by_step summary.",
			)
			if not include_steps:
				rows = [r for r in rows if _step_num(r.group) is None]

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

	# If the summary is grouped by step, show a compact legend so it’s self-explanatory.
	if is_step_grouped:
		legend_lines: List[str] = []
		for r in sorted(rows, key=lambda rr: (_step_num(rr.group) is None, int(_step_num(rr.group) or 10**9))):
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

	st.divider()
	st.subheader("Graphs")

	# For dashboard-level reporting we ignore retrieval-only runs.
	require_llm = True
	st.caption("Graphs are computed from scored, LLM-enabled runs (retrieval-only runs are ignored).")
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
		title="Avg overall over time",
	)
	st.pyplot(fig1, clear_figure=True)

	fig2 = _plot_score_hist(
		trend_rows,
		title="Avg overall distribution",
	)
	st.pyplot(fig2, clear_figure=True)

	groups = sorted([r.group for r in rows], key=_group_sort_key)
	selected = st.multiselect(
		"Show groups",
		options=groups,
		default=groups,
		format_func=_group_display,
	)
	shown = [r for r in rows if r.group in set(selected)]

	# Table
	baseline_group: Optional[str] = None
	baseline_overall: Optional[int] = None
	if is_step_grouped:
		ordered = sorted(
			[r for r in rows if _step_num(r.group) is not None],
			key=lambda rr: int(_step_num(rr.group) or 10**9),
		)
		if ordered:
			baseline_group = ordered[0].group
			baseline_overall = ordered[0].best_avg_overall if ordered[0].best_avg_overall is not None else ordered[0].worst_avg_overall

	shown_sorted = sorted(shown, key=lambda rr: _group_sort_key(rr.group))

	# Plot phase summary dynamically (so naming/sorting/filtering match the table).
	try:
		labels = [_group_display(r.group) for r in shown_sorted]
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
		pass
	_st_dataframe(
		[
			{
				"group": r.group,
				"label": _group_display(r.group),
				"delta_vs_baseline": (
					(int(r.best_avg_overall) - int(baseline_overall))
					if (is_step_grouped and baseline_overall is not None and r.best_avg_overall is not None)
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

	# Optional: show precomputed PNG if present (kept for historical parity).
	with st.expander("Show precomputed PNG"):
		png = phase_json.with_suffix(".png")
		if png.exists():
			_st_image(str(png), caption=png.name)
		else:
			st.info("No PNG found next to the JSON.")

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

	# --- Query expansion state sync (sidebar <-> quick rerun panel) ---
	# Streamlit reruns on every widget interaction. We keep the left sidebar controls and
	# the "Re-run last question" controls in sync via session_state.
	_QE_STATE_MAP: List[Tuple[str, str]] = [
		("chat_sidebar_qe_enabled", "chat_quick_qe_enabled"),
		("chat_sidebar_expand_n", "chat_quick_expand_n"),
		("chat_sidebar_expand_model", "chat_quick_expand_model"),
		("chat_sidebar_k_per_query", "chat_quick_k_per_query"),
		("chat_sidebar_rrf_k0", "chat_quick_rrf_k0"),
		("chat_sidebar_expand_cache", "chat_quick_expand_cache"),
	]

	def _qe_mark_sidebar_dirty() -> None:
		st.session_state.chat_qe_dirty_source = "sidebar"

	def _qe_mark_quick_dirty() -> None:
		st.session_state.chat_qe_dirty_source = "quick"
		# Keep the quick panel open while tweaking.
		st.session_state.chat_quick_expanded = True

	# Seed defaults (only if missing).
	st.session_state.setdefault("chat_qe_dirty_source", None)
	st.session_state.setdefault("chat_sidebar_qe_enabled", False)
	st.session_state.setdefault("chat_sidebar_expand_n", 5)
	st.session_state.setdefault("chat_sidebar_expand_model", "gpt-4.1-nano")
	st.session_state.setdefault("chat_sidebar_k_per_query", 6)
	st.session_state.setdefault("chat_sidebar_rrf_k0", 60)
	st.session_state.setdefault("chat_sidebar_expand_cache", "experiments/cache/query_expansion_cache.json")
	st.session_state.setdefault("chat_quick_expanded", False)

	# Default quick values to match sidebar.
	for sb_key, quick_key in _QE_STATE_MAP:
		if quick_key not in st.session_state:
			st.session_state[quick_key] = st.session_state.get(sb_key)

	# One-way sync based on which side changed last.
	src = st.session_state.get("chat_qe_dirty_source")
	if src == "quick":
		for sb_key, quick_key in _QE_STATE_MAP:
			if quick_key in st.session_state:
				st.session_state[sb_key] = st.session_state.get(quick_key)
		st.session_state.chat_qe_dirty_source = None
	elif src == "sidebar":
		for sb_key, quick_key in _QE_STATE_MAP:
			if sb_key in st.session_state:
				st.session_state[quick_key] = st.session_state.get(sb_key)
		st.session_state.chat_qe_dirty_source = None
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
	search_type = st.sidebar.selectbox(
		"Search type",
		options=["similarity", "mmr"],
		index=0,
		help="MMR trades off relevance vs diversity. Use lambda closer to 1.0 for more relevance.",
	)
	k = int(st.sidebar.slider("Top-k", min_value=1, max_value=20, value=12))
	fetch_k: Optional[int] = None
	lambda_mult = 0.7
	if search_type == "mmr":
		fetch_k = int(
			st.sidebar.slider(
				"MMR fetch_k",
				min_value=max(8, int(k)),
				max_value=240,
				value=max(24, int(k) * 4),
				help="Candidate pool size MMR selects from (higher = slower, sometimes better).",
			)
		)
		lambda_mult = float(
			st.sidebar.slider(
				"MMR lambda",
				min_value=0.0,
				max_value=1.0,
				value=0.7,
				help="0.0=max diversity, 1.0=max relevance.",
			)
		)
	derived_k = int(st.sidebar.slider("Derived k (routing)", min_value=1, max_value=30, value=12))
	episode_shortlist_size = int(st.sidebar.slider("Episode shortlist size", min_value=1, max_value=12, value=6))

	st.sidebar.subheader("Query expansion")
	qe_enabled = bool(
		st.sidebar.checkbox(
			"Expand queries + RRF fuse",
			key="chat_sidebar_qe_enabled",
			on_change=_qe_mark_sidebar_dirty,
			help="Generates alternate retrieval queries and fuses results via Reciprocal Rank Fusion.",
		)
	)
	expand_n = int(
		st.sidebar.slider(
			"Expand n",
			min_value=1,
			max_value=10,
			value=int(st.session_state.get("chat_sidebar_expand_n") or 5),
			disabled=(not qe_enabled),
			key="chat_sidebar_expand_n",
			on_change=_qe_mark_sidebar_dirty,
		)
	)
	expand_model = st.sidebar.selectbox(
		"Expand model",
		options=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini"],
		index=0,
		disabled=(not qe_enabled),
		key="chat_sidebar_expand_model",
		on_change=_qe_mark_sidebar_dirty,
	)
	k_per_query = int(
		st.sidebar.slider(
			"k per query",
			min_value=1,
			max_value=20,
			value=int(st.session_state.get("chat_sidebar_k_per_query") or min(int(k), 6)),
			disabled=(not qe_enabled),
			help="Docs to retrieve per expanded query before fusion.",
			key="chat_sidebar_k_per_query",
			on_change=_qe_mark_sidebar_dirty,
		)
	)
	rrf_k0 = int(
		st.sidebar.slider(
			"RRF k0",
			min_value=1,
			max_value=200,
			value=int(st.session_state.get("chat_sidebar_rrf_k0") or 60),
			disabled=(not qe_enabled),
			help="Higher reduces the impact of rank differences.",
			key="chat_sidebar_rrf_k0",
			on_change=_qe_mark_sidebar_dirty,
		)
	)
	expand_cache_path = st.sidebar.text_input(
		"Expansion cache",
		value=str(st.session_state.get("chat_sidebar_expand_cache") or "experiments/cache/query_expansion_cache.json"),
		disabled=(not qe_enabled),
		help="Set empty to disable caching.",
		key="chat_sidebar_expand_cache",
		on_change=_qe_mark_sidebar_dirty,
	)

	st.sidebar.subheader("Models")
	embed_model = DEFAULT_EMBED_MODEL
	llm_enabled = bool(st.sidebar.checkbox("Generate answer (LLM)", value=True))
	llm_model = st.sidebar.selectbox("LLM model", options=[DEFAULT_LLM_MODEL, 'gpt-4.1-nano', 'gpt-5-mini', 'gpt-5-nano'], index=0)

	temperature = float(st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0))

	# Both answer-generation and query-expansion require OpenAI.
	if (llm_enabled or qe_enabled) and not openai_ok:
		st.warning(openai_err)
		llm_enabled = False
		qe_enabled = False

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
			"Script Index",
			options=(all_local_script or DEFAULT_LOCAL_SCRIPT_PERSIST_DIRS),
			index=0,
		)
		derived_persist = st.sidebar.selectbox(
			"Derived Data Index",
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
	if "chat_last_question" not in st.session_state:
		st.session_state.chat_last_question = ""


	# Render history
	for m in st.session_state.chat_messages:
		role = str(m.get("role") or "assistant")
		content = str(m.get("content") or "")
		with st.chat_message(role):
			st.markdown(content)

	question_to_run: Optional[str] = None

	new_question = st.chat_input("Ask about The Office…")
	if new_question:
		question_to_run = str(new_question)
		st.session_state.chat_last_question = str(new_question)
		st.session_state.chat_messages.append({"role": "user", "content": question_to_run})
		with st.chat_message("user"):
			st.markdown(question_to_run)

	# Inline quick rerun controls (main pane) for the last asked question.
	last_question = str(st.session_state.get("chat_last_question") or "").strip()
	quick_backend = backend
	quick_retrieval_policy = retrieval_policy
	quick_search_type = str(search_type)
	quick_k = int(k)
	quick_fetch_k = int(fetch_k) if isinstance(fetch_k, int) else max(24, int(k) * 4)
	quick_lambda_mult = float(lambda_mult)
	quick_derived_k = int(derived_k)
	quick_episode_shortlist_size = int(episode_shortlist_size)
	quick_qe_enabled = bool(qe_enabled)
	quick_expand_n = int(expand_n)
	quick_expand_model = str(expand_model)
	quick_k_per_query = int(k_per_query)
	quick_rrf_k0 = int(rrf_k0)
	quick_expand_cache_path = str(expand_cache_path)
	quick_llm_enabled = bool(llm_enabled)
	quick_llm_model = str(llm_model)
	quick_temperature = float(temperature)

	if last_question:
		# Streamlit reruns on every widget interaction. Persist expander-open state so it
		# doesn't collapse while you're actively tweaking settings.
		if "chat_quick_expanded" not in st.session_state:
			st.session_state.chat_quick_expanded = False
		if "chat_quick_snapshot" not in st.session_state:
			st.session_state.chat_quick_snapshot = {}

		_quick_keys = [
			"chat_quick_question",
			"chat_quick_backend",
			"chat_quick_policy",
			"chat_quick_search_type",
			"chat_quick_k",
			"chat_quick_derived_k",
			"chat_quick_shortlist",
			"chat_quick_fetch_k",
			"chat_quick_lambda",
			"chat_quick_qe_enabled",
			"chat_quick_expand_n",
			"chat_quick_k_per_query",
			"chat_quick_expand_model",
			"chat_quick_rrf_k0",
			"chat_quick_expand_cache",
			"chat_quick_llm_enabled",
			"chat_quick_llm_model",
			"chat_quick_temp",
		]
		prev_snapshot = st.session_state.get("chat_quick_snapshot") or {}
		cur_snapshot_pre = {k: st.session_state.get(k) for k in _quick_keys if k in st.session_state}
		if prev_snapshot and cur_snapshot_pre and cur_snapshot_pre != prev_snapshot:
			st.session_state.chat_quick_expanded = True

		with st.expander(
			"Re-run last question with different params",
			expanded=bool(st.session_state.chat_quick_expanded),
		):
			st.caption("This lets you tweak settings right in chat without touching the left sidebar.")
			quick_question = st.text_input("Question", value=last_question, key="chat_quick_question")
			qc1, qc2, qc3 = st.columns(3)
			with qc1:
				quick_backend = st.selectbox(
					"Backend",
					options=["chroma", "qdrant"],
					index=(0 if backend == "chroma" else 1),
					key="chat_quick_backend",
				)
				quick_retrieval_policy = st.selectbox(
					"Routing Policy",
					options=["script_only", "derived_only", "derived_then_script"],
					index=["script_only", "derived_only", "derived_then_script"].index(retrieval_policy),
					key="chat_quick_policy",
				)
				quick_search_type = st.selectbox(
					"Search type",
					options=["similarity", "mmr"],
					index=["similarity", "mmr"].index(str(search_type)),
					key="chat_quick_search_type",
				)
			with qc2:
				quick_k = int(st.slider("Top-k", 1, 20, int(k), key="chat_quick_k"))
				quick_derived_k = int(st.slider("Derived k", 1, 30, int(derived_k), key="chat_quick_derived_k"))
			with qc3:
				quick_episode_shortlist_size = int(
					st.slider("Shortlist size", 1, 12, int(episode_shortlist_size), key="chat_quick_shortlist")
				)
				quick_llm_enabled = bool(
					st.checkbox("Generate answer (LLM)", value=bool(llm_enabled), key="chat_quick_llm_enabled")
				)

			qret1, qret2 = st.columns(2)
			with qret1:
				quick_fetch_k = int(
					st.slider(
						"MMR fetch_k",
						min_value=max(8, int(quick_k)),
						max_value=240,
						value=int(quick_fetch_k),
						disabled=(str(quick_search_type) != "mmr"),
						key="chat_quick_fetch_k",
					)
				)
			with qret2:
				quick_lambda_mult = float(
					st.slider(
						"MMR lambda",
						min_value=0.0,
						max_value=1.0,
						value=float(quick_lambda_mult),
						disabled=(str(quick_search_type) != "mmr"),
						key="chat_quick_lambda",
					)
				)

			st.markdown("---")
			st.caption("Optional: query expansion + fusion")
			qxe1, qxe2 = st.columns(2)
			with qxe1:
				quick_qe_enabled = bool(
					st.checkbox(
						"Expand queries + RRF fuse",
						value=bool(quick_qe_enabled),
						key="chat_quick_qe_enabled",
						on_change=_qe_mark_quick_dirty,
					)
				)
				quick_expand_n = int(
					st.slider(
						"Expand n",
						min_value=1,
						max_value=10,
						value=int(quick_expand_n),
						disabled=(not bool(quick_qe_enabled)),
						key="chat_quick_expand_n",
						on_change=_qe_mark_quick_dirty,
					)
				)
				quick_k_per_query = int(
					st.slider(
						"k per query",
						min_value=1,
						max_value=20,
						value=int(quick_k_per_query),
						disabled=(not bool(quick_qe_enabled)),
						key="chat_quick_k_per_query",
						on_change=_qe_mark_quick_dirty,
					)
				)
			with qxe2:
				quick_expand_model = st.text_input(
					"Expand model",
					value=str(quick_expand_model),
					disabled=(not bool(quick_qe_enabled)),
					key="chat_quick_expand_model",
					on_change=_qe_mark_quick_dirty,
				)
				quick_rrf_k0 = int(
					st.slider(
						"RRF k0",
						min_value=1,
						max_value=200,
						value=int(quick_rrf_k0),
						disabled=(not bool(quick_qe_enabled)),
						key="chat_quick_rrf_k0",
						on_change=_qe_mark_quick_dirty,
					)
				)
				quick_expand_cache_path = st.text_input(
					"Expansion cache",
					value=str(quick_expand_cache_path),
					disabled=(not bool(quick_qe_enabled)),
					key="chat_quick_expand_cache",
					on_change=_qe_mark_quick_dirty,
				)


			qcm1, qcm2 = st.columns([0.7, 0.3])
			with qcm1:
				quick_llm_model = st.text_input("LLM model", value=str(llm_model), key="chat_quick_llm_model")
			with qcm2:
				quick_temperature = float(
					st.slider("Temp", min_value=0.0, max_value=1.0, value=float(temperature), key="chat_quick_temp")
				)

			run_quick = st.button("Re-run in chat", key="chat_quick_rerun")
			if run_quick:
				question_to_run = str(quick_question).strip()
				if question_to_run:
					st.session_state.chat_last_question = question_to_run
					st.session_state.chat_messages.append({"role": "user", "content": question_to_run})
					with st.chat_message("user"):
						st.markdown(question_to_run)

			cc1, cc2 = st.columns([0.7, 0.3])
			with cc1:
				st.caption("Tip: changing values reruns the app; this panel should stay open now.")
			with cc2:
				if st.button("Collapse", key="chat_quick_collapse"):
					st.session_state.chat_quick_expanded = False
					st.rerun()

			# Update snapshot after widgets are created so future changes keep the panel open.
			st.session_state.chat_quick_snapshot = {k: st.session_state.get(k) for k in _quick_keys if k in st.session_state}

	if not question_to_run:
		with st.expander("Setup / expectations"):
			st.write("Local: uses Chroma under `db/`. Deployed: auto-switches to Qdrant when QDRANT_URL is set.")
			st.write("Policy `derived_then_script` uses derived cards to route into script chunks.")
		return

	# If quick rerun was used, override sidebar settings for this execution only.
	if str(st.session_state.get("chat_last_question") or "").strip() == str(question_to_run).strip():
		if "chat_quick_backend" in st.session_state:
			backend = str(quick_backend)
		if "chat_quick_policy" in st.session_state:
			retrieval_policy = str(quick_retrieval_policy)
		if "chat_quick_search_type" in st.session_state:
			search_type = str(quick_search_type)
		if "chat_quick_k" in st.session_state:
			k = int(quick_k)
		if "chat_quick_fetch_k" in st.session_state:
			fetch_k = int(quick_fetch_k)
		if "chat_quick_lambda" in st.session_state:
			lambda_mult = float(quick_lambda_mult)
		if "chat_quick_derived_k" in st.session_state:
			derived_k = int(quick_derived_k)
		if "chat_quick_shortlist" in st.session_state:
			episode_shortlist_size = int(quick_episode_shortlist_size)
		if "chat_quick_qe_enabled" in st.session_state:
			qe_enabled = bool(quick_qe_enabled)
		if "chat_quick_expand_n" in st.session_state:
			expand_n = int(quick_expand_n)
		if "chat_quick_expand_model" in st.session_state:
			expand_model = str(quick_expand_model)
		if "chat_quick_k_per_query" in st.session_state:
			k_per_query = int(quick_k_per_query)
		if "chat_quick_rrf_k0" in st.session_state:
			rrf_k0 = int(quick_rrf_k0)
		if "chat_quick_expand_cache" in st.session_state:
			expand_cache_path = str(quick_expand_cache_path)
		if "chat_quick_llm_enabled" in st.session_state:
			llm_enabled = bool(quick_llm_enabled)
		if "chat_quick_llm_model" in st.session_state:
			llm_model = str(quick_llm_model)
		if "chat_quick_temp" in st.session_state:
			temperature = float(quick_temperature)

	# Guardrail: query expansion needs OpenAI even if LLM answering is off.
	if qe_enabled and not openai_ok:
		st.warning(openai_err)
		qe_enabled = False

	# Build vectorstores lazily per request (simple + robust; can be cached later)
	try:
		with st.spinner("Initializing vectorstores…"):
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
	expanded_queries: Optional[List[str]] = None
	qe_details: Optional[Dict[str, Any]] = None
	qe_error: Optional[str] = None

	def _retrieve(db: Any, question: str, *, k_docs: int) -> List[Tuple[Any, Optional[float]]]:
		if str(search_type) == "mmr":
			return _mmr_pairs(
				db,
				question,
				k=int(k_docs),
				fetch_k=(int(fetch_k) if isinstance(fetch_k, int) else None),
				lambda_mult=float(lambda_mult),
			)
		return _similarity_pairs(db, question, k=int(k_docs))

	try:
		with st.spinner("Retrieving context…"):
			# Optionally expand the question into multiple retrieval queries.
			if qe_enabled:
				try:
					cache_path = Path(str(expand_cache_path)) if str(expand_cache_path or "").strip() else None
					expand_cfg = QueryExpansionConfig(
						enabled=True,
						n=int(expand_n),
						model=str(expand_model),
						temperature=0.0,
						cache_path=cache_path,
					)
					expander_llm = build_expander_llm(expand_cfg)
					expanded_queries = expand_queries(question=question_to_run, llm=expander_llm, config=expand_cfg)
				except Exception as e:
					# Fall back to base question if expansion fails.
					expanded_queries = [str(question_to_run)]
					qe_error = f"{type(e).__name__}: {e}"
			else:
				expanded_queries = [str(question_to_run)]

			per_q_k = int(k_per_query) if qe_enabled else int(k)
			per_q_k = max(1, min(50, per_q_k))

			def _fuse_or_single(db: Any, *, question: str) -> List[Tuple[Any, Optional[float]]]:
				"""Return top-k results, optionally using expanded queries + RRF."""
				if not qe_enabled or not expanded_queries or len(expanded_queries) <= 1:
					return _retrieve(db, question, k_docs=int(k))

				per_query: Dict[str, List[Tuple[Any, Optional[float]]]] = {}
				for q in expanded_queries:
					per_query[q] = _retrieve(db, q, k_docs=int(per_q_k))
				fused = rrf_fuse(per_query, k0=int(rrf_k0))
				return [(fd.doc, fd.fused_score) for fd in fused[: int(k)]]

			if retrieval_policy == "script_only":
				retrieved = _fuse_or_single(script_db, question=question_to_run)
			elif retrieval_policy == "derived_only":
				retrieved = _fuse_or_single(derived_db, question=question_to_run)
			else:
				# derived -> script routing (keep routing stage similarity-based for stability)
				derived_pairs = _similarity_pairs(derived_db, question_to_run, k=int(derived_k))
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

				def _retrieve_script_filtered(q: str, *, k_docs: int) -> List[Tuple[Any, Optional[float]]]:
					if not shortlist:
						return _retrieve(script_db, q, k_docs=int(k_docs))

					allowed = {s.strip().upper() for s in shortlist}
					candidate_k = max(int(k_docs) * 12, 60)
					if str(search_type) == "mmr":
						# Try grabbing a larger MMR set, then filter down.
						pairs = _mmr_pairs(
							script_db,
							q,
							k=int(max(int(k_docs) * 5, 40)),
							fetch_k=(int(fetch_k) if isinstance(fetch_k, int) else None),
							lambda_mult=float(lambda_mult),
						)
					else:
						pairs = _similarity_pairs(script_db, q, k=int(candidate_k))

					filtered: List[Tuple[Any, Optional[float]]] = []
					for d, s in pairs:
						meta = getattr(d, "metadata", None) or {}
						eid = str(meta.get("episode_id") or "").strip().upper()
						if eid and eid in allowed:
							filtered.append((d, s))
							if len(filtered) >= int(k_docs):
								break
					if filtered:
						return filtered

					if str(search_type) == "mmr":
						# Prefer a similarity fall-back when filtered MMR yields nothing.
						sim_pairs = _similarity_pairs(script_db, q, k=int(candidate_k))
						filtered2: List[Tuple[Any, Optional[float]]] = []
						for d, s in sim_pairs:
							meta = getattr(d, "metadata", None) or {}
							eid = str(meta.get("episode_id") or "").strip().upper()
							if eid and eid in allowed:
								filtered2.append((d, s))
								if len(filtered2) >= int(k_docs):
									break
						if filtered2:
							return filtered2

					# Last resort: return unfiltered results.
					return _retrieve(script_db, q, k_docs=int(k_docs))

				if not qe_enabled or not expanded_queries or len(expanded_queries) <= 1:
					retrieved = _retrieve_script_filtered(question_to_run, k_docs=int(k))
				else:
					per_query2: Dict[str, List[Tuple[Any, Optional[float]]]] = {}
					for q in expanded_queries:
						per_query2[q] = _retrieve_script_filtered(q, k_docs=int(per_q_k))
					fused2 = rrf_fuse(per_query2, k0=int(rrf_k0))
					retrieved = [(fd.doc, fd.fused_score) for fd in fused2[: int(k)]]

				routing = {
					"policy": "derived_then_script",
					"episode_shortlist": shortlist,
					"search_type": str(search_type),
					"mmr": (
						{"fetch_k": int(fetch_k) if isinstance(fetch_k, int) else None, "lambda": float(lambda_mult)}
						if str(search_type) == "mmr"
						else None
					),
					"query_expansion": (
						{
							"enabled": True,
							"n": int(expand_n),
							"model": str(expand_model),
							"k_per_query": int(per_q_k),
							"fusion": "rrf",
							"rrf_k0": int(rrf_k0),
						}
						if qe_enabled
						else {"enabled": False}
					),
					"derived_top": [
						{
							"rank": i + 1,
							"episode_id": str((getattr(d, "metadata", None) or {}).get("episode_id") or "").strip(),
							"score": (float(s) if s is not None else None),
							"source": str(
								(getattr(d, "metadata", None) or {}).get("source_file")
								or (getattr(d, "metadata", None) or {}).get("source")
								or ""
							),
						}
						for i, (d, s) in enumerate(derived_pairs[: min(len(derived_pairs), 12)])
					],
				}

			# Record query-expansion details for display even when not using derived routing.
			if qe_enabled:
				qe_details = {
					"enabled": True,
					"n": int(expand_n),
					"model": str(expand_model),
					"k_per_query": int(per_q_k),
					"fusion": "rrf",
					"rrf_k0": int(rrf_k0),
					"cache": (str(expand_cache_path) if str(expand_cache_path or "").strip() else None),
					"queries": list(expanded_queries or []),
					"error": qe_error,
				}
	except Exception as e:
		st.error(f"Retrieval failed: {type(e).__name__}: {e}")
		retrieved = []

	retrieval_ms = int((time.time() - t0) * 1000)

	if qe_enabled and expanded_queries:
		with st.expander(f"Expanded queries ({len(expanded_queries)} total)", expanded=False):
			st.caption("These are the retrieval queries generated from your question (first is always the original).")
			if isinstance(qe_details, dict) and qe_details.get("error"):
				st.warning(f"Query expansion fell back to the original question: {qe_details.get('error')}")
			st.text_area(
				"Queries",
				value="\n".join([f"{i+1}. {q}" for i, q in enumerate(expanded_queries)]),
				height=140,
			)
			if isinstance(qe_details, dict):
				cfg = {k: v for k, v in qe_details.items() if k not in {"queries"}}
				if cfg:
					st.caption("Expansion config")
					st.json(cfg)

	with st.expander(f"Retrieved context ({len(retrieved)} docs, {retrieval_ms}ms)", expanded=False):
		if routing:
			st.write("Routing:")
			st.json(routing)
		for i, (d, s) in enumerate(retrieved, start=1):
			st.markdown(f"**#{i}**  score={s if s is not None else 'n/a'}")
			st.write(_format_doc_line(d))
			st.text((getattr(d, "page_content", None) or "")[:900])

	context = "\n\n---\n\n".join([str(getattr(d, "page_content", None) or "") for (d, _s) in retrieved])

	answer_text = ""
	if not llm_enabled:
		answer_text = "(LLM disabled; retrieval only.)"
	else:
		try:
			with st.spinner("Generating answer…"):
				llm = ChatOpenAI(model=str(llm_model), temperature=float(temperature))
				sys, user = _answer_prompt(question=question_to_run, context=context)
				msg = llm.invoke([("system", sys), ("human", user)])
				answer_text = str(getattr(msg, "content", "") or "").strip()
		except Exception as e:
			answer_text = f"(LLM error: {type(e).__name__}: {e})"


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
	st.set_page_config(page_title="RAG Eval Dashboard", layout="wide")
	st.title("RAG Eval Dashboard")

	qp = _query_params()
	default_mode_raw = str(qp.get("mode") or "summary").strip().lower()

	# Stable internal mode keys (URL-friendly) with human labels.
	mode_labels: Dict[str, str] = {
		"summary": "Reports",
		"chat_debug": "Chat & Debug",
		"run_explorer": "Run Explorer",
	}

	# Back-compat for older/accidental query param values.
	legacy_map = {
		"reports": "summary",
		"report": "summary",
		"summary": "summary",
		"chat": "chat_debug",
		"chat_debug": "chat_debug",
		"chat & debug": "chat_debug",
		"chat_and_debug": "chat_debug",
		"run explorer": "run_explorer",
		"run_explorer": "run_explorer",
		"explorer": "run_explorer",
	}
	default_mode = legacy_map.get(default_mode_raw, "summary")
	if default_mode not in mode_labels:
		default_mode = "summary"

	mode = st.sidebar.radio(
		"Mode",
		options=list(mode_labels.keys()),
		index=list(mode_labels.keys()).index(default_mode),
		format_func=lambda k: mode_labels.get(str(k), str(k)),
	)

	# Keep URL query param in sync for easy sharing/reloads.
	try:
		if str(qp.get("mode") or "").strip().lower() != str(mode).strip().lower():
			try:
				st.query_params["mode"] = str(mode)  # type: ignore[attr-defined]
			except Exception:
				st.experimental_set_query_params(mode=str(mode))
	except Exception:
		pass

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
