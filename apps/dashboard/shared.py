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

import matplotlib.pyplot as plt
import streamlit as st

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

try:
	from qdrant_client import QdrantClient  # type: ignore
	from langchain_community.vectorstores import Qdrant as QdrantVS  # type: ignore

	_HAS_QDRANT = True
except Exception:
	QdrantClient = None  # type: ignore
	QdrantVS = None  # type: ignore
	_HAS_QDRANT = False


REPO_ROOT = Path(__file__).resolve().parents[2]
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
