from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Allow importing top-level packages (e.g., `rag/`) when running as a script from `experiments/`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.fusion import rrf_fuse
from rag.query_expansion import QueryExpansionConfig, build_expander_llm, expand_queries

# -----------------------------
# Defaults (adjust as needed)
# -----------------------------
DEFAULT_PERSIST_DIR = "db/chroma_db"
DEFAULT_COLLECTION_NAME: Optional[str] = None  # set if you explicitly named the collection in ingestion
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS: Optional[int] = None
DEFAULT_TIMEOUT: Optional[float] = None
DEFAULT_MAX_RETRIES: Optional[int] = None
DEFAULT_SEED: Optional[int] = None

DEFAULT_TEST_FILE = "experiments/test_queries.json"
DEFAULT_RUNS_DIR = "experiments/runs"

# If you want to record your chunking config even when baseline has metadata off,
# set these defaults to match ingestion pipeline values.
DEFAULT_CHUNK_SPLITTER = "CharacterTextSplitter"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 0

# MMR defaults
DEFAULT_FETCH_K: Optional[int] = None  # if None, we set to k*4 at runtime for mmr
DEFAULT_LAMBDA_MULT: float = 0.7

# Query expansion defaults
DEFAULT_QUERY_EXPANSION_ENABLED = False
DEFAULT_EXPAND_N = 5
DEFAULT_EXPAND_MODEL = "gpt-4.1-nano"
DEFAULT_K_PER_QUERY: Optional[int] = None  # if None, use min(k, 6)
DEFAULT_FUSION = "rrf"
DEFAULT_RRF_K0 = 60
DEFAULT_EXPAND_CACHE = "experiments/cache/query_expansion_cache.json"

# Routing defaults
DEFAULT_RETRIEVAL_POLICY = "blended"
DEFAULT_DERIVED_PERSIST_DIR = "db/chroma_db_derived_cards"
DEFAULT_DERIVED_COLLECTION_NAME = "derived_cards"
DEFAULT_DERIVED_K = 12
DEFAULT_EPISODE_SHORTLIST_SIZE = 6

# Safety: in derived_then_script, backfill with unfiltered script docs so a bad shortlist
# doesn't zero-out retrieval for quote/episode-ID questions.
DEFAULT_ROUTE_BACKFILL_UNFILTERED = True
DEFAULT_ROUTE_BACKFILL_MIN_DOCS = 3

# Blended retrieval defaults
DEFAULT_BLENDED_BASE_K_MULT = 1.0
DEFAULT_BLENDED_ROUTED_K_MULT = 1.0
DEFAULT_BLENDED_INCLUDE_DERIVED_FOR_NON_AGG = False

# Run log schema version (bump when you change log structure/semantics).
RUN_SCHEMA_VERSION = "v4"

# How much of the retrieved context to store in the run log for auditability/scoring.
# This is used by the scorer + dashboard Flight Recorder to judge answers against the
# actual context the LLM saw, rather than only short previews.
LOGGED_CONTEXT_TEXT_MAX_CHARS = 12000


def _file_fingerprint(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort fingerprint for auditability.

    Returns a dict with size/mtime/sha256 when the file exists and can be read.
    """
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None

        st = p.stat()
        size_bytes = int(st.st_size)
        mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0)

        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        return {
            "path": str(p),
            "size_bytes": size_bytes,
            "mtime_utc": mtime_utc.isoformat().replace("+00:00", "Z"),
            "sha256": h.hexdigest(),
        }
    except Exception:
        return None


def _dedupe_doc_pairs(pairs: List[Tuple[Any, Optional[float]]]) -> List[Tuple[Any, Optional[float]]]:
    """De-dupe by (source, page_content), keeping max score when available."""
    best: Dict[Tuple[Optional[str], str], Tuple[Any, Optional[float]]] = {}
    for doc, score in pairs:
        meta = getattr(doc, "metadata", None) or {}
        key = (meta.get("source"), getattr(doc, "page_content", None) or "")
        prev = best.get(key)
        if prev is None:
            best[key] = (doc, score)
            continue
        prev_score = prev[1]
        if prev_score is None and score is not None:
            best[key] = (doc, score)
        elif prev_score is not None and score is not None and float(score) > float(prev_score):
            best[key] = (doc, score)
    return list(best.values())


def _sort_pairs_best_first(pairs: List[Tuple[Any, Optional[float]]]) -> List[Tuple[Any, Optional[float]]]:
    # Scores are generally in [0,1] relevance for Chroma; treat None as lowest.
    return sorted(pairs, key=lambda ds: (-float(ds[1]) if ds[1] is not None else 1e9,))


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_sha(repo_root: Path) -> Optional[str]:
    """Best-effort git SHA for auditability.

    Returns None when git isn't available or repo_root isn't a git repo.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        sha = (res.stdout or "").strip()
        if res.returncode == 0 and sha:
            return sha
    except Exception:
        return None
    return None


def _load_derived_build_meta(derived_persist_directory: str) -> Optional[Dict[str, Any]]:
    """Load derived index build metadata if present.

    build_derived_cards_index.py writes a _build_meta.json into the persist dir.
    """
    try:
        p = Path(derived_persist_directory).expanduser().resolve() / "_build_meta.json"
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _data_version_block(
    *,
    script_persist_directory: str,
    script_collection_name: Optional[str],
    derived_persist_directory: str,
    derived_collection_name: Optional[str],
) -> Dict[str, Any]:
    derived_meta = _load_derived_build_meta(derived_persist_directory)

    # Fingerprint sqlite files when present. This gives a stable, defensible data-version
    # signal even if persist dir names are reused.
    script_sqlite_fp = _file_fingerprint(Path(script_persist_directory).expanduser().resolve() / "chroma.sqlite3")
    derived_sqlite_fp = _file_fingerprint(Path(derived_persist_directory).expanduser().resolve() / "chroma.sqlite3")

    # "build_id/tag" is best-effort: prefer explicit build_id/tag fields when present,
    # else fall back to persist-dir basename.
    derived_tag = None
    if isinstance(derived_meta, dict):
        # build_derived_cards_index.py writes created_at_utc and includes build prefixes.
        derived_tag = derived_meta.get("created_at_utc") or derived_meta.get("persist_dir")
    if not derived_tag:
        derived_tag = Path(derived_persist_directory).name

    return {
        "script": {
            "persist_directory": script_persist_directory,
            "collection_name": script_collection_name,
            "fingerprint": {
                "chroma_sqlite": script_sqlite_fp,
            },
        },
        "derived": {
            "persist_directory": derived_persist_directory,
            "collection_name": derived_collection_name,
            "build_tag": derived_tag,
            "build_meta": derived_meta,
            "fingerprint": {
                "chroma_sqlite": derived_sqlite_fp,
            },
        },
    }


def slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "run"


def preview_text(text: str, n: int = 240) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= n:
        return text
    return text[:n] + "…"


def first_line(text: str) -> str:
    if not text:
        return ""
    return text.splitlines()[0].strip()


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compute_score_stats(scores: List[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(scores)),
        "max": float(max(scores)),
        "mean": float(sum(scores) / len(scores)),
    }


def _counter_to_sorted_dict(counter: Counter[str]) -> Dict[str, int]:
    return {k: int(v) for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))}


def _top_share(counter: Counter[str], *, total: int) -> Optional[float]:
    if total <= 0 or not counter:
        return None
    return float(max(counter.values()) / total)


def _normalized_entropy(counter: Counter[str], *, total: int) -> Optional[float]:
    """Entropy normalized to [0, 1] (1.0 means perfectly uniform).

    Returns None if undefined (e.g., 0 or 1 unique values).
    """
    if total <= 0 or not counter:
        return None
    n = len(counter)
    if n <= 1:
        return None
    h = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p)
    return float(h / math.log(n))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


_AGG_QUESTION_RE = re.compile(
    r"\b(list|summari[sz]e|overview|timeline|chronolog|relationships?|girlfriends?|boyfriends?|romantic|dating|all\b|reasons?|compare|across|throughout)\b",
    re.IGNORECASE,
)


def is_aggregation_question(question: str) -> bool:
    return bool(_AGG_QUESTION_RE.search(question or ""))


def aggregation_readiness_score(
    *,
    context_docs: int,
    distinct_episode_count: int,
    top_episode_share: Optional[float],
    episode_entropy_norm: Optional[float],
) -> Optional[float]:
    """Heuristic [0, 100] score estimating whether context supports multi-episode aggregation.

    Higher is better for tasks like "list all", "summarize season", timelines, etc.
    This is intentionally simple and presentation-friendly.
    """
    if context_docs <= 0:
        return None

    diversity = float(episode_entropy_norm) if episode_entropy_norm is not None else 0.0
    concentration = float(top_episode_share) if top_episode_share is not None else 1.0
    anti_concentration = 1.0 - concentration

    # Coverage: what fraction of the context docs come from distinct episodes.
    coverage = float(distinct_episode_count / context_docs) if context_docs > 0 else 0.0

    score01 = 0.45 * _clamp01(diversity) + 0.35 * _clamp01(anti_concentration) + 0.20 * _clamp01(coverage)
    return float(round(100.0 * _clamp01(score01), 2))


def said_idk(answer: Optional[str]) -> bool:
    if not answer:
        return False
    a = answer.lower()
    return "i don't know" in a or "i do not know" in a or "not in the context" in a


_EP_CITE_RE = re.compile(r"\bS\d{2}E\d{2}\b")


def extract_episode_ids(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return sorted({m.group(0).upper() for m in _EP_CITE_RE.finditer(text)})


# Source paths look like: .../s02e12_the_injury_script.txt (underscore is a \w char)
# so word-boundary matching can fail. Keep this permissive.
_EP_IN_SOURCE_RE = re.compile(r"s(\d{2})e(\d{2})", re.IGNORECASE)


def extract_episode_ids_from_sources(sources: List[Optional[str]]) -> List[str]:
    out: set[str] = set()
    for src in sources:
        if not src:
            continue
        for m in _EP_IN_SOURCE_RE.finditer(src):
            out.add(f"S{m.group(1)}E{m.group(2)}")
    return sorted(out)


def extract_episode_ids_from_docs(docs: List[Any]) -> List[str]:
    """Extract canonical episode IDs from retrieved Document metadata.

    Falls back to parsing `source` when episode_id isn't present.
    """
    out: set[str] = set()
    sources: List[Optional[str]] = []

    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        eid = meta.get("episode_id")
        if isinstance(eid, str) and eid.strip():
            out.add(eid.strip().upper())
        sources.append(meta.get("source"))

    if not out:
        return extract_episode_ids_from_sources(sources)
    return sorted(out)


def _extract_episode_id_from_derived_doc(doc: Any) -> Optional[str]:
    meta = getattr(doc, "metadata", None) or {}
    eid = meta.get("episode_id")
    if isinstance(eid, str) and eid.strip():
        return eid.strip().upper()
    # Fallback: derived episode cards include an identity line like "Episode card: S02E12 — ..."
    txt = getattr(doc, "page_content", None) or ""
    ids = extract_episode_ids(txt)
    return ids[0] if ids else None


def _extract_episode_ids_from_derived_doc(doc: Any) -> List[str]:
    meta = getattr(doc, "metadata", None) or {}

    # Episode cards: single episode_id.
    eid = meta.get("episode_id")
    if isinstance(eid, str) and eid.strip():
        return [eid.strip().upper()]

    # Topic cards: episode_ids list.
    eids = meta.get("episode_ids")
    if isinstance(eids, list):
        out: List[str] = []
        seen: set[str] = set()
        for x in eids:
            s = str(x or "").strip().upper()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        if out:
            return out

    # Fallback: parse any SxxEyy tokens from the rendered derived card text.
    txt = getattr(doc, "page_content", None) or ""
    return extract_episode_ids(txt)


def _build_season_to_episode_ids_from_script_index(db: Chroma) -> Dict[int, List[str]]:
    """Build a mapping from season number -> episode_ids from the script index.

    We derive season from canonical episode ids like "S02E12". This is best-effort and
    safe to fail (returns {}). Used to expand season_card routing hits.
    """
    season_to: Dict[int, set[str]] = {}
    try:
        got = db._collection.get(include=["metadatas"])  # type: ignore[attr-defined]
        metas = got.get("metadatas") or []
        for md in metas:
            if not isinstance(md, dict):
                continue
            eid = md.get("episode_id")
            if not isinstance(eid, str):
                continue
            eid = eid.strip().upper()
            if not _EP_CITE_RE.fullmatch(eid):
                continue
            m = re.match(r"^S(\d{2})E(\d{2})$", eid)
            if not m:
                continue
            season = int(m.group(1))
            season_to.setdefault(season, set()).add(eid)
    except Exception:
        return {}

    return {s: sorted(list(eids)) for s, eids in sorted(season_to.items(), key=lambda kv: kv[0])}


def _episode_ids_for_shortlist(doc: Any, *, season_to_episode_ids: Optional[Dict[int, List[str]]]) -> List[str]:
    eids = _extract_episode_ids_from_derived_doc(doc)
    if eids:
        return eids

    if not season_to_episode_ids:
        return []

    meta = getattr(doc, "metadata", None) or {}
    if (meta.get("derived_type") or "") != "season_card":
        return []

    season_raw = meta.get("season")
    try:
        season = int(season_raw)
    except Exception:
        return []

    return list(season_to_episode_ids.get(season) or [])


def _doc_episode_id(doc: Any) -> Optional[str]:
    meta = getattr(doc, "metadata", None) or {}
    eid = meta.get("episode_id")
    if isinstance(eid, str) and eid.strip():
        return eid.strip().upper()

    # Fallback for legacy indexes (metadata off): parse from source path.
    src = meta.get("source")
    if isinstance(src, str) and src.strip():
        ids = extract_episode_ids_from_sources([src])
        return ids[0] if ids else None

    return None


def _doc_matches_filter(doc: Any, filt: Optional[Dict[str, Any]]) -> bool:
    if not filt:
        return True

    meta = getattr(doc, "metadata", None) or {}
    for k, v in filt.items():
        # Support only the minimal filter shapes we use in routing.
        if isinstance(v, dict) and "$in" in v:
            allowed = v.get("$in")
            if not isinstance(allowed, list):
                return False

            if k == "episode_id":
                eid = _doc_episode_id(doc)
                allowed_norm = {str(x).strip().upper() for x in allowed if str(x).strip()}
                return bool(eid and eid in allowed_norm)

            val = meta.get(k)
            return str(val) in {str(x) for x in allowed}

        # Equality
        mv = meta.get(k)
        if isinstance(v, str):
            if str(mv or "").strip().lower() != v.strip().lower():
                return False
        else:
            if mv != v:
                return False

    return True


def _similarity_search_with_scores(
    db: Chroma,
    question: str,
    *,
    k: int,
    filt: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Any, Optional[float]]]:
    if not filt:
        return [(doc, float(score)) for (doc, score) in db.similarity_search_with_relevance_scores(question, k=k)]

    # Try passing filters through to LangChain/Chroma. Different versions use different kw names.
    last_exc: Optional[Exception] = None
    for kw in ("filter", "where"):
        try:
            res = db.similarity_search_with_relevance_scores(question, k=k, **{kw: filt})
            return [(doc, float(score)) for (doc, score) in res]
        except TypeError as e:
            last_exc = e
        except Exception as e:
            # Could be a filter-shape issue; fall back.
            last_exc = e

    # Fall back to post-filtering a bigger candidate pool.
    candidate_k = max(int(k) * 12, 60)
    scored = db.similarity_search_with_relevance_scores(question, k=candidate_k)
    out: List[Tuple[Any, Optional[float]]] = []
    for doc, score in scored:
        if _doc_matches_filter(doc, filt):
            out.append((doc, float(score)))
            if len(out) >= int(k):
                break
    if out:
        return out

    # If nothing matched, return empty; include the last error in caller diagnostics if needed.
    _ = last_exc
    return []


def _retrieve_docs_with_scores(
    db: Chroma,
    question: str,
    *,
    search_type: str,
    k: int,
    fetch_k: Optional[int],
    lambda_mult: float,
    filt: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Any, Optional[float]]]:
    """Return a uniform (doc, score) shape across retrieval modes.

    Note: Chroma's MMR API returns docs without scores, so for MMR we
    backfill scores using a similarity scoring pass over the same candidate pool.
    """

    if search_type == "similarity":
        return _similarity_search_with_scores(db, question, k=int(k), filt=filt)

    if search_type == "mmr":
        effective_fetch_k = fetch_k or (int(k) * 4)

        # Try to keep filtering inside Chroma if supported.
        mmr_docs: List[Any] = []
        if filt:
            for kw in ("filter", "where"):
                try:
                    mmr_docs = db.max_marginal_relevance_search(
                        question,
                        k=int(k),
                        fetch_k=int(effective_fetch_k),
                        lambda_mult=lambda_mult,
                        **{kw: filt},
                    )
                    break
                except TypeError:
                    mmr_docs = []
                except Exception:
                    mmr_docs = []
        if not mmr_docs:
            # Either no filter, or filter not supported. Fall back to unfiltered MMR,
            # then post-filter. If filtering still yields nothing, fall back to similarity.
            base_docs = db.max_marginal_relevance_search(
                question,
                k=int(effective_fetch_k),
                fetch_k=int(max(int(effective_fetch_k) * 4, int(effective_fetch_k))),
                lambda_mult=lambda_mult,
            )
            if filt:
                mmr_docs = [d for d in base_docs if _doc_matches_filter(d, filt)][: int(k)]
            else:
                mmr_docs = base_docs[: int(k)]

            if filt and not mmr_docs:
                # Best-effort: filtered similarity instead of empty context.
                return _similarity_search_with_scores(db, question, k=int(k), filt=filt)

        # Build a best-effort score lookup from similarity scoring.
        score_map: Dict[Tuple[Optional[str], str], float] = {}
        try:
            scored = _similarity_search_with_scores(
                db,
                question,
                k=int(effective_fetch_k),
                filt=filt,
            )
            for doc, score in scored:
                if score is None:
                    continue
                key = (doc.metadata.get("source"), doc.page_content or "")
                prev = score_map.get(key)
                s = float(score)
                if prev is None or s > prev:
                    score_map[key] = s
        except Exception:
            score_map = {}

        out_pairs: List[Tuple[Any, Optional[float]]] = []
        for doc in mmr_docs:
            key = (doc.metadata.get("source"), doc.page_content or "")
            out_pairs.append((doc, score_map.get(key)))
        return out_pairs

    raise ValueError(f"Unsupported search_type: {search_type}")


_QUOTE_RE = re.compile(r"[\"\u201C\u201D]([^\"\u201C\u201D]{8,240})[\"\u201C\u201D]")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def extract_answer_quotes(answer: Optional[str]) -> List[str]:
    if not answer:
        return []
    quotes = [m.group(1).strip() for m in _QUOTE_RE.finditer(answer or "")]
    # De-dupe while keeping order.
    seen: set[str] = set()
    out: List[str] = []
    for q in quotes:
        k = normalize_for_match(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out


def any_quote_in_context(quotes: List[str], context_text: str) -> bool:
    if not quotes:
        return False
    ctx = normalize_for_match(context_text or "")
    for q in quotes:
        qn = normalize_for_match(q)
        if qn in ctx:
            return True
        # Tolerate trailing punctuation differences (e.g., "The Injury." vs "The Injury")
        q2 = (q or "").strip().rstrip(" .,!?:;\"")
        if q2 and normalize_for_match(q2) in ctx:
            return True
    return False


def cited_episode(answer: Optional[str]) -> bool:
    if not answer:
        return False
    return _EP_CITE_RE.search(answer) is not None


def safe_mean_int(values: List[int]) -> Optional[int]:
    if not values:
        return None
    return int(sum(values) / len(values))


def safe_mean_float(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def extract_usage(msg: Any) -> Dict[str, Optional[int]]:
    """
    Best-effort token usage extraction for langchain-openai 0.3.35 / langchain-core 0.3.83.
    """
    prompt = completion = total = None

    usage = getattr(msg, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion = usage.get("output_tokens") or usage.get("completion_tokens")
        total = usage.get("total_tokens")

    if prompt is None and completion is None and total is None:
        rm = getattr(msg, "response_metadata", None)
        if isinstance(rm, dict):
            for key in ("token_usage", "usage"):
                u = rm.get(key)
                if isinstance(u, dict):
                    prompt = u.get("prompt_tokens") or u.get("input_tokens")
                    completion = u.get("completion_tokens") or u.get("output_tokens")
                    total = u.get("total_tokens")
                    break

    def _to_int(x: Any) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip(), 10)
        return None

    return {
        "prompt_tokens": _to_int(prompt),
        "completion_tokens": _to_int(completion),
        "total_tokens": _to_int(total),
    }


# -----------------------------
# Test cases
# -----------------------------
@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    expected_notes: str = ""
    expected_episode_ids: Tuple[str, ...] = ()


def load_cases(test_file: Path) -> List[Case]:
    raw = read_json_file(test_file)
    if not isinstance(raw, list):
        raise ValueError(f"Test file must be a JSON array: {test_file}")

    cases: List[Case] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{i} must be an object")

        case_id = str(item.get("case_id") or f"q{i+1}")
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"Case {case_id} is missing a question")

        expected_notes = str(item.get("expected_notes") or "")
        expected_episode_ids_raw = item.get("expected_episode_ids")
        expected_episode_ids: List[str] = []
        if isinstance(expected_episode_ids_raw, list):
            for e in expected_episode_ids_raw:
                s = str(e or "").strip().upper()
                if _EP_CITE_RE.fullmatch(s):
                    expected_episode_ids.append(s)
        cases.append(
            Case(
                case_id=case_id,
                question=question,
                expected_notes=expected_notes,
                expected_episode_ids=tuple(expected_episode_ids),
            )
        )
    return cases


# -----------------------------
# Vector store
# -----------------------------
def build_vectorstore(
    persist_directory: str,
    embed_model: str,
    collection_name: Optional[str] = None,
) -> Chroma:
    embeddings = OpenAIEmbeddings(model=embed_model)
    kwargs: Dict[str, Any] = {
        "persist_directory": persist_directory,
        "embedding_function": embeddings,
    }
    if collection_name:
        kwargs["collection_name"] = collection_name
    return Chroma(**kwargs)


def get_index_counts(db: Chroma) -> Dict[str, Optional[int]]:
    try:
        n = int(db._collection.count())  # type: ignore[attr-defined]
        return {"num_vectors": n}
    except Exception:
        return {"num_vectors": None}


def _index_has_metadata_key(db: Chroma, key: str) -> bool:
    """Best-effort check for whether the underlying Chroma collection stores a metadata key."""
    try:
        got = db._collection.get(include=["metadatas"], limit=25)  # type: ignore[attr-defined]
        metas = got.get("metadatas") or []
        for md in metas:
            if isinstance(md, dict) and key in md:
                return True
        return False
    except Exception:
        return False


# -----------------------------
# LLM
# -----------------------------
def build_llm(
    *,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: Optional[float],
    max_retries: Optional[int],
    seed: Optional[int],
) -> ChatOpenAI:
    model_kwargs: Dict[str, Any] = {}
    if seed is not None:
        model_kwargs["seed"] = seed

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "model_kwargs": model_kwargs,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    return ChatOpenAI(**kwargs)


def answer_with_llm(llm: ChatOpenAI, question: str, docs_text: str) -> Any:
    system = (
        "You are a QA assistant for questions about the TV show The Office.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context does not contain the answer, say you don't know.\n"
        "When possible, cite episode identifiers present in the context (e.g., 'S02E06').\n"
        "When you make a specific claim, include at least one short exact quote from the context in double quotes."
    )
    user = f"Question: {question}\n\nContext:\n{docs_text}"
    return llm.invoke([("system", system), ("human", user)])


# -----------------------------
# Runner
# -----------------------------
def run_eval(
    *,
    test_file: Path,
    runs_dir: Path,
    persist_directory: str,
    collection_name: Optional[str],
    retrieval_policy: str,
    derived_persist_directory: str,
    derived_collection_name: Optional[str],
    derived_k: int,
    episode_shortlist_size: int,
    route_backfill_unfiltered: bool,
    route_backfill_min_docs: int,
    blended_base_k_mult: float,
    blended_routed_k_mult: float,
    blended_include_derived_for_non_agg: bool,
    embed_model: str,
    search_type: str,
    k: int,
    fetch_k: Optional[int],
    lambda_mult: float,
    query_expansion: bool,
    expand_n: int,
    expand_model: str,
    k_per_query: Optional[int],
    fusion: str,
    rrf_k0: int,
    expand_cache: Optional[str],
    llm_enabled: bool,
    llm_model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: Optional[float],
    max_retries: Optional[int],
    seed: Optional[int],
    run_name: str,
    notes: str,
    chunk_splitter: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    cases = load_cases(test_file)
    script_db = build_vectorstore(
        persist_directory=persist_directory,
        embed_model=embed_model,
        collection_name=collection_name,
    )
    index_counts = get_index_counts(script_db)

    script_index_has_episode_id = _index_has_metadata_key(script_db, "episode_id")
    season_to_episode_ids: Optional[Dict[int, List[str]]] = None
    if script_index_has_episode_id:
        season_to_episode_ids = _build_season_to_episode_ids_from_script_index(script_db)

    derived_db: Optional[Chroma] = None
    derived_index_counts: Optional[Dict[str, Optional[int]]] = None
    if retrieval_policy in {"derived_only", "derived_then_script", "auto", "blended"}:
        derived_db = build_vectorstore(
            persist_directory=derived_persist_directory,
            embed_model=embed_model,
            collection_name=derived_collection_name,
        )
        derived_index_counts = get_index_counts(derived_db)

    llm = None
    if llm_enabled:
        llm = build_llm(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
        )

    expander_llm = None
    expand_cfg = QueryExpansionConfig(
        enabled=bool(query_expansion),
        n=int(expand_n),
        model=str(expand_model),
        temperature=0.0,
        max_retries=max_retries,
        seed=seed,
        cache_path=(Path(expand_cache) if expand_cache else None),
    )
    if query_expansion:
        expander_llm = build_expander_llm(expand_cfg)

    created_at = utc_now_iso()
    run_id = f"{created_at.replace(':', '-')}_{slug(run_name)}"

    code_version = {
        "git_sha": _git_sha(_REPO_ROOT),
    }

    data_version = _data_version_block(
        script_persist_directory=str(persist_directory),
        script_collection_name=collection_name,
        derived_persist_directory=str(derived_persist_directory),
        derived_collection_name=derived_collection_name,
    )

    out: Dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "run_name": run_name,
            "created_at_utc": created_at,
            "notes": notes,
        },
        "config": {
            "run_schema_version": RUN_SCHEMA_VERSION,
            "code_version": code_version,
            "data_version": data_version,
            "vectorstore": {
                "kind": "chroma",
                "persist_directory": persist_directory,
                "collection_name": collection_name,
                **index_counts,
            },
            "derived_vectorstore": (
                {
                    "kind": "chroma",
                    "persist_directory": derived_persist_directory,
                    "collection_name": derived_collection_name,
                    **(derived_index_counts or {}),
                }
                if derived_db is not None
                else None
            ),
            "embedding": {"provider": "openai", "model": embed_model},
            "chunking": {
                "splitter": chunk_splitter,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "retrieval": {
                "policy": retrieval_policy,
                "search_type": search_type,
                "k": k,
                "fetch_k": (fetch_k if search_type == "mmr" else None),
                "lambda_mult": (lambda_mult if search_type == "mmr" else None),
                "routing": (
                    {
                        "derived_k": int(derived_k),
                        "episode_shortlist_size": int(episode_shortlist_size),
                        "route_backfill_unfiltered": bool(route_backfill_unfiltered),
                        "route_backfill_min_docs": int(route_backfill_min_docs),
                        "blended": {
                            "base_k_mult": float(blended_base_k_mult),
                            "routed_k_mult": float(blended_routed_k_mult),
                            "include_derived_for_non_agg": bool(blended_include_derived_for_non_agg),
                        }
                    }
                    if retrieval_policy in {"derived_then_script", "auto", "blended"}
                    else None
                ),
                "query_expansion": {
                    "enabled": bool(query_expansion),
                    "n": int(expand_n) if query_expansion else None,
                    "model": str(expand_model) if query_expansion else None,
                    "k_per_query": (int(k_per_query) if k_per_query is not None else None) if query_expansion else None,
                    "fusion": str(fusion) if query_expansion else None,
                    "rrf_k0": int(rrf_k0) if (query_expansion and fusion == "rrf") else None,
                    "cache": str(expand_cache) if (query_expansion and expand_cache) else None,
                },
            },
            "llm": {
                "enabled": llm_enabled,
                "provider": "openai" if llm_enabled else None,
                "model": llm_model if llm_enabled else None,
                "temperature": temperature if llm_enabled else None,
                "max_tokens": max_tokens if llm_enabled else None,
                "timeout": timeout if llm_enabled else None,
                "max_retries": max_retries if llm_enabled else None,
                "seed": seed if llm_enabled else None,
            },
        },
        "cases": [],
        "summary": {},
    }

    answer_latencies: List[int] = []
    retrieval_latencies: List[int] = []
    total_context_chars: List[int] = []
    total_context_docs: List[int] = []

    distinct_episode_counts: List[int] = []
    distinct_source_counts: List[int] = []
    top_episode_shares: List[float] = []
    episode_entropy_norms: List[float] = []
    cited_not_in_context_counts: List[int] = []
    agg_readiness_scores: List[float] = []
    agg_readiness_scores_agg_questions: List[float] = []

    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []

    errors: List[str] = []

    retrieval_failures: List[bool] = []
    grounding_failures: List[bool] = []
    quote_match_rates: List[bool] = []

    def retrieve_docs_with_scores_routed(
        question: str,
        *,
        k_override: Optional[int] = None,
    ) -> Tuple[List[Tuple[Any, Optional[float]]], Optional[Dict[str, Any]]]:
        effective_k = int(k_override) if k_override is not None else int(k)

        if retrieval_policy == "script_only":
            pairs = _retrieve_docs_with_scores(
                script_db,
                question,
                search_type=search_type,
                k=effective_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filt=None,
            )
            return pairs, None

        if retrieval_policy == "derived_only":
            if derived_db is None:
                raise RuntimeError("derived_only policy requires derived_db")
            pairs = _retrieve_docs_with_scores(
                derived_db,
                question,
                search_type=search_type,
                k=effective_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filt=None,
            )
            return pairs, {"policy": "derived_only"}

        if retrieval_policy in {"derived_then_script", "auto"}:
            if derived_db is None:
                raise RuntimeError("derived_then_script policy requires derived_db")

            effective_policy = "derived_then_script"
            if retrieval_policy == "auto":
                # Only apply routing to aggregation-like questions; otherwise stay script-only.
                if not is_aggregation_question(question):
                    effective_policy = "script_only"

            if effective_policy == "script_only":
                pairs = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type=search_type,
                    k=effective_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=None,
                )
                return pairs, {"policy": "auto", "effective_policy": "script_only"}

            # Stage 1: use derived episode cards to propose which episodes matter.
            derived_pairs = _similarity_search_with_scores(
                derived_db,
                question,
                k=int(derived_k),
                filt={"derived_type": {"$in": ["episode_card", "season_card", "topic_card"]}},
            )

            shortlist: List[str] = []
            seen: set[str] = set()
            for d, _s in derived_pairs:
                for eid in _episode_ids_for_shortlist(d, season_to_episode_ids=season_to_episode_ids):
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    shortlist.append(eid)
                    if len(shortlist) >= int(episode_shortlist_size):
                        break
                if len(shortlist) >= int(episode_shortlist_size):
                    break

            # Stage 2: retrieve scripts/summaries constrained to that episode shortlist.
            shortlist_set = {s.strip().upper() for s in shortlist if str(s).strip()}

            script_filter: Optional[Dict[str, Any]] = None
            script_filter_mode: str = "none"

            # Prefer metadata pushdown when episode_id exists in the script index.
            if shortlist_set and script_index_has_episode_id:
                script_filter = {"episode_id": {"$in": sorted(shortlist_set)}}
                script_filter_mode = "metadata"
                script_pairs = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type=search_type,
                    k=effective_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=script_filter,
                )
            elif shortlist_set:
                # Legacy script index: only `source` is present. Retrieve a bigger candidate pool
                # and filter by parsing episode from source.
                script_filter_mode = "source_parse"
                candidate_k = max(int(effective_k) * 14, 80)
                candidates = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type="similarity",  # stable ordering for filtering
                    k=int(candidate_k),
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=None,
                )
                filtered = [(d, s) for (d, s) in candidates if (_doc_episode_id(d) or "") in shortlist_set]
                script_pairs = filtered[: int(effective_k)]
                # Backfill with unfiltered docs to avoid catastrophic misses.
                if route_backfill_unfiltered:
                    min_docs = max(int(route_backfill_min_docs), 0)
                    if len(script_pairs) < min_docs:
                        seen_keys = {
                            (
                                (getattr(d, "metadata", {}) or {}).get("source"),
                                getattr(d, "page_content", None) or "",
                            )
                            for (d, _s) in script_pairs
                        }
                        for d, s in candidates:
                            key = ((getattr(d, "metadata", {}) or {}).get("source"), getattr(d, "page_content", None) or "")
                            if key in seen_keys:
                                continue
                            script_pairs.append((d, s))
                            seen_keys.add(key)
                            if len(script_pairs) >= int(effective_k):
                                break
            else:
                script_pairs = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type=search_type,
                    k=effective_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=None,
                )

            routing = {
                "policy": ("auto" if retrieval_policy == "auto" else "derived_then_script"),
                "effective_policy": effective_policy,
                "derived": {
                    "k": int(derived_k),
                    "filter": {"derived_type": {"$in": ["episode_card", "season_card", "topic_card"]}},
                    "included_in_context": bool(is_aggregation_question(question)),
                    "results": [
                        {
                            "rank": i + 1,
                            "episode_ids": _episode_ids_for_shortlist(d, season_to_episode_ids=season_to_episode_ids),
                            "derived_type": (getattr(d, "metadata", {}) or {}).get("derived_type"),
                            "season": (getattr(d, "metadata", {}) or {}).get("season"),
                            "title": (getattr(d, "metadata", {}) or {}).get("title"),
                            "score": (float(s) if s is not None else None),
                        }
                        for i, (d, s) in enumerate(derived_pairs[: min(len(derived_pairs), 12)])
                    ],
                },
                "episode_shortlist": shortlist,
                "script_filter": script_filter,
                "script_filter_mode": script_filter_mode,
                "script_index_has_episode_id": bool(script_index_has_episode_id),
                "backfill": {
                    "enabled": bool(route_backfill_unfiltered),
                    "min_docs": int(route_backfill_min_docs),
                },
            }
            # For aggregation-like questions, include a handful of derived cards directly
            # in the context so the model can aggregate from rollups first, then use scripts
            # for supporting quotes.
            if is_aggregation_question(question):
                derived_context_k = min(len(derived_pairs), max(2, min(6, int(round(0.4 * float(effective_k))))))
                derived_context_pairs = derived_pairs[: int(derived_context_k)]
            else:
                derived_context_pairs = []

            combined = _dedupe_doc_pairs(derived_context_pairs + script_pairs)
            combined = _sort_pairs_best_first(combined)
            combined = combined[: int(effective_k)]
            return combined, routing

        if retrieval_policy == "blended":
            # Baseline scripts first (high recall for pinpoint questions).
            base_k = max(1, int(round(float(blended_base_k_mult) * float(effective_k))))
            base_pairs = _retrieve_docs_with_scores(
                script_db,
                question,
                search_type=search_type,
                k=base_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filt=None,
            )

            agg_like = is_aggregation_question(question)
            include_derived = bool(agg_like or blended_include_derived_for_non_agg)
            if not include_derived:
                # Pure baseline for non-aggregation questions.
                return base_pairs[: int(effective_k)], {"policy": "blended", "effective_policy": "script_only", "agg_like": agg_like}

            if derived_db is None:
                # Should not happen due to earlier setup, but keep safe.
                return base_pairs[: int(effective_k)], {"policy": "blended", "effective_policy": "script_only", "agg_like": agg_like, "note": "derived_db_missing"}

            # Use derived episode cards to propose which episodes to expand with scripts.
            derived_pairs = _similarity_search_with_scores(
                derived_db,
                question,
                k=int(derived_k),
                filt={"derived_type": {"$in": ["episode_card", "season_card", "topic_card"]}},
            )

            # For aggregation-like questions, include derived rollups directly in context.
            if agg_like:
                derived_context_k = min(len(derived_pairs), max(2, min(6, int(round(0.4 * float(effective_k))))))
                derived_context_pairs = derived_pairs[: int(derived_context_k)]
            else:
                derived_context_pairs = []

            shortlist: List[str] = []
            seen: set[str] = set()
            for d, _s in derived_pairs:
                for eid in _episode_ids_for_shortlist(d, season_to_episode_ids=season_to_episode_ids):
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    shortlist.append(eid)
                    if len(shortlist) >= int(episode_shortlist_size):
                        break
                if len(shortlist) >= int(episode_shortlist_size):
                    break

            routed_k = max(1, int(round(float(blended_routed_k_mult) * float(effective_k))))

            routed_pairs: List[Tuple[Any, Optional[float]]]
            script_filter: Optional[Dict[str, Any]] = None
            script_filter_mode: str = "none"

            if shortlist and script_index_has_episode_id:
                script_filter = {"episode_id": {"$in": [s.strip().upper() for s in shortlist]}}
                script_filter_mode = "metadata"
                routed_pairs = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type=search_type,
                    k=routed_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=script_filter,
                )
            elif shortlist:
                script_filter_mode = "source_parse"
                candidate_k = max(int(routed_k) * 14, 80)
                candidates = _retrieve_docs_with_scores(
                    script_db,
                    question,
                    search_type="similarity",
                    k=int(candidate_k),
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filt=None,
                )
                shortlist_set = {s.strip().upper() for s in shortlist}
                routed_pairs = [(d, s) for (d, s) in candidates if (_doc_episode_id(d) or "") in shortlist_set][: int(routed_k)]
            else:
                routed_pairs = []

            combined = _dedupe_doc_pairs(derived_context_pairs + base_pairs + routed_pairs)
            combined = _sort_pairs_best_first(combined)
            combined = combined[: int(effective_k)]

            routing = {
                "policy": "blended",
                "effective_policy": "blended",
                "agg_like": bool(agg_like),
                "base": {"k": int(base_k), "returned": len(base_pairs)},
                "derived": {
                    "k": int(derived_k),
                    "filter": {"derived_type": {"$in": ["episode_card", "season_card", "topic_card"]}},
                    "included_in_context": bool(agg_like),
                    "results": [
                        {
                            "rank": i + 1,
                            # Back-compat: keep a single episode_id field while also logging
                            # full episode_ids (topic cards can map to many episodes).
                            "episode_id": _extract_episode_id_from_derived_doc(d),
                            "episode_ids": _episode_ids_for_shortlist(d, season_to_episode_ids=season_to_episode_ids),
                            "derived_type": (getattr(d, "metadata", {}) or {}).get("derived_type"),
                            "topic_id": (getattr(d, "metadata", {}) or {}).get("topic_id"),
                            "topic_type": (getattr(d, "metadata", {}) or {}).get("topic_type"),
                            "season": (getattr(d, "metadata", {}) or {}).get("season"),
                            "title": (getattr(d, "metadata", {}) or {}).get("title"),
                            "score": (float(s) if s is not None else None),
                        }
                        for i, (d, s) in enumerate(derived_pairs[: min(len(derived_pairs), 10)])
                    ],
                },
                "episode_shortlist": shortlist,
                "routed": {
                    "k": int(routed_k),
                    "script_filter": script_filter,
                    "script_filter_mode": script_filter_mode,
                    "returned": len(routed_pairs),
                },
                "combined": {"returned": len(combined)},
            }
            return combined, routing

        raise ValueError(f"Unsupported retrieval_policy: {retrieval_policy}")

    for case in cases:
        # Retrieval
        t0 = time.time()
        try:
            expanded_queries: Optional[List[str]] = None
            multi_query_details: Optional[Dict[str, Any]] = None

            if query_expansion:
                if expander_llm is None:
                    raise RuntimeError("query_expansion enabled but expander_llm is None")

                expanded_queries = expand_queries(question=case.question, llm=expander_llm, config=expand_cfg)
                per_q_k = int(k_per_query) if k_per_query is not None else min(int(k), 6)

                per_query_results: Dict[str, List[Tuple[Any, Optional[float]]]] = {}
                per_query_routing: Dict[str, Any] = {}
                for q in expanded_queries:
                    pairs, routing = retrieve_docs_with_scores_routed(q, k_override=per_q_k)
                    per_query_results[q] = pairs
                    if routing is not None:
                        # Keep it compact; routing logs can get large quickly.
                        per_query_routing[q] = {
                            "policy": routing.get("policy"),
                            "episode_shortlist": routing.get("episode_shortlist"),
                        }

                if fusion == "rrf":
                    fused = rrf_fuse(per_query_results, k0=int(rrf_k0))
                    retrieved = [(fd.doc, fd.fused_score) for fd in fused[: int(k)]]
                    multi_query_details = {
                        "expanded_queries": expanded_queries,
                        "k_per_query": per_q_k,
                        "fusion": "rrf",
                        "rrf_k0": int(rrf_k0),
                        "candidates": sum(len(v) for v in per_query_results.values()),
                        "deduped": len(fused),
                        "routing": (per_query_routing if per_query_routing else None),
                        "provenance": [
                            {
                                "rank": i + 1,
                                "retrieved_by": fd.retrieved_by,
                                "ranks_by_query": fd.ranks_by_query,
                                "best_base_score": fd.best_base_score,
                                "fused_score": fd.fused_score,
                                "source": getattr(fd.doc, "metadata", {}).get("source"),
                            }
                            for i, fd in enumerate(fused[: int(k)])
                        ],
                    }
                else:
                    raise ValueError(f"Unsupported fusion: {fusion}")
            else:
                retrieved, routing_details = retrieve_docs_with_scores_routed(case.question)

            retrieval_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            retrieval_ms = int((time.time() - t0) * 1000)
            retrieved = []
            routing_details = None
            errors.append(f"{case.case_id}: retrieval_error: {type(e).__name__}: {e}")

        retrieval_latencies.append(retrieval_ms)

        results = []
        context_parts = []
        scores: List[float] = []

        result_sources: List[Optional[str]] = []

        for rank, (doc, score) in enumerate(retrieved, start=1):
            src = doc.metadata.get("source")
            txt = doc.page_content or ""
            result_sources.append(src)
            if score is not None:
                scores.append(float(score))
            results.append(
                {
                    "rank": rank,
                    "source": src,
                    "episode_id": doc.metadata.get("episode_id"),
                    "doc_type": doc.metadata.get("doc_type"),
                    "derived_type": doc.metadata.get("derived_type"),
                    "topic_id": doc.metadata.get("topic_id"),
                    "topic_type": doc.metadata.get("topic_type"),
                    "chunk_type": doc.metadata.get("chunk_type"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "first_line": first_line(txt),
                    "preview": preview_text(txt, n=240),
                    "score": (float(score) if score is not None else None),
                }
            )
            context_parts.append(txt)

        context_text = "\n\n---\n\n".join(context_parts)
        context_chars = len(context_text)
        context_docs = len(context_parts)
        total_context_chars.append(context_chars)
        total_context_docs.append(context_docs)

        # Answer (optional)
        answer_text: Optional[str] = None
        answer_latency_ms: Optional[int] = None
        usage: Dict[str, Optional[int]] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        answer_error: Optional[str] = None

        if llm is not None:
            t1 = time.time()
            try:
                msg = answer_with_llm(llm, case.question, context_text)
                answer_latency_ms = int((time.time() - t1) * 1000)
                answer_text = getattr(msg, "content", None) or None
                usage = extract_usage(msg)

                if usage["prompt_tokens"] is not None:
                    prompt_tokens.append(usage["prompt_tokens"])
                if usage["completion_tokens"] is not None:
                    completion_tokens.append(usage["completion_tokens"])
                if usage["total_tokens"] is not None:
                    total_tokens.append(usage["total_tokens"])
            except Exception as e:
                answer_latency_ms = int((time.time() - t1) * 1000)
                answer_error = f"{type(e).__name__}: {e}"
                errors.append(f"{case.case_id}: answer_error: {answer_error}")

            if answer_latency_ms is not None:
                answer_latencies.append(answer_latency_ms)

        logged_context_text = context_text
        context_text_truncated = False
        if len(logged_context_text) > int(LOGGED_CONTEXT_TEXT_MAX_CHARS):
            logged_context_text = logged_context_text[: int(LOGGED_CONTEXT_TEXT_MAX_CHARS)] + "\n… (truncated)"
            context_text_truncated = True

        out_case = {
            "case_id": case.case_id,
            "question": case.question,
            "expected_notes": case.expected_notes,
            "expected": {
                "episode_ids": list(case.expected_episode_ids) if case.expected_episode_ids else None,
            },
            "retrieval": {
                "top_k": k,
                "latency_ms": retrieval_ms,
                "stats": compute_score_stats(scores),
                "results": results,
                "multi_query": multi_query_details,
                "routing": (routing_details if (not query_expansion) else None),
            },
            "answer": {
                "text": answer_text,
                "latency_ms": answer_latency_ms,
                "error": answer_error,
                "usage": usage,
                "context_text": (logged_context_text if logged_context_text else None),
                "context_text_truncated": (context_text_truncated if logged_context_text else None),
                "stats": {
                    "context_docs": context_docs,
                    "context_chars": context_chars,
                    "answer_chars": len(answer_text) if answer_text else 0,
                },
            },
            "heuristics": {
                "said_idk": said_idk(answer_text),
                "cited_episode": cited_episode(answer_text),
                "retrieved_any": context_docs > 0,
            },
        }

        # --- Diagnostics / labels ---
        context_episode_ids = extract_episode_ids_from_docs([d for (d, _s) in retrieved])
        cited_episode_ids = extract_episode_ids(answer_text)
        expected_episode_ids = list(case.expected_episode_ids)

        # Context diversity / redundancy (use raw docs to preserve multiplicity)
        episode_vals: List[str] = []
        source_vals: List[str] = []
        doc_type_vals: List[str] = []
        chunk_type_vals: List[str] = []
        for (doc, _score) in retrieved:
            meta = getattr(doc, "metadata", None) or {}
            eid = meta.get("episode_id")
            if not isinstance(eid, str) or not eid.strip():
                # Fall back to parsing from source if metadata isn't present.
                src0 = meta.get("source")
                if isinstance(src0, str) and src0:
                    m = _EP_IN_SOURCE_RE.search(src0)
                    if m:
                        eid = f"S{m.group(1)}E{m.group(2)}"
            if isinstance(eid, str) and eid.strip():
                episode_vals.append(eid.strip().upper())

            src = meta.get("source")
            if isinstance(src, str) and src.strip():
                source_vals.append(src.strip())

            dt = meta.get("doc_type")
            if isinstance(dt, str) and dt.strip():
                doc_type_vals.append(dt.strip())

            ct = meta.get("chunk_type")
            if isinstance(ct, str) and ct.strip():
                chunk_type_vals.append(ct.strip())

        episode_counter: Counter[str] = Counter(episode_vals)
        source_counter: Counter[str] = Counter(source_vals)
        doc_type_counter: Counter[str] = Counter(doc_type_vals)
        chunk_type_counter: Counter[str] = Counter(chunk_type_vals)

        distinct_episode_count = len(episode_counter)
        distinct_source_count = len(source_counter)
        distinct_episode_counts.append(distinct_episode_count)
        distinct_source_counts.append(distinct_source_count)

        top_ep_share = _top_share(episode_counter, total=context_docs)
        if top_ep_share is not None:
            top_episode_shares.append(top_ep_share)

        ep_entropy = _normalized_entropy(episode_counter, total=context_docs)
        if ep_entropy is not None:
            episode_entropy_norms.append(ep_entropy)

        agg_like = is_aggregation_question(case.question)
        agg_score = aggregation_readiness_score(
            context_docs=context_docs,
            distinct_episode_count=distinct_episode_count,
            top_episode_share=top_ep_share,
            episode_entropy_norm=ep_entropy,
        )
        if agg_score is not None:
            agg_readiness_scores.append(float(agg_score))
            if agg_like:
                agg_readiness_scores_agg_questions.append(float(agg_score))

        correct_episode_retrieved: Optional[bool] = None
        correct_episode_cited: Optional[bool] = None
        if expected_episode_ids:
            correct_episode_retrieved = any(e in context_episode_ids for e in expected_episode_ids)
            correct_episode_cited = any(e in cited_episode_ids for e in expected_episode_ids) if cited_episode_ids else False

        quotes = extract_answer_quotes(answer_text)
        quoted_context = any_quote_in_context(quotes, context_text)
        quote_match_rates.append(quoted_context if quotes else False)

        cited_not_in_context = [e for e in cited_episode_ids if e not in context_episode_ids]
        cited_not_in_context_counts.append(len(cited_not_in_context))

        retrieval_failure: Optional[bool] = None
        if expected_episode_ids:
            retrieval_failure = not bool(correct_episode_retrieved)

        grounding_failure: Optional[bool] = None
        # Only meaningful when an answer was generated.
        if llm_enabled and answer_text is not None and not said_idk(answer_text) and context_docs > 0:
            grounding_failure = False
            # If the answer cites episodes that aren't in the retrieved context, it's very likely ungrounded.
            if cited_not_in_context:
                grounding_failure = True
            # If it provided quotes but none appear in context, that's also likely ungrounded.
            if quotes and not quoted_context:
                grounding_failure = True
            # If we have ground truth and the answer cites an episode, but not the expected one, flag grounding.
            if expected_episode_ids and cited_episode_ids and not bool(correct_episode_cited):
                grounding_failure = True

        out_case["diagnostics"] = {
            "context_episode_ids": context_episode_ids,
            "cited_episode_ids": cited_episode_ids,
            "cited_episode_ids_not_in_context": cited_not_in_context,
            "cited_episode_ids_not_in_context_count": len(cited_not_in_context),
            "correct_episode_retrieved_in_top_k": correct_episode_retrieved,
            "correct_episode_cited": correct_episode_cited,
            "aggregation": {
                "is_aggregation_like_question": agg_like,
                "readiness_score_0_100": agg_score,
            },
            "context_diversity": {
                "distinct_episode_count": distinct_episode_count,
                "distinct_source_count": distinct_source_count,
                "top_episode_share": top_ep_share,
                "episode_entropy_norm": ep_entropy,
                "episode_counts": _counter_to_sorted_dict(episode_counter),
                "source_counts": _counter_to_sorted_dict(source_counter),
                "doc_type_counts": _counter_to_sorted_dict(doc_type_counter),
                "chunk_type_counts": _counter_to_sorted_dict(chunk_type_counter),
                "source_dup_rate": (
                    float(1.0 - (distinct_source_count / context_docs)) if context_docs > 0 else None
                ),
                "episode_dup_rate": (
                    float(1.0 - (distinct_episode_count / context_docs)) if context_docs > 0 else None
                ),
            },
            "answer_quotes": {
                "count": len(quotes),
                "any_quote_in_context": quoted_context if quotes else None,
                "examples": [q[:120] for q in quotes[:2]] if quotes else [],
            },
        }

        out_case["labels"] = {
            "retrieval_failure": retrieval_failure,
            "grounding_failure": grounding_failure,
            # Reasoning correctness requires gold answers or human labels.
            "reasoning_failure": None,
        }

        if retrieval_failure is not None:
            retrieval_failures.append(bool(retrieval_failure))
        if grounding_failure is not None:
            grounding_failures.append(bool(grounding_failure))

        out["cases"].append(out_case)

    out["summary"] = {
        "cases": len(cases),
        "answered_with_llm": bool(llm_enabled),
        "avg_retrieval_latency_ms": safe_mean_int(retrieval_latencies),
        "avg_context_docs": safe_mean_float([float(x) for x in total_context_docs]) if total_context_docs else None,
        "avg_context_chars": safe_mean_float([float(x) for x in total_context_chars]) if total_context_chars else None,
        "avg_distinct_episodes_in_context": safe_mean_float([float(x) for x in distinct_episode_counts])
        if distinct_episode_counts
        else None,
        "avg_distinct_sources_in_context": safe_mean_float([float(x) for x in distinct_source_counts])
        if distinct_source_counts
        else None,
        "avg_top_episode_share_in_context": safe_mean_float(top_episode_shares) if top_episode_shares else None,
        "avg_episode_entropy_norm_in_context": safe_mean_float(episode_entropy_norms)
        if episode_entropy_norms
        else None,
        "avg_cited_episode_ids_not_in_context": safe_mean_float([float(x) for x in cited_not_in_context_counts])
        if cited_not_in_context_counts
        else None,
        "avg_aggregation_readiness_score": safe_mean_float(agg_readiness_scores) if agg_readiness_scores else None,
        "avg_aggregation_readiness_score_agg_questions": (
            safe_mean_float(agg_readiness_scores_agg_questions) if agg_readiness_scores_agg_questions else None
        ),
        "avg_answer_latency_ms": safe_mean_int(answer_latencies) if answer_latencies else None,
        "avg_prompt_tokens": safe_mean_int(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": safe_mean_int(completion_tokens) if completion_tokens else None,
        "avg_total_tokens": safe_mean_int(total_tokens) if total_tokens else None,
        "idk_rate": (
            float(sum(1 for c in out["cases"] if c["heuristics"]["said_idk"]) / len(out["cases"]))
            if out["cases"]
            else None
        ),
        "episode_citation_rate": (
            float(sum(1 for c in out["cases"] if c["heuristics"]["cited_episode"]) / len(out["cases"]))
            if out["cases"]
            else None
        ),
        "retrieval_empty_rate": (
            float(sum(1 for c in out["cases"] if not c["heuristics"]["retrieved_any"]) / len(out["cases"]))
            if out["cases"]
            else None
        ),
        "retrieval_failure_rate": (
            float(sum(1 for x in retrieval_failures if x) / len(retrieval_failures)) if retrieval_failures else None
        ),
        "grounding_failure_rate": (
            float(sum(1 for x in grounding_failures if x) / len(grounding_failures)) if grounding_failures else None
        ),
        "quote_in_context_rate": (
            float(sum(1 for x in quote_match_rates if x) / len(quote_match_rates)) if quote_match_rates else None
        ),
        "errors_count": len(errors),
        "errors_sample": errors[:10],
    }

    out_path = runs_dir / f"{run_id}.json"
    write_json_file(out_path, out)
    return out_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run RAG eval and log results to JSON.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to test_queries.json")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory to write run logs")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name (optional)")

    parser.add_argument(
        "--retrieval-policy",
        default=DEFAULT_RETRIEVAL_POLICY,
        choices=["script_only", "derived_only", "derived_then_script", "auto", "blended"],
        help="Routing policy: script-only baseline, derived-only, or derived->script two-stage",
    )
    parser.add_argument(
        "--derived-persist-dir",
        default=DEFAULT_DERIVED_PERSIST_DIR,
        help="Chroma persist directory for derived-cards index (used by derived_* policies)",
    )
    parser.add_argument(
        "--derived-collection-name",
        default=DEFAULT_DERIVED_COLLECTION_NAME,
        help="Chroma collection name for derived-cards index",
    )
    parser.add_argument("--derived-k", type=int, default=DEFAULT_DERIVED_K, help="Top-k derived docs (routing stage 1)")
    parser.add_argument(
        "--episode-shortlist-size",
        type=int,
        default=DEFAULT_EPISODE_SHORTLIST_SIZE,
        help="How many episode_ids to shortlist from derived results (routing stage 1)",
    )
    parser.add_argument(
        "--route-backfill-unfiltered",
        action="store_true",
        default=DEFAULT_ROUTE_BACKFILL_UNFILTERED,
        help="In derived_then_script routing, backfill with unfiltered script docs when the shortlist yields too few hits",
    )
    parser.add_argument(
        "--route-backfill-min-docs",
        type=int,
        default=DEFAULT_ROUTE_BACKFILL_MIN_DOCS,
        help="Minimum number of docs to ensure even if routing shortlist is poor (only when backfill enabled)",
    )

    parser.add_argument(
        "--blended-base-k-mult",
        type=float,
        default=DEFAULT_BLENDED_BASE_K_MULT,
        help="For blended policy: retrieve base scripts with k = round(k * mult)",
    )
    parser.add_argument(
        "--blended-routed-k-mult",
        type=float,
        default=DEFAULT_BLENDED_ROUTED_K_MULT,
        help="For blended policy: retrieve routed scripts with k = round(k * mult)",
    )
    parser.add_argument(
        "--blended-include-derived-for-non-agg",
        action="store_true",
        default=DEFAULT_BLENDED_INCLUDE_DERIVED_FOR_NON_AGG,
        help="For blended policy: also use derived episode routing for non-aggregation questions",
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")

    parser.add_argument(
        "--search-type",
        default="similarity",
        choices=["similarity", "mmr"],
        help="Retrieval type",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieved docs")
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K, help="MMR candidate pool size (only for mmr)")
    parser.add_argument("--lambda-mult", type=float, default=DEFAULT_LAMBDA_MULT, help="MMR relevance/diversity tradeoff")

    # Query expansion (multi-query retrieval)
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        default=DEFAULT_QUERY_EXPANSION_ENABLED,
        help="Enable query expansion + fusion (multi-query retrieval)",
    )
    parser.add_argument("--expand-n", type=int, default=DEFAULT_EXPAND_N, help="Number of alternate queries to generate")
    parser.add_argument("--expand-model", default=DEFAULT_EXPAND_MODEL, help="LLM model used for query expansion")
    parser.add_argument(
        "--k-per-query",
        type=int,
        default=DEFAULT_K_PER_QUERY,
        help="Docs to retrieve per expanded query (default: min(k, 6))",
    )
    parser.add_argument(
        "--fusion",
        default=DEFAULT_FUSION,
        choices=["rrf"],
        help="Fusion method for combining per-query results",
    )
    parser.add_argument("--rrf-k0", type=int, default=DEFAULT_RRF_K0, help="RRF constant k0 (higher reduces rank impact)")
    parser.add_argument(
        "--expand-cache",
        default=DEFAULT_EXPAND_CACHE,
        help="Path to query expansion cache JSON (set empty string to disable)",
    )

    parser.add_argument("--no-llm", action="store_true", help="Skip LLM answering; log retrieval only")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens (optional)")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout seconds (optional)")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries (optional)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for determinism if supported (optional)")

    parser.add_argument("--run-name", default="baseline_similarity_k3", help="Run name stored in the log")
    parser.add_argument("--notes", default="", help="Optional notes stored in the log")

    # Chunking config (logged for reproducibility; doesn't change anything here)
    parser.add_argument("--chunk-splitter", default=DEFAULT_CHUNK_SPLITTER, help="Chunk splitter used in ingestion")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size used in ingestion")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap used in ingestion")

    args = parser.parse_args()

    out_path = run_eval(
        test_file=Path(args.test_file),
        runs_dir=Path(args.runs_dir),
        persist_directory=args.persist_dir,
        collection_name=args.collection_name if args.collection_name else None,
        retrieval_policy=str(args.retrieval_policy),
        derived_persist_directory=str(args.derived_persist_dir),
        derived_collection_name=str(args.derived_collection_name) if args.derived_collection_name else None,
        derived_k=int(args.derived_k),
        episode_shortlist_size=int(args.episode_shortlist_size),
        route_backfill_unfiltered=bool(args.route_backfill_unfiltered),
        route_backfill_min_docs=int(args.route_backfill_min_docs),
        blended_base_k_mult=float(args.blended_base_k_mult),
        blended_routed_k_mult=float(args.blended_routed_k_mult),
        blended_include_derived_for_non_agg=bool(args.blended_include_derived_for_non_agg),
        embed_model=args.embed_model,
        search_type=args.search_type,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        query_expansion=bool(args.query_expansion),
        expand_n=int(args.expand_n),
        expand_model=str(args.expand_model),
        k_per_query=(int(args.k_per_query) if args.k_per_query is not None else None),
        fusion=str(args.fusion),
        rrf_k0=int(args.rrf_k0),
        expand_cache=(str(args.expand_cache) if str(args.expand_cache).strip() else None),
        llm_enabled=not args.no_llm,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        seed=args.seed,
        run_name=args.run_name,
        notes=args.notes,
        chunk_splitter=args.chunk_splitter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Wrote run log: {out_path}")


if __name__ == "__main__":
    main()