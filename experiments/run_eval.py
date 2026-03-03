from __future__ import annotations

import argparse
import json
import re
import sys
import time
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


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def said_idk(answer: Optional[str]) -> bool:
    if not answer:
        return False
    a = answer.lower()
    return "i don't know" in a or "i do not know" in a or "not in the context" in a


_EP_CITE_RE = re.compile(r"\bS\d{2}E\d{2}\b")


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
        cases.append(Case(case_id=case_id, question=question, expected_notes=expected_notes))
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
        "When possible, cite episode identifiers present in the context (e.g., 'S02E06')."
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
    db = build_vectorstore(
        persist_directory=persist_directory,
        embed_model=embed_model,
        collection_name=collection_name,
    )
    index_counts = get_index_counts(db)

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

    out: Dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "run_name": run_name,
            "created_at_utc": created_at,
            "notes": notes,
        },
        "config": {
            "vectorstore": {
                "kind": "chroma",
                "persist_directory": persist_directory,
                "collection_name": collection_name,
                **index_counts,
            },
            "embedding": {"provider": "openai", "model": embed_model},
            "chunking": {
                "splitter": chunk_splitter,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "retrieval": {
                "search_type": search_type,
                "k": k,
                "fetch_k": (fetch_k if search_type == "mmr" else None),
                "lambda_mult": (lambda_mult if search_type == "mmr" else None),
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

    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []

    errors: List[str] = []

    def retrieve_docs_with_scores(question: str, *, k_override: Optional[int] = None) -> List[Tuple[Any, Optional[float]]]:
        """Return a uniform (doc, score) shape across retrieval modes.

        Note: Chroma's MMR API returns docs without scores, so for MMR we
        backfill scores using a similarity scoring pass over the same candidate pool.
        """

        effective_k = int(k_override) if k_override is not None else int(k)

        if search_type == "similarity":
            return [
                (doc, float(score))
                for (doc, score) in db.similarity_search_with_relevance_scores(question, k=effective_k)
            ]

        if search_type == "mmr":
            effective_fetch_k = fetch_k or (effective_k * 4)
            mmr_docs = db.max_marginal_relevance_search(
                question,
                k=effective_k,
                fetch_k=effective_fetch_k,
                lambda_mult=lambda_mult,
            )

            # Build a best-effort score lookup from similarity scoring.
            score_map: Dict[Tuple[Optional[str], str], float] = {}
            try:
                scored = db.similarity_search_with_relevance_scores(question, k=effective_fetch_k)
                for doc, score in scored:
                    key = (doc.metadata.get("source"), doc.page_content or "")
                    # Keep the max score if duplicates occur.
                    prev = score_map.get(key)
                    s = float(score)
                    if prev is None or s > prev:
                        score_map[key] = s
            except Exception:
                # If scoring fails for any reason, we still return the MMR docs.
                score_map = {}

            out_pairs: List[Tuple[Any, Optional[float]]] = []
            for doc in mmr_docs:
                key = (doc.metadata.get("source"), doc.page_content or "")
                out_pairs.append((doc, score_map.get(key)))
            return out_pairs

        raise ValueError(f"Unsupported search_type: {search_type}")

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
                for q in expanded_queries:
                    per_query_results[q] = retrieve_docs_with_scores(q, k_override=per_q_k)

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
                retrieved = retrieve_docs_with_scores(case.question)

            retrieval_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            retrieval_ms = int((time.time() - t0) * 1000)
            retrieved = []
            errors.append(f"{case.case_id}: retrieval_error: {type(e).__name__}: {e}")

        retrieval_latencies.append(retrieval_ms)

        results = []
        context_parts = []
        scores: List[float] = []

        for rank, (doc, score) in enumerate(retrieved, start=1):
            src = doc.metadata.get("source")
            txt = doc.page_content or ""
            if score is not None:
                scores.append(float(score))
            results.append(
                {
                    "rank": rank,
                    "source": src,
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

        out_case = {
            "case_id": case.case_id,
            "question": case.question,
            "expected_notes": case.expected_notes,
            "retrieval": {
                "top_k": k,
                "latency_ms": retrieval_ms,
                "stats": compute_score_stats(scores),
                "results": results,
                "multi_query": multi_query_details,
            },
            "answer": {
                "text": answer_text,
                "latency_ms": answer_latency_ms,
                "error": answer_error,
                "usage": usage,
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

        out["cases"].append(out_case)

    out["summary"] = {
        "cases": len(cases),
        "answered_with_llm": bool(llm_enabled),
        "avg_retrieval_latency_ms": safe_mean_int(retrieval_latencies),
        "avg_context_docs": safe_mean_float([float(x) for x in total_context_docs]) if total_context_docs else None,
        "avg_context_chars": safe_mean_float([float(x) for x in total_context_chars]) if total_context_chars else None,
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