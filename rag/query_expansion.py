from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class QueryExpansionConfig:
    enabled: bool = False
    n: int = 5
    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_retries: Optional[int] = None
    seed: Optional[int] = None
    cache_path: Optional[Path] = Path("experiments/cache/query_expansion_cache.json")
    prompt_version: str = "v2"


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from an LLM response."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    # First try raw parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Then try to find a JSON object inside the text.
    match = _JSON_OBJECT_RE.search(text)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj

    raise ValueError("could not parse JSON object")


def _normalize_queries(queries: List[Any], *, limit: int) -> List[str]:
    out: List[str] = []
    for q in queries:
        s = str(q or "").strip()
        if not s:
            continue
        # keep queries compact
        s = re.sub(r"\s+", " ", s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _cache_key(*, question: str, config: QueryExpansionConfig) -> str:
    return "|".join(
        [
            "query_expansion",
            config.prompt_version,
            config.model,
            str(config.n),
            question.strip(),
        ]
    )


def _load_cache(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            out: Dict[str, List[str]] = {}
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [str(x) for x in v if str(x).strip()]
            return out
    except Exception:
        return {}
    return {}


def _write_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_expander_llm(config: QueryExpansionConfig) -> ChatOpenAI:
    model_kwargs: Dict[str, Any] = {}
    if config.seed is not None:
        model_kwargs["seed"] = config.seed

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "model_kwargs": model_kwargs,
    }
    if config.max_retries is not None:
        kwargs["max_retries"] = config.max_retries

    return ChatOpenAI(**kwargs)


def expand_queries(
    *,
    question: str,
    llm: ChatOpenAI,
    config: QueryExpansionConfig,
) -> List[str]:
    """Generate alternative retrieval queries.

    Returns a list of length 1..(n+1) including the original question first.
    Uses a cache when configured.
    """
    base = (question or "").strip()
    if not base:
        return []

    if not config.enabled:
        return [base]

    key = _cache_key(question=base, config=config)

    cache: Dict[str, List[str]] = {}
    if config.cache_path is not None:
        cache = _load_cache(config.cache_path)
        cached = cache.get(key)
        if cached:
            # Always keep original question first.
            return [base] + _normalize_queries(cached, limit=config.n)

    system = (
        "You generate search queries for retrieving evidence from a corpus of TV scripts and episode summaries.\n"
        "Do NOT answer the user. Do NOT add facts not present in the question.\n"
        "Keep each query short (<= 12 words) and entity-preserving.\n"
        "Preserve important constraints/qualifiers from the original question (e.g., 'serious', 'how it ended', 'in which episode').\n"
        "Do NOT broaden the scope (e.g., don't turn 'serious girlfriends' into 'dating history') unless the original question is broad.\n"
        'Return ONLY valid JSON: {"queries": [..]}.'
    )

    user = (
        f"Original question: {base}\n\n" f"Generate exactly {config.n} alternative search queries."
    )

    msg = llm.invoke([("system", system), ("human", user)])
    content = str(getattr(msg, "content", "") or "")

    obj = _extract_json(content)
    queries = obj.get("queries")
    if not isinstance(queries, list):
        raise ValueError("expander returned JSON without a 'queries' list")

    alts = _normalize_queries(queries, limit=config.n)

    # Cache only the alternates.
    if config.cache_path is not None:
        cache[key] = alts
        _write_cache(config.cache_path, cache)

    return [base] + alts
