from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class FusedDoc:
    doc: Any
    fused_score: float
    best_base_score: Optional[float]
    retrieved_by: List[str]
    ranks_by_query: Dict[str, int]


def _doc_key(doc: Any) -> Tuple[Optional[str], str]:
    """Stable-ish dedupe key for a LangChain Document.

    Uses (source, sha1(page_content)). This works without relying on chunk ids.
    """
    src = None
    try:
        src = doc.metadata.get("source")
    except Exception:
        src = None

    content = getattr(doc, "page_content", "") or ""
    h = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
    return (src, h)


def rrf_fuse(
    per_query_results: Dict[str, List[Tuple[Any, Optional[float]]]],
    *,
    k0: int = 60,
) -> List[FusedDoc]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    per_query_results: {query_string: [(doc, base_score_or_None), ...]} where list order is rank.

    Returns fused docs sorted by fused_score desc.
    """
    if k0 <= 0:
        raise ValueError("k0 must be > 0")

    agg: Dict[Tuple[Optional[str], str], FusedDoc] = {}

    for query, results in per_query_results.items():
        for idx, (doc, base_score) in enumerate(results, start=1):
            key = _doc_key(doc)
            inc = 1.0 / float(k0 + idx)

            existing = agg.get(key)
            if existing is None:
                agg[key] = FusedDoc(
                    doc=doc,
                    fused_score=inc,
                    best_base_score=(float(base_score) if base_score is not None else None),
                    retrieved_by=[query],
                    ranks_by_query={query: idx},
                )
            else:
                existing.fused_score += inc
                existing.ranks_by_query.setdefault(query, idx)
                if query not in existing.retrieved_by:
                    existing.retrieved_by.append(query)
                if base_score is not None:
                    s = float(base_score)
                    if existing.best_base_score is None or s > existing.best_base_score:
                        existing.best_base_score = s

    fused = list(agg.values())

    # Deterministic sort: fused_score, then best_base_score, then source.
    def _sort_key(item: FusedDoc) -> Tuple[float, float, str]:
        src = None
        try:
            src = item.doc.metadata.get("source")
        except Exception:
            src = None
        return (
            float(item.fused_score),
            float(item.best_base_score) if item.best_base_score is not None else -1.0,
            str(src or ""),
        )

    fused.sort(key=_sort_key, reverse=True)
    return fused
