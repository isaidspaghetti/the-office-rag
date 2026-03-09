from __future__ import annotations

import re
from typing import Any, Optional


_EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


def safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def safe_int(x: Any) -> Optional[int]:
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


def safe_float(x: Any) -> Optional[float]:
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


def normalize_episode_id(s: Any) -> Optional[str]:
    if not s:
        return None
    m = _EP_RE.search(str(s).strip().upper())
    return m.group(0).upper() if m else None


def question_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if not q:
        return "(missing)"
    if re.search(r"\b(which|what)\s+episode\b|\bin\s+which\s+episode\b|\bepisode\s+is\b", q):
        return "Episode lookup"
    if re.search(r"\bquote\b|\bexact\s+quote\b|\bwhat\s+did\b.+\bsay\b", q):
        return "Quote / line"
    if re.search(
        r"\b(list|summari[sz]e|overview|timeline|chronolog|across|throughout|all\b|compare)\b",
        q,
    ):
        return "Aggregation"
    if re.search(r"\bwhy\b|\bhow\b", q):
        return "Explanation"
    return "Factual"
