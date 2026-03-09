from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _iter_episode_cards(glob_pattern: str) -> Iterable[Path]:
    # Pattern is interpreted under repo root if relative.
    base = _REPO_ROOT
    patt = glob_pattern
    if str(Path(glob_pattern)).startswith("/"):
        base = Path("/")
        patt = glob_pattern.lstrip("/")
    yield from sorted(base.glob(patt))


@dataclass(frozen=True)
class PairStats:
    pair: Tuple[str, str]
    episode_ids: Set[str]
    evidence_snippets: List[str]


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return ("", "")
    return tuple(sorted([a, b], key=lambda x: x.lower()))  # type: ignore[return-value]


def _load_main_cast_names(character_config_path: Optional[Path]) -> Set[str]:
    if not character_config_path:
        return set()
    try:
        cfg = _read_json(character_config_path)
    except Exception:
        return set()
    out: Set[str] = set()
    for c in cfg.get("characters") or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if name:
            out.add(name)
        for a in c.get("aliases") or []:
            aa = str(a).strip()
            if aa:
                out.add(aa)
    return out


def _make_queries(a: str, b: str) -> List[str]:
    # Keep this small: each query triggers an embedding call.
    # Use one general relationship query + one breakup/dating-ish query that helps romantic pairs.
    a_s = a.strip()
    b_s = b.strip()
    return [
        f"{a_s} {b_s} relationship",
        f"{a_s} {b_s} dating breakup",
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate relationship TopicSpec list by mining EpisodeDerivedCardV1 artifacts"
    )
    p.add_argument(
        "--episode-cards-glob",
        default="derived/artifacts/**/episode_cards/*.json",
        help="Glob for episode card JSONs (relative to repo root by default)",
    )
    p.add_argument(
        "--character-config",
        default="derived/entity_corpora/config.major_relationships.json",
        help="Optional config JSON with a 'characters' list (used for main-cast filtering)",
    )
    p.add_argument(
        "--require-main-cast",
        action="store_true",
        help="If set, only include pairs where at least one side is in character-config's characters/aliases",
    )
    p.add_argument(
        "--min-episodes",
        type=int,
        default=2,
        help="Minimum distinct episodes a pair must appear in",
    )
    p.add_argument("--top-n", type=int, default=30, help="Max number of pairs to include")
    p.add_argument(
        "--out",
        default="derived/topic_cards/topics.major_relationships.generated.json",
        help="Output topics JSON path",
    )

    args = p.parse_args()

    character_config_path = (
        _resolve_under_repo(str(args.character_config)) if args.character_config else None
    )
    main_cast = _load_main_cast_names(character_config_path)

    pair_episodes: DefaultDict[Tuple[str, str], Set[str]] = defaultdict(set)

    # We keep a couple snippets per pair for debug/inspection only.
    pair_snippets: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    for path in _iter_episode_cards(str(args.episode_cards_glob)):
        try:
            card = _read_json(path)
        except Exception:
            continue
        if str(card.get("schema") or "") != "EpisodeDerivedCardV1":
            continue
        episode_id = str(card.get("episode_id") or "").strip().upper()
        if not episode_id:
            continue

        for rel in card.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            pair = rel.get("pair")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a = str(pair[0] or "").strip()
            b = str(pair[1] or "").strip()
            aa, bb = _normalize_pair(a, b)
            if not aa or not bb or aa.lower() == bb.lower():
                continue

            if args.require_main_cast and main_cast:
                if (aa not in main_cast) and (bb not in main_cast):
                    continue

            pair_episodes[(aa, bb)].add(episode_id)

            status = str(rel.get("status") or "").strip()
            if status and len(pair_snippets[(aa, bb)]) < 2:
                pair_snippets[(aa, bb)].append(status)

    ranked: List[Tuple[Tuple[str, str], int]] = []
    for pair, eps in pair_episodes.items():
        ranked.append((pair, len(eps)))

    ranked.sort(key=lambda x: x[1], reverse=True)

    topics: List[Dict[str, Any]] = []
    included = 0
    for (a, b), n_eps in ranked:
        if n_eps < int(args.min_episodes):
            continue
        topic_id = f"relationship_{_slug(a)}_{_slug(b)}"
        topics.append(
            {
                "topic_id": topic_id,
                "topic_type": "relationship",
                "entity_names": [a, b],
                "queries": _make_queries(a, b),
                "mined_stats": {
                    "num_episodes": n_eps,
                    "episode_ids": sorted(pair_episodes[(a, b)]),
                    "example_statuses": pair_snippets.get((a, b), []),
                },
            }
        )
        included += 1
        if included >= int(args.top_n):
            break

    out_path = _resolve_under_repo(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema": "TopicSpecsV1",
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "generator": {
            "script": "derived/topic_cards/generate_relationship_topics.py",
            "episode_cards_glob": str(args.episode_cards_glob),
            "character_config": str(character_config_path) if character_config_path else None,
            "require_main_cast": bool(args.require_main_cast),
            "min_episodes": int(args.min_episodes),
            "top_n": int(args.top_n),
        },
        "topics": topics,
    }
    out_path.write_text(_safe_json(payload) + "\n", encoding="utf-8")

    print(f"Wrote topics: {out_path}")
    print(
        _safe_json(
            {
                "num_topics": len(topics),
                "min_episodes": int(args.min_episodes),
                "top_n": int(args.top_n),
            }
        )
    )


if __name__ == "__main__":
    main()
