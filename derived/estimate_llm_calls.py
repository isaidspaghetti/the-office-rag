from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Allow importing sibling modules when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from derived.segment_episode import approx_tokens_from_chars, iter_episode_scripts, segment_episode_text

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"


@dataclass(frozen=True)
class EpisodeCallPlan:
    episode_id: str
    title: str
    body_chars: int
    body_approx_tokens: int
    segment_target_tokens: int
    min_tokens: int
    segment_count: int
    map_calls: int
    reduce_calls: int
    total_calls: int


def _percentile(sorted_vals: Sequence[int], p: float) -> int:
    if not sorted_vals:
        return 0
    if p <= 0:
        return int(sorted_vals[0])
    if p >= 100:
        return int(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return int(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return int(round(d0 + d1))


def _print_summary(label: str, values: List[int]) -> None:
    if not values:
        print(f"{label}: <no data>")
        return

    vals = sorted(values)
    mean = int(round(statistics.mean(vals)))
    med = int(round(statistics.median(vals)))

    print(f"{label}: count={len(vals)}")
    print(f"  min  : {vals[0]:>6}")
    print(f"  p50  : {med:>6}")
    print(f"  mean : {mean:>6}")
    print(f"  p90  : {_percentile(vals, 90):>6}")
    print(f"  p95  : {_percentile(vals, 95):>6}")
    print(f"  p99  : {_percentile(vals, 99):>6}")
    print(f"  max  : {vals[-1]:>6}")


def iter_call_plans(
    *,
    docs_dir: Path,
    segment_target_tokens: int,
    min_tokens: int,
    include_reduce: bool,
    limit_episodes: Optional[int] = None,
) -> Iterable[EpisodeCallPlan]:
    for i, ep in enumerate(iter_episode_scripts(docs_dir=docs_dir)):
        if limit_episodes is not None and i >= int(limit_episodes):
            break

        segs = segment_episode_text(
            episode_id=ep.episode_id,
            body_text=ep.body_text,
            target_tokens=int(segment_target_tokens),
            min_tokens=int(min_tokens),
        )

        map_calls = len(segs)
        reduce_calls = 1 if include_reduce else 0

        yield EpisodeCallPlan(
            episode_id=ep.episode_id,
            title=ep.title,
            body_chars=len(ep.body_text),
            body_approx_tokens=approx_tokens_from_chars(len(ep.body_text)),
            segment_target_tokens=int(segment_target_tokens),
            min_tokens=int(min_tokens),
            segment_count=len(segs),
            map_calls=map_calls,
            reduce_calls=reduce_calls,
            total_calls=map_calls + reduce_calls,
        )


def _sort_key(p: EpisodeCallPlan) -> Tuple[int, str]:
    # episode_id is SxxExx so lexical sorting works.
    return (p.total_calls, p.episode_id)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate LLM call counts for the derived pipeline (map: segment summaries, reduce: episode cards)."
    )
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="Normalized docs dir")
    p.add_argument("--segment-target-tokens", type=int, default=1800, help="Approx target tokens per segment")
    p.add_argument("--min-tokens", type=int, default=400, help="Approx min tokens before splitting")
    p.add_argument(
        "--include-reduce",
        action="store_true",
        help="Include 1 additional reduce call per episode (episode card build)",
    )
    p.add_argument("--top-n", type=int, default=15, help="Print the episodes with the most calls")
    p.add_argument("--limit-episodes", type=int, default=None, help="Only consider the first N episodes")
    p.add_argument("--csv-out", default=None, help="Optional path to write per-episode call plan CSV")

    args = p.parse_args()

    plans = list(
        iter_call_plans(
            docs_dir=Path(args.docs_dir),
            segment_target_tokens=int(args.segment_target_tokens),
            min_tokens=int(args.min_tokens),
            include_reduce=bool(args.include_reduce),
            limit_episodes=(int(args.limit_episodes) if args.limit_episodes is not None else None),
        )
    )

    if not plans:
        raise SystemExit("No episodes found.")

    total_calls = sum(p.total_calls for p in plans)
    map_calls = sum(p.map_calls for p in plans)
    reduce_calls = sum(p.reduce_calls for p in plans)

    seg_counts = [p.segment_count for p in plans]
    call_counts = [p.total_calls for p in plans]

    print("=== Derived pipeline LLM call estimate (dry run) ===")
    print(f"Episodes: {len(plans)}")
    print(f"Segment target (approx tokens): {int(args.segment_target_tokens)}")
    print(f"Include reduce calls: {bool(args.include_reduce)}")
    print()

    print(f"Total map calls   (segment summaries): {map_calls}")
    print(f"Total reduce calls (episode cards)    : {reduce_calls}")
    print(f"TOTAL LLM calls                       : {total_calls}")

    print()
    _print_summary("Segments per episode", seg_counts)
    _print_summary("Total calls per episode", call_counts)

    top_n = max(0, int(args.top_n))
    if top_n:
        print()
        print(f"=== Top {top_n} episodes by total calls ===")
        for p2 in sorted(plans, key=lambda x: x.total_calls, reverse=True)[:top_n]:
            title = f" — {p2.title}" if p2.title else ""
            print(
                f"{p2.episode_id}{title}\n"
                f"  segments={p2.segment_count}  map_calls={p2.map_calls}  reduce_calls={p2.reduce_calls}  total_calls={p2.total_calls}"
            )

    if args.csv_out:
        out_path = Path(str(args.csv_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode_id",
                    "title",
                    "body_chars",
                    "body_approx_tokens",
                    "segment_target_tokens",
                    "min_tokens",
                    "segment_count",
                    "map_calls",
                    "reduce_calls",
                    "total_calls",
                ]
            )
            for p2 in plans:
                w.writerow(
                    [
                        p2.episode_id,
                        p2.title,
                        p2.body_chars,
                        p2.body_approx_tokens,
                        p2.segment_target_tokens,
                        p2.min_tokens,
                        p2.segment_count,
                        p2.map_calls,
                        p2.reduce_calls,
                        p2.total_calls,
                    ]
                )
        print()
        print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
