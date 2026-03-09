from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Allow importing `ingestion.*` when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.load_documents import load_documents


@dataclass(frozen=True)
class EpisodeSize:
    episode_id: str
    season: Optional[int]
    episode: Optional[int]
    title: str
    source: str
    char_count: int
    word_count: int
    approx_tokens: int


def _approx_tokens_from_chars(char_count: int) -> int:
    # Rule-of-thumb for English: ~4 chars/token.
    # Good enough for planning segment sizes.
    return int(math.ceil(char_count / 4.0))


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


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def iter_episode_script_sizes(*, docs_dir: Path) -> Iterable[EpisodeSize]:
    docs = load_documents(docs_path=str(docs_dir), use_metadata=True, show_progress=True)

    for d in docs:
        meta = d.metadata or {}
        if meta.get("doc_type") != "script":
            continue

        episode_id = str(meta.get("episode_id") or "").strip().upper()
        if not episode_id:
            continue

        title = str(meta.get("title") or "").strip()
        source = str(meta.get("source") or "").strip()
        text = d.page_content or ""

        char_count = len(text)
        word_count = len(text.split())
        approx_tokens = _approx_tokens_from_chars(char_count)

        yield EpisodeSize(
            episode_id=episode_id,
            season=_safe_int(meta.get("season")),
            episode=_safe_int(meta.get("episode")),
            title=title,
            source=source,
            char_count=char_count,
            word_count=word_count,
            approx_tokens=approx_tokens,
        )


def _sort_key(e: EpisodeSize) -> Tuple[int, int, str]:
    return (
        int(e.season or 99),
        int(e.episode or 999),
        e.episode_id,
    )


def _print_summary(label: str, values: List[int]) -> None:
    if not values:
        print(f"{label}: <no data>")
        return

    vals = sorted(values)
    mean = int(round(statistics.mean(vals)))
    med = int(round(statistics.median(vals)))

    print(f"{label}: count={len(vals)}")
    print(f"  min  : {vals[0]:>8}")
    print(f"  p50  : {med:>8}")
    print(f"  mean : {mean:>8}")
    print(f"  p90  : {_percentile(vals, 90):>8}")
    print(f"  p95  : {_percentile(vals, 95):>8}")
    print(f"  p99  : {_percentile(vals, 99):>8}")
    print(f"  max  : {vals[-1]:>8}")


def _segments_needed(approx_tokens: int, segment_target_tokens: int) -> int:
    if segment_target_tokens <= 0:
        return 0
    return int(math.ceil(approx_tokens / float(segment_target_tokens)))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze per-episode script sizes for segmentation planning."
    )
    p.add_argument(
        "--docs-dir",
        default="ingestion/normalized_docs_txt",
        help="Path to normalized docs directory (containing scripts/)",
    )
    p.add_argument(
        "--segment-target-tokens",
        type=int,
        default=1800,
        help="Target token size per segment (for estimating segments/episode)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Print the N largest episodes by approx tokens",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write a CSV of per-episode sizes",
    )

    args = p.parse_args()

    docs_dir = Path(args.docs_dir)
    sizes = list(iter_episode_script_sizes(docs_dir=docs_dir))
    sizes.sort(key=_sort_key)

    if not sizes:
        raise SystemExit(f"No script documents found under: {docs_dir}")

    char_counts = [s.char_count for s in sizes]
    token_counts = [s.approx_tokens for s in sizes]
    word_counts = [s.word_count for s in sizes]

    print("=== Episode script size summary (planning) ===")
    print(f"Scripts analyzed: {len(sizes)}")
    print(f"Segment target (approx tokens): {int(args.segment_target_tokens)}")
    print()

    _print_summary("Characters", char_counts)
    _print_summary("Words", word_counts)
    _print_summary("Approx tokens", token_counts)

    print()
    sizes_by_tokens = sorted(sizes, key=lambda s: s.approx_tokens, reverse=True)
    top_n = max(0, int(args.top_n))
    if top_n:
        print(f"=== Top {top_n} longest episodes (by approx tokens) ===")
        for s in sizes_by_tokens[:top_n]:
            segs = _segments_needed(s.approx_tokens, int(args.segment_target_tokens))
            title = (s.title or "").strip()
            title_part = f" — {title}" if title else ""
            print(
                f"{s.episode_id}{title_part}\n"
                f"  approx_tokens={s.approx_tokens}  chars={s.char_count}  words={s.word_count}  est_segments={segs}"
            )

    if args.csv_out:
        out_path = Path(str(args.csv_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode_id",
                    "season",
                    "episode",
                    "title",
                    "source",
                    "char_count",
                    "word_count",
                    "approx_tokens",
                    "segment_target_tokens",
                    "est_segments",
                ]
            )
            for s in sizes:
                w.writerow(
                    [
                        s.episode_id,
                        s.season,
                        s.episode,
                        s.title,
                        s.source,
                        s.char_count,
                        s.word_count,
                        s.approx_tokens,
                        int(args.segment_target_tokens),
                        _segments_needed(s.approx_tokens, int(args.segment_target_tokens)),
                    ]
                )
        print()
        print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
