from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

HEADER_DIVIDER = "----"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_header_block(text: str) -> Tuple[Dict[str, str], str]:
    """Parse KEY: value lines until divider '----'. Returns (header_dict_lowercase, body_text)."""
    lines = text.splitlines()
    header: Dict[str, str] = {}

    divider_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == HEADER_DIVIDER:
            divider_idx = i
            break

    if divider_idx is None:
        return {}, text

    for line in lines[:divider_idx]:
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        header[k.strip().lower()] = v.strip()

    body_lines = lines[divider_idx + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines = body_lines[1:]

    body = "\n".join(body_lines)
    return header, body


def _coerce_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = str(s).strip()
    return int(s) if s.isdigit() else None


def _episode_id_from_header(header: Dict[str, str]) -> Optional[str]:
    season = _coerce_int(header.get("season"))
    episode = _coerce_int(header.get("episode"))
    if season is None or episode is None:
        return None
    return f"S{season:02d}E{episode:02d}"


def approx_tokens_from_chars(char_count: int) -> int:
    # Rule-of-thumb. Use real tokenization later if we want tighter packing.
    return int(math.ceil(char_count / 4.0))


@dataclass(frozen=True)
class EpisodeScript:
    episode_id: str
    title: str
    source_path: str
    body_text: str


@dataclass(frozen=True)
class EpisodeSegment:
    episode_id: str
    segment_id: str
    segment_index: int
    segment_char_start: int
    segment_char_end: int
    approx_tokens: int


def iter_episode_scripts(*, docs_dir: Path) -> Iterable[EpisodeScript]:
    scripts_root = docs_dir / "scripts"
    if not scripts_root.exists():
        raise FileNotFoundError(f"Missing scripts directory: {scripts_root}")

    for path in sorted(scripts_root.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        header, body = parse_header_block(raw)

        episode_id = _episode_id_from_header(header)
        if not episode_id:
            continue

        title = (header.get("title") or "").strip()
        yield EpisodeScript(
            episode_id=episode_id,
            title=title,
            source_path=str(path.as_posix()),
            body_text=body,
        )


_PARAGRAPH_SEP_RE = re.compile(r"\n{2,}")


def _piece_spans(text: str) -> List[Tuple[int, int]]:
    """Return spans that cover the full text, preferring paragraph boundaries.

    Pieces include the paragraph separator at the end of each piece (except the last)
    so that concatenating all pieces recreates the original text exactly.
    """
    if not text:
        return [(0, 0)]

    seps = [(m.start(), m.end()) for m in _PARAGRAPH_SEP_RE.finditer(text)]
    if not seps:
        return [(0, len(text))]

    pieces: List[Tuple[int, int]] = []
    cursor = 0
    for sep_start, sep_end in seps:
        if sep_end <= cursor:
            continue
        pieces.append((cursor, sep_end))
        cursor = sep_end
    if cursor < len(text):
        pieces.append((cursor, len(text)))

    # Filter empty pieces (can happen with leading separators)
    pieces = [(a, b) for (a, b) in pieces if b > a]
    return pieces if pieces else [(0, len(text))]


def segment_episode_text(
    *,
    episode_id: str,
    body_text: str,
    target_tokens: int = 1800,
    min_tokens: int = 400,
) -> List[EpisodeSegment]:
    """Deterministically segment an episode body into chunks sized ~target_tokens.

    Offsets are relative to the *body_text* (header removed), so they remain stable
    as long as the normalized source file doesn't change.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if min_tokens <= 0:
        raise ValueError("min_tokens must be > 0")
    if min_tokens > target_tokens:
        raise ValueError("min_tokens must be <= target_tokens")

    text = body_text or ""
    pieces = _piece_spans(text)

    segments: List[EpisodeSegment] = []
    seg_start: Optional[int] = None
    seg_end: Optional[int] = None
    seg_tokens = 0

    def _flush() -> None:
        nonlocal seg_start, seg_end, seg_tokens
        if seg_start is None or seg_end is None:
            return
        seg_index = len(segments)
        segment_id = f"{episode_id}:seg:{seg_index:03d}"
        segments.append(
            EpisodeSegment(
                episode_id=episode_id,
                segment_id=segment_id,
                segment_index=seg_index,
                segment_char_start=int(seg_start),
                segment_char_end=int(seg_end),
                approx_tokens=int(seg_tokens),
            )
        )
        seg_start = None
        seg_end = None
        seg_tokens = 0

    for piece_start, piece_end in pieces:
        piece_len = piece_end - piece_start
        piece_tokens = approx_tokens_from_chars(piece_len)

        if seg_start is None:
            seg_start = piece_start
            seg_end = piece_end
            seg_tokens = piece_tokens
            continue

        # If adding this piece would exceed target and we already have enough content,
        # flush and start a new segment.
        if seg_tokens >= min_tokens and (seg_tokens + piece_tokens) > target_tokens:
            _flush()
            seg_start = piece_start
            seg_end = piece_end
            seg_tokens = piece_tokens
            continue

        # Otherwise, extend current segment.
        seg_end = piece_end
        seg_tokens += piece_tokens

        # If a segment grows beyond target because one piece is huge, we accept it.

    _flush()

    # Ensure coverage for empty text
    if not segments:
        segments.append(
            EpisodeSegment(
                episode_id=episode_id,
                segment_id=f"{episode_id}:seg:000",
                segment_index=0,
                segment_char_start=0,
                segment_char_end=0,
                approx_tokens=0,
            )
        )

    return segments


def _find_episode(*, docs_dir: Path, episode_id: str) -> EpisodeScript:
    target = episode_id.strip().upper()
    for ep in iter_episode_scripts(docs_dir=docs_dir):
        if ep.episode_id == target:
            return ep
    raise FileNotFoundError(f"Episode {target} not found under {docs_dir}/scripts")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deterministically segment an episode script for map/reduce summarization."
    )
    p.add_argument(
        "--docs-dir", default="ingestion/normalized_docs_txt", help="Normalized docs dir"
    )
    p.add_argument("--episode-id", required=False, help="Episode ID like S07E24")
    p.add_argument("--list", action="store_true", help="List available episode IDs and titles")
    p.add_argument(
        "--target-tokens", type=int, default=1800, help="Target tokens per segment (approx)"
    )
    p.add_argument(
        "--min-tokens", type=int, default=400, help="Minimum tokens before splitting (approx)"
    )
    p.add_argument("--out", default=None, help="Optional path to write segments JSON")
    p.add_argument(
        "--include-text", action="store_true", help="Include full segment text in output JSON"
    )
    p.add_argument(
        "--print-n", type=int, default=3, help="Print the first N segments as a sanity check"
    )

    args = p.parse_args()

    docs_dir = Path(args.docs_dir)

    if args.list:
        eps = sorted(iter_episode_scripts(docs_dir=docs_dir), key=lambda e: e.episode_id)
        for e in eps:
            title = f" — {e.title}" if e.title else ""
            print(f"{e.episode_id}{title}")
        return

    if not args.episode_id:
        raise SystemExit("Provide --episode-id (e.g., S02E11) or use --list")

    ep = _find_episode(docs_dir=docs_dir, episode_id=str(args.episode_id))
    segs = segment_episode_text(
        episode_id=ep.episode_id,
        body_text=ep.body_text,
        target_tokens=int(args.target_tokens),
        min_tokens=int(args.min_tokens),
    )

    print("=== Episode segmentation ===")
    print(f"Episode: {ep.episode_id} — {ep.title}")
    print(f"Source : {ep.source_path}")
    print(
        f"Body chars: {len(ep.body_text)}  approx_tokens={approx_tokens_from_chars(len(ep.body_text))}"
    )
    print(
        f"Segments: {len(segs)}  target_tokens={int(args.target_tokens)}  min_tokens={int(args.min_tokens)}"
    )

    n = max(0, int(args.print_n))
    if n:
        for s in segs[:n]:
            text = ep.body_text[s.segment_char_start : s.segment_char_end]
            preview = text[:240].replace("\n", "\\n")
            print(
                f"\n{s.segment_id}  chars={s.segment_char_end - s.segment_char_start}  approx_tokens={s.approx_tokens}"
            )
            print(f"  offsets: [{s.segment_char_start}, {s.segment_char_end})")
            print(f"  preview: {preview}...")

    if args.out:
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema": "EpisodeSegmentsV1",
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "episode_id": ep.episode_id,
            "title": ep.title,
            "source": ep.source_path,
            "target_tokens": int(args.target_tokens),
            "min_tokens": int(args.min_tokens),
            "body_char_count": len(ep.body_text),
            "body_approx_tokens": approx_tokens_from_chars(len(ep.body_text)),
            "segments": [],
        }

        for s in segs:
            item = {
                "segment_id": s.segment_id,
                "segment_index": s.segment_index,
                "segment_char_start": s.segment_char_start,
                "segment_char_end": s.segment_char_end,
                "approx_tokens": s.approx_tokens,
            }
            if args.include_text:
                item["text"] = ep.body_text[s.segment_char_start : s.segment_char_end]
            payload["segments"].append(item)

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
