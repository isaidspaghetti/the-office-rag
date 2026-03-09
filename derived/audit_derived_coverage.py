from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_EP_RE = re.compile(r"s(\d{2})e(\d{2})", re.IGNORECASE)


def _ep_id_from_path(path: Path) -> Optional[str]:
    m = _EP_RE.search(path.as_posix())
    if not m:
        return None
    return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"


def _season_from_ep(episode_id: str) -> Optional[int]:
    m = re.fullmatch(r"S(\d{2})E(\d{2})", (episode_id or "").strip().upper())
    if not m:
        return None
    return int(m.group(1))


def _group_by_season(episode_ids: Iterable[str]) -> Dict[int, List[str]]:
    by: Dict[int, List[str]] = defaultdict(list)
    for eid in sorted(set(episode_ids)):
        s = _season_from_ep(eid)
        if s is None:
            continue
        by[s].append(eid)
    return dict(sorted(by.items(), key=lambda kv: kv[0]))


def _scan_expected_episodes(docs_dir: Path) -> Dict[int, List[str]]:
    """Expected episodes based on normalized script files present."""
    script_dir = docs_dir / "scripts"
    if not script_dir.exists():
        raise FileNotFoundError(f"scripts dir not found: {script_dir}")

    eps: Set[str] = set()
    for p in script_dir.rglob("*.txt"):
        eid = _ep_id_from_path(p)
        if eid:
            eps.add(eid)

    return _group_by_season(eps)


def _auto_detect_prefixes(out_root: Path, glob_pat: str) -> List[str]:
    out: List[str] = []
    for p in sorted(out_root.glob(glob_pat)):
        if not p.is_dir():
            continue

        # For segsummary/episodecard we want the base season dir (no per-episode suffix)
        # because the per-episode artifacts live under {base}_SxxEyy/...
        if "_S" in p.name:
            continue

        # Require a season manifest for build roots.
        if not (p / "season_manifest.json").exists():
            continue

        out.append(p.name)
    return out


def _latest_prefix_by_season(prefixes: Sequence[str], *, kind: str) -> Dict[int, str]:
    """Pick one prefix per season.

    Heuristic: prefer prefixes that include 'seasonXX', then sort lexicographically.
    This is good enough for a local audit report.
    """
    by: Dict[int, List[str]] = defaultdict(list)
    for pref in prefixes:
        m = re.search(r"season(\d{2})", pref)
        if not m:
            continue
        by[int(m.group(1))].append(pref)

    chosen: Dict[int, str] = {}
    for season, items in by.items():
        chosen[season] = sorted(items)[-1]
    return chosen


def _episode_cards_present(out_root: Path, episode_prefix: str, season: int) -> Set[str]:
    # Layout: {prefix}_SxxEyy/episode_cards/SxxEyy.json
    glob_pat = f"{episode_prefix}_S??E??/episode_cards/S??E??.json"
    present: Set[str] = set()
    for p in out_root.glob(glob_pat):
        eid = _ep_id_from_path(p)
        if eid and _season_from_ep(eid) == season:
            present.add(eid)
    return present


def _seg_summaries_present(out_root: Path, seg_prefix: str, season: int) -> Set[str]:
    # Layout: {prefix}_SxxEyy/segment_summaries/SxxEyy/*.json
    glob_pat = f"{seg_prefix}_S??E??/segment_summaries/S??E??/*.json"
    present: Set[str] = set()
    for p in out_root.glob(glob_pat):
        eid = _ep_id_from_path(p)
        if eid and _season_from_ep(eid) == season:
            present.add(eid)
    return present


def _segments_present(out_root: Path, season: int) -> Set[str]:
    # Layout: segments/SxxEyy.json
    present: Set[str] = set()
    seg_dir = out_root / "segments"
    if not seg_dir.exists():
        return present
    for p in seg_dir.glob("S??E??.json"):
        eid = _ep_id_from_path(p)
        if eid and _season_from_ep(eid) == season:
            present.add(eid)
    return present


def _season_card_present(out_root: Path, season_prefix: str, season: int) -> bool:
    # Layout: {prefix}/season_cards/seasonXX.json
    p = out_root / season_prefix / "season_cards" / f"season{season:02d}.json"
    return p.exists()


def _fmt_status(have: int, expected: int) -> str:
    if expected <= 0:
        return "(no episodes?)"
    if have == expected:
        return f"OK ({have}/{expected})"
    if have == 0:
        return f"MISSING (0/{expected})"
    return f"PARTIAL ({have}/{expected})"


def audit(*, docs_dir: Path, out_root: Path) -> Dict[str, object]:
    expected_by_season = _scan_expected_episodes(docs_dir)

    seg_prefixes = _auto_detect_prefixes(out_root, "derived_segsummary_season*")
    ep_prefixes = _auto_detect_prefixes(out_root, "derived_episodecard_season*")

    season_prefixes: List[str] = []
    for p in sorted(out_root.glob("derived_seasoncard_season*")):
        if not p.is_dir():
            continue
        if (p / "season_cards").exists() or (p / "season_manifest.json").exists():
            season_prefixes.append(p.name)

    seg_by_season = _latest_prefix_by_season(seg_prefixes, kind="segsummary")
    ep_by_season = _latest_prefix_by_season(ep_prefixes, kind="episodecard")
    season_by_season = _latest_prefix_by_season(season_prefixes, kind="seasoncard")

    seasons = sorted(expected_by_season.keys())

    rows: List[Dict[str, object]] = []
    for s in seasons:
        expected_eps = expected_by_season.get(s, [])
        expected_n = len(expected_eps)

        segs_present = _segments_present(out_root, s)

        seg_pref = seg_by_season.get(s)
        segsum_present = _seg_summaries_present(out_root, seg_pref, s) if seg_pref else set()

        ep_pref = ep_by_season.get(s)
        epcards_present = _episode_cards_present(out_root, ep_pref, s) if ep_pref else set()

        season_pref = season_by_season.get(s)
        season_card_ok = _season_card_present(out_root, season_pref, s) if season_pref else False

        rows.append(
            {
                "season": s,
                "expected_episodes": expected_n,
                "segments": {
                    "present": len(segs_present),
                    "status": _fmt_status(len(segs_present), expected_n),
                },
                "segment_summaries": {
                    "build_prefix": seg_pref,
                    "present": len(segsum_present),
                    "status": _fmt_status(len(segsum_present), expected_n),
                },
                "episode_cards": {
                    "build_prefix": ep_pref,
                    "present": len(epcards_present),
                    "status": _fmt_status(len(epcards_present), expected_n),
                },
                "season_card": {
                    "build_prefix": season_pref,
                    "present": bool(season_card_ok),
                },
                "missing": {
                    "segments": sorted(set(expected_eps) - set(segs_present)),
                    "segment_summaries": sorted(set(expected_eps) - set(segsum_present)),
                    "episode_cards": sorted(set(expected_eps) - set(epcards_present)),
                },
            }
        )

    return {
        "docs_dir": str(docs_dir),
        "out_root": str(out_root),
        "seasons": seasons,
        "rows": rows,
    }


def _to_markdown(report: Dict[str, object]) -> str:
    rows = report.get("rows") or []

    lines: List[str] = []
    lines.append("# Derived coverage audit")
    lines.append("")
    lines.append(f"- docs_dir: `{report.get('docs_dir')}`")
    lines.append(f"- out_root: `{report.get('out_root')}`")
    lines.append("")
    lines.append("## Status by season")
    lines.append("")
    lines.append(
        "| Season | Expected eps | Segments | Segment summaries | Episode cards | Season card |"
    )
    lines.append("|---:|---:|---|---|---|---|")

    for r in rows:
        season = int(r["season"])  # type: ignore[index]
        exp = int(r["expected_episodes"])  # type: ignore[index]
        seg = r["segments"]["status"]  # type: ignore[index]
        segsum = r["segment_summaries"]["status"]  # type: ignore[index]
        epcards = r["episode_cards"]["status"]  # type: ignore[index]
        sc = "OK" if r["season_card"]["present"] else "MISSING"  # type: ignore[index]
        lines.append(f"| {season:02d} | {exp} | {seg} | {segsum} | {epcards} | {sc} |")

    # Show only a small missing list to keep this skimmable.
    lines.append("")
    lines.append("## Missing (first 8 per category)")
    lines.append("")

    for r in rows:
        season = int(r["season"])  # type: ignore[index]
        miss = r.get("missing") or {}
        seg_m = list(miss.get("segments") or [])[:8]
        segs_m = list(miss.get("segment_summaries") or [])[:8]
        ep_m = list(miss.get("episode_cards") or [])[:8]

        if not (seg_m or segs_m or ep_m) and bool(r["season_card"]["present"]):  # type: ignore[index]
            continue

        lines.append(f"### Season {season:02d}")
        if seg_m:
            lines.append(f"- segments missing: {', '.join(seg_m)}")
        if segs_m:
            lines.append(f"- seg summaries missing: {', '.join(segs_m)}")
        if ep_m:
            lines.append(f"- episode cards missing: {', '.join(ep_m)}")
        if not bool(r["season_card"]["present"]):  # type: ignore[index]
            lines.append("- season card missing")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Audit derived artifact coverage by season")
    p.add_argument("--docs-dir", default="ingestion/normalized_docs_txt")
    p.add_argument("--out-root", default="derived/artifacts")
    p.add_argument("--write-json", default="")
    p.add_argument("--write-md", default="")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = (
        (repo_root / args.docs_dir).resolve()
        if not Path(args.docs_dir).is_absolute()
        else Path(args.docs_dir)
    )
    out_root = (
        (repo_root / args.out_root).resolve()
        if not Path(args.out_root).is_absolute()
        else Path(args.out_root)
    )

    report = audit(docs_dir=docs_dir, out_root=out_root)

    if str(args.write_json).strip():
        path = (
            (repo_root / args.write_json).resolve()
            if not Path(args.write_json).is_absolute()
            else Path(args.write_json)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = _to_markdown(report)
    if str(args.write_md).strip():
        path = (
            (repo_root / args.write_md).resolve()
            if not Path(args.write_md).is_absolute()
            else Path(args.write_md)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")

    print(md)


if __name__ == "__main__":
    main()
