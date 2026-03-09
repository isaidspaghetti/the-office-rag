from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from derived.segment_episode import (
    EpisodeScript,
    approx_tokens_from_chars,
    iter_episode_scripts,
    segment_episode_text,
)
from derived.summarize_segments import SummarizeConfig, summarize_episode_segments

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"
DEFAULT_OUT_ROOT = "derived/artifacts"
DEFAULT_SEGMENTS_ROOT = "derived/artifacts/segments"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _season_from_episode_id(episode_id: str) -> Optional[int]:
    s = (episode_id or "").strip().upper()
    if len(s) < 4 or not s.startswith("S"):
        return None
    try:
        return int(s[1:3])
    except Exception:
        return None


def _iter_season_episodes(*, docs_dir: Path, season: int) -> List[EpisodeScript]:
    out: List[EpisodeScript] = []
    for ep in iter_episode_scripts(docs_dir=docs_dir):
        s = _season_from_episode_id(ep.episode_id)
        if s == int(season):
            out.append(ep)
    return sorted(out, key=lambda e: e.episode_id)


def _write_episode_segments_file(
    *,
    episode: EpisodeScript,
    segments_root: Path,
    target_tokens: int,
    min_tokens: int,
    force: bool,
) -> Path:
    segments_root.mkdir(parents=True, exist_ok=True)
    out_path = segments_root / f"{episode.episode_id}.json"
    if out_path.exists() and not force:
        return out_path

    segments = segment_episode_text(
        episode_id=episode.episode_id,
        body_text=episode.body_text,
        target_tokens=int(target_tokens),
        min_tokens=int(min_tokens),
    )

    obj: Dict[str, Any] = {
        "schema": "EpisodeSegmentsV1",
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "episode_id": episode.episode_id,
        "title": episode.title,
        "source": episode.source_path,
        "target_tokens": int(target_tokens),
        "min_tokens": int(min_tokens),
        "body_char_count": len(episode.body_text),
        "body_approx_tokens": approx_tokens_from_chars(len(episode.body_text)),
        "segments": [
            {
                "segment_id": s.segment_id,
                "segment_index": s.segment_index,
                "segment_char_start": s.segment_char_start,
                "segment_char_end": s.segment_char_end,
                "approx_tokens": s.approx_tokens,
            }
            for s in segments
        ],
    }

    out_path.write_text(_safe_json(obj) + "\n", encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class SeasonRunConfig:
    season: int
    docs_dir: Path
    out_root: Path
    segments_root: Path
    llm_model: str
    target_tokens: int
    min_tokens: int
    build_prefix: Optional[str]
    force_segments: bool
    force_summaries: bool


def summarize_season(cfg: SeasonRunConfig) -> Path:
    created_at = utc_now_iso()
    build_prefix = (cfg.build_prefix or "").strip()
    if not build_prefix:
        ts = created_at.replace(":", "-")
        build_prefix = f"derived_segsummary_season{int(cfg.season):02d}_nano_{ts}"

    season_dir = cfg.out_root / build_prefix
    season_dir.mkdir(parents=True, exist_ok=True)
    season_manifest_path = season_dir / "season_manifest.json"

    episodes = _iter_season_episodes(docs_dir=cfg.docs_dir, season=int(cfg.season))
    if not episodes:
        raise FileNotFoundError(f"No episodes found for season {cfg.season} under {cfg.docs_dir}")

    # Resume if manifest exists; otherwise create a fresh plan.
    if season_manifest_path.exists():
        manifest = json.loads(season_manifest_path.read_text(encoding="utf-8"))
        # Ensure we keep going rather than overwriting history.
        manifest.setdefault("execution", {})
        manifest["execution"]["status"] = "running"
        manifest["execution"].setdefault("started_at_utc", utc_now_iso())
        manifest["execution"]["ended_at_utc"] = None
        manifest.setdefault("config", {})
        manifest["config"].update(
            {
                "docs_dir": str(cfg.docs_dir.as_posix()),
                "out_root": str(cfg.out_root.as_posix()),
                "segments_root": str(cfg.segments_root.as_posix()),
                "llm_model": cfg.llm_model,
                "target_tokens": int(cfg.target_tokens),
                "min_tokens": int(cfg.min_tokens),
                "force_segments": bool(cfg.force_segments),
                "force_summaries": bool(cfg.force_summaries),
            }
        )
        manifest["resumed_at_utc"] = utc_now_iso()
    else:
        # Plan
        plan_eps: List[Dict[str, Any]] = []
        planned_calls = 0
        for ep in episodes:
            segs = segment_episode_text(
                episode_id=ep.episode_id,
                body_text=ep.body_text,
                target_tokens=int(cfg.target_tokens),
                min_tokens=int(cfg.min_tokens),
            )
            planned_calls += len(segs)
            plan_eps.append(
                {
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "planned_segments": len(segs),
                    "planned_llm_calls": len(segs),
                    "segment_file": f"segments/{ep.episode_id}.json",
                    "segment_summaries_build_id": f"{build_prefix}_{ep.episode_id}",
                }
            )

        manifest = {
            "schema": "SeasonSegmentSummaryRunManifestV1",
            "schema_version": 1,
            "build_prefix": build_prefix,
            "created_at_utc": created_at,
            "season": int(cfg.season),
            "config": {
                "docs_dir": str(cfg.docs_dir.as_posix()),
                "out_root": str(cfg.out_root.as_posix()),
                "segments_root": str(cfg.segments_root.as_posix()),
                "llm_model": cfg.llm_model,
                "target_tokens": int(cfg.target_tokens),
                "min_tokens": int(cfg.min_tokens),
                "force_segments": bool(cfg.force_segments),
                "force_summaries": bool(cfg.force_summaries),
            },
            "plan": {
                "planned_episodes": plan_eps,
                "planned_total_llm_calls": planned_calls,
            },
            "execution": {
                "status": "running",
                "started_at_utc": utc_now_iso(),
                "ended_at_utc": None,
                "episodes_done": 0,
                "segment_summaries_written": 0,
                "errors": 0,
                "episodes": [],
            },
        }

    season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    # Build a quick lookup so we can skip episodes already done.
    prior_status: Dict[str, str] = {}
    for e in manifest.get("execution", {}).get("episodes", []) or []:
        if isinstance(e, dict) and e.get("episode_id"):
            prior_status[str(e["episode_id"]).strip().upper()] = str(e.get("status") or "")

    # Execute
    for idx, ep in enumerate(episodes, start=1):
        # Skip fully completed episodes on resume unless forcing summaries.
        if prior_status.get(ep.episode_id) == "done" and not cfg.force_summaries:
            print(f"[{idx}/{len(episodes)}] {ep.episode_id} skip (already done)")
            continue
        t0 = time.time()
        ep_status: Dict[str, Any] = {
            "episode_id": ep.episode_id,
            "title": ep.title,
            "status": "running",
            "segment_file": None,
            "segment_summaries_build_id": f"{build_prefix}_{ep.episode_id}",
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "duration_ms": None,
            "error": None,
        }

        try:
            seg_path = _write_episode_segments_file(
                episode=ep,
                segments_root=cfg.segments_root,
                target_tokens=int(cfg.target_tokens),
                min_tokens=int(cfg.min_tokens),
                force=bool(cfg.force_segments),
            )
            ep_status["segment_file"] = str(seg_path.as_posix())

            # Per-episode build id so summarize_segments.py's manifest.json doesn't collide.
            ep_build_id = ep_status["segment_summaries_build_id"]
            out_dir = summarize_episode_segments(
                SummarizeConfig(
                    docs_dir=cfg.docs_dir,
                    out_root=cfg.out_root,
                    episode_id=ep.episode_id,
                    llm_model=cfg.llm_model,
                    target_tokens=int(cfg.target_tokens),
                    min_tokens=int(cfg.min_tokens),
                    max_segments=None,
                    segment_index=None,
                    build_id=ep_build_id,
                    force=bool(cfg.force_summaries),
                    prompt_only=False,
                )
            )

            # Count what was written for this episode in its own manifest.
            ep_manifest_path = cfg.out_root / ep_build_id / "manifest.json"
            written = 0
            if ep_manifest_path.exists():
                ep_manifest = json.loads(ep_manifest_path.read_text(encoding="utf-8"))
                written = int(ep_manifest.get("execution", {}).get("written", 0) or 0)

            manifest["execution"]["episodes_done"] += 1
            manifest["execution"]["segment_summaries_written"] += written
            ep_status["status"] = "done"
            ep_status["out_dir"] = str(out_dir.as_posix())

        except Exception as e:
            manifest["execution"]["errors"] += 1
            ep_status["status"] = "error"
            ep_status["error"] = f"{type(e).__name__}: {e}"

        ep_status["ended_at_utc"] = utc_now_iso()
        ep_status["duration_ms"] = int((time.time() - t0) * 1000)
        manifest["execution"]["episodes"].append(ep_status)

        season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")
        print(
            f"[{idx}/{len(episodes)}] {ep.episode_id} {ep_status['status']} ({ep_status['duration_ms']}ms)"
        )

    # Recompute aggregate counters based on the latest status per episode.
    latest_by_episode: Dict[str, Dict[str, Any]] = {}
    for e in manifest.get("execution", {}).get("episodes", []) or []:
        if isinstance(e, dict) and e.get("episode_id"):
            latest_by_episode[str(e["episode_id"]).strip().upper()] = e

    done_count = sum(1 for e in latest_by_episode.values() if e.get("status") == "done")
    error_count = sum(1 for e in latest_by_episode.values() if e.get("status") == "error")

    written_sum = 0
    for episode_id in sorted(latest_by_episode.keys()):
        ep_build_id = f"{build_prefix}_{episode_id}"
        ep_manifest_path = cfg.out_root / ep_build_id / "manifest.json"
        if ep_manifest_path.exists():
            try:
                ep_manifest = json.loads(ep_manifest_path.read_text(encoding="utf-8"))
                written_sum += int(ep_manifest.get("execution", {}).get("written", 0) or 0)
            except Exception:
                pass

    manifest["execution"]["episodes_done"] = done_count
    manifest["execution"]["errors"] = error_count
    manifest["execution"]["segment_summaries_written"] = written_sum

    # Convenience view: latest status per episode (useful after resume/retries).
    manifest["execution"]["latest_episodes"] = [
        latest_by_episode[episode_id] for episode_id in sorted(latest_by_episode.keys())
    ]

    manifest["execution"]["status"] = "done" if error_count == 0 else "done_with_errors"
    manifest["execution"]["ended_at_utc"] = utc_now_iso()
    season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    return season_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize all episode segments for one season (map step)"
    )
    p.add_argument("--season", type=int, required=True, help="Season number (1-9)")
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--segments-root", default=DEFAULT_SEGMENTS_ROOT)
    p.add_argument("--llm-model", default=None, help="Defaults to summarize_segments.py default")
    p.add_argument("--target-tokens", type=int, default=1800)
    p.add_argument("--min-tokens", type=int, default=400)
    p.add_argument("--build-prefix", default=None, help="Folder prefix under out-root")
    p.add_argument("--force-segments", action="store_true")
    p.add_argument("--force-summaries", action="store_true")

    args = p.parse_args()

    docs_dir = _resolve_under_repo(str(args.docs_dir))
    out_root = _resolve_under_repo(str(args.out_root))
    segments_root = _resolve_under_repo(str(args.segments_root))

    # If unset, reuse summarize_segments.py's default.
    llm_model = str(args.llm_model).strip() if args.llm_model else None
    if not llm_model:
        from derived.summarize_segments import DEFAULT_LLM_MODEL as SUM_DEFAULT

        llm_model = SUM_DEFAULT

    out_dir = summarize_season(
        SeasonRunConfig(
            season=int(args.season),
            docs_dir=docs_dir,
            out_root=out_root,
            segments_root=segments_root,
            llm_model=llm_model,
            target_tokens=int(args.target_tokens),
            min_tokens=int(args.min_tokens),
            build_prefix=(str(args.build_prefix) if args.build_prefix else None),
            force_segments=bool(args.force_segments),
            force_summaries=bool(args.force_summaries),
        )
    )

    print(f"Done. Season artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
