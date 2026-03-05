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

from derived.reduce_episode import ReduceConfig, reduce_episode
from derived.segment_episode import EpisodeScript, iter_episode_scripts

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


@dataclass(frozen=True)
class ReduceSeasonConfig:
    season: int
    docs_dir: Path
    out_root: Path
    segments_root: Path
    segment_summaries_root: Path
    llm_model: str
    segments_build_prefix: str
    build_prefix: Optional[str]
    force: bool


def reduce_season(cfg: ReduceSeasonConfig) -> Path:
    created_at = utc_now_iso()
    build_prefix = (cfg.build_prefix or "").strip()
    if not build_prefix:
        ts = created_at.replace(":", "-")
        build_prefix = f"derived_episodecard_season{int(cfg.season):02d}_nano_{ts}"

    season_dir = cfg.out_root / build_prefix
    season_dir.mkdir(parents=True, exist_ok=True)
    season_manifest_path = season_dir / "season_manifest.json"

    episodes = _iter_season_episodes(docs_dir=cfg.docs_dir, season=int(cfg.season))
    if not episodes:
        raise FileNotFoundError(f"No episodes found for season {cfg.season} under {cfg.docs_dir}")

    plan_eps: List[Dict[str, Any]] = []
    for ep in episodes:
        seg_build_id = f"{cfg.segments_build_prefix}_{ep.episode_id}"
        plan_eps.append(
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "segments_build_id": seg_build_id,
                "episode_card_build_id": f"{build_prefix}_{ep.episode_id}",
                "out_file": f"episode_cards/{ep.episode_id}.json",
            }
        )

    manifest: Dict[str, Any] = {
        "schema": "SeasonEpisodeCardRunManifestV1",
        "schema_version": 1,
        "build_prefix": build_prefix,
        "created_at_utc": created_at,
        "season": int(cfg.season),
        "config": {
            "docs_dir": str(cfg.docs_dir.as_posix()),
            "out_root": str(cfg.out_root.as_posix()),
            "segments_root": str(cfg.segments_root.as_posix()),
            "segment_summaries_root": str(cfg.segment_summaries_root.as_posix()),
            "llm_model": cfg.llm_model,
            "segments_build_prefix": cfg.segments_build_prefix,
            "force": bool(cfg.force),
        },
        "plan": {
            "planned_episodes": plan_eps,
            "planned_total_llm_calls": len(episodes),
        },
        "execution": {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "episodes_done": 0,
            "episode_cards_written": 0,
            "errors": 0,
            "episodes": [],
        },
    }

    season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    for idx, ep in enumerate(episodes, start=1):
        t0 = time.time()
        ep_build_id = f"{build_prefix}_{ep.episode_id}"
        seg_build_id = f"{cfg.segments_build_prefix}_{ep.episode_id}"

        ep_status: Dict[str, Any] = {
            "episode_id": ep.episode_id,
            "title": ep.title,
            "status": "running",
            "segments_build_id": seg_build_id,
            "episode_card_build_id": ep_build_id,
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "duration_ms": None,
            "error": None,
        }

        try:
            reduce_episode(
                ReduceConfig(
                    docs_dir=cfg.docs_dir,
                    segments_root=cfg.segments_root,
                    segment_summaries_root=cfg.segment_summaries_root,
                    out_root=cfg.out_root,
                    episode_id=ep.episode_id,
                    segments_build_id=seg_build_id,
                    llm_model=cfg.llm_model,
                    build_id=ep_build_id,
                    force=bool(cfg.force),
                    prompt_only=False,
                )
            )
            manifest["execution"]["episodes_done"] += 1
            manifest["execution"]["episode_cards_written"] += 1
            ep_status["status"] = "done"

        except Exception as e:
            manifest["execution"]["errors"] += 1
            ep_status["status"] = "error"
            ep_status["error"] = f"{type(e).__name__}: {e}"

        ep_status["ended_at_utc"] = utc_now_iso()
        ep_status["duration_ms"] = int((time.time() - t0) * 1000)
        manifest["execution"]["episodes"].append(ep_status)
        season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")
        print(f"[{idx}/{len(episodes)}] {ep.episode_id} {ep_status['status']} ({ep_status['duration_ms']}ms)")

    manifest["execution"]["status"] = "done" if manifest["execution"]["errors"] == 0 else "done_with_errors"
    manifest["execution"]["ended_at_utc"] = utc_now_iso()
    season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    return season_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Reduce one season's SegmentSummaryV1 into EpisodeDerivedCardV1")
    p.add_argument("--season", type=int, required=True, help="Season number (1-9)")
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--segments-root", default=DEFAULT_SEGMENTS_ROOT)
    p.add_argument("--segment-summaries-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--llm-model", default=None, help="Defaults to reduce_episode.py default")
    p.add_argument(
        "--segments-build-prefix",
        required=True,
        help="Prefix used for segment summary build ids, e.g. derived_segsummary_season01_nano_2026-03-04",
    )
    p.add_argument("--build-prefix", default=None, help="Output build prefix under out-root")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()

    docs_dir = _resolve_under_repo(str(args.docs_dir))
    out_root = _resolve_under_repo(str(args.out_root))
    segments_root = _resolve_under_repo(str(args.segments_root))
    segsum_root = _resolve_under_repo(str(args.segment_summaries_root))

    llm_model = (str(args.llm_model).strip() if args.llm_model else None)
    if not llm_model:
        from derived.reduce_episode import DEFAULT_LLM_MODEL as REDUCE_DEFAULT

        llm_model = REDUCE_DEFAULT

    out_dir = reduce_season(
        ReduceSeasonConfig(
            season=int(args.season),
            docs_dir=docs_dir,
            out_root=out_root,
            segments_root=segments_root,
            segment_summaries_root=segsum_root,
            llm_model=llm_model,
            segments_build_prefix=str(args.segments_build_prefix).strip(),
            build_prefix=(str(args.build_prefix) if args.build_prefix else None),
            force=bool(args.force),
        )
    )

    print(f"Done. Season episode cards under: {out_dir}")


if __name__ == "__main__":
    main()
