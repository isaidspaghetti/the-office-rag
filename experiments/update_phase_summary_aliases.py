from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _pick_latest(
    candidates: list[Path], *, prefer_ctxfix: bool = True, prefer_llm: bool = True
) -> Optional[Path]:
    if not candidates:
        return None

    def _is_ctxfix(p: Path) -> bool:
        return "ctxfix" in p.name.lower()

    def _is_llm(p: Path) -> bool:
        return "_llm" in p.name.lower()

    pool = list(candidates)

    if prefer_ctxfix:
        ctxfix = [p for p in pool if _is_ctxfix(p)]
        if ctxfix:
            pool = ctxfix

    if prefer_llm:
        llm = [p for p in pool if _is_llm(p)]
        if llm:
            pool = llm

    return sorted(pool)[-1]


def _copy_sidecars(src_json: Path, dst_json: Path, *, overwrite: bool) -> None:
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst: Path) -> None:
        if dst.exists() and not overwrite:
            return
        shutil.copy2(src, dst)

    _copy(src_json, dst_json)

    for ext in (".csv", ".png"):
        sidecar = src_json.with_suffix(ext)
        if sidecar.exists():
            _copy(sidecar, dst_json.with_suffix(ext))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Create stable alias files for the latest phase score summary artifacts. "
            "Copies JSON/CSV/PNG into fixed filenames under experiments/."
        )
    )
    p.add_argument(
        "--experiments-dir",
        default=str(EXPERIMENTS_DIR),
        help="Path to experiments/ directory (default: repo/experiments)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing alias files (default: false)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done; do not write any files",
    )

    args = p.parse_args()

    exp_dir = Path(args.experiments_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise SystemExit(f"experiments dir not found: {exp_dir}")

    jsons = sorted(exp_dir.glob("phase_score_summary_*.json"))
    if not jsons:
        raise SystemExit(f"No phase_score_summary_*.json under: {exp_dir}")

    by_step = [p for p in jsons if "by_step" in p.name.lower()]
    phase_level = [p for p in jsons if "by_step" not in p.name.lower()]

    latest_phase = (
        _pick_latest(phase_level, prefer_ctxfix=True, prefer_llm=True) or sorted(phase_level)[-1]
    )
    latest_step = _pick_latest(by_step, prefer_ctxfix=True, prefer_llm=True) if by_step else None

    alias_phase = exp_dir / "phase_score_summary_latest.json"
    alias_step = exp_dir / "phase_score_summary_latest_by_step.json"

    def _print(msg: str) -> None:
        print(msg)

    _print(f"latest_phase: {latest_phase.name}")
    _print(f"alias_phase: {alias_phase.name}")
    if latest_step:
        _print(f"latest_by_step: {latest_step.name}")
        _print(f"alias_by_step: {alias_step.name}")

    if args.dry_run:
        return

    _copy_sidecars(latest_phase, alias_phase, overwrite=bool(args.overwrite))
    if latest_step:
        _copy_sidecars(latest_step, alias_step, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
