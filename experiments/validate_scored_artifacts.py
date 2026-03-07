from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Row:
    path: Path
    judge_mode: str
    cases_scored: int
    avg_overall: Optional[int]
    run_id: str
    run_name: str


def _read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("not a dict")
    return obj


def _safe_int(x: Any) -> Optional[int]:
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


def summarize_scored_dir(scored_dir: Path) -> Tuple[List[Row], List[Tuple[Path, str]]]:
    rows: List[Row] = []
    errors: List[Tuple[Path, str]] = []

    for path in sorted(scored_dir.glob("*.scored.json")):
        try:
            obj = _read_json(path)

            scoring_meta = obj.get("scoring_meta") if isinstance(obj.get("scoring_meta"), dict) else {}
            judge_mode = str(scoring_meta.get("judge_mode") or "missing").strip()

            score_summary = obj.get("score_summary") if isinstance(obj.get("score_summary"), dict) else {}
            cases_scored = _safe_int(score_summary.get("cases_scored")) or 0
            avg_overall = _safe_int(score_summary.get("avg_overall"))

            run = obj.get("run") if isinstance(obj.get("run"), dict) else {}
            run_id = str(run.get("run_id") or "").strip()
            run_name = str(run.get("run_name") or "").strip()

            rows.append(
                Row(
                    path=path,
                    judge_mode=judge_mode,
                    cases_scored=int(cases_scored),
                    avg_overall=avg_overall,
                    run_id=run_id,
                    run_name=run_name,
                )
            )
        except Exception as e:
            errors.append((path, f"{type(e).__name__}: {e}"))

    return rows, errors


def main() -> None:
    p = argparse.ArgumentParser(description="Validate scored run artifacts for Streamlit dashboards")
    p.add_argument(
        "--scored-dir",
        default="experiments/scored_runs_two_pass",
        help="Directory containing *.scored.json files",
    )
    p.add_argument(
        "--require-judge-mode",
        default="two_pass",
        help="Expected scoring_meta.judge_mode (default: two_pass)",
    )
    p.add_argument(
        "--min-cases-scored",
        type=int,
        default=1,
        help="Fail if any artifact has fewer cases_scored than this",
    )
    p.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Print N sample files",
    )
    args = p.parse_args()

    scored_dir = Path(args.scored_dir).expanduser().resolve()
    if not scored_dir.exists() or not scored_dir.is_dir():
        raise SystemExit(f"scored_dir does not exist or is not a dir: {scored_dir}")

    rows, errors = summarize_scored_dir(scored_dir)

    mode_counts: Dict[str, int] = {}
    cases = [r.cases_scored for r in rows]

    bad_mode = [r for r in rows if r.judge_mode != str(args.require_judge_mode)]
    low_cases = [r for r in rows if r.cases_scored < int(args.min_cases_scored)]

    for r in rows:
        mode_counts[r.judge_mode] = mode_counts.get(r.judge_mode, 0) + 1

    mean_cases = (sum(cases) / len(cases)) if cases else None
    min_cases = min(cases) if cases else None
    max_cases = max(cases) if cases else None

    print("=== Scored artifacts validation ===")
    print("scored_dir:", scored_dir)
    print("files:", len(rows))
    print("judge_mode_counts:", mode_counts)
    print("cases_scored:", {"min": min_cases, "max": max_cases, "mean": (round(mean_cases, 2) if mean_cases is not None else None)})
    print("json_errors:", len(errors))

    if errors:
        print("\n--- JSON read errors (first 10) ---")
        for path, msg in errors[:10]:
            print("-", path.name, msg)

    if bad_mode:
        print("\n--- FAIL: wrong judge_mode (first 10) ---")
        for r in bad_mode[:10]:
            print("-", r.path.name, "judge_mode=", r.judge_mode)

    if low_cases:
        print("\n--- FAIL: too few cases_scored (first 10) ---")
        for r in low_cases[:10]:
            print("-", r.path.name, "cases_scored=", r.cases_scored, "run_name=", r.run_name)

    if int(args.show_samples) > 0 and rows:
        print("\n--- Samples ---")
        for r in rows[: int(args.show_samples)]:
            print(
                "-",
                r.path.name,
                "run_id=",
                (r.run_id or "(missing)"),
                "run_name=",
                (r.run_name or "(missing)"),
                "avg_overall=",
                r.avg_overall,
                "judge_mode=",
                r.judge_mode,
                "cases_scored=",
                r.cases_scored,
            )

    ok = True
    if errors:
        ok = False
    if bad_mode:
        ok = False
    if low_cases:
        ok = False

    if not ok:
        raise SystemExit(2)

    print("\nOK: scored artifacts look deploy-ready")


if __name__ == "__main__":
    main()
