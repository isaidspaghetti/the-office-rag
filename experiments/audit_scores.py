from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_json_files(root: Path, *, suffix: str = ".json") -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return
    yield from sorted(root.rglob(f"*{suffix}"))


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


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
        return float(str(x).strip())
    except Exception:
        return None


@dataclass(frozen=True)
class Row:
    run_file: Path
    scored_file: Optional[Path]
    run_id: str
    run_name: str
    cases: int
    answered_with_llm: Optional[bool]
    cases_scored: Optional[int]
    avg_overall: Optional[float]
    judge_cases: Optional[int]
    judge_error_cases: Optional[int]
    flags: Tuple[str, ...]


def _judge_case_counts(scored_obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(scored_obj, dict):
        return None, None
    scored_cases = scored_obj.get("scored_cases")
    if not isinstance(scored_cases, list):
        return None, None

    judge_cases = 0
    judge_error_cases = 0
    for r in scored_cases:
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("judge"), dict):
            judge_cases += 1
        if r.get("judge_error"):
            judge_error_cases += 1
    return judge_cases, judge_error_cases


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit run logs and scored outputs for suspicious pairings.")
    ap.add_argument("--runs-dir", default="experiments/runs")
    ap.add_argument("--scored-dir", default="experiments/scored_runs")
    ap.add_argument("--out", default="experiments/_score_audit.txt")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    scored_dir = Path(args.scored_dir).expanduser().resolve()

    scored_by_id: Dict[str, Path] = {}
    for p in _iter_json_files(scored_dir, suffix=".scored.json"):
        rid = p.name.replace(".scored.json", "")
        scored_by_id[rid] = p

    rows: List[Row] = []

    for run_path in _iter_json_files(runs_dir, suffix=".json"):
        run_obj, err = _read_json(run_path)
        if err is not None or not isinstance(run_obj, dict):
            # cleanup_artifacts.py handles these.
            continue

        run_id = str(safe_get(run_obj, "run.run_id", run_path.stem) or run_path.stem)
        run_name = str(safe_get(run_obj, "run.run_name", run_path.stem) or run_path.stem)
        cases_obj = run_obj.get("cases")
        cases = len(cases_obj) if isinstance(cases_obj, list) else 0

        answered_with_llm = safe_get(run_obj, "summary.answered_with_llm", None)
        if answered_with_llm is None:
            answered_with_llm = safe_get(run_obj, "config.llm.enabled", None)
        answered_bool = bool(answered_with_llm) if isinstance(answered_with_llm, bool) else None

        scored_path = scored_by_id.get(run_id)
        cases_scored = avg_overall = None
        judge_cases = judge_error_cases = None

        flags: List[str] = []

        if scored_path is None:
            flags.append("missing_scored")
        else:
            scored_obj, s_err = _read_json(scored_path)
            if s_err is not None:
                flags.append("scored_broken_json")
            else:
                cases_scored = safe_int(safe_get(scored_obj, "score_summary.cases_scored", None))
                avg_overall = safe_float(safe_get(scored_obj, "score_summary.avg_overall", None))
                judge_cases, judge_error_cases = _judge_case_counts(scored_obj)

                if answered_bool is False:
                    flags.append("scored_for_no_llm_run")
                if cases_scored == 0:
                    flags.append("zero_cases_scored")
                if avg_overall is not None and not (0.0 <= float(avg_overall) <= 100.0):
                    flags.append("avg_overall_out_of_range")
                if judge_error_cases is not None and judge_error_cases > 0:
                    flags.append("judge_errors_present")

        rows.append(
            Row(
                run_file=run_path,
                scored_file=scored_path,
                run_id=run_id,
                run_name=run_name,
                cases=cases,
                answered_with_llm=answered_bool,
                cases_scored=cases_scored,
                avg_overall=avg_overall,
                judge_cases=judge_cases,
                judge_error_cases=judge_error_cases,
                flags=tuple(flags),
            )
        )

    # Sort: most suspicious first.
    def _rank(r: Row) -> Tuple[int, int, str]:
        score = 0
        for f in r.flags:
            if f in {"scored_for_no_llm_run", "scored_broken_json", "avg_overall_out_of_range"}:
                score += 100
            elif f in {"zero_cases_scored"}:
                score += 50
            elif f in {"judge_errors_present"}:
                score += 10
            elif f in {"missing_scored"}:
                score += 1
        return (-score, -(r.judge_error_cases or 0), r.run_id)

    rows_sorted = sorted(rows, key=_rank)

    out_lines: List[str] = []
    out_lines.append(f"Runs scanned: {len(rows)}")

    flagged = [r for r in rows_sorted if r.flags]
    out_lines.append(f"Runs with any flags: {len(flagged)}")
    out_lines.append("")

    for r in flagged[:60]:
        out_lines.append(
            "\t".join(
                [
                    r.run_id,
                    r.run_name,
                    f"cases={r.cases}",
                    f"answered_with_llm={r.answered_with_llm}",
                    f"cases_scored={r.cases_scored}",
                    f"avg_overall={r.avg_overall}",
                    f"judge_cases={r.judge_cases}",
                    f"judge_error_cases={r.judge_error_cases}",
                    f"flags={','.join(r.flags)}",
                ]
            )
        )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
