from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _to_int(x: Any) -> Optional[int]:
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


_STEP_RE = re.compile(r"step(?P<n>\d{2})", re.IGNORECASE)


def _extract_step(run_name: str) -> Optional[int]:
    m = _STEP_RE.search(run_name or "")
    if not m:
        return None
    try:
        return int(m.group("n"), 10)
    except Exception:
        return None


def infer_phase_from_run_name(run_name: str) -> str:
    """Heuristic phase classifier for run names without explicit stepNN markers."""
    s = (run_name or "").lower()

    has_qe = "qe" in s or "query_expansion" in s
    has_rrf = "rrf" in s or "fusion" in s
    has_mmr = "mmr" in s

    if "baseline" in s:
        return "baseline"

    if has_qe and has_rrf and has_mmr:
        return "qe+fusion+mmr"
    if has_qe and has_rrf:
        return "qe+fusion"
    if has_qe:
        return "qe"
    if has_mmr:
        return "mmr"

    # Routing-ish labels
    if "blended" in s or "hybrid" in s:
        return "routing_blended"
    if "derived_then_script" in s or "derived->script" in s:
        return "routing_derived_then_script"
    if "derived_only" in s:
        return "routing_derived_only"
    if "script_only" in s:
        return "script_only"

    return "other"


@dataclass(frozen=True)
class ScoredRun:
    run_id: str
    run_name: str
    created_at_utc: str
    step: Optional[int]
    phase: str
    avg_overall: Optional[int]
    scored_file: str


def iter_scored_runs(
    *,
    runs_dir: Path,
    scored_dir: Path,
    run_name_prefix: str,
    group_by: str,
    require_llm: bool,
) -> Iterable[ScoredRun]:
    for rf in sorted(runs_dir.rglob("*.json")):
        try:
            run_obj = _read_json(rf)
        except Exception:
            continue
        if not isinstance(run_obj, dict):
            continue

        if require_llm:
            llm_enabled = _safe_get(run_obj, "config.llm.enabled", None)
            if not bool(llm_enabled):
                continue

        run_id = str(_safe_get(run_obj, "run.run_id", "") or "").strip()
        run_name = str(_safe_get(run_obj, "run.run_name", "") or "")
        created_at_utc = str(_safe_get(run_obj, "run.created_at_utc", "") or "")

        if run_name_prefix and not run_name.startswith(run_name_prefix):
            continue
        if not run_id:
            continue

        scored_path = (scored_dir / f"{run_id}.scored.json").expanduser().resolve()
        if not scored_path.exists():
            continue

        try:
            scored_obj = _read_json(scored_path)
        except Exception:
            continue

        avg_overall = _to_int(_safe_get(scored_obj, "score_summary.avg_overall", None))

        step = _extract_step(run_name)
        phase = f"step{step:02d}" if step is not None else infer_phase_from_run_name(run_name)

        if group_by == "step" and step is None:
            # Skip if caller explicitly requested step grouping.
            continue

        yield ScoredRun(
            run_id=run_id,
            run_name=run_name,
            created_at_utc=created_at_utc,
            step=step,
            phase=phase,
            avg_overall=avg_overall,
            scored_file=str(scored_path.as_posix()),
        )


def summarize_groups(runs: List[ScoredRun], *, group_by: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[ScoredRun]] = {}
    for r in runs:
        key = f"step{r.step:02d}" if (group_by == "step" and r.step is not None) else r.phase
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        scored = [r for r in rows if isinstance(r.avg_overall, int)]
        best = max(scored, key=lambda r: int(r.avg_overall)) if scored else None
        worst = min(scored, key=lambda r: int(r.avg_overall)) if scored else None

        out.append(
            {
                "group": key,
                "runs_total": len(rows),
                "runs_with_overall": len(scored),
                "best_avg_overall": (best.avg_overall if best else None),
                "best_run_id": (best.run_id if best else None),
                "best_run_name": (best.run_name if best else None),
                "worst_avg_overall": (worst.avg_overall if worst else None),
                "worst_run_id": (worst.run_id if worst else None),
                "worst_run_name": (worst.run_name if worst else None),
            }
        )

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
        g = str(row.get("group") or "")
        m = re.match(r"^step(\d{2})$", g)
        if m:
            return (0, m.group(1))
        return (1, g)

    out.sort(key=_sort_key)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def try_plot_png(path: Path, rows: List[Dict[str, Any]], *, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    labels = [str(r["group"]) for r in rows]
    best = [r.get("best_avg_overall") for r in rows]
    worst = [r.get("worst_avg_overall") for r in rows]

    # Replace None with NaN so matplotlib gaps them.
    def _nan(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    best_y = [_nan(x) for x in best]
    worst_y = [_nan(x) for x in worst]

    fig_w = max(9.0, 0.55 * len(labels))
    plt.figure(figsize=(fig_w, 4.8))
    plt.plot(labels, best_y, marker="o", label="best avg_overall")
    plt.plot(labels, worst_y, marker="o", label="worst avg_overall")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.as_posix(), dpi=160)
    plt.close()
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize best/worst judge scores per phase and optionally plot a PNG.")
    p.add_argument("--runs-dir", default="experiments/runs")
    p.add_argument("--scored-dir", default="experiments/scored_runs_two_pass")
    p.add_argument("--run-name-prefix", default="", help="Optional filter for run.run_name")
    p.add_argument("--group-by", default="phase", choices=["phase", "step"], help="Group runs by inferred phase or stepNN")
    p.add_argument(
        "--require-llm",
        action="store_true",
        help="Only include runs where config.llm.enabled is true (exclude retrieval-only runs).",
    )
    p.add_argument("--out-csv", default="experiments/phase_score_summary.csv")
    p.add_argument("--out-json", default="experiments/phase_score_summary.json")
    p.add_argument("--out-png", default="experiments/phase_score_summary.png")
    p.add_argument("--no-png", action="store_true", help="Skip PNG generation")

    args = p.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    scored_dir = Path(args.scored_dir).expanduser().resolve()

    runs = list(
        iter_scored_runs(
            runs_dir=runs_dir,
            scored_dir=scored_dir,
            run_name_prefix=str(args.run_name_prefix or ""),
            group_by=str(args.group_by),
            require_llm=bool(args.require_llm),
        )
    )

    if not runs:
        raise SystemExit("No scored runs found (check --scored-dir and --run-name-prefix)")

    summary = summarize_groups(runs, group_by=str(args.group_by))

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    write_csv(out_csv, summary)
    write_json(
        out_json,
        {
            "group_by": str(args.group_by),
            "run_name_prefix": str(args.run_name_prefix or ""),
            "require_llm": bool(args.require_llm),
            "rows": summary,
        },
    )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")

    if not bool(args.no_png):
        out_png = Path(args.out_png).expanduser().resolve()
        ok = try_plot_png(out_png, summary, title=f"Best/Worst avg_overall by {args.group_by}")
        if ok:
            print(f"Wrote: {out_png}")
        else:
            print("PNG skipped: matplotlib not installed")


if __name__ == "__main__":
    main()
