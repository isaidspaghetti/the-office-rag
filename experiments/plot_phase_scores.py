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

    has_meta = "meta_" in s or "_meta_" in s or "chroma_db_meta" in s

    # In our timeline, "meta index" is effectively a second derived phase we liked.
    # Make it its own bucket even if it also includes QE/MMR.
    if has_meta:
        return "derived_meta"

    # Early phase: baseline then bump k (usually from k=3 -> k=12).
    is_k_jump = (
        ("_t1_" in s or "similarity_k12" in s or re.search(r"(?:^|_)k12(?:_|$)", s) is not None)
        and ("similarity" in s)
        and (not has_qe)
        and (not has_mmr)
    )

    if "baseline" in s:
        return "baseline"

    if is_k_jump:
        return "k12"

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


_PHASE_ORDER: Dict[str, int] = {
    "baseline": 10,
    "k12": 20,
    "mmr": 30,
    "qe+fusion": 40,
    "qe+fusion+mmr": 50,
    "routing_blended": 55,
    "routing_derived_only": 60,
    "routing_derived_then_script": 70,
    "derived_meta": 80,
    "script_only": 90,
    "qe": 95,
    "other": 999,
}


_PHASE_DISPLAY: Dict[str, str] = {
    "baseline": "Baseline",
    "k12": "Increase retrieval k (k=12)",
    "mmr": "MMR",
    "qe+fusion": "QE + fusion (RRF)",
    "qe+fusion+mmr": "QE + fusion (RRF) + MMR",
    "routing_blended": "Routing blended (derived + script)",
    "routing_derived_then_script": "Routing: derived → script",
    "routing_derived_only": "Derived-only",
    "derived_meta": "Derived v2 (metadata index)",
    "script_only": "Script-only",
    "qe": "Query expansion (no fusion)",
    "other": "Other",
}


_STEP_FRIENDLY: Dict[str, str] = {
    "step01": "Baseline (script-only; no derived)",
    "step02": "Derived-only retrieval",
    "step03": "Routing: derived → script",
    "step04": "Higher k / MMR tuning",
    "step05": "Query expansion + fusion (RRF)",
    "step06": "Best combined",
}


def _display_group(group: str, *, group_by: str) -> str:
    g = str(group or "").strip()
    gl = g.lower()
    if group_by == "step" and gl in _STEP_FRIENDLY:
        return f"{gl} — {_STEP_FRIENDLY[gl]}"
    if group_by == "phase" and gl in _PHASE_DISPLAY:
        return _PHASE_DISPLAY[gl]
    return g


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
        # Important: for group_by=phase, do NOT keep stepNN as its own group.
        # Steps belong in the dedicated group_by=step artifact; the phase artifact should
        # show the broader timeline buckets (baseline/mmr/qe+fusion/.../routing).
        if group_by == "phase":
            phase = infer_phase_from_run_name(run_name)
        else:
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

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
        g = str(row.get("group") or "")
        gl = g.lower()
        if group_by == "step":
            m = re.match(r"^step(\d{2})$", gl)
            if m:
                return (0, int(m.group(1)), gl)
            return (1, 10**9, gl)

        # group_by == phase
        if gl in _PHASE_ORDER:
            return (0, int(_PHASE_ORDER[gl]), gl)
        return (1, 10**9, gl)

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


def try_plot_png(path: Path, rows: List[Dict[str, Any]], *, title: str, group_by: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    # Prefer user-friendly display labels on the PNG while keeping stable group keys in JSON.
    labels = [_display_group(str(r["group"]), group_by=str(group_by)) for r in rows]
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
    p = argparse.ArgumentParser(
        description="Summarize best/worst judge scores per phase and optionally plot a PNG."
    )
    p.add_argument("--runs-dir", default="experiments/runs")
    p.add_argument("--scored-dir", default="experiments/scored_runs_two_pass")
    p.add_argument("--run-name-prefix", default="", help="Optional filter for run.run_name")
    p.add_argument(
        "--group-by",
        default="phase",
        choices=["phase", "step"],
        help="Group runs by inferred phase or stepNN",
    )
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
        ok = try_plot_png(
            out_png,
            summary,
            title=f"Best/Worst avg_overall by {args.group_by}",
            group_by=str(args.group_by),
        )
        if ok:
            print(f"Wrote: {out_png}")
        else:
            print("PNG skipped: matplotlib not installed")


if __name__ == "__main__":
    main()
