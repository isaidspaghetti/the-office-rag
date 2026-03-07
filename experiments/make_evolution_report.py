from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


_STEP_RE = re.compile(r"step(?P<n>\d{2})", re.IGNORECASE)


def _extract_step(run_name: str) -> Optional[int]:
    m = _STEP_RE.search(run_name or "")
    if not m:
        return None
    try:
        return int(m.group("n"), 10)
    except Exception:
        return None


def _fmt(x: Any, *, digits: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


@dataclass(frozen=True)
class RunRow:
    step: int
    run_file: str
    run_id: str
    run_name: str
    created_at_utc: str
    retrieval_policy: str
    search_type: str
    k: Optional[int]

    avg_total_tokens: Optional[int]

    avg_retrieval_latency_ms: Optional[int]
    avg_context_docs: Optional[float]
    avg_context_chars: Optional[float]

    avg_distinct_episodes_in_context: Optional[float]
    avg_top_episode_share_in_context: Optional[float]
    avg_episode_entropy_norm_in_context: Optional[float]
    avg_aggregation_readiness_score: Optional[float]

    retrieval_empty_rate: Optional[float]
    retrieval_failure_rate: Optional[float]
    grounding_failure_rate: Optional[float]
    quote_in_context_rate: Optional[float]

    errors_count: Optional[int]

    scored_file: Optional[str]

    # Optional judge-scored metrics (from experiments/scored_runs_two_pass/<run_id>.scored.json)
    judge_cases_scored: Optional[int]
    judge_avg_overall: Optional[int]
    det_episode_ok_rate: Optional[float]
    det_must_include_ok_rate: Optional[float]
    det_forbidden_hit_rate: Optional[float]


def _load_scored_obj(*, scored_dir: Optional[Path], run_id: str) -> Dict[str, Any]:
    """Load a scored run JSON by run_id.

    Expected file layout (from experiments/score_runs.py):
      <scored_dir>/<run_id>.scored.json
    """
    if scored_dir is None:
        return {}
    rid = str(run_id or "").strip()
    if not rid:
        return {}

    p = (scored_dir / f"{rid}.scored.json").expanduser().resolve()
    if not p.exists() or not p.is_file():
        return {}
    try:
        obj = _read_json(p)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        return float(s)
    except Exception:
        return None


def load_rows(*, runs_dir: Path, run_name_prefix: str, scored_dir: Optional[Path] = None) -> List[RunRow]:
    rows: List[RunRow] = []

    for rf in sorted(runs_dir.rglob("*.json")):
        try:
            obj = _read_json(rf)
        except Exception:
            continue
        if not (isinstance(obj, dict) and isinstance(obj.get("run"), dict) and isinstance(obj.get("summary"), dict)):
            continue

        run_name = str(_safe_get(obj, "run.run_name", "") or "")
        if not run_name.startswith(run_name_prefix):
            continue

        step = _extract_step(run_name)
        if step is None:
            continue

        run_id = str(_safe_get(obj, "run.run_id", "") or "")
        scored_path: Optional[Path] = None
        if scored_dir is not None and run_id.strip():
            candidate = (scored_dir / f"{run_id}.scored.json").expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                scored_path = candidate

        scored = _load_scored_obj(scored_dir=scored_dir, run_id=run_id)

        rows.append(
            RunRow(
                step=int(step),
                run_file=str(rf.as_posix()),
                run_id=run_id,
                run_name=run_name,
                created_at_utc=str(_safe_get(obj, "run.created_at_utc", "") or ""),
                retrieval_policy=str(_safe_get(obj, "config.retrieval.policy", "") or ""),
                search_type=str(_safe_get(obj, "config.retrieval.search_type", "") or ""),
                k=_to_int(_safe_get(obj, "config.retrieval.k", None)),

                avg_total_tokens=_to_int(_safe_get(obj, "summary.avg_total_tokens", None)),

                avg_retrieval_latency_ms=_to_int(_safe_get(obj, "summary.avg_retrieval_latency_ms", None)),
                avg_context_docs=_to_float(_safe_get(obj, "summary.avg_context_docs", None)),
                avg_context_chars=_to_float(_safe_get(obj, "summary.avg_context_chars", None)),

                avg_distinct_episodes_in_context=_to_float(_safe_get(obj, "summary.avg_distinct_episodes_in_context", None)),
                avg_top_episode_share_in_context=_to_float(_safe_get(obj, "summary.avg_top_episode_share_in_context", None)),
                avg_episode_entropy_norm_in_context=_to_float(_safe_get(obj, "summary.avg_episode_entropy_norm_in_context", None)),
                avg_aggregation_readiness_score=_to_float(_safe_get(obj, "summary.avg_aggregation_readiness_score", None)),

                retrieval_empty_rate=_to_float(_safe_get(obj, "summary.retrieval_empty_rate", None)),
                retrieval_failure_rate=_to_float(_safe_get(obj, "summary.retrieval_failure_rate", None)),
                grounding_failure_rate=_to_float(_safe_get(obj, "summary.grounding_failure_rate", None)),
                quote_in_context_rate=_to_float(_safe_get(obj, "summary.quote_in_context_rate", None)),

                errors_count=_to_int(_safe_get(obj, "summary.errors_count", None)),

                scored_file=(str(scored_path.as_posix()) if scored_path is not None else None),

                judge_cases_scored=_to_int(_safe_get(scored, "score_summary.cases_scored", None)),
                judge_avg_overall=_to_int(_safe_get(scored, "score_summary.avg_overall", None)),
                det_episode_ok_rate=_to_float(_safe_get(scored, "deterministic_summary.episode_ok_rate", None)),
                det_must_include_ok_rate=_to_float(_safe_get(scored, "deterministic_summary.must_include_ok_rate", None)),
                det_forbidden_hit_rate=_to_float(_safe_get(scored, "deterministic_summary.forbidden_hit_rate", None)),
            )
        )

    rows.sort(key=lambda r: (r.step, r.created_at_utc, r.run_id))
    return rows


def render_markdown(*, rows: List[RunRow], run_name_prefix: str, notes: str) -> str:
    lines: List[str] = []
    lines.append(f"# Evolution report: {run_name_prefix}*")
    lines.append("")
    if notes:
        lines.append(notes.strip())
        lines.append("")

    has_scores = any((r.judge_avg_overall is not None) or (r.judge_cases_scored is not None) for r in rows)

    if has_scores:
        lines.append("## Summary table (scored)")
        lines.append("")
        lines.append(
            "| step | run_name | policy | search | k | avg_overall | judge_cases | det_episode_ok | det_must_include_ok | det_forbidden_hit | avg_total_tokens | grounding_fail | quote_in_ctx |"
        )
        lines.append(
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )

        for r in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{r.step:02d}",
                        r.run_name,
                        r.retrieval_policy,
                        r.search_type,
                        _fmt(r.k, digits=0),
                        _fmt(r.judge_avg_overall, digits=0),
                        _fmt(r.judge_cases_scored, digits=0),
                        _fmt(r.det_episode_ok_rate, digits=3),
                        _fmt(r.det_must_include_ok_rate, digits=3),
                        _fmt(r.det_forbidden_hit_rate, digits=3),
                        _fmt(r.avg_total_tokens, digits=0),
                        _fmt(r.grounding_failure_rate, digits=3),
                        _fmt(r.quote_in_context_rate, digits=3),
                    ]
                )
                + " |"
            )

        lines.append("")

    lines.append("## Summary table (retrieval)")
    lines.append("")
    lines.append(
        "| step | run_name | policy | search | k | avg_ctx_docs | avg_distinct_eps | top_ep_share | ep_entropy | agg_readiness | empty_rate | retrieval_fail_rate | avg_retrieval_ms | errors |"
    )
    lines.append(
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r.step:02d}",
                    r.run_name,
                    r.retrieval_policy,
                    r.search_type,
                    _fmt(r.k, digits=0),
                    _fmt(r.avg_context_docs, digits=2),
                    _fmt(r.avg_distinct_episodes_in_context, digits=2),
                    _fmt(r.avg_top_episode_share_in_context, digits=3),
                    _fmt(r.avg_episode_entropy_norm_in_context, digits=3),
                    _fmt(r.avg_aggregation_readiness_score, digits=1),
                    _fmt(r.retrieval_empty_rate, digits=3),
                    _fmt(r.retrieval_failure_rate, digits=3),
                    _fmt(r.avg_retrieval_latency_ms, digits=0),
                    _fmt(r.errors_count, digits=0),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Run files")
    lines.append("")
    for r in rows:
        lines.append(f"- step {r.step:02d}: {r.run_file}")

    if has_scores:
        scored_files = [r.scored_file for r in rows if r.scored_file]
        if scored_files:
            lines.append("")
            lines.append("## Scored files")
            lines.append("")
            for p in scored_files:
                lines.append(f"- {p}")

    lines.append("")
    if has_scores:
        lines.append("## Notes")
        lines.append("")
        lines.append("- This report already includes judge scoring (avg_overall) and deterministic checks.")
        lines.append("- If you re-run new steps, re-run scoring so the scored files stay in sync.")
    else:
        lines.append("## Next (when OPENAI_API_KEY is set)")
        lines.append("")
        lines.append("- Re-run the same steps with LLM answering enabled (`--llm-model gpt-4.1-mini`) and then judge-score them with `experiments/score_runs.py`.")
        lines.append("- That will populate `avg_overall` and other judge metrics for a true end-to-end evolution story.")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a markdown evolution report from run logs")
    p.add_argument("--runs-dir", default="experiments/runs")
    p.add_argument(
        "--scored-dir",
        default="experiments/scored_runs_two_pass",
        help=(
            "Directory containing scored run JSONs (default: experiments/scored_runs_two_pass). "
            "Set to empty string to omit judge metrics."
        ),
    )
    p.add_argument("--run-name-prefix", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    scored_dir = Path(args.scored_dir).expanduser().resolve() if str(args.scored_dir).strip() else None
    rows = load_rows(runs_dir=runs_dir, run_name_prefix=str(args.run_name_prefix), scored_dir=scored_dir)
    if not rows:
        raise SystemExit(f"No runs found with prefix: {args.run_name_prefix}")

    md = render_markdown(rows=rows, run_name_prefix=str(args.run_name_prefix), notes=str(args.notes))

    out = str(args.out).strip()
    if out:
        out_path = Path(out).expanduser().resolve()
    else:
        safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(args.run_name_prefix)).strip("_")
        out_path = Path("experiments/evolution_reports") / f"{safe_prefix}.md"
        out_path = out_path.expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
