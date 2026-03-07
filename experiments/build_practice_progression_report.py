from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_k_from_name(run_name: str) -> Optional[int]:
    import re

    m = re.search(r"\bk(?P<k>\d{1,3})\b", str(run_name or ""))
    if not m:
        return None
    try:
        return int(m.group("k"))
    except Exception:
        return None


def _infer_search_type(*, run_name: str, raw: Optional[str]) -> str:
    if raw and str(raw).strip():
        return str(raw).strip()
    name = str(run_name or "").lower()
    if "mmr" in name:
        return "mmr"
    return "similarity"


def _infer_qe_enabled(*, run_name: str, cfg_retrieval: Any) -> bool:
    # Newer schemas store retrieval.query_expansion.enabled.
    qe = safe_get(cfg_retrieval, "query_expansion.enabled", None)
    if isinstance(qe, bool):
        return bool(qe)
    # Older schemas: fall back to run name.
    name = str(run_name or "").lower()
    return bool("qe_" in name or "query_expansion" in name or "rrf" in name)


def _infer_policy(
    *,
    run_name: str,
    cfg_retrieval: Any,
    has_derived_vectorstore: bool,
) -> str:
    raw = safe_get(cfg_retrieval, "policy", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    # Older run logs didn't have a policy field; they were effectively script-only.
    name = str(run_name or "").lower()
    if name.startswith("topiccards_") or "topiccards" in name:
        return "blended"  # topiccards runs are blended by construction
    if "blended" in name:
        return "blended"
    if "derived_only" in name:
        return "derived_only"
    if "derived_then_script" in name or "derived_then" in name:
        return "derived_then_script"
    if name.startswith("auto_") or "_auto_" in name or "auto_" in name:
        return "auto"
    if "script_only" in name:
        return "script_only"

    # Best-effort: if the run used a derived vectorstore but didn't log a policy,
    # assume derived_then_script (most common routed policy).
    if has_derived_vectorstore:
        return "derived_then_script"

    return "script_only"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        s = str(x).strip()
        return float(s)
    except Exception:
        return None


def _parse_utc_iso(ts: Any) -> Optional[datetime]:
    s = str(ts or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _iter_scored_files(scored_dir: Path) -> Iterable[Path]:
    yield from sorted(scored_dir.glob("*.scored.json"))


@dataclass(frozen=True)
class RunRow:
    run_id: str
    run_name: str
    created_at_utc: str
    retrieval_policy: str
    search_type: str
    k: Optional[int]
    qe_enabled: bool
    script_persist_directory: str
    derived_persist_directory: str
    has_derived_index: bool
    uses_meta_index: bool
    avg_overall: Optional[int]
    cases_scored: Optional[int]
    avg_total_tokens: Optional[int]
    judge_mode: Optional[str]


def _bool_meta_index(script_persist: str) -> bool:
    # Be strict: we only want to mark runs that explicitly used the metadata index.
    s = (script_persist or "").replace("\\", "/").lower()
    return ("/chroma_db_meta" in s) or s.endswith("chroma_db_meta") or ("db/chroma_db_meta" in s)


def _load_run_summary_tokens(run_log: Optional[Dict[str, Any]]) -> Optional[int]:
    # Prefer run-level summary if present.
    return safe_int(safe_get(run_log, "summary.avg_total_tokens", None))


def load_rows(*, runs_dir: Path, scored_dir: Path) -> List[RunRow]:
    out: List[RunRow] = []
    for p in _iter_scored_files(scored_dir):
        try:
            scored = _read_json(p)
            if not isinstance(scored, dict):
                continue
        except Exception:
            continue

        run_id = str(safe_get(scored, "run.run_id", "") or "").strip() or p.name[: -len(".scored.json")]
        run_name = str(safe_get(scored, "run.run_name", "") or "").strip() or run_id
        created_at_utc = str(safe_get(scored, "run.created_at_utc", "") or "").strip()

        judge_mode = str(safe_get(scored, "scoring_meta.judge_mode", "") or "").strip().lower() or None
        cases_scored = safe_int(safe_get(scored, "score_summary.cases_scored", None))
        avg_overall = safe_int(safe_get(scored, "score_summary.avg_overall", None))

        cfg_retrieval = safe_get(scored, "config.retrieval", {})
        cfg_derived_vs = safe_get(scored, "config.derived_vectorstore", None)
        has_derived_vs = bool(cfg_derived_vs)

        retrieval_policy = _infer_policy(
            run_name=run_name,
            cfg_retrieval=cfg_retrieval,
            has_derived_vectorstore=has_derived_vs,
        )

        search_type = _infer_search_type(
            run_name=run_name,
            raw=(safe_get(cfg_retrieval, "search_type", None) if isinstance(cfg_retrieval, dict) else None),
        )

        k = safe_int(safe_get(cfg_retrieval, "k", None))
        if k is None:
            k = _parse_k_from_name(run_name)

        qe_enabled = _infer_qe_enabled(run_name=run_name, cfg_retrieval=cfg_retrieval)

        script_persist = str(safe_get(scored, "config.data_version.script.persist_directory", "") or "").strip()
        if not script_persist:
            script_persist = str(safe_get(scored, "config.vectorstore.persist_directory", "") or "").strip()

        derived_persist = str(safe_get(scored, "config.data_version.derived.persist_directory", "") or "").strip()
        has_derived = bool(cfg_derived_vs)
        if not derived_persist:
            derived_persist = str(safe_get(scored, "config.derived_vectorstore.persist_directory", "") or "").strip()

        uses_meta_index = _bool_meta_index(script_persist)

        run_log_path = runs_dir / f"{run_id}.json"
        run_log = None
        if run_log_path.exists() and run_log_path.is_file():
            try:
                run_log = _read_json(run_log_path)
            except Exception:
                run_log = None

        avg_total_tokens = _load_run_summary_tokens(run_log)

        out.append(
            RunRow(
                run_id=run_id,
                run_name=run_name,
                created_at_utc=created_at_utc,
                retrieval_policy=retrieval_policy,
                search_type=search_type,
                k=k,
                qe_enabled=bool(qe_enabled),
                script_persist_directory=script_persist,
                derived_persist_directory=derived_persist,
                has_derived_index=bool(has_derived),
                uses_meta_index=bool(uses_meta_index),
                avg_overall=avg_overall,
                cases_scored=cases_scored,
                avg_total_tokens=avg_total_tokens,
                judge_mode=judge_mode,
            )
        )
    return out


PRACTICE_ORDER = [
    "baseline",
    "higher_k",
    "mmr",
    "query_expansion_rrf",
    "metadata_index",
    "derived_routing",
    "blended",
    "topiccards",
]


PRACTICE_LABELS = {
    "baseline": "Baseline (similarity, small k)",
    "higher_k": "Increase k (more recall)",
    "mmr": "MMR (diversify context)",
    "query_expansion_rrf": "Query expansion + RRF (recall)",
    "metadata_index": "Metadata-enabled filtering", 
    "derived_routing": "Derived routing (episode shortlisting)",
    "blended": "Hybrid retrieval (baseline + routed)",
    "topiccards": "Topic cards (derived evidence) + Hybrid",
}


def classify_primary_practice(r: RunRow) -> str:
    name = (r.run_name or "").lower()
    policy = (r.retrieval_policy or "").lower()
    search = (r.search_type or "").lower()
    k = int(r.k) if r.k is not None else None

    if name.startswith("topiccards_"):
        return "topiccards"
    if policy == "blended":
        return "blended"
    if policy in {"derived_then_script", "auto", "derived_only"} or r.has_derived_index:
        return "derived_routing"
    if r.uses_meta_index:
        return "metadata_index"
    if r.qe_enabled:
        return "query_expansion_rrf"
    if search == "mmr":
        return "mmr"
    if k is not None and k >= 10:
        return "higher_k"
    if policy == "script_only" and search == "similarity" and (k is not None and k <= 4) and (not r.qe_enabled):
        return "baseline"
    return "baseline"


def pick_best(rows: List[RunRow]) -> Optional[RunRow]:
    if not rows:
        return None

    def _key(r: RunRow) -> Tuple[int, int, float]:
        score = int(r.avg_overall) if r.avg_overall is not None else -1
        cases = int(r.cases_scored) if r.cases_scored is not None else 0
        # Prefer runs that have token stats present for cost tradeoffs.
        has_tokens = 1 if r.avg_total_tokens is not None else 0
        return (score, cases, has_tokens)

    return max(rows, key=_key)


def format_int(x: Optional[int]) -> str:
    return "—" if x is None else str(int(x))


def format_bool(x: bool) -> str:
    return "yes" if x else "no"


def render_markdown(*, rows: List[RunRow]) -> str:
    # Build practice buckets.
    by_practice: Dict[str, List[RunRow]] = {k: [] for k in PRACTICE_ORDER}
    for r in rows:
        pr = classify_primary_practice(r)
        by_practice.setdefault(pr, []).append(r)

    best_by_practice: List[Tuple[str, RunRow]] = []
    for pr in PRACTICE_ORDER:
        best = pick_best(by_practice.get(pr) or [])
        if best is not None:
            best_by_practice.append((pr, best))

    baseline = None
    for pr, r in best_by_practice:
        if pr == "baseline":
            baseline = r
            break

    best_overall = pick_best(rows)

    lines: List[str] = []
    lines.append("# Practice progression (two-pass scored)\n")
    lines.append("This report is generated from historical runs and focuses on *RAG practices*, not filenames.")
    lines.append("Scoring uses the **two-pass judge**: (1) context-only groundedness, (2) gold-only correctness, then merged.")
    lines.append("")

    if best_overall is not None:
        lines.append("## Snapshot")
        if baseline and baseline.avg_overall is not None and best_overall.avg_overall is not None:
            delta = int(best_overall.avg_overall) - int(baseline.avg_overall)
            lines.append(
                f"- Best run: **{best_overall.run_name}** (avg_overall={best_overall.avg_overall}, Δ vs baseline={delta:+d})"
            )
        else:
            lines.append(f"- Best run: **{best_overall.run_name}** (avg_overall={format_int(best_overall.avg_overall)})")
        lines.append(f"- Runs scanned: {len(rows)}")
        lines.append("")

    lines.append("## Milestones (representative best per practice)")
    lines.append(
        "| practice | representative run_name | policy | search | k | qe | meta_index | derived_index | avg_overall | avg_total_tokens | cases_scored | created_at_utc |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for pr, r in best_by_practice:
        lines.append(
            "| "
            + " | ".join(
                [
                    PRACTICE_LABELS.get(pr, pr),
                    r.run_name,
                    r.retrieval_policy,
                    r.search_type,
                    str(r.k or "—"),
                    format_bool(r.qe_enabled),
                    format_bool(r.uses_meta_index),
                    format_bool(r.has_derived_index),
                    format_int(r.avg_overall),
                    format_int(r.avg_total_tokens),
                    format_int(r.cases_scored),
                    r.created_at_utc or "",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## What changed (plain-English)")
    lines.append("- **Increase k**: trades cost for recall (more chunks in context).")
    lines.append("- **MMR**: diversifies retrieved chunks to reduce redundancy and cover more evidence." )
    lines.append("- **Query expansion + RRF**: generates alternate retrieval queries and fuses results for higher recall on ambiguous queries.")
    lines.append("- **Metadata-enabled filtering**: allows hard constraints (e.g., season/episode) to prevent drift across irrelevant episodes.")
    lines.append("- **Derived routing**: uses higher-level derived cards to shortlist episodes, then retrieves scripts within that shortlist.")
    lines.append("- **Hybrid retrieval**: combines baseline recall with routed expansions, improving robustness across question types.")
    lines.append("")

    lines.append("## Notes / gotchas")
    lines.append("- `avg_total_tokens` is pulled from the original run log when available (missing if the run didn’t log usage).")
    lines.append("- Some runs match multiple practices; this report assigns a single **primary** practice by priority to keep the story simple.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a practice-based progression report from scored runs")
    p.add_argument("--runs-dir", default="experiments/runs", help="Directory with run logs")
    p.add_argument(
        "--scored-dir",
        default="experiments/scored_runs_two_pass",
        help="Directory with *.scored.json files (two-pass by default)",
    )
    p.add_argument(
        "--out",
        default="experiments/evolution_reports/practice_progression_scored_two_pass.md",
        help="Output markdown path",
    )
    p.add_argument(
        "--require-two-pass",
        action="store_true",
        default=True,
        help="If set, include only scored runs with scoring_meta.judge_mode == two_pass",
    )
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    scored_dir = Path(args.scored_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not scored_dir.exists() or not scored_dir.is_dir():
        raise SystemExit(f"scored-dir not found: {scored_dir}")

    rows = load_rows(runs_dir=runs_dir, scored_dir=scored_dir)

    if args.require_two_pass:
        rows = [r for r in rows if (r.judge_mode or "") == "two_pass"]

    # Keep only rows that actually have judge scores.
    rows = [r for r in rows if r.avg_overall is not None]

    # Sort for determinism: newest-first within equal scores is not desired; keep stable by created_at.
    rows.sort(key=lambda r: (_parse_utc_iso(r.created_at_utc) or datetime(1970, 1, 1), r.run_id))

    md = render_markdown(rows=rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
