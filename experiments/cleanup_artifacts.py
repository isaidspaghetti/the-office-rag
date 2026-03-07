from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%d_%H%M%SZ")


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return
    yield from sorted(root.rglob("*.json"))


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
        s = str(x).strip()
        return float(s)
    except Exception:
        return None


@dataclass(frozen=True)
class RunAudit:
    path: Path
    ok: bool
    run_id: Optional[str]
    run_name: Optional[str]
    reasons: Tuple[str, ...]
    cases: int
    retrieved_any_count: int
    total_results_count: int
    answered_with_llm: Optional[bool]
    answers_present_count: int
    answer_error_count: int
    retrieval_empty_rate: Optional[float]
    avg_context_docs: Optional[float]


@dataclass(frozen=True)
class ScoredAudit:
    path: Path
    ok: bool
    run_id_guess: Optional[str]
    cases_scored: Optional[int]
    judge_cases: Optional[int]
    reasons: Tuple[str, ...]


def audit_run_file(path: Path) -> RunAudit:
    obj, err = _read_json(path)
    if err is not None:
        return RunAudit(
            path=path,
            ok=False,
            run_id=None,
            run_name=None,
            reasons=("broken_json",),
            cases=0,
            retrieved_any_count=0,
            total_results_count=0,
            answered_with_llm=None,
            answers_present_count=0,
            answer_error_count=0,
            retrieval_empty_rate=None,
            avg_context_docs=None,
        )

    if not isinstance(obj, dict):
        return RunAudit(
            path=path,
            ok=False,
            run_id=None,
            run_name=None,
            reasons=("not_object",),
            cases=0,
            retrieved_any_count=0,
            total_results_count=0,
            answered_with_llm=None,
            answers_present_count=0,
            answer_error_count=0,
            retrieval_empty_rate=None,
            avg_context_docs=None,
        )

    cases_obj = obj.get("cases")
    if not isinstance(cases_obj, list):
        # Some non-run jsons can live under runs/; quarantine them.
        rid = str(safe_get(obj, "run.run_id", "") or "") or None
        rname = str(safe_get(obj, "run.run_name", "") or "") or None
        return RunAudit(
            path=path,
            ok=False,
            run_id=rid,
            run_name=rname,
            reasons=("not_run_log",),
            cases=0,
            retrieved_any_count=0,
            total_results_count=0,
            answered_with_llm=None,
            answers_present_count=0,
            answer_error_count=0,
            retrieval_empty_rate=None,
            avg_context_docs=None,
        )

    rid = str(safe_get(obj, "run.run_id", "") or "") or None
    rname = str(safe_get(obj, "run.run_name", "") or "") or None

    if len(cases_obj) == 0:
        return RunAudit(
            path=path,
            ok=False,
            run_id=rid,
            run_name=rname,
            reasons=("empty_cases",),
            cases=0,
            retrieved_any_count=0,
            total_results_count=0,
            answered_with_llm=None,
            answers_present_count=0,
            answer_error_count=0,
            retrieval_empty_rate=None,
            avg_context_docs=None,
        )

    reasons: List[str] = []

    retrieved_any_count = 0
    total_results_count = 0
    answers_present_count = 0
    answer_error_count = 0

    answered_with_llm = safe_get(obj, "summary.answered_with_llm", None)
    if answered_with_llm is None:
        answered_with_llm = safe_get(obj, "config.llm.enabled", None)
    answered_with_llm_bool: Optional[bool] = None
    if isinstance(answered_with_llm, bool):
        answered_with_llm_bool = bool(answered_with_llm)

    retrieval_empty_rate = safe_float(safe_get(obj, "summary.retrieval_empty_rate", None))
    avg_context_docs = safe_float(safe_get(obj, "summary.avg_context_docs", None))

    for c in cases_obj:
        if not isinstance(c, dict):
            continue
        results = safe_get(c, "retrieval.results", [])
        if isinstance(results, list):
            total_results_count += len(results)
            if len(results) > 0:
                retrieved_any_count += 1

        ans = c.get("answer") if isinstance(c.get("answer"), dict) else {}
        ans_text = ans.get("text")
        if isinstance(ans_text, str) and ans_text.strip():
            answers_present_count += 1
        ans_err = ans.get("error")
        if ans_err is not None and str(ans_err).strip():
            answer_error_count += 1

    if total_results_count == 0:
        reasons.append("no_retrieval_results")

    # If the run claims it answered with an LLM, but every answer is empty,
    # the run is likely broken / non-actionable for scoring.
    if answered_with_llm_bool is True and answers_present_count == 0:
        reasons.append("answers_empty_all_cases")

    # Heuristic: if every case had a retrieval error recorded.
    errors_count = safe_int(safe_get(obj, "summary.errors_count", None))
    if errors_count is not None and errors_count > 0:
        reasons.append("errors_present")

    # These are strong signals that the artifact is not useful.
    hard_bad = {"broken_json", "not_run_log", "empty_cases", "no_retrieval_results", "answers_empty_all_cases"}
    ok = not any(r in hard_bad for r in reasons)

    # Keep runs that only have errors_present (they can still be informative).
    if reasons == ["errors_present"]:
        ok = True

    return RunAudit(
        path=path,
        ok=ok,
        run_id=rid,
        run_name=rname,
        reasons=tuple(reasons) if reasons else ("ok",),
        cases=len(cases_obj),
        retrieved_any_count=retrieved_any_count,
        total_results_count=total_results_count,
        answered_with_llm=answered_with_llm_bool,
        answers_present_count=answers_present_count,
        answer_error_count=answer_error_count,
        retrieval_empty_rate=retrieval_empty_rate,
        avg_context_docs=avg_context_docs,
    )


def audit_scored_file(path: Path) -> ScoredAudit:
    obj, err = _read_json(path)
    run_id_guess = path.name.replace(".scored.json", "")

    if err is not None:
        return ScoredAudit(
            path=path,
            ok=False,
            run_id_guess=run_id_guess,
            cases_scored=None,
            judge_cases=None,
            reasons=("broken_json",),
        )

    if not isinstance(obj, dict):
        return ScoredAudit(path=path, ok=False, run_id_guess=run_id_guess, cases_scored=None, judge_cases=None, reasons=("not_object",))

    cases_scored = safe_int(safe_get(obj, "score_summary.cases_scored", None))
    scored_cases = obj.get("scored_cases")

    judge_cases = 0
    if isinstance(scored_cases, list):
        for r in scored_cases:
            if isinstance(r, dict) and isinstance(r.get("judge"), dict):
                judge_cases += 1

    reasons: List[str] = []

    if not isinstance(scored_cases, list):
        reasons.append("missing_scored_cases")
    elif len(scored_cases) == 0:
        reasons.append("empty_scored_cases")

    if cases_scored is None:
        reasons.append("missing_cases_scored")
    elif cases_scored == 0:
        reasons.append("zero_cases_scored")

    ok = not reasons
    if not reasons:
        reasons = ["ok"]

    return ScoredAudit(
        path=path,
        ok=ok,
        run_id_guess=run_id_guess,
        cases_scored=cases_scored,
        judge_cases=judge_cases,
        reasons=tuple(reasons),
    )


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__dup{i}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique name for {dest}")


def move_paths(paths: List[Path], *, base_dir: Path, trash_dir: Path, dry_run: bool) -> List[Tuple[Path, Path]]:
    moved: List[Tuple[Path, Path]] = []
    for p in paths:
        try:
            rel = p.relative_to(base_dir)
        except Exception:
            rel = Path(p.name)

        dest = trash_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = _unique_dest(dest)

        moved.append((p, dest))
        if dry_run:
            continue

        shutil.move(str(p), str(dest))

    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit and quarantine broken run/scored artifacts.")
    ap.add_argument("--runs-dir", default="experiments/runs")
    ap.add_argument("--scored-dir", default="experiments/scored_runs_two_pass")
    ap.add_argument("--trash-root", default="experiments/_trash")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this, does a dry-run and prints a report.",
    )
    ap.add_argument(
        "--also-quarantine-nonrun-json",
        action="store_true",
        default=True,
        help="Also quarantine non-run JSON files found under runs-dir (default: true).",
    )
    ap.add_argument(
        "--keep-errorful-runs",
        action="store_true",
        default=True,
        help="Keep runs that only have errors_present but still retrieved docs (default: true).",
    )

    ap.add_argument(
        "--quarantine-scored-for-no-llm-runs",
        action="store_true",
        default=True,
        help="If a run was executed with --no-llm, quarantine any corresponding scored file (default: true).",
    )

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    scored_dir = Path(args.scored_dir).expanduser().resolve()
    trash_root = Path(args.trash_root).expanduser().resolve()

    run_audits: List[RunAudit] = []
    for p in _iter_json_files(runs_dir):
        run_audits.append(audit_run_file(p))

    scored_audits: List[ScoredAudit] = []
    for p in _iter_json_files(scored_dir):
        scored_audits.append(audit_scored_file(p))

    # Index audits by run_id / scored run_id guess.
    run_by_id: Dict[str, RunAudit] = {}
    for a in run_audits:
        if a.run_id:
            run_by_id[a.run_id] = a

    scored_by_id: Dict[str, ScoredAudit] = {a.run_id_guess or "": a for a in scored_audits if a.run_id_guess}

    # Decide quarantine set.
    quarantine_runs: List[RunAudit] = []
    for a in run_audits:
        if a.ok:
            continue
        if (not args.also_quarantine_nonrun_json) and ("not_run_log" in a.reasons):
            continue
        quarantine_runs.append(a)

    quarantine_scored: List[ScoredAudit] = [a for a in scored_audits if not a.ok]

    # If a run is quarantined, quarantine its scored file too (even if scored looks ok).
    quarantine_scored_paths: set[Path] = {a.path for a in quarantine_scored}
    for ra in quarantine_runs:
        if not ra.run_id:
            continue
        sc = scored_by_id.get(ra.run_id)
        if sc is not None:
            quarantine_scored_paths.add(sc.path)

    # If the run didn't generate an LLM answer, any scored output is generally meaningless.
    if bool(args.quarantine_scored_for_no_llm_runs):
        for ra in run_audits:
            if not ra.run_id:
                continue
            if ra.answered_with_llm is False:
                sc = scored_by_id.get(ra.run_id)
                if sc is not None:
                    quarantine_scored_paths.add(sc.path)

    quarantine_run_paths = [a.path for a in quarantine_runs]

    # Report summary.
    run_reason_counts: Counter[str] = Counter()
    for a in quarantine_runs:
        for r in a.reasons:
            run_reason_counts[r] += 1

    scored_reason_counts: Counter[str] = Counter()
    for a in quarantine_scored:
        for r in a.reasons:
            scored_reason_counts[r] += 1

    print("=== Audit summary ===")
    print(f"Runs scanned:   {len(run_audits)}")
    print(f"Scored scanned: {len(scored_audits)}")
    print(f"Runs to quarantine:   {len(quarantine_run_paths)}")
    print(f"Scored to quarantine: {len(quarantine_scored_paths)}")

    print("\n--- Runs quarantine reasons ---")
    for k, v in sorted(run_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")

    print("\n--- Scored quarantine reasons ---")
    for k, v in sorted(scored_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")

    # Choose trash dir.
    stamp = utc_now_compact()
    trash_dir = trash_root / stamp
    trash_runs = trash_dir / "runs"
    trash_scored = trash_dir / "scored_runs"

    print("\n=== Quarantine destination ===")
    print(trash_dir)

    # Show a small sample.
    def _sample(paths: List[Path], n: int = 8) -> List[str]:
        return [str(p) for p in paths[:n]]

    print("\nSample runs to quarantine:")
    for s in _sample(quarantine_run_paths):
        print(" -", s)

    print("\nSample scored to quarantine:")
    for s in _sample(sorted(list(quarantine_scored_paths))):
        print(" -", s)

    if not args.apply:
        print("\n(dry-run; pass --apply to move files)")
        return

    moved_runs = move_paths(quarantine_run_paths, base_dir=runs_dir, trash_dir=trash_runs, dry_run=False)
    moved_scored = move_paths(sorted(list(quarantine_scored_paths)), base_dir=scored_dir, trash_dir=trash_scored, dry_run=False)

    print("\n=== Moved ===")
    print(f"Runs moved:   {len(moved_runs)}")
    print(f"Scored moved: {len(moved_scored)}")


if __name__ == "__main__":
    main()
