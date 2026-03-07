from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Find cases where judge context likely lacked supporting quotes.")
    p.add_argument("--runs-dir", default="experiments/runs")
    p.add_argument("--scored-dir", default="experiments/scored_runs_two_pass_rescored_2026-03-07")
    p.add_argument("--max", type=int, default=10)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    scored_dir = Path(args.scored_dir)

    run_by_id: Dict[str, Dict[str, Any]] = {}
    for rp in sorted(runs_dir.glob("*.json")):
        try:
            obj = read_json(rp)
        except Exception:
            continue
        rid = str((obj.get("run") or {}).get("run_id") or "").strip()
        if rid:
            run_by_id[rid] = obj

    hits: List[Tuple[str, str, str, List[str], List[str]]] = []

    for sp in sorted(scored_dir.glob("*.scored.json")):
        try:
            sobj = read_json(sp)
        except Exception:
            continue

        rid = str((sobj.get("run") or {}).get("run_id") or "").strip()
        if not rid:
            continue
        run = run_by_id.get(rid)
        if not run:
            continue

        cases = run.get("cases") or []
        cases_by_id = {
            c.get("case_id"): c for c in cases if isinstance(c, dict) and str(c.get("case_id") or "").strip()
        }

        for row in sobj.get("scored_cases") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("case_id") or "").strip()
            if not cid:
                continue
            c = cases_by_id.get(cid)
            if not c:
                continue

            # Run-eval diagnostics computed against full context_text at eval time.
            aq = ((c.get("diagnostics") or {}).get("answer_quotes") or {})
            any_in_ctx = aq.get("any_quote_in_context")
            if any_in_ctx is not True:
                continue

            judge = row.get("judge") or {}
            unsupported = judge.get("unsupported_claims") or []
            unsupported_str = " ".join(str(x) for x in unsupported).lower()

            # Heuristic: judge claims quote unsupported / not in context.
            if "not in the context" in unsupported_str or "not present in the retrieved context" in unsupported_str:
                examples = aq.get("examples") or []
                hits.append(
                    (
                        rid,
                        cid,
                        sp.name,
                        [str(x) for x in unsupported[:2]],
                        [str(x) for x in examples[:2]],
                    )
                )

    print(f"potential_mismatch_hits: {len(hits)}")
    for rid, cid, fname, unsupported2, examples2 in hits[: int(args.max)]:
        print("\n---")
        print("run_id:", rid)
        print("case_id:", cid)
        print("scored_file:", fname)
        print("unsupported_claims:", unsupported2)
        print("quote_examples:", examples2)


if __name__ == "__main__":
    main()
