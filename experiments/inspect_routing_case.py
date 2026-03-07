from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to a run JSON file")
    p.add_argument("--case", required=True, help="Case id, e.g. q1")
    args = p.parse_args()

    run_path = Path(args.run)
    obj = _load(run_path)

    cases = obj.get("cases") or []
    case = None
    for c in cases:
        if c.get("case_id") == args.case:
            case = c
            break
    if not case:
        raise SystemExit(f"case_id not found: {args.case}")

    expected = (case.get("expected") or {}).get("episode_ids")
    diag = case.get("diagnostics") or {}
    routing: Optional[Dict[str, Any]] = (case.get("retrieval") or {}).get("routing")

    print(f"=== {run_path.name} {case['case_id']} ===")
    print("Q:", case.get("question"))
    print("expected:", expected)

    if routing:
        print("policy/effective:", routing.get("policy"), routing.get("effective_policy"))
        print("shortlist:", routing.get("episode_shortlist"))
        print("script_filter_mode:", routing.get("script_filter_mode"))
        derived = routing.get("derived") or {}
        top_rows = (derived.get("results") or [])[:8]
        top = [r.get("episode_id") for r in top_rows]
        top_multi = [r.get("episode_ids") for r in top_rows if r.get("episode_ids")]
        print("derived_top:", top)
        typed = [
            {
                "derived_type": r.get("derived_type"),
                "topic_id": r.get("topic_id"),
                "episode_id": r.get("episode_id"),
            }
            for r in top_rows
        ]
        print("derived_top_typed:", typed)
        if top_multi:
            print("derived_top_episode_ids:", top_multi)

    print("context_episode_ids:", diag.get("context_episode_ids"))

    results = (case.get("retrieval") or {}).get("results") or []
    print("results:", len(results))
    for r in results[:5]:
        print(" -", r.get("source"))


if __name__ == "__main__":
    main()
