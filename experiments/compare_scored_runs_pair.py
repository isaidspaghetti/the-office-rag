from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _case_map(scored_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in scored_obj.get("scored_cases", []) or []:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("case_id") or "").strip()
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        if not cid:
            continue
        out[cid] = {
            "overall": judge.get("overall"),
            "verdict": judge.get("verdict"),
            "question": judge.get("question") or "",
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two *.scored.json outputs case-by-case.")
    p.add_argument("--scored-dir", required=True, help="Directory containing <run_id>.scored.json")
    p.add_argument("--best-run-id", required=True, help="run_id for the best run")
    p.add_argument("--worst-run-id", required=True, help="run_id for the worst run")
    p.add_argument("--out", required=True, help="Output JSON report path")
    p.add_argument("--top", type=int, default=15, help="How many top deltas to include")

    args = p.parse_args()

    scored_dir = Path(args.scored_dir).expanduser().resolve()
    best_path = scored_dir / f"{args.best_run_id}.scored.json"
    worst_path = scored_dir / f"{args.worst_run_id}.scored.json"

    best = _read_json(best_path)
    worst = _read_json(worst_path)

    mb = _case_map(best)
    mw = _case_map(worst)

    rows: List[Dict[str, Any]] = []
    for cid in sorted(set(mb) & set(mw)):
        ob = mb[cid].get("overall")
        ow = mw[cid].get("overall")
        if not isinstance(ob, int) or not isinstance(ow, int):
            continue
        rows.append(
            {
                "case_id": cid,
                "delta": int(ob - ow),
                "best_overall": int(ob),
                "worst_overall": int(ow),
                "best_verdict": mb[cid].get("verdict"),
                "worst_verdict": mw[cid].get("verdict"),
                "question": mb[cid].get("question") or "",
            }
        )

    rows.sort(key=lambda r: int(r["delta"]), reverse=True)

    top_deltas = rows[: int(args.top)]
    regressions = sorted([r for r in rows if int(r["delta"]) < 0], key=lambda r: int(r["delta"]))[:10]

    report = {
        "best_run_id": str(args.best_run_id),
        "worst_run_id": str(args.worst_run_id),
        "scored_dir": str(scored_dir),
        "cases_compared": len(rows),
        "top_deltas": top_deltas,
        "top_regressions": regressions,
    }

    out_path = Path(args.out).expanduser().resolve()
    _write_json(out_path, report)

    print(f"Wrote {out_path}")
    print("Top deltas:")
    for r in top_deltas[:8]:
        print(
            f"- {r['case_id']} delta={r['delta']} best={r['best_overall']}({r['best_verdict']}) "
            f"worst={r['worst_overall']}({r['worst_verdict']})"
        )


if __name__ == "__main__":
    main()
