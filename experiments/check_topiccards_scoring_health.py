from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    scored_dir = Path("experiments/scored_runs")
    rows: List[Tuple[str, Any, Any, int, int, Any]] = []

    for p in sorted(scored_dir.glob("*.scored.json")):
        name = p.name
        if "topiccards_blended_" not in name:
            continue

        try:
            obj = _read_json(p)
        except Exception as e:
            rows.append((name, "BROKEN_JSON", f"{type(e).__name__}: {e}", 0, 0, None))
            continue

        summ = obj.get("score_summary") or {}
        cases_scored = summ.get("cases_scored")
        avg_overall = summ.get("avg_overall")

        scored_cases = obj.get("scored_cases")
        judge_cases = 0
        judge_errors = 0
        if isinstance(scored_cases, list):
            for r in scored_cases:
                if not isinstance(r, dict):
                    continue
                if isinstance(r.get("judge"), dict):
                    judge_cases += 1
                if r.get("judge_error"):
                    judge_errors += 1

        rows.append(
            (
                name,
                cases_scored,
                avg_overall,
                judge_cases,
                judge_errors,
                (len(scored_cases) if isinstance(scored_cases, list) else None),
            )
        )

    ok = [
        r
        for r in rows
        if isinstance(r[1], int) and r[1] > 0 and isinstance(r[3], int) and r[3] > 0 and r[4] == 0
    ]
    bad = [r for r in rows if r not in ok]

    out = Path("experiments/_topiccards_scoring_health.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"topiccards_scored_files={len(rows)}\n")
        f.write(f"ok={len(ok)} bad={len(bad)}\n\n")
        f.write(
            "name\tcases_scored\tavg_overall\tjudge_cases\tjudge_errors\tscored_cases_len\n"
        )
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
