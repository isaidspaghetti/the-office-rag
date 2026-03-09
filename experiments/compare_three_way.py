from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pick(rows: List[Dict[str, str]], run_name: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if (r.get("run_name") or "") == run_name:
            return r
    return None


def f(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.3f}"


def main() -> None:
    rows = load_rows(Path("experiments/run_metrics.csv"))

    triples = [
        ("baseline_similarity_k3", "routed_baseline_similarity_k3", "auto_baseline_similarity_k3"),
        ("t1_similarity_k12", "routed_t1_similarity_k12", "auto_t1_similarity_k12"),
        (
            "t2_mmr_k12_fetch40_l07_fixed",
            "routed_t2_mmr_k12_fetch40_l07_fixed",
            "auto_t2_mmr_k12_fetch40_l07_fixed",
        ),
        ("qe_rrf_similarity_k12", "routed_qe_rrf_similarity_k12", "auto_qe_rrf_similarity_k12"),
        ("qe_rrf_mmr_k12", "routed_qe_rrf_mmr_k12", "auto_qe_rrf_mmr_k12"),
    ]

    cols = [
        "idk_rate",
        "retrieval_failure_rate",
        "quote_in_context_rate",
        "avg_aggregation_readiness_score_agg_questions",
        "avg_distinct_episodes_in_context",
    ]

    for base, routed, auto in triples:
        b = pick(rows, base)
        r = pick(rows, routed)
        a = pick(rows, auto)
        if not b or not r or not a:
            print("missing:", base, routed, auto)
            continue

        print(f"\n=== {base} vs {routed} vs {auto} ===")
        for c in cols:
            bv = f(b.get(c))
            rv = f(r.get(c))
            av = f(a.get(c))
            print(f"{c}: base={fmt(bv)} routed={fmt(rv)} auto={fmt(av)}")


if __name__ == "__main__":
    main()
