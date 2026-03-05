from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _pick(rows: List[Dict[str, str]], run_name: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if (r.get("run_name") or "") == run_name:
            return r
    return None


def _f(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main() -> None:
    rows = _load_rows(Path("experiments/run_metrics.csv"))

    pairs: List[Tuple[str, str]] = [
        ("baseline_similarity_k3", "routed_baseline_similarity_k3"),
        ("t1_similarity_k12", "routed_t1_similarity_k12"),
        ("t2_mmr_k12_fetch40_l07_fixed", "routed_t2_mmr_k12_fetch40_l07_fixed"),
        ("qe_rrf_similarity_k12", "routed_qe_rrf_similarity_k12"),
        ("qe_rrf_mmr_k12", "routed_qe_rrf_mmr_k12"),
    ]

    cols = [
        "avg_context_docs",
        "avg_distinct_episodes_in_context",
        "avg_episode_entropy_norm_in_context",
        "avg_aggregation_readiness_score_agg_questions",
        "grounding_failure_rate",
        "quote_in_context_rate",
        "idk_rate",
        "retrieval_failure_rate",
        "avg_cited_episode_ids_not_in_context",
    ]

    for base, routed in pairs:
        b = _pick(rows, base)
        r = _pick(rows, routed)
        if not b or not r:
            print(f"missing pair: {base} vs {routed}")
            continue

        print(f"\n=== {base} vs {routed} ===")
        for c in cols:
            bv = _f(b.get(c))
            rv = _f(r.get(c))
            if bv is None and rv is None:
                continue
            if bv is None or rv is None:
                print(f"{c}: base={bv} routed={rv}")
                continue
            delta = rv - bv
            print(f"{c}: base={bv:.3f} routed={rv:.3f} delta={delta:+.3f}")


if __name__ == "__main__":
    main()
