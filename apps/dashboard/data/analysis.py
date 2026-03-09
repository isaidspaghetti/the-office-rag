from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from apps.dashboard.data.loaders import index_scored_cases_by_id, load_scored_obj
from apps.dashboard.data.stage_classifier import (
    avg_total_tokens,
    classify_failure_mode,
    classify_stage,
    chunking_signature,
    derived_persist_dir,
    query_expansion_enabled,
    retrieval_policy,
    script_persist_dir,
    search_type,
)
from apps.dashboard.data.transforms import question_type, safe_float, safe_get, safe_int


def failure_mode_counts_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_dir: str,
) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    total = 0

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        cases = ro.get("cases")
        if not isinstance(cases, list) or not cases:
            continue

        scored_obj = load_scored_obj(scored_dir=scored_dir, run_id=rid)
        judge_by_case_id = index_scored_cases_by_id(scored_obj)

        for c in cases:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("case_id") or "").strip()
            judge = judge_by_case_id.get(cid)
            mode = classify_failure_mode(case=c, judge=judge)
            counts[mode] += 1
            total += 1

    rows: List[Dict[str, Any]] = []
    for mode, n in counts.most_common():
        rows.append({"failure_mode": mode, "count": int(n), "share": (float(n / total) if total else None)})
    return rows


def failure_modes_by_question_type_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_dir: str,
) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str]] = Counter()
    totals_by_qtype: Counter[str] = Counter()

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        cases = ro.get("cases")
        if not isinstance(cases, list) or not cases:
            continue

        scored_obj = load_scored_obj(scored_dir=scored_dir, run_id=rid)
        judge_by_case_id = index_scored_cases_by_id(scored_obj)

        for c in cases:
            if not isinstance(c, dict):
                continue
            q = str(c.get("question") or "")
            qtype = question_type(q)
            cid = str(c.get("case_id") or "").strip()
            judge = judge_by_case_id.get(cid)
            mode = classify_failure_mode(case=c, judge=judge)
            counts[(qtype, mode)] += 1
            totals_by_qtype[qtype] += 1

    mode_order = {
        "ok": 0,
        "retrieval_miss": 1,
        "grounding": 2,
        "aggregation": 3,
        "reasoning": 4,
        "unknown": 999,
    }
    qtype_order = {
        "Factual": 0,
        "Episode lookup": 1,
        "Aggregation": 2,
        "Explanation": 3,
        "Quote / line": 4,
        "(missing)": 999,
    }

    rows: List[Dict[str, Any]] = []
    for (qtype, mode), n in sorted(
        counts.items(),
        key=lambda kv: (
            qtype_order.get(kv[0][0], 50),
            mode_order.get(kv[0][1], 500),
            kv[0][0],
            kv[0][1],
        ),
    ):
        total = int(totals_by_qtype.get(qtype, 0))
        rows.append(
            {
                "question_type": qtype,
                "failure_mode": mode,
                "count": int(n),
                "share_within_type": (float(n / total) if total else None),
            }
        )
    return rows


def cost_quality_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue

        tokens = avg_total_tokens(ro)
        if tokens is None:
            continue

        run = ro.get("run") if isinstance(ro.get("run"), dict) else {}
        run_name = str(run.get("run_name") or "")

        rows.append(
            {
                "run_id": rid,
                "run_name": run_name,
                "stage": classify_stage(ro, baseline_llm_model=baseline_llm_model, baseline_k=baseline_k),
                "avg_overall": int(avg_overall),
                "avg_total_tokens": float(tokens),
                "retrieval_policy": retrieval_policy(ro),
                "search_type": search_type(ro),
                "k": safe_int(safe_get(ro, "config.retrieval.k", None)),
            }
        )
    return rows


def stage_timeline_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> List[Dict[str, Any]]:
    stage_scores: Dict[str, List[int]] = {}
    stage_counts: Counter[str] = Counter()

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue

        stage = classify_stage(ro, baseline_llm_model=baseline_llm_model, baseline_k=baseline_k)
        stage_scores.setdefault(stage, []).append(int(avg_overall))
        stage_counts[stage] += 1

    stage_order: Dict[str, int] = {
        "Baseline": 10,
        "Increased Recall": 20,
        "MMR / Diversity": 30,
        "Scene Chunking": 40,
        "Query Expansion + Fusion": 50,
        "Metadata / Routing": 60,
        "Derived Summaries / Cards": 70,
        "Model Upgrades": 80,
        "Other / Uncategorized": 999,
    }

    rows: List[Dict[str, Any]] = []
    for stage, vals in sorted(stage_scores.items(), key=lambda kv: (stage_order.get(kv[0], 500), kv[0])):
        if not vals:
            continue
        avg_score = float(sum(vals) / len(vals))
        rows.append({"stage": stage, "avg_score": round(avg_score, 1), "runs": int(stage_counts.get(stage, 0))})

    prev: Optional[float] = None
    for r in rows:
        cur = safe_float(r.get("avg_score"))
        if prev is None or cur is None:
            r["delta_vs_previous"] = None
        else:
            r["delta_vs_previous"] = round(float(cur - prev), 1)
        prev = float(cur) if cur is not None else prev

    return rows


def technique_impact_rows_for_runs(
    *,
    run_ids: List[str],
    run_obj_by_id: Dict[str, Dict[str, Any]],
    scored_summary_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tech_defs: List[Tuple[str, str]] = [
        ("metadata_index", "Metadata index"),
        ("scene_chunking", "Scene chunking"),
        ("derived_routing", "Derived routing/cards"),
        ("qe_fusion", "Query expansion + fusion"),
        ("mmr", "MMR retrieval"),
        ("high_k", "High k (>=10)"),
    ]

    enabled: Dict[str, List[int]] = {k: [] for k, _ in tech_defs}
    disabled: Dict[str, List[int]] = {k: [] for k, _ in tech_defs}

    for rid in run_ids:
        ro = run_obj_by_id.get(rid)
        if not isinstance(ro, dict):
            continue
        scored_summary = scored_summary_by_id.get(rid) or {}
        avg_overall = scored_summary.get("avg_overall")
        if not isinstance(avg_overall, int):
            continue

        name = str(safe_get(ro, "run.run_name", "") or "").lower()
        policy = retrieval_policy(ro)
        search = search_type(ro)
        qe = query_expansion_enabled(ro)
        persist = script_persist_dir(ro).lower()
        derived_persist = derived_persist_dir(ro).lower()
        chunk_sig = chunking_signature(ro)
        k = safe_int(safe_get(ro, "config.retrieval.k", None))

        has_metadata = ("chroma_db_meta" in persist) or ("metadata" in name)
        has_scene = ("scene" in chunk_sig) or ("scene" in name)
        has_derived = (
            policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}
            or "derived" in derived_persist
            or "topic" in name
        )
        has_qe_fusion = qe or ("rrf" in name) or ("fusion" in name)
        has_mmr = (search == "mmr") or ("mmr" in name)
        has_high_k = bool(k is not None and int(k) >= 10)

        flags: Dict[str, bool] = {
            "metadata_index": bool(has_metadata),
            "scene_chunking": bool(has_scene),
            "derived_routing": bool(has_derived),
            "qe_fusion": bool(has_qe_fusion),
            "mmr": bool(has_mmr),
            "high_k": bool(has_high_k),
        }

        for key, _label in tech_defs:
            if flags.get(key) is True:
                enabled[key].append(int(avg_overall))
            else:
                disabled[key].append(int(avg_overall))

    out: List[Dict[str, Any]] = []
    for key, label in tech_defs:
        en = enabled.get(key) or []
        dis = disabled.get(key) or []
        if not en or not dis:
            continue
        en_avg = float(sum(en) / len(en))
        dis_avg = float(sum(dis) / len(dis))
        out.append(
            {
                "technique": label,
                "delta_avg_overall": round(en_avg - dis_avg, 1),
                "enabled_n": int(len(en)),
                "disabled_n": int(len(dis)),
                "enabled_avg": round(en_avg, 1),
                "disabled_avg": round(dis_avg, 1),
            }
        )

    out.sort(key=lambda r: float(r.get("delta_avg_overall") or 0.0), reverse=True)
    return out
