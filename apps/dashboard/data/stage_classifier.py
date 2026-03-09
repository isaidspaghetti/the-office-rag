from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from apps.dashboard.data.transforms import safe_float, safe_get, safe_int, question_type


def avg_total_tokens(run_obj: Dict[str, Any]) -> Optional[float]:
    # Prefer run summary.
    v = safe_float(safe_get(run_obj, "summary.avg_total_tokens", None))
    if v is not None:
        return float(v)

    cases = run_obj.get("cases")
    if not isinstance(cases, list) or not cases:
        return None

    vals: List[float] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        usage = safe_get(c, "answer.usage", {})
        if not isinstance(usage, dict):
            continue
        t = safe_float(usage.get("total_tokens"))
        if t is not None:
            vals.append(float(t))
    return float(sum(vals) / len(vals)) if vals else None


def script_persist_dir(run_obj: Dict[str, Any]) -> str:
    dv = safe_get(run_obj, "config.data_version.script.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(safe_get(run_obj, "config.vectorstore.persist_directory", "") or "")


def derived_persist_dir(run_obj: Dict[str, Any]) -> str:
    dv = safe_get(run_obj, "config.data_version.derived.persist_directory", None)
    if isinstance(dv, str) and dv.strip():
        return dv
    return str(safe_get(run_obj, "config.derived_vectorstore.persist_directory", "") or "")


def retrieval_policy(run_obj: Dict[str, Any]) -> str:
    return str(safe_get(run_obj, "config.retrieval.policy", "") or "").strip().lower()


def search_type(run_obj: Dict[str, Any]) -> str:
    return str(safe_get(run_obj, "config.retrieval.search_type", "") or "").strip().lower()


def query_expansion_enabled(run_obj: Dict[str, Any]) -> bool:
    return bool(safe_get(run_obj, "config.retrieval.query_expansion.enabled", False))


def chunking_signature(run_obj: Dict[str, Any]) -> str:
    splitter = str(safe_get(run_obj, "config.chunking.splitter", "") or "")
    persist = script_persist_dir(run_obj)
    return f"{splitter}|{persist}".lower()


def llm_model(run_obj: Dict[str, Any]) -> Optional[str]:
    m = safe_get(run_obj, "config.llm.model", None)
    if isinstance(m, str) and m.strip():
        return m.strip()
    return None


def baseline_signature_from_runs(run_objs: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    llm_models: List[str] = []
    ks: List[int] = []
    for o in run_objs:
        m = llm_model(o)
        if m:
            llm_models.append(m)
        k = safe_int(safe_get(o, "config.retrieval.k", None))
        if k is not None:
            ks.append(int(k))

    baseline_llm_model: Optional[str] = None
    if llm_models:
        baseline_llm_model = Counter(llm_models).most_common(1)[0][0]

    baseline_k: Optional[int] = min(ks) if ks else None
    return baseline_llm_model, baseline_k


def classify_stage(
    run_obj: Dict[str, Any],
    *,
    baseline_llm_model: Optional[str],
    baseline_k: Optional[int],
) -> str:
    name = str(safe_get(run_obj, "run.run_name", "") or "").lower()
    policy = retrieval_policy(run_obj)
    search = search_type(run_obj)
    qe = query_expansion_enabled(run_obj)
    persist = script_persist_dir(run_obj).lower()
    derived_persist = derived_persist_dir(run_obj).lower()
    chunk_sig = chunking_signature(run_obj)

    k = safe_int(safe_get(run_obj, "config.retrieval.k", None))

    if (
        policy in {"derived_only", "derived_then_script", "auto", "blended", "hybrid"}
        or "derived" in derived_persist
    ):
        return "Derived Summaries / Cards"
    if "chroma_db_meta" in persist or "metadata" in name or "routing" in name:
        return "Metadata / Routing"
    if qe or "qe" in name or "query_expansion" in name or "rrf" in name or "fusion" in name:
        return "Query Expansion + Fusion"
    if "scene" in chunk_sig or "scene" in name:
        return "Scene Chunking"
    if search == "mmr" or "mmr" in name:
        return "MMR / Diversity"
    if baseline_k is not None and k is not None and k > baseline_k:
        return "Increased Recall"
    if k is not None and k >= 10:
        return "Increased Recall"

    m = llm_model(run_obj)
    if baseline_llm_model and m and m != baseline_llm_model:
        return "Model Upgrades"
    if "gpt-" in name and baseline_llm_model and baseline_llm_model.replace(".", "_") not in name:
        return "Model Upgrades"

    if "baseline" in name or policy in {"script_only", "script", "baseline", ""}:
        return "Baseline"
    return "Other / Uncategorized"


def classify_failure_mode(*, case: Dict[str, Any], judge: Optional[Dict[str, Any]]) -> str:
    """Return one of: retrieval_miss | grounding | aggregation | reasoning | ok | unknown."""

    labels = case.get("labels")
    if isinstance(labels, dict):
        if labels.get("retrieval_failure") is True:
            return "retrieval_miss"
        if labels.get("grounding_failure") is True:
            return "grounding"

    qtype = question_type(str(case.get("question") or ""))

    if isinstance(judge, dict):
        corr = safe_int(judge.get("correctness"))
        g = safe_int(judge.get("groundedness"))
        comp = safe_int(judge.get("completeness"))

        if qtype == "Aggregation" and comp is not None and comp <= 2:
            return "aggregation"
        if g is not None and g <= 2:
            return "grounding"
        if corr is not None and corr <= 2:
            return "reasoning"

        overall = safe_int(judge.get("overall"))
        if overall is not None and overall >= 80:
            return "ok"
        if overall is not None and overall <= 40:
            if qtype == "Aggregation":
                return "aggregation"
            return "reasoning"

    return "unknown"
