from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


"""Summarize run JSON files into flat tables for analysis/charting."""

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Any, path: List[str]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _iter_run_files(runs_dir: Path) -> Iterable[Path]:
    # Use rglob so we don't silently miss runs if we later nest folders by date/tag.
    yield from sorted(runs_dir.rglob("*.json"))


def _is_run_log(obj: Any) -> bool:
    return isinstance(obj, dict) and "run" in obj and "cases" in obj


def _flatten_run_row(run_obj: Dict[str, Any], *, run_file: Path) -> Dict[str, Any]:
    run = run_obj.get("run", {}) if isinstance(run_obj, dict) else {}
    cfg = run_obj.get("config", {}) if isinstance(run_obj, dict) else {}
    summary = run_obj.get("summary", {}) if isinstance(run_obj, dict) else {}

    # Prefer explicit provenance block when present; fall back to legacy fields.
    data_version = cfg.get("data_version", {}) if isinstance(cfg, dict) else {}
    script_dv = data_version.get("script", {}) if isinstance(data_version, dict) else {}
    derived_dv = data_version.get("derived", {}) if isinstance(data_version, dict) else {}

    script_fp = (script_dv.get("fingerprint", {}) if isinstance(script_dv, dict) else {})
    script_sqlite_fp = (script_fp.get("chroma_sqlite") if isinstance(script_fp, dict) else None) or {}

    derived_fp = (derived_dv.get("fingerprint", {}) if isinstance(derived_dv, dict) else {})
    derived_sqlite_fp = (derived_fp.get("chroma_sqlite") if isinstance(derived_fp, dict) else None) or {}

    retrieval = cfg.get("retrieval", {}) if isinstance(cfg, dict) else {}
    qe = retrieval.get("query_expansion", {}) if isinstance(retrieval, dict) else {}

    vectorstore = cfg.get("vectorstore", {}) if isinstance(cfg, dict) else {}
    derived_vectorstore = cfg.get("derived_vectorstore", {}) if isinstance(cfg, dict) else {}

    return {
        "run_file": str(run_file.as_posix()),
        "run_id": run.get("run_id"),
        "run_name": run.get("run_name"),
        "created_at_utc": run.get("created_at_utc"),
        "notes": run.get("notes"),

        # --- Auditability / provenance (added in run schema v4) ---
        "run_schema_version": cfg.get("run_schema_version"),
        "git_sha": _safe_get(cfg, ["code_version", "git_sha"]),

        "script_persist_directory": (
            (script_dv.get("persist_directory") if isinstance(script_dv, dict) else None)
            or _safe_get(cfg, ["vectorstore", "persist_directory"])
        ),
        "script_collection_name": (
            (script_dv.get("collection_name") if isinstance(script_dv, dict) else None)
            or _safe_get(cfg, ["vectorstore", "collection_name"])
        ),
        "script_chroma_sqlite_size_bytes": (script_sqlite_fp.get("size_bytes") if isinstance(script_sqlite_fp, dict) else None),
        "script_chroma_sqlite_mtime_utc": (script_sqlite_fp.get("mtime_utc") if isinstance(script_sqlite_fp, dict) else None),
        "script_chroma_sqlite_sha256": (script_sqlite_fp.get("sha256") if isinstance(script_sqlite_fp, dict) else None),

        "derived_persist_directory": (
            (derived_dv.get("persist_directory") if isinstance(derived_dv, dict) else None)
            or (derived_vectorstore.get("persist_directory") if isinstance(derived_vectorstore, dict) else None)
        ),
        "derived_collection_name": (
            (derived_dv.get("collection_name") if isinstance(derived_dv, dict) else None)
            or (derived_vectorstore.get("collection_name") if isinstance(derived_vectorstore, dict) else None)
        ),
        "derived_build_tag": (derived_dv.get("build_tag") if isinstance(derived_dv, dict) else None),
        "derived_chroma_sqlite_size_bytes": (derived_sqlite_fp.get("size_bytes") if isinstance(derived_sqlite_fp, dict) else None),
        "derived_chroma_sqlite_mtime_utc": (derived_sqlite_fp.get("mtime_utc") if isinstance(derived_sqlite_fp, dict) else None),
        "derived_chroma_sqlite_sha256": (derived_sqlite_fp.get("sha256") if isinstance(derived_sqlite_fp, dict) else None),

        "persist_directory": _safe_get(cfg, ["vectorstore", "persist_directory"]),
        "collection_name": _safe_get(cfg, ["vectorstore", "collection_name"]),
        "doc_count": vectorstore.get("doc_count"),
        "vector_count": vectorstore.get("vector_count"),
        "search_type": retrieval.get("search_type"),
        "k": retrieval.get("k"),
        "fetch_k": retrieval.get("fetch_k"),
        "lambda_mult": retrieval.get("lambda_mult"),
        "qe_enabled": qe.get("enabled"),
        "qe_n": qe.get("n"),
        "qe_model": qe.get("model"),
        "k_per_query": qe.get("k_per_query"),
        "fusion": qe.get("fusion"),
        "rrf_k0": qe.get("rrf_k0"),
        "answered_with_llm": summary.get("answered_with_llm"),
        "cases": summary.get("cases"),
        "avg_retrieval_latency_ms": summary.get("avg_retrieval_latency_ms"),
        "avg_context_docs": summary.get("avg_context_docs"),
        "avg_context_chars": summary.get("avg_context_chars"),
        "avg_distinct_episodes_in_context": summary.get("avg_distinct_episodes_in_context"),
        "avg_distinct_sources_in_context": summary.get("avg_distinct_sources_in_context"),
        "avg_top_episode_share_in_context": summary.get("avg_top_episode_share_in_context"),
        "avg_episode_entropy_norm_in_context": summary.get("avg_episode_entropy_norm_in_context"),
        "avg_aggregation_readiness_score": summary.get("avg_aggregation_readiness_score"),
        "avg_aggregation_readiness_score_agg_questions": summary.get("avg_aggregation_readiness_score_agg_questions"),
        "episode_citation_rate": summary.get("episode_citation_rate"),
        "quote_in_context_rate": summary.get("quote_in_context_rate"),
        "grounding_failure_rate": summary.get("grounding_failure_rate"),
        "retrieval_empty_rate": summary.get("retrieval_empty_rate"),
        "retrieval_failure_rate": summary.get("retrieval_failure_rate"),
        "idk_rate": summary.get("idk_rate"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_cited_episode_ids_not_in_context": summary.get("avg_cited_episode_ids_not_in_context"),
        "errors_count": summary.get("errors_count"),
    }


def _flatten_case_rows(run_obj: Dict[str, Any], *, run_file: Path) -> List[Dict[str, Any]]:
    run = run_obj.get("run", {}) if isinstance(run_obj, dict) else {}
    cfg = run_obj.get("config", {}) if isinstance(run_obj, dict) else {}
    retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg, dict) else {}

    data_version = cfg.get("data_version", {}) if isinstance(cfg, dict) else {}
    script_dv = data_version.get("script", {}) if isinstance(data_version, dict) else {}
    derived_dv = data_version.get("derived", {}) if isinstance(data_version, dict) else {}

    script_fp = (script_dv.get("fingerprint", {}) if isinstance(script_dv, dict) else {})
    script_sqlite_fp = (script_fp.get("chroma_sqlite") if isinstance(script_fp, dict) else None) or {}

    derived_fp = (derived_dv.get("fingerprint", {}) if isinstance(derived_dv, dict) else {})
    derived_sqlite_fp = (derived_fp.get("chroma_sqlite") if isinstance(derived_fp, dict) else None) or {}

    rows: List[Dict[str, Any]] = []
    cases = run_obj.get("cases", []) if isinstance(run_obj, dict) else []
    if not isinstance(cases, list):
        return rows

    for c in cases:
        if not isinstance(c, dict):
            continue
        diag = c.get("diagnostics", {}) if isinstance(c.get("diagnostics"), dict) else {}
        div = diag.get("context_diversity", {}) if isinstance(diag.get("context_diversity"), dict) else {}
        agg = diag.get("aggregation", {}) if isinstance(diag.get("aggregation"), dict) else {}
        heur = c.get("heuristics", {}) if isinstance(c.get("heuristics"), dict) else {}

        rows.append(
            {
                "run_file": str(run_file.as_posix()),
                "run_id": run.get("run_id"),
                "run_name": run.get("run_name"),
                "created_at_utc": run.get("created_at_utc"),

                # --- Auditability / provenance (added in run schema v4) ---
                "run_schema_version": cfg.get("run_schema_version"),
                "git_sha": _safe_get(cfg, ["code_version", "git_sha"]),

                "script_persist_directory": (
                    (script_dv.get("persist_directory") if isinstance(script_dv, dict) else None)
                    or _safe_get(cfg, ["vectorstore", "persist_directory"])
                ),
                "script_collection_name": (
                    (script_dv.get("collection_name") if isinstance(script_dv, dict) else None)
                    or _safe_get(cfg, ["vectorstore", "collection_name"])
                ),
                "script_chroma_sqlite_sha256": (
                    script_sqlite_fp.get("sha256") if isinstance(script_sqlite_fp, dict) else None
                ),

                "derived_persist_directory": (derived_dv.get("persist_directory") if isinstance(derived_dv, dict) else None),
                "derived_collection_name": (derived_dv.get("collection_name") if isinstance(derived_dv, dict) else None),
                "derived_build_tag": (derived_dv.get("build_tag") if isinstance(derived_dv, dict) else None),
                "derived_chroma_sqlite_sha256": (
                    derived_sqlite_fp.get("sha256") if isinstance(derived_sqlite_fp, dict) else None
                ),

                "persist_directory": _safe_get(cfg, ["vectorstore", "persist_directory"]),
                "search_type": retrieval_cfg.get("search_type"),
                "k": retrieval_cfg.get("k"),
                "qe_enabled": _safe_get(retrieval_cfg, ["query_expansion", "enabled"]),
                "case_id": c.get("case_id"),
                "question": c.get("question"),
                "retrieved_any": heur.get("retrieved_any"),
                "said_idk": heur.get("said_idk"),
                "cited_episode": heur.get("cited_episode"),
                "context_docs": _safe_get(c, ["answer", "stats", "context_docs"]),
                "context_chars": _safe_get(c, ["answer", "stats", "context_chars"]),
                "distinct_episode_count": div.get("distinct_episode_count"),
                "distinct_source_count": div.get("distinct_source_count"),
                "top_episode_share": div.get("top_episode_share"),
                "episode_entropy_norm": div.get("episode_entropy_norm"),
                "source_dup_rate": div.get("source_dup_rate"),
                "episode_dup_rate": div.get("episode_dup_rate"),
                "aggregation_like": agg.get("is_aggregation_like_question"),
                "aggregation_readiness_score_0_100": agg.get("readiness_score_0_100"),
                "cited_episode_ids_not_in_context_count": diag.get("cited_episode_ids_not_in_context_count"),
                "grounding_failure": _safe_get(c, ["labels", "grounding_failure"]),
                "retrieval_failure": _safe_get(c, ["labels", "retrieval_failure"]),
            }
        )

    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize experiments/runs/*.json into flat tables for charting.")
    p.add_argument("--runs-dir", default="experiments/runs", help="Directory containing run JSONs")
    p.add_argument("--out-run-csv", default="experiments/run_metrics.csv", help="Output CSV path (run-level)")
    p.add_argument("--out-run-jsonl", default="experiments/run_metrics.jsonl", help="Output JSONL path (run-level)")
    p.add_argument(
        "--out-case-csv",
        default="experiments/case_metrics.csv",
        help="Output CSV path (case-level; one row per question per run)",
    )
    p.add_argument(
        "--out-case-jsonl",
        default="experiments/case_metrics.jsonl",
        help="Output JSONL path (case-level; one row per question per run)",
    )

    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    run_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for run_file in _iter_run_files(runs_dir):
        try:
            obj = _read_json(run_file)
            if not _is_run_log(obj):
                skipped.append({"run_file": str(run_file.as_posix()), "reason": "not_a_run_log"})
                continue
            run_rows.append(_flatten_run_row(obj, run_file=run_file))
            case_rows.extend(_flatten_case_rows(obj, run_file=run_file))
        except Exception as e:
            # Keep going even if a run file is malformed, but record it.
            skipped.append(
                {
                    "run_file": str(run_file.as_posix()),
                    "reason": f"error:{type(e).__name__}",
                }
            )
            continue

    _write_csv(Path(args.out_run_csv), run_rows)
    _write_jsonl(Path(args.out_run_jsonl), run_rows)
    _write_csv(Path(args.out_case_csv), case_rows)
    _write_jsonl(Path(args.out_case_jsonl), case_rows)

    print(f"Wrote run-level rows: {len(run_rows)}")
    print(f"Wrote case-level rows: {len(case_rows)}")
    if skipped:
        print(f"Skipped files: {len(skipped)}")
        for row in skipped[:10]:
            print(" -", row["reason"], row["run_file"])


if __name__ == "__main__":
    main()
