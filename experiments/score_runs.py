from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def normalize_text(s: str) -> str:
    return (s or "").lower()


def find_episode_ids(text: str) -> List[str]:
    return [m.group(0).upper() for m in EP_RE.finditer(text or "")]


def any_substring_present(text: str, needles: List[str]) -> bool:
    t = normalize_text(text)
    return any(n.lower() in t for n in needles)


def all_substrings_present(text: str, needles: List[str]) -> bool:
    t = normalize_text(text)
    return all(n.lower() in t for n in needles)


def said_idk(answer: Optional[str]) -> bool:
    if not answer:
        return True
    a = normalize_text(answer)
    return "i don't know" in a or "i do not know" in a or "not in the context" in a


def _must_include_needed(total: int) -> int:
    """Heuristic threshold for must_include term hits.

    Empirically matches existing scored artifacts (e.g., 6 terms -> need 4, 3 terms -> need 2).
    """
    if total <= 0:
        return 0
    # ~2/3 of terms, rounded up.
    return int((2 * total + 2) // 3)


def deterministic_score(
    *,
    answer: str,
    expected_episode_ids: List[str],
    must_include: List[str],
    must_not_include: List[str],
) -> Dict[str, Any]:
    """
    Returns a small deterministic scoring bundle.
    This is intentionally lightweight: it provides guardrails and makes eval defensible.
    """
    ans = answer or ""
    ans_eps = set(find_episode_ids(ans))

    expected_eps = {e.upper() for e in (expected_episode_ids or [])}
    episode_ok = True
    if expected_eps:
        episode_ok = bool(ans_eps & expected_eps) or any(
            e.lower() in normalize_text(ans) for e in expected_eps
        )

    must_include_hits = 0
    for term in must_include or []:
        if str(term or "").strip() and term.lower() in normalize_text(ans):
            must_include_hits += 1

    must_include_total = len(list(must_include or []))
    must_include_needed = _must_include_needed(must_include_total)
    must_ok = True
    if must_include_total:
        must_ok = must_include_hits >= must_include_needed

    forbidden_hit = False
    forbidden_terms_hit: List[str] = []
    for term in must_not_include or []:
        if term.lower() in normalize_text(ans):
            forbidden_hit = True
            forbidden_terms_hit.append(term)

    return {
        "answer_episode_ids": sorted(ans_eps),
        "expected_episode_ids": sorted(expected_eps),
        "episode_ok": episode_ok,
        "must_include_ok": must_ok,
        "must_include": list(must_include or []),
        "must_include_hits": int(must_include_hits),
        "must_include_needed": int(must_include_needed),
        "must_include_total": int(must_include_total),
        "forbidden_hit": forbidden_hit,
        "forbidden_terms_hit": forbidden_terms_hit,
    }


def build_context_from_results(
    case: Dict[str, Any],
    *,
    max_docs: Optional[int] = None,
    max_chars: int = 8000,
) -> str:
    """Build judge context for scoring.

    Preference order:
      1) If the run log contains the full context the answering model saw (answer.context_text), use it.
      2) Otherwise, fall back to retrieval previews.

    Important: we want the judge to see the *same evidence* the answer model saw. Using only a
    handful of previews can drop supporting quotes/claims and make the judge appear "wrong".
    """

    full_ctx = safe_get(case, "answer.context_text", "")
    if isinstance(full_ctx, str) and full_ctx.strip():
        ctx = full_ctx.strip()
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n… (truncated)"
        return ctx

    results = safe_get(case, "retrieval.results", [])
    if not isinstance(results, list) or not results:
        return ""

    take_n = int(max_docs) if isinstance(max_docs, int) and max_docs > 0 else len(results)

    parts: List[str] = []
    for r in results[:take_n]:
        src = safe_get(r, "source", "")
        eid = safe_get(r, "episode_id", "")
        dt = safe_get(r, "doc_type", "")
        prev = safe_get(r, "preview", "")
        block = f"SOURCE: {src}\nEPISODE_ID: {eid}\nDOC_TYPE: {dt}\n\n{prev}".strip()
        parts.append(block)

    ctx = "\n\n---\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n… (truncated)"
    return ctx


def _overall_from_components(
    *, correctness: int, groundedness: int, completeness: int, hallucination: int
) -> int:
    # Weighted: correctness 40%, groundedness 30%, completeness 20%, hallucination 10%.
    # Each component is 0..5.
    raw = (40 * correctness) + (30 * groundedness) + (20 * completeness) + (10 * hallucination)
    return int(round(raw / 5.0))


def _verdict_from_overall(*, overall: int, idk: bool, context_sufficiency: Optional[str]) -> str:
    if idk and (context_sufficiency in {"empty", "insufficient", None}):
        return "idk_preferred"
    if overall >= 90:
        return "correct"
    if overall >= 60:
        return "partially_correct"
    return "incorrect"


def judge_prompt(
    *,
    question: str,
    answer: str,
    gold_answer: str,
    expected_episode_ids: List[str],
    context: str,
) -> Tuple[str, str]:
    system = (
        "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
        "You must grade the candidate answer against a gold reference AND the retrieved context.\n"
        "If the candidate answer claims facts not supported by the retrieved context, penalize GROUNDEDNESS.\n"
        "If the retrieved context does not contain enough to answer, the best answer is 'I don't know' (and that can score well on groundedness).\n"
        "Return ONLY valid JSON with the required schema.\n"
    )

    user = {
        "question": question,
        "gold_answer": gold_answer,
        "expected_episode_ids": expected_episode_ids,
        "candidate_answer": answer,
        "retrieved_context": context,
        "rubric": {
            "correctness_0_to_5": "Matches gold key facts and does not contradict.",
            "groundedness_0_to_5": "Claims are supported by retrieved_context; penalize unsupported claims.",
            "completeness_0_to_5": "Covers the required parts of the question.",
            "hallucination_penalty_0_to_5": "5 means no hallucinations; 0 means major hallucinations.",
            "overall_0_to_100": "Weighted: correctness 40%, groundedness 30%, completeness 20%, hallucination 10%.",
        },
        "required_output_schema": {
            "correctness": "int 0..5",
            "groundedness": "int 0..5",
            "completeness": "int 0..5",
            "hallucination": "int 0..5",
            "overall": "int 0..100",
            "verdict": "one of: correct | partially_correct | incorrect | idk_preferred",
            "unsupported_claims": "list of short strings",
            "missing_points": "list of short strings",
            "notes": "short string",
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def grounding_prompt(*, question: str, answer: str, context: str) -> Tuple[str, str]:
    system = (
        "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
        "Evaluate ONLY groundedness relative to the retrieved context.\n"
        "Do NOT use outside knowledge.\n"
        "Return ONLY valid JSON with the required schema.\n"
    )
    user = {
        "question": question,
        "candidate_answer": answer,
        "retrieved_context": context,
        "required_output_schema": {
            "groundedness": "int 0..5",
            "hallucination": "int 0..5",
            "context_sufficiency": "one of: sufficient | insufficient | empty",
            "unsupported_claims": "list of short strings",
            "notes": "short string",
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def correctness_prompt(
    *,
    question: str,
    answer: str,
    gold_answer: str,
    expected_episode_ids: List[str],
    context: str,
) -> Tuple[str, str]:
    system = (
        "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
        "Evaluate correctness and completeness against the gold reference.\n"
        "If the retrieved context is insufficient, it's acceptable to prefer 'I don't know'.\n"
        "Return ONLY valid JSON with the required schema.\n"
    )
    user = {
        "question": question,
        "gold_answer": gold_answer,
        "expected_episode_ids": expected_episode_ids,
        "candidate_answer": answer,
        "retrieved_context": context,
        "required_output_schema": {
            "correctness": "int 0..5",
            "completeness": "int 0..5",
            "missing_points": "list of short strings",
            "notes": "short string",
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def parse_judge_json(text: str) -> Dict[str, Any]:
    """
    Be robust to the model occasionally wrapping JSON in text.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        # try to extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validate_int_range(
    obj: Dict[str, Any], key: str, lo: int, hi: int, errors: List[str]
) -> Optional[int]:
    v = obj.get(key)
    if not isinstance(v, int):
        errors.append(f"{key} must be int")
        return None
    if v < lo or v > hi:
        errors.append(f"{key} out of range {lo}..{hi}")
        return None
    return int(v)


def _validate_list_str(obj: Dict[str, Any], key: str, errors: List[str]) -> List[str]:
    v = obj.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        errors.append(f"{key} must be list")
        return []
    out: List[str] = []
    for it in v:
        s = str(it or "").strip()
        if s:
            out.append(s)
    return out


def _validate_grounding(obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    groundedness = _validate_int_range(obj, "groundedness", 0, 5, errors)
    hallucination = _validate_int_range(obj, "hallucination", 0, 5, errors)
    cs = obj.get("context_sufficiency")
    if cs not in {"sufficient", "insufficient", "empty"}:
        errors.append("context_sufficiency must be one of: sufficient|insufficient|empty")
        cs = None
    unsupported_claims = _validate_list_str(obj, "unsupported_claims", errors)
    notes = str(obj.get("notes") or "").strip()
    if groundedness is None or hallucination is None or cs is None:
        return None, errors
    return {
        "groundedness": groundedness,
        "hallucination": hallucination,
        "context_sufficiency": cs,
        "unsupported_claims": unsupported_claims,
        "notes": notes,
    }, errors


def _validate_correctness(obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    correctness = _validate_int_range(obj, "correctness", 0, 5, errors)
    completeness = _validate_int_range(obj, "completeness", 0, 5, errors)
    missing_points = _validate_list_str(obj, "missing_points", errors)
    notes = str(obj.get("notes") or "").strip()
    if correctness is None or completeness is None:
        return None, errors
    return {
        "correctness": correctness,
        "completeness": completeness,
        "missing_points": missing_points,
        "notes": notes,
    }, errors


def score_case_two_pass(
    llm: ChatOpenAI,
    *,
    case: Dict[str, Any],
    gold: Dict[str, Any],
) -> Dict[str, Any]:
    question = str(safe_get(case, "question", ""))
    answer = str(safe_get(case, "answer.text", ""))

    gold_answer = str(gold.get("gold_answer", ""))
    expected_episode_ids = list(gold.get("expected_episode_ids", []) or [])
    must_include = list(gold.get("must_include", []) or [])
    must_not_include = list(gold.get("must_not_include", []) or [])

    det = deterministic_score(
        answer=answer,
        expected_episode_ids=expected_episode_ids,
        must_include=must_include,
        must_not_include=must_not_include,
    )

    has_full_ctx = isinstance(safe_get(case, "answer.context_text", None), str) and bool(
        str(safe_get(case, "answer.context_text", "")).strip()
    )
    context_source = "answer_context_text" if has_full_ctx else "retrieval_previews"
    context = build_context_from_results(case)
    if not context.strip():
        context_sufficiency = "empty"
    else:
        context_sufficiency = None

    sys_g, user_g = grounding_prompt(question=question, answer=answer, context=context)
    raw_g = str(llm.invoke([("system", sys_g), ("human", user_g)]).content or "")
    g_obj = parse_judge_json(raw_g)
    g_valid, g_errors = _validate_grounding(g_obj)

    sys_c, user_c = correctness_prompt(
        question=question,
        answer=answer,
        gold_answer=gold_answer,
        expected_episode_ids=expected_episode_ids,
        context=context,
    )
    raw_c = str(llm.invoke([("system", sys_c), ("human", user_c)]).content or "")
    c_obj = parse_judge_json(raw_c)
    c_valid, c_errors = _validate_correctness(c_obj)

    merged_errors: List[str] = []
    merged_ok = True
    if g_valid is None:
        merged_ok = False
        merged_errors.extend(["grounding_invalid"] + g_errors)
        g_valid = {
            "groundedness": 0,
            "hallucination": 0,
            "context_sufficiency": context_sufficiency or "insufficient",
            "unsupported_claims": [],
            "notes": "",
        }
    if c_valid is None:
        merged_ok = False
        merged_errors.extend(["correctness_invalid"] + c_errors)
        c_valid = {
            "correctness": 0,
            "completeness": 0,
            "missing_points": [],
            "notes": "",
        }

    if context_sufficiency is not None:
        g_valid["context_sufficiency"] = context_sufficiency

    overall = _overall_from_components(
        correctness=int(c_valid["correctness"]),
        groundedness=int(g_valid["groundedness"]),
        completeness=int(c_valid["completeness"]),
        hallucination=int(g_valid["hallucination"]),
    )
    verdict = _verdict_from_overall(
        overall=int(overall),
        idk=said_idk(answer),
        context_sufficiency=str(g_valid.get("context_sufficiency")) if g_valid else None,
    )

    judge = {
        "correctness": int(c_valid["correctness"]),
        "groundedness": int(g_valid["groundedness"]),
        "completeness": int(c_valid["completeness"]),
        "hallucination": int(g_valid["hallucination"]),
        "overall": int(overall),
        "verdict": verdict,
        "unsupported_claims": list(g_valid.get("unsupported_claims") or []),
        "missing_points": list(c_valid.get("missing_points") or []),
        "context_sufficiency": str(g_valid.get("context_sufficiency") or ""),
        "notes": f"grounding: {str(g_valid.get('notes') or '').strip()} | correctness: {str(c_valid.get('notes') or '').strip()}".strip(),
        "question": question,
    }

    judge_validation_errors: List[str] = []
    if not isinstance(judge.get("overall"), int) or judge["overall"] < 0 or judge["overall"] > 100:
        judge_validation_errors.append("overall must be int 0..100")
    judge_ok = not judge_validation_errors

    return {
        "deterministic": det,
        "judge": judge,
        "judge_context_source": context_source,
        "judge_two_pass": {
            "grounding": g_valid,
            "correctness": c_valid,
            "raw": {"grounding": raw_g, "correctness": raw_c},
            "validation": {
                "grounding_ok": g_valid is not None and not g_errors,
                "grounding_errors": g_errors,
                "correctness_ok": c_valid is not None and not c_errors,
                "correctness_errors": c_errors,
                "merged_ok": bool(merged_ok and not merged_errors),
                "merged_errors": merged_errors,
            },
        },
        "judge_raw_text": None,
        "judge_validation": {"ok": bool(judge_ok), "errors": judge_validation_errors},
    }


def score_case(
    llm: ChatOpenAI,
    *,
    case: Dict[str, Any],
    gold: Dict[str, Any],
) -> Dict[str, Any]:
    question = str(safe_get(case, "question", ""))
    answer = str(safe_get(case, "answer.text", ""))

    gold_answer = str(gold.get("gold_answer", ""))
    expected_episode_ids = list(gold.get("expected_episode_ids", []) or [])
    must_include = list(gold.get("must_include", []) or [])
    must_not_include = list(gold.get("must_not_include", []) or [])

    det = deterministic_score(
        answer=answer,
        expected_episode_ids=expected_episode_ids,
        must_include=must_include,
        must_not_include=must_not_include,
    )

    has_full_ctx = isinstance(safe_get(case, "answer.context_text", None), str) and bool(
        str(safe_get(case, "answer.context_text", "")).strip()
    )
    context_source = "answer_context_text" if has_full_ctx else "retrieval_previews"
    context = build_context_from_results(case)

    system, user = judge_prompt(
        question=question,
        answer=answer,
        gold_answer=gold_answer,
        expected_episode_ids=expected_episode_ids,
        context=context,
    )

    resp = llm.invoke([("system", system), ("human", user)]).content
    judge = parse_judge_json(resp)

    return {
        "deterministic": det,
        "judge": judge,
        "judge_context_source": context_source,
    }


def _deterministic_summary(scored_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(scored_cases)
    if total <= 0:
        return {
            "cases_total": 0,
            "episode_ok_rate": None,
            "must_include_ok_rate": None,
            "forbidden_hit_rate": None,
        }

    ep_ok = 0
    must_ok = 0
    forb_hit = 0

    for row in scored_cases:
        det = row.get("deterministic") or {}
        if bool(det.get("episode_ok")):
            ep_ok += 1
        if bool(det.get("must_include_ok")):
            must_ok += 1
        if bool(det.get("forbidden_hit")):
            forb_hit += 1

    return {
        "cases_total": total,
        "episode_ok_rate": float(ep_ok / total),
        "must_include_ok_rate": float(must_ok / total),
        "forbidden_hit_rate": float(forb_hit / total),
    }


def score_run(
    llm: ChatOpenAI,
    *,
    run_obj: Dict[str, Any],
    gold_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cases = safe_get(run_obj, "cases", [])
    if not isinstance(cases, list):
        raise ValueError("Run file missing cases[]")

    scored_cases: List[Dict[str, Any]] = []
    overall_scores: List[int] = []

    for c in cases:
        cid = str(safe_get(c, "case_id", ""))
        gold = gold_by_id.get(cid)
        if not gold:
            scored_cases.append({"case_id": cid, "error": "missing_gold"})
            continue

        scored = score_case(llm, case=c, gold=gold)
        scored_cases.append({"case_id": cid, **scored})

        overall = scored.get("judge", {}).get("overall")
        if isinstance(overall, int):
            overall_scores.append(overall)

    avg_overall = round(sum(overall_scores) / len(overall_scores)) if overall_scores else None

    return {
        "run": safe_get(run_obj, "run", {}),
        "config": safe_get(run_obj, "config", {}),
        "score_summary": {
            "cases_scored": len(overall_scores),
            "avg_overall": avg_overall,
            "overall_scores": overall_scores,
        },
        "deterministic_summary": _deterministic_summary(scored_cases),
        "scored_cases": scored_cases,
    }


@dataclass(frozen=True)
class ScoreConfig:
    judge_mode: str  # one_pass|two_pass|none
    judge_model: str
    temperature: float
    timeout_s: float
    judge_max_retries: int
    openai_max_retries: int
    deterministic_only: bool


def _iter_run_files(runs_dir: Path) -> List[Path]:
    return sorted(runs_dir.rglob("*.json"))


def _score_one_run(
    *,
    run_obj: Dict[str, Any],
    cfg: ScoreConfig,
    gold_by_id: Dict[str, Dict[str, Any]],
    llm: Optional[ChatOpenAI],
) -> Dict[str, Any]:
    if cfg.deterministic_only or cfg.judge_mode == "none":
        cases = safe_get(run_obj, "cases", [])
        scored_cases: List[Dict[str, Any]] = []
        for c in cases if isinstance(cases, list) else []:
            cid = str(safe_get(c, "case_id", ""))
            gold = gold_by_id.get(cid)
            if not gold:
                scored_cases.append({"case_id": cid, "error": "missing_gold"})
                continue
            answer = str(safe_get(c, "answer.text", ""))
            det = deterministic_score(
                answer=answer,
                expected_episode_ids=list(gold.get("expected_episode_ids", []) or []),
                must_include=list(gold.get("must_include", []) or []),
                must_not_include=list(gold.get("must_not_include", []) or []),
            )
            scored_cases.append({"case_id": cid, "deterministic": det, "judge": None})

        return {
            "run": safe_get(run_obj, "run", {}),
            "config": safe_get(run_obj, "config", {}),
            "scoring_meta": {
                "scored_at_utc": utc_now_iso(),
                "scoring_schema_version": "v2",
                "judge_mode": "none",
                "judge_model": None,
                "judge_temperature": None,
                "judge_timeout_s": cfg.timeout_s,
                "judge_max_retries": cfg.judge_max_retries,
                "openai_max_retries": cfg.openai_max_retries,
                "deterministic_only": True,
            },
            "score_summary": {"cases_scored": 0, "avg_overall": None, "overall_scores": []},
            "deterministic_summary": _deterministic_summary(scored_cases),
            "scored_cases": scored_cases,
        }

    if llm is None:
        raise RuntimeError("LLM is required for judging")

    if cfg.judge_mode == "two_pass":
        cases = safe_get(run_obj, "cases", [])
        scored_cases: List[Dict[str, Any]] = []
        overall_scores: List[int] = []
        for c in cases if isinstance(cases, list) else []:
            cid = str(safe_get(c, "case_id", ""))
            gold = gold_by_id.get(cid)
            if not gold:
                scored_cases.append({"case_id": cid, "error": "missing_gold"})
                continue

            scored = score_case_two_pass(llm, case=c, gold=gold)
            scored_cases.append({"case_id": cid, **scored})

            overall = (scored.get("judge") or {}).get("overall")
            if isinstance(overall, int):
                overall_scores.append(overall)

        avg_overall = round(sum(overall_scores) / len(overall_scores)) if overall_scores else None
        return {
            "run": safe_get(run_obj, "run", {}),
            "config": safe_get(run_obj, "config", {}),
            "scoring_meta": {
                "scored_at_utc": utc_now_iso(),
                "scoring_schema_version": "v2",
                "judge_mode": "two_pass",
                "judge_model": cfg.judge_model,
                "judge_temperature": float(cfg.temperature),
                "judge_timeout_s": float(cfg.timeout_s),
                "judge_max_retries": int(cfg.judge_max_retries),
                "openai_max_retries": int(cfg.openai_max_retries),
                "deterministic_only": False,
            },
            "score_summary": {
                "cases_scored": len(overall_scores),
                "avg_overall": avg_overall,
                "overall_scores": overall_scores,
            },
            "deterministic_summary": _deterministic_summary(scored_cases),
            "scored_cases": scored_cases,
        }

    # Fall back to legacy one-pass judge behavior.
    if cfg.judge_mode == "one_pass":
        scored = score_run(llm, run_obj=run_obj, gold_by_id=gold_by_id)
        scored["scoring_meta"] = {
            "scored_at_utc": utc_now_iso(),
            "scoring_schema_version": "v2",
            "judge_mode": "one_pass",
            "judge_model": cfg.judge_model,
            "judge_temperature": float(cfg.temperature),
            "judge_timeout_s": float(cfg.timeout_s),
            "judge_max_retries": int(cfg.judge_max_retries),
            "openai_max_retries": int(cfg.openai_max_retries),
            "deterministic_only": False,
        }
        return scored

    raise ValueError(f"Unsupported judge_mode: {cfg.judge_mode}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Score RAG run logs using deterministic checks and an optional judge."
    )
    parser.add_argument(
        "--runs-dir", default="experiments/runs", help="Directory containing run JSON logs"
    )
    parser.add_argument(
        "--gold", default="experiments/gold_answers.json", help="Gold answers JSON file"
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/scored_runs_two_pass",
        help="Where to write scored JSON files (default: experiments/scored_runs_two_pass)",
    )

    parser.add_argument(
        "--run-name-prefix",
        default="",
        help="If set, only score runs whose run_name starts with this prefix.",
    )
    parser.add_argument(
        "--judge-mode",
        default="two_pass",
        choices=["two_pass", "one_pass", "none"],
        help="Judge scoring mode. Use 'none' for deterministic-only scoring.",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4.1-mini", help="OpenAI model to use as judge"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Judge request timeout (seconds)"
    )
    parser.add_argument("--judge-max-retries", type=int, default=2, help="Retries for judge calls")
    parser.add_argument(
        "--openai-max-retries", type=int, default=0, help="OpenAI client max retries"
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Skip judge calls; write deterministic-only outputs",
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scored files")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap for number of runs to score (0 = no cap)",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    gold_path = Path(args.gold).expanduser().resolve()

    gold_list = read_json(gold_path)
    if not isinstance(gold_list, list):
        raise ValueError("gold file must be a JSON array")

    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for g in gold_list:
        if not isinstance(g, dict):
            continue
        cid = str(g.get("case_id") or "").strip()
        if cid:
            gold_by_id[cid] = g

    cfg = ScoreConfig(
        judge_mode=str(args.judge_mode),
        judge_model=str(args.judge_model),
        temperature=float(args.temperature),
        timeout_s=float(args.timeout),
        judge_max_retries=int(args.judge_max_retries),
        openai_max_retries=int(args.openai_max_retries),
        deterministic_only=bool(args.deterministic_only),
    )

    llm: Optional[ChatOpenAI] = None
    if not cfg.deterministic_only and cfg.judge_mode != "none":
        # Note: `max_retries` here is for HTTP-level retries in langchain-openai.
        # We also do application-level retries around scoring.
        llm = ChatOpenAI(
            model=cfg.judge_model,
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
            max_retries=cfg.openai_max_retries,
        )

    run_files = _iter_run_files(runs_dir)
    if not run_files:
        raise FileNotFoundError(f"No run json files under {runs_dir}")

    name_prefix = str(args.run_name_prefix or "").strip()

    scored_count = 0
    for rf in run_files:
        run_obj = read_json(rf)
        if not (isinstance(run_obj, dict) and "run" in run_obj and "cases" in run_obj):
            continue

        run_name = str(safe_get(run_obj, "run.run_name", "") or "")
        if name_prefix and not run_name.startswith(name_prefix):
            continue

        run_id = str(safe_get(run_obj, "run.run_id", rf.stem))
        out_path = out_dir / f"{run_id}.scored.json"

        if out_path.exists() and not bool(args.overwrite):
            print(f"Skipping existing: {out_path}")
            continue

        # Retry around transient judge failures.
        attempts = 0
        last_err: Optional[Exception] = None
        while attempts <= cfg.judge_max_retries:
            try:
                scored = _score_one_run(run_obj=run_obj, cfg=cfg, gold_by_id=gold_by_id, llm=llm)
                # Enrich meta with provenance fields when present.
                if isinstance(scored.get("scoring_meta"), dict):
                    scored["scoring_meta"].setdefault("gold_file", str(gold_path))
                    scored["scoring_meta"].setdefault("runs_dir", str(runs_dir))
                write_json(out_path, scored)
                print(f"Wrote: {out_path}")
                scored_count += 1
                break
            except Exception as e:
                last_err = e
                attempts += 1
                if attempts > cfg.judge_max_retries:
                    raise
                sleep_s = min(6.0, 0.5 * (2 ** (attempts - 1)))
                print(f"Retrying {run_id} after {type(e).__name__}: {e} (sleep {sleep_s:.1f}s)")
                time.sleep(sleep_s)

        if int(args.max_runs or 0) > 0 and scored_count >= int(args.max_runs):
            break


if __name__ == "__main__":
    main()
