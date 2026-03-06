from __future__ import annotations

import argparse
import json
import os
import re
import time
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def said_idk(text: str) -> bool:
    t = normalize_text(text or "")
    return "i don't know" in t or "i do not know" in t or "not in the context" in t


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


def count_substrings_present(text: str, needles: List[str]) -> int:
    t = normalize_text(text)
    hits = 0
    for n in needles or []:
        ns = str(n or "").strip().lower()
        if not ns:
            continue
        if ns in t:
            hits += 1
    return hits


def must_include_threshold(n: int) -> int:
    """How many `must_include` items should be present to pass.

    Gold rows often include stems (e.g. "impersonat") and multiple salient entities.
    Requiring *all* items is often too strict, but requiring *any* is too lenient.
    """
    n = int(n)
    if n <= 0:
        return 0
    if n <= 2:
        return n
    return int(max(1, math.ceil(0.6 * float(n))))


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
        episode_ok = bool(ans_eps & expected_eps) or any(e.lower() in normalize_text(ans) for e in expected_eps)

    must_hits = count_substrings_present(ans, must_include)
    must_n = len([str(x).strip() for x in (must_include or []) if str(x).strip()])
    must_need = must_include_threshold(must_n)
    must_ok = True
    if must_n > 0:
        must_ok = must_hits >= must_need

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
        "must_include": must_include,
        "must_include_hits": int(must_hits),
        "must_include_needed": int(must_need),
        "must_include_total": int(must_n),
        "forbidden_hit": forbidden_hit,
        "forbidden_terms_hit": forbidden_terms_hit,
    }


def build_context_from_results(case: Dict[str, Any], *, max_docs: int = 6, max_chars: int = 8000) -> str:
    """
    Prefer the actual logged context text (when present). Otherwise fall back to retrieval previews.

    Using the logged context keeps judging aligned with what the model actually saw.
    """
    ctx_logged = safe_get(case, "answer.context_text", None)
    if isinstance(ctx_logged, str) and ctx_logged.strip():
        # Keep the judge prompt bounded.
        if len(ctx_logged) > int(max_chars):
            return ctx_logged[: int(max_chars)] + "\n… (truncated)"
        return ctx_logged

    results = safe_get(case, "retrieval.results", [])
    if not isinstance(results, list) or not results:
        return ""

    parts: List[str] = []
    for r in results[:max_docs]:
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
            "notes": "short string"
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def judge_grounding_prompt(
    *,
    question: str,
    answer: str,
    context: str,
) -> Tuple[str, str]:
    """Judge groundedness against retrieved context only.

    This avoids 'gold leakage' bias into groundedness scoring.
    """
    system = (
        "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
        "You must grade the candidate answer ONLY against the retrieved context.\n"
        "Do NOT use prior knowledge. Do NOT use any gold reference.\n"
        "If the context is insufficient, prefer marking context_sufficiency as insufficient and do not invent facts.\n"
        "Return ONLY valid JSON with the required schema.\n"
    )

    user = {
        "question": question,
        "candidate_answer": answer,
        "retrieved_context": context,
        "rubric": {
            "groundedness_0_to_5": "Claims are supported by retrieved_context; penalize unsupported claims.",
            "hallucination_0_to_5": "5 means no hallucinations beyond retrieved_context.",
            "context_sufficiency": "One of: sufficient | insufficient | unclear (is there enough info in retrieved_context to answer?)",
        },
        "required_output_schema": {
            "groundedness": "int 0..5",
            "hallucination": "int 0..5",
            "context_sufficiency": "one of: sufficient | insufficient | unclear",
            "unsupported_claims": "list of short strings",
            "notes": "short string",
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def judge_correctness_prompt(
    *,
    question: str,
    answer: str,
    gold_answer: str,
    expected_episode_ids: List[str],
) -> Tuple[str, str]:
    """Judge correctness/completeness against gold only.

    This avoids the judge 'excusing' wrong answers due to missing retrieval.
    """
    system = (
        "You are a strict evaluator for a QA system.\n"
        "You must grade the candidate answer against the gold reference ONLY.\n"
        "Ignore any retrieved context (you will not be shown any).\n"
        "Return ONLY valid JSON with the required schema.\n"
    )

    user = {
        "question": question,
        "gold_answer": gold_answer,
        "expected_episode_ids": expected_episode_ids,
        "candidate_answer": answer,
        "rubric": {
            "correctness_0_to_5": "Matches gold key facts and does not contradict.",
            "completeness_0_to_5": "Covers the required parts of the question relative to the gold.",
        },
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


def validate_judge_output(obj: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty if valid enough)."""
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["judge_output_not_dict"]

    def _int_in_range(key: str, lo: int, hi: int) -> None:
        v = obj.get(key)
        if not isinstance(v, int):
            errs.append(f"{key}:not_int")
            return
        if v < lo or v > hi:
            errs.append(f"{key}:out_of_range:{v}")

    for k in ("correctness", "groundedness", "completeness", "hallucination"):
        _int_in_range(k, 0, 5)
    _int_in_range("overall", 0, 100)

    verdict = obj.get("verdict")
    if verdict not in {"correct", "partially_correct", "incorrect", "idk_preferred"}:
        errs.append("verdict:invalid")

    for k in ("unsupported_claims", "missing_points"):
        v = obj.get(k)
        if not isinstance(v, list):
            errs.append(f"{k}:not_list")

    notes = obj.get("notes")
    if not (notes is None or isinstance(notes, str)):
        errs.append("notes:not_str")

    return errs


def validate_grounding_output(obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["judge_output_not_dict"]

    def _int_in_range(key: str, lo: int, hi: int) -> None:
        v = obj.get(key)
        if not isinstance(v, int):
            errs.append(f"{key}:not_int")
            return
        if v < lo or v > hi:
            errs.append(f"{key}:out_of_range:{v}")

    _int_in_range("groundedness", 0, 5)
    _int_in_range("hallucination", 0, 5)

    cs = obj.get("context_sufficiency")
    if cs not in {"sufficient", "insufficient", "unclear"}:
        errs.append("context_sufficiency:invalid")

    v = obj.get("unsupported_claims")
    if not isinstance(v, list):
        errs.append("unsupported_claims:not_list")

    notes = obj.get("notes")
    if not (notes is None or isinstance(notes, str)):
        errs.append("notes:not_str")

    return errs


def validate_correctness_output(obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["judge_output_not_dict"]

    def _int_in_range(key: str, lo: int, hi: int) -> None:
        v = obj.get(key)
        if not isinstance(v, int):
            errs.append(f"{key}:not_int")
            return
        if v < lo or v > hi:
            errs.append(f"{key}:out_of_range:{v}")

    _int_in_range("correctness", 0, 5)
    _int_in_range("completeness", 0, 5)

    v = obj.get("missing_points")
    if not isinstance(v, list):
        errs.append("missing_points:not_list")

    notes = obj.get("notes")
    if not (notes is None or isinstance(notes, str)):
        errs.append("notes:not_str")

    return errs


def _compute_overall_0_100(*, correctness: int, groundedness: int, completeness: int, hallucination: int) -> int:
    c = max(0, min(5, int(correctness)))
    g = max(0, min(5, int(groundedness)))
    comp = max(0, min(5, int(completeness)))
    h = max(0, min(5, int(hallucination)))
    score = (c / 5.0) * 40.0 + (g / 5.0) * 30.0 + (comp / 5.0) * 20.0 + (h / 5.0) * 10.0
    return int(round(max(0.0, min(100.0, score))))


def _merge_two_pass(
    *,
    question: str,
    answer: str,
    grounding: Dict[str, Any],
    correctness: Dict[str, Any],
) -> Dict[str, Any]:
    groundedness = int(grounding.get("groundedness"))
    hallucination = int(grounding.get("hallucination"))
    context_sufficiency = str(grounding.get("context_sufficiency") or "").strip().lower()
    unsupported_claims = grounding.get("unsupported_claims") if isinstance(grounding.get("unsupported_claims"), list) else []

    corr = int(correctness.get("correctness"))
    completeness = int(correctness.get("completeness"))
    missing_points = correctness.get("missing_points") if isinstance(correctness.get("missing_points"), list) else []

    overall = _compute_overall_0_100(
        correctness=corr,
        groundedness=groundedness,
        completeness=completeness,
        hallucination=hallucination,
    )

    # Verdict heuristic (keep stable categories used by dashboard).
    verdict: str
    if said_idk(answer) and groundedness >= 4 and context_sufficiency in {"insufficient", "unclear"}:
        verdict = "idk_preferred"
    elif corr >= 4 and completeness >= 4 and groundedness >= 4 and hallucination >= 4:
        verdict = "correct"
    elif corr >= 3 and groundedness >= 3:
        verdict = "partially_correct"
    else:
        verdict = "incorrect"

    # Keep notes short and audit-friendly.
    g_note = str(grounding.get("notes") or "").strip()
    c_note = str(correctness.get("notes") or "").strip()
    notes = ""
    if g_note and c_note:
        notes = f"grounding: {g_note} | correctness: {c_note}"
    else:
        notes = g_note or c_note
    if len(notes) > 500:
        notes = notes[:500] + "…"

    # Include context_sufficiency as an extra key; consumers can ignore it.
    return {
        "correctness": int(max(0, min(5, corr))),
        "groundedness": int(max(0, min(5, groundedness))),
        "completeness": int(max(0, min(5, completeness))),
        "hallucination": int(max(0, min(5, hallucination))),
        "overall": int(overall),
        "verdict": verdict,
        "unsupported_claims": unsupported_claims,
        "missing_points": missing_points,
        "context_sufficiency": context_sufficiency,
        "notes": notes,
        "question": question,
    }


def judge_with_retries(
    llm: ChatOpenAI,
    *,
    system: str,
    user: str,
    max_retries: int = 2,
) -> Tuple[Dict[str, Any], str]:
    """Invoke the judge and return (parsed_json, raw_text).

    The judge occasionally returns non-JSON despite instructions; retry a couple times
    to avoid failing an entire run.
    """
    last_text = ""
    last_exc: Optional[Exception] = None
    for attempt in range(int(max_retries) + 1):
        try:
            last_text = str(llm.invoke([("system", system), ("human", user)]).content or "")
            return parse_judge_json(last_text), last_text
        except Exception as e:
            last_exc = e
            if attempt >= int(max_retries):
                raise

            # Lightweight backoff to avoid hammering when rate-limited.
            # Keep it short so the scorer remains responsive.
            sleep_s = min(8.0, 0.75 * (2 ** attempt))
            time.sleep(sleep_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def score_case(
    llm: ChatOpenAI,
    *,
    case: Dict[str, Any],
    gold: Dict[str, Any],
    deterministic_only: bool = False,
    judge_max_retries: int = 2,
    judge_mode: str = "single",
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

    if deterministic_only:
        return {"deterministic": det}

    context = build_context_from_results(case)

    judge_mode_norm = str(judge_mode or "single").strip().lower()
    if judge_mode_norm not in {"single", "two_pass", "two-pass", "2pass"}:
        judge_mode_norm = "single"

    if judge_mode_norm == "single":
        system, user = judge_prompt(
            question=question,
            answer=answer,
            gold_answer=gold_answer,
            expected_episode_ids=expected_episode_ids,
            context=context,
        )

        try:
            judge, raw = judge_with_retries(llm, system=system, user=user, max_retries=int(judge_max_retries))
            validation_errors = validate_judge_output(judge if isinstance(judge, dict) else {})
            return {
                "deterministic": det,
                "judge": judge,
                "judge_raw_text": raw,
                "judge_validation": {
                    "ok": (len(validation_errors) == 0),
                    "errors": validation_errors,
                },
            }
        except Exception as e:
            # Keep going even if the judge fails on this case.
            return {
                "deterministic": det,
                "judge_error": f"{type(e).__name__}: {e}",
            }

    # --- Two-pass judge ---
    # Pass A: groundedness + hallucination vs retrieved context only.
    sys_a, user_a = judge_grounding_prompt(question=question, answer=answer, context=context)
    # Pass B: correctness + completeness vs gold only (no retrieved context).
    sys_b, user_b = judge_correctness_prompt(
        question=question,
        answer=answer,
        gold_answer=gold_answer,
        expected_episode_ids=expected_episode_ids,
    )

    try:
        a_obj, a_raw = judge_with_retries(llm, system=sys_a, user=user_a, max_retries=int(judge_max_retries))
        a_errors = validate_grounding_output(a_obj if isinstance(a_obj, dict) else {})

        b_obj, b_raw = judge_with_retries(llm, system=sys_b, user=user_b, max_retries=int(judge_max_retries))
        b_errors = validate_correctness_output(b_obj if isinstance(b_obj, dict) else {})

        merged = _merge_two_pass(question=question, answer=answer, grounding=a_obj, correctness=b_obj)
        merged_errors = validate_judge_output(merged)

        return {
            "deterministic": det,
            "judge": merged,
            "judge_two_pass": {
                "grounding": a_obj,
                "correctness": b_obj,
                "raw": {
                    "grounding": a_raw,
                    "correctness": b_raw,
                },
                "validation": {
                    "grounding_ok": (len(a_errors) == 0),
                    "grounding_errors": a_errors,
                    "correctness_ok": (len(b_errors) == 0),
                    "correctness_errors": b_errors,
                    "merged_ok": (len(merged_errors) == 0),
                    "merged_errors": merged_errors,
                },
            },
            # Keep legacy fields too (useful in dashboards expecting them).
            "judge_raw_text": None,
            "judge_validation": {
                "ok": (len(merged_errors) == 0),
                "errors": merged_errors,
            },
        }
    except Exception as e:
        return {
            "deterministic": det,
            "judge_error": f"two_pass:{type(e).__name__}: {e}",
        }


def score_run(
    llm: ChatOpenAI,
    *,
    run_obj: Dict[str, Any],
    gold_by_id: Dict[str, Dict[str, Any]],
    deterministic_only: bool = False,
    judge_max_retries: int = 2,
    judge_mode: str = "single",
    scoring_meta: Optional[Dict[str, Any]] = None,
    progress_every: int = 0,
) -> Dict[str, Any]:
    cases = safe_get(run_obj, "cases", [])
    if not isinstance(cases, list):
        raise ValueError("Run file missing cases[]")

    scored_cases: List[Dict[str, Any]] = []
    overall_scores: List[int] = []

    det_episode_ok: List[bool] = []
    det_must_include_ok: List[bool] = []
    det_forbidden_hit: List[bool] = []

    run_name = str(safe_get(run_obj, "run.run_name", "") or "")
    run_id = str(safe_get(run_obj, "run.run_id", "") or "")

    pe = int(progress_every)
    if pe < 0:
        pe = 0

    if pe and not deterministic_only:
        print(f"Scoring run: {run_name or run_id or '(unknown)'} | cases={len(cases)} | judge_mode={judge_mode}", flush=True)

    for i, c in enumerate(cases, start=1):
        cid = str(safe_get(c, "case_id", ""))
        gold = gold_by_id.get(cid)
        if not gold:
            scored_cases.append({"case_id": cid, "error": "missing_gold"})
            continue

        if pe and (i == 1 or i % pe == 0) and not deterministic_only:
            print(f"  case {i}/{len(cases)}: {cid}", flush=True)

        try:
            scored = score_case(
                llm,
                case=c,
                gold=gold,
                deterministic_only=bool(deterministic_only),
                judge_max_retries=int(judge_max_retries),
                judge_mode=str(judge_mode),
            )
        except Exception as e:
            scored = {"error": f"{type(e).__name__}: {e}"}

        scored_cases.append({"case_id": cid, **scored})

        det0 = scored.get("deterministic")
        if isinstance(det0, dict):
            if isinstance(det0.get("episode_ok"), bool):
                det_episode_ok.append(bool(det0["episode_ok"]))
            if isinstance(det0.get("must_include_ok"), bool):
                det_must_include_ok.append(bool(det0["must_include_ok"]))
            if isinstance(det0.get("forbidden_hit"), bool):
                det_forbidden_hit.append(bool(det0["forbidden_hit"]))

        overall = scored.get("judge", {}).get("overall")
        if isinstance(overall, int):
            overall_scores.append(overall)

    avg_overall = round(sum(overall_scores) / len(overall_scores)) if overall_scores else None

    def _rate(xs: List[bool]) -> Optional[float]:
        if not xs:
            return None
        return float(sum(1 for x in xs if x) / len(xs))

    out = {
        "run": safe_get(run_obj, "run", {}),
        "config": safe_get(run_obj, "config", {}),
        "scoring_meta": (scoring_meta or None),
        "score_summary": {
            "cases_scored": len(overall_scores),
            "avg_overall": avg_overall,
            "overall_scores": overall_scores,
        },
        "deterministic_summary": {
            "cases_total": len(scored_cases),
            "episode_ok_rate": _rate(det_episode_ok),
            "must_include_ok_rate": _rate(det_must_include_ok),
            "forbidden_hit_rate": _rate(det_forbidden_hit),
        },
        "scored_cases": scored_cases,
    }

    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Score RAG run logs using gold answers + OpenAI judge.")
    parser.add_argument("--runs-dir", default="experiments/runs", help="Directory containing run JSON logs")
    parser.add_argument("--gold", default="experiments/gold_answers.json", help="Gold answers JSON file")
    parser.add_argument("--out-dir", default="experiments/scored_runs", help="Where to write scored JSON files")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scoring if the scored output file already exists (useful for resuming a bulk run)",
    )

    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="OpenAI model to use as judge")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Skip LLM judge calls and write deterministic checks only (useful when quota is unavailable)",
    )

    parser.add_argument(
        "--judge-mode",
        default="single",
        choices=["single", "two_pass"],
        help=(
            "Judge mode. 'single' uses gold+context in one call. "
            "'two_pass' runs context-only groundedness + gold-only correctness and merges results."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N cases per run (0 disables; recommended 5-10 for long judge runs)",
    )
    parser.add_argument("--judge-timeout", type=float, default=60.0, help="Judge request timeout (seconds)")
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=2,
        help="Max retries for judge call/parsing (does not include OpenAI SDK internal retries)",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=0,
        help="Max retries inside OpenAI client (0 avoids long sleep loops on 429)",
    )

    parser.add_argument(
        "--run-name-prefix",
        action="append",
        default=[],
        help=(
            "Only score runs whose run.run_name starts with this prefix. "
            "Can be provided multiple times. If omitted, scores all runs found."
        ),
    )
    parser.add_argument(
        "--run-id-prefix",
        action="append",
        default=[],
        help=(
            "Only score runs whose run.run_id starts with this prefix. "
            "Can be provided multiple times. If omitted, no run_id filtering is applied."
        ),
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

    llm = ChatOpenAI(
        model=args.judge_model,
        temperature=args.temperature,
        timeout=float(args.judge_timeout) if args.judge_timeout is not None else None,
        max_retries=int(args.openai_max_retries) if args.openai_max_retries is not None else None,
    )

    run_files = sorted(runs_dir.rglob("*.json"))
    if not run_files:
        raise FileNotFoundError(f"No run json files under {runs_dir}")

    for rf in run_files:
        try:
            run_obj = read_json(rf)
            if not (isinstance(run_obj, dict) and "run" in run_obj and "cases" in run_obj):
                continue

            run_name = str(safe_get(run_obj, "run.run_name", "") or "")
            run_id = str(safe_get(run_obj, "run.run_id", rf.stem) or "")

            name_prefixes = [str(x) for x in (args.run_name_prefix or []) if str(x).strip()]
            if name_prefixes and not any(run_name.startswith(p) for p in name_prefixes):
                continue

            id_prefixes = [str(x) for x in (args.run_id_prefix or []) if str(x).strip()]
            if id_prefixes and not any(run_id.startswith(p) for p in id_prefixes):
                continue

            out_path = out_dir / f"{run_id}.scored.json"
            if bool(args.skip_existing) and out_path.exists():
                print(f"Skipping existing: {out_path}")
                continue

            scored = score_run(
                llm,
                run_obj=run_obj,
                gold_by_id=gold_by_id,
                deterministic_only=bool(args.deterministic_only),
                judge_max_retries=int(args.judge_max_retries),
                judge_mode=str(args.judge_mode),
                scoring_meta={
                    "scored_at_utc": utc_now_iso(),
                    "scoring_schema_version": "v2",
                    "judge_mode": str(args.judge_mode),
                    "judge_model": str(args.judge_model),
                    "judge_temperature": float(args.temperature),
                    "judge_timeout_s": float(args.judge_timeout) if args.judge_timeout is not None else None,
                    "judge_max_retries": int(args.judge_max_retries),
                    "openai_max_retries": int(args.openai_max_retries) if args.openai_max_retries is not None else None,
                    "deterministic_only": bool(args.deterministic_only),
                    "gold_file": str(gold_path),
                    "runs_dir": str(runs_dir),
                },
                progress_every=int(args.progress_every),
            )
            write_json(out_path, scored)
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"ERROR scoring {rf}: {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    main()