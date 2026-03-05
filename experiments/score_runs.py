from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


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

    must_ok = True
    if must_include:
        # not strict "all" because stems like "impersonat" are allowed
        must_ok = any_substring_present(ans, must_include)

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
        "forbidden_hit": forbidden_hit,
        "forbidden_terms_hit": forbidden_terms_hit,
    }


def build_context_from_results(case: Dict[str, Any], *, max_docs: int = 6, max_chars: int = 8000) -> str:
    """
    Use retrieval previews as judge context. This keeps scoring cheap and avoids disk dependency.
    """
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
    for attempt in range(max_retries + 1):
        try:
            last_text = str(llm.invoke([("system", system), ("human", user)]).content or "")
            return parse_judge_json(last_text), last_text
        except Exception as e:
            if attempt >= max_retries:
                raise e
    raise RuntimeError("unreachable")


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

    context = build_context_from_results(case)

    system, user = judge_prompt(
        question=question,
        answer=answer,
        gold_answer=gold_answer,
        expected_episode_ids=expected_episode_ids,
        context=context,
    )

    try:
        judge, raw = judge_with_retries(llm, system=system, user=user, max_retries=2)
        return {
            "deterministic": det,
            "judge": judge,
        }
    except Exception as e:
        # Keep going even if the judge fails on this case.
        return {
            "deterministic": det,
            "judge_error": f"{type(e).__name__}: {e}",
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

        try:
            scored = score_case(llm, case=c, gold=gold)
        except Exception as e:
            scored = {"error": f"{type(e).__name__}: {e}"}

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
        "scored_cases": scored_cases,
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Score RAG run logs using gold answers + OpenAI judge.")
    parser.add_argument("--runs-dir", default="experiments/runs", help="Directory containing run JSON logs")
    parser.add_argument("--gold", default="experiments/gold_answers.json", help="Gold answers JSON file")
    parser.add_argument("--out-dir", default="experiments/scored_runs", help="Where to write scored JSON files")

    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="OpenAI model to use as judge")
    parser.add_argument("--temperature", type=float, default=0.0)

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

    llm = ChatOpenAI(model=args.judge_model, temperature=args.temperature)

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

            scored = score_run(llm, run_obj=run_obj, gold_by_id=gold_by_id)

            out_path = out_dir / f"{run_id}.scored.json"
            write_json(out_path, scored)
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"ERROR scoring {rf}: {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    main()