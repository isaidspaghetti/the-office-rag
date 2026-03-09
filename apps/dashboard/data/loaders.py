from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from apps.dashboard.data.transforms import safe_get


@dataclass(frozen=True)
class RunEntry:
    path: str
    obj: Dict[str, Any]


@st.cache_data(show_spinner=False)
def load_run_entries(*, runs_dir: str, max_files: int = 800) -> List[RunEntry]:
    p = Path(runs_dir)
    if not p.exists():
        return []

    files = sorted(p.rglob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if max_files and len(files) > int(max_files):
        files = files[: int(max_files)]

    out: List[RunEntry] = []
    for rf in files:
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "run" in obj and "cases" in obj:
                out.append(RunEntry(path=str(rf), obj=obj))
        except Exception:
            continue
    return out


def run_id_from_entry(entry: RunEntry) -> str:
    run = entry.obj.get("run") if isinstance(entry.obj.get("run"), dict) else {}
    rid = str((run or {}).get("run_id") or "").strip() if isinstance(run, dict) else ""
    if rid:
        return rid
    try:
        return Path(entry.path).stem
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_scored_summary(*, scored_dir: str) -> Dict[str, Dict[str, Any]]:
    p = Path(scored_dir)
    if not p.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for sf in sorted(p.glob("*.scored.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            obj = json.loads(sf.read_text(encoding="utf-8"))
            run = obj.get("run") if isinstance(obj, dict) else None
            rid = str((run or {}).get("run_id") or "") if isinstance(run, dict) else ""
            rid = rid.strip() or sf.name[: -len(".scored.json")]
            avg_overall = safe_get(obj, "score_summary.avg_overall", None)
            out[rid] = {
                "path": str(sf),
                "avg_overall": (int(avg_overall) if isinstance(avg_overall, int) else None),
            }
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def load_scored_obj(*, scored_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    p = Path(scored_dir) / f"{run_id}.scored.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def index_scored_cases_by_id(scored_obj: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(scored_obj, dict):
        return {}
    cases = scored_obj.get("scored_cases")
    if not isinstance(cases, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in cases:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("case_id") or "").strip()
        judge = row.get("judge")
        if not cid or not isinstance(judge, dict):
            continue
        out[cid] = judge
    return out
