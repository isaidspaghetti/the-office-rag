from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from derived.segment_episode import EpisodeScript, iter_episode_scripts

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"
DEFAULT_OUT_ROOT = "derived/artifacts"
DEFAULT_LLM_MODEL = "gpt-4.1-nano"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _season_from_episode_id(episode_id: str) -> Optional[int]:
    s = (episode_id or "").strip().upper()
    if len(s) < 4 or not s.startswith("S"):
        return None
    try:
        return int(s[1:3])
    except Exception:
        return None


def _iter_season_episodes(*, docs_dir: Path, season: int) -> List[EpisodeScript]:
    out: List[EpisodeScript] = []
    for ep in iter_episode_scripts(docs_dir=docs_dir):
        s = _season_from_episode_id(ep.episode_id)
        if s == int(season):
            out.append(ep)
    return sorted(out, key=lambda e: e.episode_id)


def _escape_control_chars_in_json_strings(s: str) -> str:
    """Best-effort repair for JSON strings containing raw control characters."""

    if not s:
        return s

    out: List[str] = []
    in_string = False
    escape = False

    for ch in s:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            code = ord(ch)
            if code < 0x20:
                # Replace raw control chars inside strings.
                mapping = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
                out.append(mapping.get(ch, f"\\u{code:04x}"))
            else:
                out.append(ch)
            continue

        # not in string
        if ch == '"':
            out.append(ch)
            in_string = True
        else:
            out.append(ch)

    return "".join(out)


def _try_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        repaired = _escape_control_chars_in_json_strings(text)
        return json.loads(repaired)


def _compact_evidence(ev: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = str(ev.get("episode_id") or "").strip().upper()
    segment_id = str(ev.get("segment_id") or "").strip()
    snippet = str(ev.get("snippet") or "").strip().replace("\n", " ")
    if len(snippet) > 180:
        snippet = snippet[:177] + "..."

    out: Dict[str, Any] = {"episode_id": episode_id}
    if segment_id:
        out["segment_id"] = segment_id
    if snippet:
        out["snippet"] = snippet
    return out


def _compact_episode_card(card: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = str(card.get("episode_id") or "").strip().upper()
    title = str(card.get("title") or "").strip()

    synopsis = str(card.get("one_paragraph_synopsis") or "").strip().replace("\n", " ")
    if len(synopsis) > 600:
        synopsis = synopsis[:597] + "..."

    threads_in: List[Dict[str, Any]] = list(card.get("main_threads") or [])
    threads: List[Dict[str, Any]] = []
    for t in threads_in[:6]:
        thread = str(t.get("thread") or "").strip()
        evidence = [
            _compact_evidence(e)
            for e in (list(t.get("evidence") or [])[:2])
            if isinstance(e, dict)
        ]
        if thread:
            threads.append({"thread": thread, "evidence": evidence})

    chars_in: List[Dict[str, Any]] = list(card.get("character_highlights") or [])
    chars: List[Dict[str, Any]] = []
    for c in chars_in[:10]:
        character = str(c.get("character") or "").strip()
        what_changes = str(c.get("what_changes") or "").strip().replace("\n", " ")
        if len(what_changes) > 300:
            what_changes = what_changes[:297] + "..."
        evidence = [
            _compact_evidence(e)
            for e in (list(c.get("evidence") or [])[:2])
            if isinstance(e, dict)
        ]
        if character and what_changes:
            chars.append({"character": character, "what_changes": what_changes, "evidence": evidence})

    return {
        "episode_id": episode_id,
        "title": title,
        "synopsis": synopsis,
        "main_threads": threads,
        "character_highlights": chars,
    }


@dataclass(frozen=True)
class ReduceSeasonCardConfig:
    season: int
    docs_dir: Path
    out_root: Path
    episode_cards_build_prefix: str
    llm_model: str
    build_prefix: Optional[str]
    force: bool


def _episode_card_path(*, out_root: Path, episode_cards_build_prefix: str, episode_id: str) -> Path:
    ep = str(episode_id).strip().upper()
    return out_root / f"{episode_cards_build_prefix}_{ep}" / "episode_cards" / f"{ep}.json"


def reduce_season_card(cfg: ReduceSeasonCardConfig) -> Path:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    created_at = utc_now_iso()
    build_prefix = (cfg.build_prefix or "").strip()
    if not build_prefix:
        ts = created_at.replace(":", "-")
        build_prefix = f"derived_seasoncard_season{int(cfg.season):02d}_nano_{ts}"

    season_dir = cfg.out_root / build_prefix
    season_dir.mkdir(parents=True, exist_ok=True)

    out_file = season_dir / "season_cards" / f"season{int(cfg.season):02d}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    season_manifest_path = season_dir / "season_manifest.json"

    if out_file.exists() and not cfg.force:
        return season_dir

    episodes = _iter_season_episodes(docs_dir=cfg.docs_dir, season=int(cfg.season))

    manifest: Dict[str, Any] = {
        "schema": "SeasonSeasonCardRunManifestV1",
        "schema_version": 1,
        "build_prefix": build_prefix,
        "created_at_utc": created_at,
        "season": int(cfg.season),
        "config": {
            "docs_dir": str(cfg.docs_dir),
            "out_root": str(cfg.out_root),
            "episode_cards_build_prefix": str(cfg.episode_cards_build_prefix),
            "llm_model": str(cfg.llm_model),
            "force": bool(cfg.force),
        },
        "plan": {
            "planned_episodes": [
                {
                    "episode_id": e.episode_id,
                    "title": e.title,
                    "episode_card_file": str(
                        _episode_card_path(
                            out_root=cfg.out_root,
                            episode_cards_build_prefix=cfg.episode_cards_build_prefix,
                            episode_id=e.episode_id,
                        )
                    ),
                }
                for e in episodes
            ],
            "planned_total_llm_calls": 1,
        },
        "execution": {
            "status": "running",
            "started_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "episodes_loaded": 0,
            "episodes_missing": 0,
            "errors": 0,
            "error": None,
        },
    }

    season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    t0 = time.time()

    # Load episode cards
    compact_cards: List[Dict[str, Any]] = []
    missing: List[str] = []

    for ep in episodes:
        p = _episode_card_path(
            out_root=cfg.out_root,
            episode_cards_build_prefix=cfg.episode_cards_build_prefix,
            episode_id=ep.episode_id,
        )
        if not p.exists():
            missing.append(ep.episode_id)
            continue
        card = json.loads(p.read_text(encoding="utf-8"))
        compact_cards.append(_compact_episode_card(card))

    manifest["execution"]["episodes_loaded"] = len(compact_cards)
    manifest["execution"]["episodes_missing"] = len(missing)

    if not compact_cards:
        manifest["execution"]["status"] = "error"
        manifest["execution"]["errors"] = 1
        manifest["execution"]["error"] = (
            "No episode cards found to reduce. "
            "Check --episode-cards-build-prefix and confirm episode card files exist."
        )
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")
        raise FileNotFoundError(manifest["execution"]["error"])

    system = (
        "You are building a season-level derived card for a RAG system. "
        "You MUST ONLY use the provided episode-card inputs. "
        "If a claim is not supported, omit it or put it under open_questions. "
        "Return ONLY valid JSON. No markdown.\n\n"
        "CITATIONS: For every non-trivial claim, include at least one evidence pointer "
        "with episode_id and (when available) segment_id.\n\n"
        "STYLE: Be compact and factual; do not invent details."
    )

    user = {
        "season": int(cfg.season),
        "episodes_available": [c["episode_id"] for c in compact_cards],
        "episodes_missing": missing,
        "episode_cards": compact_cards,
        "output_schema": {
            "overview": "string",
            "major_arcs": [
                {
                    "arc": "string",
                    "summary": "string",
                    "episode_ids": ["SxxEyy"],
                    "evidence": [{"episode_id": "SxxEyy", "segment_id": "SxxEyy:seg:NNN"}],
                }
            ],
            "character_arcs": [
                {
                    "character": "string",
                    "summary": "string",
                    "episode_ids": ["SxxEyy"],
                    "evidence": [{"episode_id": "SxxEyy", "segment_id": "SxxEyy:seg:NNN"}],
                }
            ],
            "running_gags": [
                {
                    "gag": "string",
                    "summary": "string",
                    "episode_ids": ["SxxEyy"],
                    "evidence": [{"episode_id": "SxxEyy", "segment_id": "SxxEyy:seg:NNN"}],
                }
            ],
            "open_questions": ["string"],
        },
        "caps": {
            "major_arcs_max": 10,
            "character_arcs_max": 12,
            "running_gags_max": 12,
            "open_questions_max": 12,
        },
    }

    llm = ChatOpenAI(model=cfg.llm_model, temperature=0)
    try:
        llm = llm.bind(response_format={"type": "json_object"})
    except Exception:
        pass

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = _try_parse_json(text)

        season_card: Dict[str, Any] = {
            "schema": "SeasonDerivedCardV1",
            "schema_version": 1,
            "build_id": build_prefix,
            "created_at_utc": created_at,
            "llm_model": str(cfg.llm_model),
            "doc_type": "derived",
            "derived_type": "season_card",
            "season": int(cfg.season),
            "episode_ids": [e.episode_id for e in episodes],
            "episode_ids_included": [c["episode_id"] for c in compact_cards],
            "episode_ids_missing": missing,
            "source_episode_cards_build_prefix": str(cfg.episode_cards_build_prefix),
            "overview": str(body.get("overview") or "").strip(),
            "major_arcs": body.get("major_arcs") or [],
            "character_arcs": body.get("character_arcs") or [],
            "running_gags": body.get("running_gags") or [],
            "open_questions": body.get("open_questions") or [],
        }

        out_file.write_text(_safe_json(season_card) + "\n", encoding="utf-8")
        manifest["execution"]["status"] = "done" if manifest["execution"]["errors"] == 0 else "done_with_errors"

    except Exception as e:
        manifest["execution"]["status"] = "error"
        manifest["execution"]["errors"] = int(manifest["execution"]["errors"]) + 1
        manifest["execution"]["error"] = f"{type(e).__name__}: {e}"
        raise

    finally:
        manifest["execution"]["ended_at_utc"] = utc_now_iso()
        manifest["execution"]["duration_ms"] = int((time.time() - t0) * 1000)
        season_manifest_path.write_text(_safe_json(manifest) + "\n", encoding="utf-8")

    return season_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Reduce EpisodeDerivedCardV1 files into a SeasonDerivedCardV1")
    p.add_argument("--season", type=int, required=True, help="Season number (1-9)")
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--episode-cards-build-prefix", required=True)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--build-prefix", default=None, help="Output build prefix under out-root")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()

    out_root = _resolve_under_repo(str(args.out_root))
    docs_dir = _resolve_under_repo(str(args.docs_dir))

    out_dir = reduce_season_card(
        ReduceSeasonCardConfig(
            season=int(args.season),
            docs_dir=docs_dir,
            out_root=out_root,
            episode_cards_build_prefix=str(args.episode_cards_build_prefix).strip(),
            llm_model=str(args.llm_model).strip(),
            build_prefix=(str(args.build_prefix) if args.build_prefix else None),
            force=bool(args.force),
        )
    )

    print(f"Done. Season card under: {out_dir}")


if __name__ == "__main__":
    main()
