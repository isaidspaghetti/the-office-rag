from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

# Allow importing repo modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT_ROOT = "derived/artifacts"
DEFAULT_PERSIST_DIR = "db/chroma_db_derived_cards"
DEFAULT_COLLECTION_NAME = "derived_cards"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def parse_csv_list(s: str) -> List[str]:
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


def _season_episode_from_id(episode_id: str) -> Tuple[Optional[int], Optional[int]]:
    s = (episode_id or "").strip().upper()
    if len(s) != 6 or not s.startswith("S") or "E" not in s:
        return None, None
    try:
        return int(s[1:3]), int(s[4:6])
    except Exception:
        return None, None


def _render_episode_card(card: Dict[str, Any]) -> str:
    episode_id = str(card.get("episode_id") or "").strip().upper()
    title = str(card.get("title") or "").strip()
    synopsis = str(card.get("one_paragraph_synopsis") or "").strip()

    lines: List[str] = []
    lines.append(f"Episode card: {episode_id} — {title}".strip())
    if synopsis:
        lines.append("Synopsis:")
        lines.append(synopsis)

    threads = list(card.get("main_threads") or [])
    if threads:
        lines.append("Main threads:")
        for t in threads[:10]:
            thread = str(t.get("thread") or "").strip()
            if not thread:
                continue
            lines.append(f"- {thread}")

    chars = list(card.get("character_highlights") or [])
    if chars:
        lines.append("Character highlights:")
        for c in chars[:20]:
            character = str(c.get("character") or "").strip()
            what_changes = str(c.get("what_changes") or "").strip()
            if not character or not what_changes:
                continue
            lines.append(f"- {character}: {what_changes}")

    # Keep compact evidence pointers for back-routing.
    seg_ids: List[str] = []
    for t in threads[:8]:
        for ev in list(t.get("evidence") or [])[:2]:
            if not isinstance(ev, dict):
                continue
            seg = str(ev.get("segment_id") or "").strip()
            if seg:
                seg_ids.append(seg)
    if seg_ids:
        lines.append("Evidence segment_ids:")
        lines.append(", ".join(sorted(set(seg_ids))[:120]))

    return "\n".join(lines).strip() + "\n"


def _render_season_card(card: Dict[str, Any]) -> str:
    season = int(card.get("season") or 0)
    overview = str(card.get("overview") or "").strip()

    lines: List[str] = []
    lines.append(f"Season card: Season {season:02d}")
    if overview:
        lines.append("Overview:")
        lines.append(overview)

    def _section(title: str, items: List[Dict[str, Any]], key1: str) -> None:
        if not items:
            return
        lines.append(f"{title}:")
        for it in items[:20]:
            a = str(it.get(key1) or "").strip()
            summary = str(it.get("summary") or "").strip()
            if not a or not summary:
                continue
            lines.append(f"- {a}: {summary}")

    _section("Major arcs", list(card.get("major_arcs") or []), "arc")
    _section("Character arcs", list(card.get("character_arcs") or []), "character")
    _section("Running gags", list(card.get("running_gags") or []), "gag")

    oq = list(card.get("open_questions") or [])
    if oq:
        lines.append("Open questions:")
        for q in oq[:20]:
            qs = str(q or "").strip()
            if qs:
                lines.append(f"- {qs}")

    return "\n".join(lines).strip() + "\n"


def _render_topic_card(card: Dict[str, Any]) -> str:
    topic_id = str(card.get("topic_id") or "").strip()
    topic_type = str(card.get("topic_type") or "").strip()
    entities = [str(x).strip() for x in (card.get("entity_names") or []) if str(x).strip()]
    episode_ids = [str(x).strip().upper() for x in (card.get("episode_ids") or []) if str(x).strip()]

    lines: List[str] = []
    header = "Topic card"
    if topic_type:
        header += f" ({topic_type})"
    if topic_id:
        header += f": {topic_id}"
    lines.append(header)
    if entities:
        lines.append("Entities: " + ", ".join(entities))
    if episode_ids:
        lines.append("Episode IDs: " + ", ".join(episode_ids[:80]))

    facts = list(card.get("bullet_facts") or [])
    if facts:
        lines.append("Facts:")
        for f in facts[:30]:
            if not isinstance(f, dict):
                continue
            fact = str(f.get("fact") or "").strip()
            if fact:
                lines.append(f"- {fact}")

    missing = [str(x).strip() for x in (card.get("missing") or []) if str(x).strip()]
    if missing:
        lines.append("Missing:")
        for m in missing[:20]:
            lines.append(f"- {m}")

    return "\n".join(lines).strip() + "\n"


def _iter_episode_card_files(out_root: Path, build_prefix: str) -> Iterable[Path]:
    # Layout: {prefix}_SxxEyy/episode_cards/SxxEyy.json
    pattern = f"{build_prefix}_S??E??/episode_cards/S??E??.json"
    yield from sorted(out_root.glob(pattern))


def _iter_season_card_files(out_root: Path, build_prefix: str) -> Iterable[Path]:
    # Layout: {prefix}/season_cards/seasonXX.json
    yield from sorted((out_root / build_prefix).glob("season_cards/season*.json"))


def _iter_topic_card_files(out_root: Path, build_prefix: str) -> Iterable[Path]:
    # Layout: {build_prefix}/topic_cards/<topic_id>.json
    yield from sorted((out_root / build_prefix).glob("topic_cards/*.json"))


def _auto_detect_episode_card_prefixes(out_root: Path) -> List[str]:
    # Only include base season dirs, not per-episode dirs.
    prefixes: List[str] = []
    for p in sorted(out_root.glob("derived_episodecard_*")):
        if not p.is_dir():
            continue
        if "_S" in p.name:
            continue
        if (p / "season_manifest.json").exists():
            prefixes.append(p.name)
    return prefixes


def _auto_detect_season_card_prefixes(out_root: Path) -> List[str]:
    prefixes: List[str] = []
    for p in sorted(out_root.glob("derived_seasoncard_*")):
        if not p.is_dir():
            continue
        if (p / "season_cards").exists():
            prefixes.append(p.name)
    return prefixes


def _auto_detect_topic_card_prefixes(out_root: Path) -> List[str]:
    prefixes: List[str] = []
    for p in sorted(out_root.glob("topiccards_*")):
        if not p.is_dir():
            continue
        if (p / "topic_cards").exists():
            # Require at least one card.
            if list((p / "topic_cards").glob("*.json")):
                prefixes.append(p.name)
    return prefixes


@dataclass(frozen=True)
class BuildConfig:
    out_root: Path
    persist_dir: Path
    collection_name: str
    embed_model: str
    episode_cards_build_prefixes: List[str]
    season_cards_build_prefixes: List[str]
    topic_cards_build_prefixes: List[str]
    include_episode_cards: bool
    include_season_cards: bool
    include_topic_cards: bool
    seasons_filter: Optional[List[int]]
    reset: bool
    dry_run: bool


def _passes_season_filter(season: Optional[int], seasons_filter: Optional[List[int]]) -> bool:
    if not seasons_filter:
        return True
    if season is None:
        return False
    return int(season) in set(int(x) for x in seasons_filter)


def build_derived_cards_index(cfg: BuildConfig) -> Dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv()

    if not os.environ.get("OPENAI_API_KEY") and not cfg.dry_run:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or your environment.")

    if cfg.reset and cfg.persist_dir.exists():
        shutil.rmtree(cfg.persist_dir)

    docs: List[Dict[str, Any]] = []

    if cfg.include_episode_cards:
        for pref in cfg.episode_cards_build_prefixes:
            paths = list(_iter_episode_card_files(cfg.out_root, pref))
            if not paths:
                continue
            for p in paths:
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "EpisodeDerivedCardV1":
                    continue

                episode_id = str(card.get("episode_id") or "").strip().upper()
                season, episode = _season_episode_from_id(episode_id)
                if not _passes_season_filter(season, cfg.seasons_filter):
                    continue

                title = str(card.get("title") or "").strip()
                build_id = str(card.get("build_id") or "").strip()
                stable_build_tag = build_id or pref

                docs.append(
                    {
                        "page_content": _render_episode_card(card),
                        "metadata": {
                            "doc_type": "derived",
                            "derived_type": "episode_card",
                            "episode_id": episode_id,
                            "season": season,
                            "episode": episode,
                            "title": title,
                            "build_id": build_id,
                            "source_file": str(p),
                            "episode_cards_build_prefix": pref,
                            "stable_doc_id": f"episode_card:{episode_id}:{stable_build_tag}",
                        },
                    }
                )

    if cfg.include_season_cards:
        for pref in cfg.season_cards_build_prefixes:
            paths = list(_iter_season_card_files(cfg.out_root, pref))
            if not paths:
                continue
            for p in paths:
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "SeasonDerivedCardV1":
                    continue

                season = int(card.get("season") or 0) or None
                if not _passes_season_filter(season, cfg.seasons_filter):
                    continue

                build_id = str(card.get("build_id") or "").strip()
                stable_build_tag = build_id or pref

                docs.append(
                    {
                        "page_content": _render_season_card(card),
                        "metadata": {
                            "doc_type": "derived",
                            "derived_type": "season_card",
                            "season": season,
                            "build_id": build_id,
                            "source_file": str(p),
                            "season_cards_build_prefix": pref,
                            "stable_doc_id": f"season_card:{int(season):02d}:{stable_build_tag}",
                        },
                    }
                )

    if cfg.include_topic_cards:
        for pref in cfg.topic_cards_build_prefixes:
            paths = list(_iter_topic_card_files(cfg.out_root, pref))
            if not paths:
                continue
            for p in paths:
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "TopicCardV1":
                    continue

                topic_id = str(card.get("topic_id") or "").strip()
                topic_type = str(card.get("topic_type") or "").strip()
                entities = [str(x).strip() for x in (card.get("entity_names") or []) if str(x).strip()]
                episode_ids = [str(x).strip().upper() for x in (card.get("episode_ids") or []) if str(x).strip()]
                build_id = str(card.get("build_id") or "").strip()
                stable_build_tag = build_id or pref

                docs.append(
                    {
                        "page_content": _render_topic_card(card),
                        "metadata": {
                            "doc_type": "derived",
                            "derived_type": "topic_card",
                            "topic_id": topic_id,
                            "topic_type": topic_type,
                            "entity_names": entities,
                            "episode_ids": episode_ids,
                            "build_id": build_id,
                            "source_file": str(p),
                            "topic_cards_build_prefix": pref,
                            "stable_doc_id": f"topic_card:{topic_id}:{stable_build_tag}",
                        },
                    }
                )

    summary: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "out_root": str(cfg.out_root),
        "persist_dir": str(cfg.persist_dir),
        "collection_name": str(cfg.collection_name),
        "embed_model": str(cfg.embed_model),
        "include_episode_cards": bool(cfg.include_episode_cards),
        "include_season_cards": bool(cfg.include_season_cards),
        "include_topic_cards": bool(cfg.include_topic_cards),
        "episode_cards_build_prefixes": cfg.episode_cards_build_prefixes,
        "season_cards_build_prefixes": cfg.season_cards_build_prefixes,
        "topic_cards_build_prefixes": cfg.topic_cards_build_prefixes,
        "seasons_filter": cfg.seasons_filter,
        "docs_planned": len(docs),
        "dry_run": bool(cfg.dry_run),
    }

    if cfg.dry_run:
        return summary

    try:
        from langchain_chroma import Chroma  # type: ignore
        from langchain_core.documents import Document  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing indexing dependencies. Install with: pip install langchain-chroma langchain-openai langchain-core chromadb"
        ) from e

    embeddings = OpenAIEmbeddings(model=str(cfg.embed_model))

    lc_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]
    ids = [str(d["metadata"].get("stable_doc_id")) for d in docs]

    _ = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        persist_directory=str(cfg.persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
        collection_name=str(cfg.collection_name),
        ids=ids,
    )

    summary["docs_indexed"] = len(docs)

    cfg.persist_dir.mkdir(parents=True, exist_ok=True)
    (cfg.persist_dir / "_build_meta.json").write_text(_safe_json(summary) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build a Chroma index from derived card JSON artifacts")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    p.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)

    p.add_argument(
        "--episode-cards-build-prefixes",
        default="",
        help="Comma-separated episode-card build prefixes. If omitted, auto-detect under out-root.",
    )
    p.add_argument(
        "--season-cards-build-prefixes",
        default="",
        help="Comma-separated season-card build prefixes. If omitted, auto-detect under out-root.",
    )

    p.add_argument(
        "--topic-cards-build-prefixes",
        default="",
        help="Comma-separated topic-card build prefixes. If omitted, auto-detect under out-root.",
    )

    p.add_argument("--no-episode-cards", action="store_true", help="Do not index EpisodeDerivedCardV1")
    p.add_argument("--no-season-cards", action="store_true", help="Do not index SeasonDerivedCardV1")
    p.add_argument("--no-topic-cards", action="store_true", help="Do not index TopicCardV1")

    p.add_argument(
        "--seasons",
        default="",
        help="Optional comma-separated season numbers to include (e.g. 1,2,4).",
    )

    p.add_argument("--reset", action="store_true", help="Delete persist dir before rebuilding")
    p.add_argument("--dry-run", action="store_true", help="Print what would be indexed; do not embed/write")

    args = p.parse_args()

    out_root = _resolve_under_repo(str(args.out_root))
    persist_dir = _resolve_under_repo(str(args.persist_dir))

    episode_prefixes = parse_csv_list(str(args.episode_cards_build_prefixes))
    season_prefixes = parse_csv_list(str(args.season_cards_build_prefixes))
    topic_prefixes = parse_csv_list(str(args.topic_cards_build_prefixes))

    include_episode_cards = not bool(args.no_episode_cards)
    include_season_cards = not bool(args.no_season_cards)
    include_topic_cards = not bool(args.no_topic_cards)

    if include_episode_cards and not episode_prefixes:
        episode_prefixes = _auto_detect_episode_card_prefixes(out_root)

    if include_season_cards and not season_prefixes:
        season_prefixes = _auto_detect_season_card_prefixes(out_root)

    if include_topic_cards and not topic_prefixes:
        topic_prefixes = _auto_detect_topic_card_prefixes(out_root)

    seasons_filter: Optional[List[int]] = None
    seasons_s = parse_csv_list(str(args.seasons))
    if seasons_s:
        seasons_filter = [int(x) for x in seasons_s]

    summary = build_derived_cards_index(
        BuildConfig(
            out_root=out_root,
            persist_dir=persist_dir,
            collection_name=str(args.collection_name).strip() or DEFAULT_COLLECTION_NAME,
            embed_model=str(args.embed_model),
            episode_cards_build_prefixes=episode_prefixes,
            season_cards_build_prefixes=season_prefixes,
            topic_cards_build_prefixes=topic_prefixes,
            include_episode_cards=include_episode_cards,
            include_season_cards=include_season_cards,
            include_topic_cards=include_topic_cards,
            seasons_filter=seasons_filter,
            reset=bool(args.reset),
            dry_run=bool(args.dry_run),
        )
    )

    print(_safe_json(summary))


if __name__ == "__main__":
    main()
