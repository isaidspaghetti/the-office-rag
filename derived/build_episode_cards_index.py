from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Allow importing repo modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT_ROOT = "derived/artifacts"
DEFAULT_PERSIST_DIR = "db/chroma_db_episode_cards"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _iter_episode_card_files(*, out_root: Path, build_prefix: str) -> Iterable[Path]:
    # Layout produced by derived.reduce_season: {prefix}_SxxEyy/episode_cards/SxxEyy.json
    pattern = f"{build_prefix}_S??E??/episode_cards/S??E??.json"
    yield from sorted(out_root.glob(pattern))


def _season_episode_from_id(episode_id: str) -> Tuple[Optional[int], Optional[int]]:
    s = (episode_id or "").strip().upper()
    if len(s) != 6 or not s.startswith("S") or "E" not in s:
        return None, None
    try:
        return int(s[1:3]), int(s[4:6])
    except Exception:
        return None, None


def _render_episode_card_for_index(card: Dict[str, Any]) -> str:
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

    # Keep evidence pointers (compact) so retrieval can route back.
    ev_ids: List[str] = []
    for t in threads[:8]:
        for ev in list(t.get("evidence") or [])[:2]:
            if not isinstance(ev, dict):
                continue
            seg = str(ev.get("segment_id") or "").strip()
            if seg:
                ev_ids.append(seg)
    if ev_ids:
        ev_ids = sorted(set(ev_ids))
        lines.append("Evidence segment_ids:")
        lines.append(", ".join(ev_ids[:80]))

    return "\n".join(lines).strip() + "\n"


def build_episode_cards_index(
    *,
    out_root: Path,
    persist_dir: Path,
    collection_name: Optional[str],
    embed_model: str,
    episode_cards_build_prefix: str,
    reset: bool,
) -> None:
    load_dotenv()

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    paths = list(_iter_episode_card_files(out_root=out_root, build_prefix=episode_cards_build_prefix))
    if not paths:
        raise FileNotFoundError(
            f"No episode card files found under {out_root} for build prefix {episode_cards_build_prefix}."
        )

    docs: List[Document] = []
    for p in paths:
        card = json.loads(p.read_text(encoding="utf-8"))
        if str(card.get("schema") or "") != "EpisodeDerivedCardV1":
            continue

        episode_id = str(card.get("episode_id") or "").strip().upper()
        season, episode = _season_episode_from_id(episode_id)
        title = str(card.get("title") or "").strip()
        build_id = str(card.get("build_id") or "").strip()

        docs.append(
            Document(
                page_content=_render_episode_card_for_index(card),
                metadata={
                    "doc_type": "derived",
                    "derived_type": "episode_card",
                    "episode_id": episode_id,
                    "season": season,
                    "episode": episode,
                    "title": title,
                    "build_id": build_id,
                    "source_file": str(p),
                    "episode_cards_build_prefix": episode_cards_build_prefix,
                },
            )
        )

    embeddings = OpenAIEmbeddings(model=str(embed_model))

    kwargs: Dict[str, Any] = {
        "documents": docs,
        "embedding": embeddings,
        "persist_directory": str(persist_dir),
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if collection_name:
        kwargs["collection_name"] = collection_name

    _ = Chroma.from_documents(**kwargs)

    meta = {
        "episode_cards_build_prefix": episode_cards_build_prefix,
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embed_model": str(embed_model),
        "docs_indexed": len(docs),
    }
    (persist_dir / "_build_meta.json").write_text(_safe_json(meta) + "\n", encoding="utf-8")

    print(f"Indexed {len(docs)} episode cards into {persist_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a Chroma index from EpisodeDerivedCardV1 JSON artifacts")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    p.add_argument("--collection-name", default=None)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--episode-cards-build-prefix", required=True)
    p.add_argument("--reset", action="store_true")

    args = p.parse_args()

    out_root = _resolve_under_repo(str(args.out_root))
    persist_dir = _resolve_under_repo(str(args.persist_dir))

    build_episode_cards_index(
        out_root=out_root,
        persist_dir=persist_dir,
        collection_name=(str(args.collection_name).strip() if args.collection_name else None),
        embed_model=str(args.embed_model),
        episode_cards_build_prefix=str(args.episode_cards_build_prefix).strip(),
        reset=bool(args.reset),
    )


if __name__ == "__main__":
    main()
