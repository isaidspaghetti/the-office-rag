from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Allow importing repo modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT_ROOT = "derived/artifacts"
DEFAULT_PERSIST_DIR = "db/chroma_db_season_cards"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _resolve_under_repo(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _iter_season_card_files(*, out_root: Path, build_prefix: str) -> Iterable[Path]:
    # Layout produced by derived.reduce_season_card: {prefix}/season_cards/seasonXX.json
    yield from sorted((out_root / build_prefix).glob("season_cards/season*.json"))


def _render_season_card_for_index(card: Dict[str, Any]) -> str:
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


def build_season_cards_index(
    *,
    out_root: Path,
    persist_dir: Path,
    collection_name: Optional[str],
    embed_model: str,
    season_cards_build_prefix: str,
    reset: bool,
) -> None:
    load_dotenv()

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    paths = list(_iter_season_card_files(out_root=out_root, build_prefix=season_cards_build_prefix))
    if not paths:
        raise FileNotFoundError(
            f"No season card files found under {out_root}/{season_cards_build_prefix}/season_cards/."
        )

    docs: List[Document] = []
    for p in paths:
        card = json.loads(p.read_text(encoding="utf-8"))
        if str(card.get("schema") or "") != "SeasonDerivedCardV1":
            continue

        season = int(card.get("season") or 0)
        build_id = str(card.get("build_id") or "").strip()

        docs.append(
            Document(
                page_content=_render_season_card_for_index(card),
                metadata={
                    "doc_type": "derived",
                    "derived_type": "season_card",
                    "season": season,
                    "build_id": build_id,
                    "source_file": str(p),
                    "season_cards_build_prefix": season_cards_build_prefix,
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
        "season_cards_build_prefix": season_cards_build_prefix,
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embed_model": str(embed_model),
        "docs_indexed": len(docs),
    }
    (persist_dir / "_build_meta.json").write_text(_safe_json(meta) + "\n", encoding="utf-8")

    print(f"Indexed {len(docs)} season cards into {persist_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a Chroma index from SeasonDerivedCardV1 JSON artifacts"
    )
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    p.add_argument("--collection-name", default=None)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--season-cards-build-prefix", required=True)
    p.add_argument("--reset", action="store_true")

    args = p.parse_args()

    out_root = _resolve_under_repo(str(args.out_root))
    persist_dir = _resolve_under_repo(str(args.persist_dir))

    build_season_cards_index(
        out_root=out_root,
        persist_dir=persist_dir,
        collection_name=(str(args.collection_name).strip() if args.collection_name else None),
        embed_model=str(args.embed_model),
        season_cards_build_prefix=str(args.season_cards_build_prefix).strip(),
        reset=bool(args.reset),
    )


if __name__ == "__main__":
    main()
