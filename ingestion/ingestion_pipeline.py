from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from chunk_documents import chunk_documents_by_scene
from load_documents import load_documents

# ---- Config you can edit ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DOCS_DIR = Path(__file__).resolve().parent / "normalized_docs_txt"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "db" / "chroma_db"
EMBED_MODEL = "text-embedding-3-small"


_EPISODE_IN_PATH_RE = re.compile(r"(?:^|[\\/])s(?P<season>\d{2})e(?P<episode>\d{2})(?:[_.-]|$)", re.IGNORECASE)


def _detect_doc_type_from_source(source: str) -> str:
    s = source.replace("\\", "/").lower()
    if "/summaries/" in s:
        return "summary"
    if "/scripts/" in s:
        return "script"
    return "unknown"


def _parse_episode_from_source(source: str) -> tuple[int | None, int | None, str | None]:
    """Best-effort parse of season/episode from our normalized filenames.

    Examples:
      .../scripts/season_06/s06e23_the_chump_script.txt -> (6, 23, "S06E23")
      .../summaries/season_07/s07e04_sex_ed_summary.txt -> (7, 4, "S07E04")
    """
    if not source:
        return None, None, None

    m = _EPISODE_IN_PATH_RE.search(source.replace("\\", "/"))
    if not m:
        return None, None, None

    try:
        season = int(m.group("season"))
        episode = int(m.group("episode"))
    except Exception:
        return None, None, None

    return season, episode, f"S{season:02d}E{episode:02d}"


def _source_relpath(source: str) -> str | None:
    if not source:
        return None
    try:
        p = Path(source)
        if p.is_absolute():
            rel = p.resolve().relative_to(PROJECT_ROOT)
            return rel.as_posix()
    except Exception:
        return None
    return None


def add_chunk_metadata(chunks: List[Document], *, default_chunk_type: str = "char") -> List[Document]:
    """Ensure each chunk has stable chunk-level metadata.

    - Preserves existing metadata (episode_id, doc_type, speakers, etc.)
    - Adds:
        - chunk_type (if missing)
        - chunk_index (monotonic per source file)

    Why this is useful: later we can cite/group/debug by (episode_id, source, chunk_index)
    and build episode-diverse retrieval policies.
    """
    next_index_by_source: dict[str, int] = {}
    out: List[Document] = []

    for chunk in chunks:
        meta = dict(chunk.metadata)
        meta.setdefault("chunk_type", default_chunk_type)

        # Ensure doc-level metadata exists even when --use-metadata is off.
        # This is critical for episode routing + drill-down.
        source = str(meta.get("source", ""))
        meta.setdefault("doc_type", _detect_doc_type_from_source(source))

        if not meta.get("episode_id"):
            season, episode_num, episode_id = _parse_episode_from_source(source)
            if season is not None:
                meta.setdefault("season", season)
            if episode_num is not None:
                # Keep naming consistent with other metadata: episode number within season.
                meta.setdefault("episode", episode_num)
            if episode_id:
                meta["episode_id"] = episode_id

        rel = _source_relpath(source)
        if rel:
            meta.setdefault("source_relpath", rel)

        idx = next_index_by_source.get(source, 0)
        meta["chunk_index"] = idx
        next_index_by_source[source] = idx + 1

        out.append(Document(page_content=chunk.page_content, metadata=meta))

    return out


def create_vector_store(
    chunks: List[Document],
    persist_directory: Path,
    *,
    collection_name: str | None = None,
) -> Chroma:
    """
    Create and persist Chroma vector store.

    Note: Chroma.from_documents() writes a new collection to the persist dir.
    Keep persist dirs separate per experiment to avoid mixing chunk strategies.
    """
    # Important: keep embedding requests under provider per-request token limits and
    # be resilient to transient TPM rate limits during large ingestions.
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, chunk_size=64, max_retries=6)

    persist_directory.mkdir(parents=True, exist_ok=True)

    print("--- Creating ChromaDB vector store ---")
    chroma_kwargs = {
        "persist_directory": str(persist_directory),
        "embedding_function": embeddings,
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if collection_name:
        chroma_kwargs["collection_name"] = collection_name

    vs = Chroma(**chroma_kwargs)

    batch_size = 128
    max_attempts = 8
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        attempt = 0
        while True:
            try:
                vs.add_documents(batch)
                break
            except Exception as e:
                # OpenAI rate limits can happen during large ingestions; retry with backoff.
                attempt += 1
                if attempt >= max_attempts:
                    raise
                sleep_s = min(5.0, 0.25 * (2 ** (attempt - 1)))
                print(
                    f"Rate/temporary error while embedding batch {start}-{start + len(batch) - 1} "
                    f"(attempt {attempt}/{max_attempts}); sleeping {sleep_s:.2f}s: {type(e).__name__}"
                )
                time.sleep(sleep_s)

        if (start // batch_size) % 25 == 0:
            print(f"  Progress: {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    print("--- Finished creating vector store ---")
    print(f"Vector store saved to: {persist_directory}")
    return vs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest normalized Office docs into Chroma")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Chroma persist directory (default matches experiments/run_eval.py)",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional Chroma collection name (leave empty to use default)",
    )
    parser.add_argument(
        "--chunking",
        choices=["character", "scene", "scene_window"],
        default="character",
        help="Chunking strategy. Default is 'character' (previous behavior).",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Character chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Character chunk overlap")
    parser.add_argument(
        "--scene-window-size",
        type=int,
        default=1,
        help="For scene_window chunking, include +/- N scenes around the current scene",
    )
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Parse headers into doc.metadata (recommended once stable)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the persist dir before ingesting (DANGEROUS; dev only)",
    )

    args = parser.parse_args()

    load_dotenv()

    if not NORMALIZED_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Normalized docs not found at {NORMALIZED_DOCS_DIR}. "
            "Run ingestion/normalize_docs.py first."
        )

    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = (PROJECT_ROOT / persist_dir).resolve()

    if args.reset and persist_dir.exists():
        print(f"--- Resetting Chroma persist dir: {persist_dir} ---")
        shutil.rmtree(persist_dir)

    # Load episode-level and summary docs (still doc-level at this point).
    documents = load_documents(
        docs_path=str(NORMALIZED_DOCS_DIR),
        use_metadata=bool(args.use_metadata),
    )
    print(f"Loaded {len(documents)} documents from {NORMALIZED_DOCS_DIR}")

    if args.chunking == "character":
        splitter = CharacterTextSplitter(
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
        )
        chunks = splitter.split_documents(documents)
        chunks = add_chunk_metadata(chunks, default_chunk_type="char")
        print(f"Created {len(chunks)} chunks (character splitter)")
    elif args.chunking == "scene":
        chunks = chunk_documents_by_scene(
            documents,
            window_size=0,
            fallback_chunk_size=int(args.chunk_size),
            fallback_chunk_overlap=int(args.chunk_overlap),
        )
        chunks = add_chunk_metadata(chunks, default_chunk_type="char")
        print(f"Created {len(chunks)} chunks (scene-based; no windowing)")
    else:  # scene_window
        chunks = chunk_documents_by_scene(
            documents,
            window_size=int(args.scene_window_size),
            fallback_chunk_size=int(args.chunk_size),
            fallback_chunk_overlap=int(args.chunk_overlap),
        )
        chunks = add_chunk_metadata(chunks, default_chunk_type="char")
        print(
            f"Created {len(chunks)} chunks (scene windows; window_size={int(args.scene_window_size)})"
        )

    # Persist a NEW index for this chunking strategy.
    create_vector_store(
        chunks,
        persist_dir,
        collection_name=args.collection_name if args.collection_name else None,
    )


if __name__ == "__main__":
    main()