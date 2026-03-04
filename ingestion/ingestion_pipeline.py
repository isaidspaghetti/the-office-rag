from __future__ import annotations

import argparse
import shutil
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

        source = str(meta.get("source", ""))
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
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    persist_directory.mkdir(parents=True, exist_ok=True)

    print("--- Creating ChromaDB vector store ---")
    kwargs = {
        "documents": chunks,
        "embedding": embeddings,
        "persist_directory": str(persist_directory),
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if collection_name:
        kwargs["collection_name"] = collection_name

    vs = Chroma.from_documents(
        **kwargs,
    )
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