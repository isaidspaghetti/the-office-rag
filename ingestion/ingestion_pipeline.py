from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from chunk_documents import chunk_documents_by_scene
from load_documents import load_documents

# ---- Config you can edit ----
NORMALIZED_DOCS_DIR = Path("ingestion/normalized_docs_txt")
PERSIST_DIR = Path("db/chroma_db_scene_w1")  # use a NEW db dir for scene-chunk index
EMBED_MODEL = "text-embedding-3-small"


def create_vector_store(chunks: List[Document], persist_directory: Path) -> Chroma:
    """
    Create and persist Chroma vector store.

    Note: Chroma.from_documents() writes a new collection to the persist dir.
    Keep persist dirs separate per experiment to avoid mixing chunk strategies.
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    persist_directory.mkdir(parents=True, exist_ok=True)

    print("--- Creating ChromaDB vector store ---")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store saved to: {persist_directory}")
    return vs


def main() -> None:
    print("Main function called")
    load_dotenv()

    if not NORMALIZED_DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Normalized docs not found at {NORMALIZED_DOCS_DIR}. "
            "Run ingestion/normalize_docs.py first."
        )

    # Load episode-level and summary docs (still doc-level at this point).
    documents = load_documents(
        docs_path=str(NORMALIZED_DOCS_DIR),
        use_metadata=False,  # keep off for now; turn on later when you're ready
    )
    print(f"Loaded {len(documents)} documents from {NORMALIZED_DOCS_DIR}")

    # Chunk scripts by scene (summaries remain whole or char-chunk if large).
    chunks = chunk_documents_by_scene(
        documents,
        fallback_chunk_size=1000,
        fallback_chunk_overlap=150,
    )
    print(f"Created {len(chunks)} chunks (scene-based where available)")

    # Persist a NEW index for this chunking strategy.
    create_vector_store(chunks, PERSIST_DIR)


if __name__ == "__main__":
    main()