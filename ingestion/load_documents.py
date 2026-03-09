from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

HEADER_DIVIDER = "----"


def _parse_header_block(text: str) -> Tuple[Dict[str, str], str]:
    """
    Parse KEY: value lines until divider '----'.
    Returns (header_dict_lowercase, body_text).
    """
    lines = text.splitlines()
    header: Dict[str, str] = {}

    divider_idx = None
    for i, line in enumerate(lines):
        if line.strip() == HEADER_DIVIDER:
            divider_idx = i
            break

    if divider_idx is None:
        return {}, text

    for line in lines[:divider_idx]:
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        header[k.strip().lower()] = v.strip()

    body_lines = lines[divider_idx + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines = body_lines[1:]

    body = "\n".join(body_lines).strip()
    return header, body


def _coerce_int(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    return int(s) if s.isdigit() else None


def _detect_doc_type_from_source(source: str) -> str:
    s = source.replace("\\", "/").lower()
    if "/summaries/" in s:
        return "summary"
    if "/scripts/" in s:
        return "script"
    return "unknown"


def enrich_documents_with_metadata(docs: List[Document]) -> List[Document]:
    """
    Parse our normalized TXT header into doc.metadata and strip the header from page_content.
    Keeps a compact identity line in page_content for readability/citations.
    """
    enriched: List[Document] = []

    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        doc_type = _detect_doc_type_from_source(source)

        header, body = _parse_header_block(doc.page_content)

        meta = dict(doc.metadata)
        meta["doc_type"] = doc_type

        season = _coerce_int(header.get("season"))
        episode = _coerce_int(header.get("episode"))
        title = header.get("title") or None

        if season is not None:
            meta["season"] = season
        if episode is not None:
            meta["episode"] = episode
        if title:
            meta["title"] = title

        # Canonical episode_id used throughout retrieval/eval for grouping and citations.
        # We store any raw header episode_id (e.g., "02-11") separately.
        if season is not None and episode is not None:
            meta["episode_id"] = f"S{season:02d}E{episode:02d}"

        if doc_type == "script":
            if header.get("episode_id"):
                meta["episode_id_raw"] = header["episode_id"]
            if header.get("speakers"):
                meta["speakers"] = [s.strip() for s in header["speakers"].split(",") if s.strip()]

        if doc_type == "summary":
            if header.get("airdate"):
                meta["airdate"] = header["airdate"]
            if header.get("tvmaze_id"):
                meta["tvmaze_id"] = header["tvmaze_id"]
            if header.get("tvmaze_url"):
                meta["tvmaze_url"] = header["tvmaze_url"]

        # Minimal identity line so retrieved chunks self-identify.
        ident_parts = ["The Office"]
        if season is not None and episode is not None:
            ident_parts.append(f"S{season:02d}E{episode:02d}")
        if title:
            ident_parts.append(title)
        ident_line = " — ".join(ident_parts)

        new_content = ident_line + "\n\n" + (body if body else doc.page_content)

        enriched.append(Document(page_content=new_content, metadata=meta))

    return enriched


def load_documents(
    docs_path: str = "./normalized_docs_txt",
    *,
    use_metadata: bool = False,
    show_progress: bool = True,
) -> List[Document]:
    """
    Load normalized .txt docs recursively. Optionally enrich with metadata by parsing headers.
    """
    docs_dir = Path(docs_path)
    print(f"Loading documents from {docs_dir.resolve()}")

    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Directory {docs_dir} does not exist. "
            f"Please run ingestion/normalize_docs.py to create it."
        )

    loader = DirectoryLoader(
        path=str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=show_progress,
        use_multithreading=True,
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt documents found under {docs_dir}. " f"Please run ingestion/normalize_docs.py."
        )

    if use_metadata:
        documents = enrich_documents_with_metadata(documents)

    # Debug: show first 2 docs
    for i, doc in enumerate(documents[:2]):

        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Metadata keys: {sorted(doc.metadata.keys())}")
        preview = doc.page_content[:1400].replace("\n", "\\n")
        print(f"  Preview: {preview}...")

    return documents
