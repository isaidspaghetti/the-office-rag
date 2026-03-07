from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# Matches lines like: === SCENE 000 ===
_SCENE_HEADER_RE = re.compile(r"^===\s*SCENE\s+(\d{3})\s*===\s*$", re.MULTILINE)


def split_episode_script_into_scenes(doc: Document) -> List[Document]:
    """Split an episode-level script Document into one Document per scene.

    Uses scene markers produced by ingestion/normalize_docs.py.
    If no scene markers are found, returns [doc].
    """
    text = doc.page_content or ""
    matches = list(_SCENE_HEADER_RE.finditer(text))
    if not matches:
        return [doc]

    chunks: List[Document] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        scene_index = int(match.group(1))
        scene_text = text[start:end].strip()
        if not scene_text:
            continue

        meta = dict(doc.metadata)
        meta["chunk_type"] = "scene"
        meta["scene_index"] = scene_index

        chunks.append(Document(page_content=scene_text, metadata=meta))

    return chunks


def split_episode_script_into_scene_windows(doc: Document, *, window_size: int = 1) -> List[Document]:
    """Split an episode script into overlapping scene windows.

    With window_size=1 (default), each output chunk includes:
      previous scene + current scene + next scene (when available).

    If no scene markers are found, returns [doc].
    """
    if window_size < 0:
        raise ValueError("window_size must be >= 0")

    # First split into atomic scenes.
    scenes = split_episode_script_into_scenes(doc)
    if len(scenes) <= 1 or window_size == 0:
        return scenes

    out: List[Document] = []
    n = len(scenes)

    for i in range(n):
        start_i = max(0, i - window_size)
        end_i = min(n - 1, i + window_size)
        window_docs = scenes[start_i : end_i + 1]

        window_text = "\n\n".join((d.page_content or "").strip() for d in window_docs).strip()
        if not window_text:
            continue

        meta = dict(doc.metadata)
        meta["chunk_type"] = "scene_window"
        meta["window_size"] = window_size
        meta["scene_window_start"] = int(window_docs[0].metadata.get("scene_index", start_i))
        meta["scene_window_end"] = int(window_docs[-1].metadata.get("scene_index", end_i))
        meta["scene_index"] = int(scenes[i].metadata.get("scene_index", i))

        out.append(Document(page_content=window_text, metadata=meta))

    return out


def chunk_documents_by_scene(
    documents: List[Document],
    *,
    window_size: int = 1,
    fallback_chunk_size: int = 1000,
    fallback_chunk_overlap: int = 150,
) -> List[Document]:
    """Chunk documents with a scene-first strategy.

        - For script docs containing scene markers, split into overlapping scene windows.
    - For docs without scene markers (e.g., summaries), keep as one doc if short;
      otherwise fall back to a CharacterTextSplitter.
    """
    fallback_splitter = CharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=fallback_chunk_overlap,
    )

    out: List[Document] = []

    for doc in documents:
        text = doc.page_content or ""

        if _SCENE_HEADER_RE.search(text):
            out.extend(split_episode_script_into_scene_windows(doc, window_size=window_size))
            continue

        if len(text) <= fallback_chunk_size:
            meta = dict(doc.metadata)
            meta.setdefault("chunk_type", "full")
            out.append(Document(page_content=text, metadata=meta))
        else:
            for chunk in fallback_splitter.split_documents([doc]):
                meta = dict(chunk.metadata)
                meta.setdefault("chunk_type", "char")
                out.append(Document(page_content=chunk.page_content, metadata=meta))

    return out