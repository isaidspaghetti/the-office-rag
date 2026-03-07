from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        # Avoid python-dotenv edge cases in interactive/stdin contexts by
        # preferring an explicit repo-root `.env` path when present.
        repo_root = Path(__file__).resolve().parent
        env_path = repo_root / ".env"
        if env_path.exists() and env_path.is_file():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except Exception:
        return


_REPO_ROOT = Path(__file__).resolve().parent

_EP_IN_PATH_RE = re.compile(r"(?:^|[\\/])s(?P<season>\d{2})e(?P<episode>\d{2})(?:[_.-]|$)", re.IGNORECASE)
_HEADER_DIVIDER = "----"


def _as_posix_relpath(p: Path) -> str:
    try:
        return p.resolve().relative_to(_REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _detect_doc_type_from_relpath(relpath: str) -> str:
    s = (relpath or "").replace("\\", "/").lower()
    if "/summaries/" in s:
        return "summary"
    if "/scripts/" in s:
        return "script"
    return "unknown"


def _parse_episode_from_relpath(relpath: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if not relpath:
        return None, None, None

    m = _EP_IN_PATH_RE.search(relpath.replace("\\", "/"))
    if not m:
        return None, None, None

    try:
        season = int(m.group("season"))
        episode = int(m.group("episode"))
    except Exception:
        return None, None, None

    return season, episode, f"S{season:02d}E{episode:02d}"


def _parse_header_block(text: str) -> Tuple[Dict[str, str], str]:
    lines = (text or "").splitlines()
    header: Dict[str, str] = {}

    divider_idx = None
    for i, line in enumerate(lines):
        if line.strip() == _HEADER_DIVIDER:
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


def _coerce_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    t = str(s).strip()
    return int(t) if t.isdigit() else None


def _stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


_QDRANT_ID_NAMESPACE = uuid.UUID("4f43b669-63d9-4efe-8f6c-8a729f7d7b3b")


def _qdrant_point_id(stable_id: str) -> str:
    """Convert an arbitrary stable string id into a Qdrant-valid UUID point id.

    Qdrant point IDs must be an unsigned integer or a UUID. Our internal stable IDs
    are arbitrary strings (sha1 hex, or "type:SxxEyy:tag"), so we deterministically
    map them to UUIDs via uuid5.
    """
    s = str(stable_id or "").strip()
    if not s:
        # Should never happen; keep deterministic.
        return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, "(empty)"))

    # If it's already a UUID, keep it.
    try:
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, s))


def _chunk_text_char(text: str, *, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    t = text or ""
    n = len(t)
    if n == 0:
        return

    cs = max(1, int(chunk_size))
    ov = max(0, int(chunk_overlap))
    step = max(1, cs - ov)

    for start in range(0, n, step):
        end = min(n, start + cs)
        chunk = t[start:end]
        if chunk.strip():
            yield chunk
        if end >= n:
            break


@dataclass(frozen=True)
class QdrantConn:
    url: str
    api_key: Optional[str]


def _build_qdrant_client(conn: QdrantConn) -> Any:
    from qdrant_client import QdrantClient  # type: ignore

    kwargs: Dict[str, Any] = {"url": str(conn.url)}
    if conn.api_key:
        kwargs["api_key"] = str(conn.api_key)

    # Keep timeouts reasonable.
    kwargs["timeout"] = 60.0
    return QdrantClient(**kwargs)


def _build_embedder(*, embed_model: str) -> Any:
    from langchain_openai import OpenAIEmbeddings  # type: ignore

    # chunk_size controls batch size inside the embedder.
    return OpenAIEmbeddings(model=str(embed_model), chunk_size=128, max_retries=6)


def _ensure_collection(client: Any, *, collection_name: str, vector_size: int, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore

    name = str(collection_name)

    if recreate:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
        )
        return

    # Create if missing.
    try:
        client.get_collection(name)
        return
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
        )


def _iter_normalized_txt_files(docs_dir: Path) -> Iterable[Path]:
    yield from sorted(docs_dir.rglob("*.txt"))


def _build_doc_identity_line(*, season: Optional[int], episode: Optional[int], title: Optional[str]) -> str:
    parts = ["The Office"]
    if season is not None and episode is not None:
        parts.append(f"S{season:02d}E{episode:02d}")
    if title:
        parts.append(str(title).strip())
    return " — ".join([p for p in parts if p]).strip()


def _iter_script_chunks(
    *,
    docs_dir: Path,
    use_metadata_headers: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """Yield (point_id, payload, chunk_text)."""

    for path in _iter_normalized_txt_files(docs_dir):
        relpath = _as_posix_relpath(path)
        raw = path.read_text(encoding="utf-8", errors="replace")

        header: Dict[str, str] = {}
        body = raw
        if use_metadata_headers:
            header, body = _parse_header_block(raw)

        season = _coerce_int(header.get("season"))
        episode = _coerce_int(header.get("episode"))
        title = (header.get("title") or "").strip() or None

        doc_type = _detect_doc_type_from_relpath(relpath)
        season2, episode2, eid2 = _parse_episode_from_relpath(relpath)

        # Prefer header for season/episode if present, else fallback to filename.
        if season is None:
            season = season2
        if episode is None:
            episode = episode2
        episode_id = None
        if season is not None and episode is not None:
            episode_id = f"S{season:02d}E{episode:02d}"
        elif eid2:
            episode_id = eid2

        ident = _build_doc_identity_line(season=season, episode=episode, title=title)
        page_content = (ident + "\n\n" + body.strip()).strip() if ident else body.strip()

        base_meta: Dict[str, Any] = {
            "doc_type": doc_type,
            "episode_id": episode_id,
            "season": season,
            "episode": episode,
            "title": title,
            "source": relpath,
            "source_relpath": relpath,
        }

        for idx, chunk in enumerate(
            _chunk_text_char(page_content, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        ):
            payload = dict(base_meta)
            payload["chunk_type"] = "char"
            payload["chunk_index"] = int(idx)
            payload["text"] = chunk

            pid = _stable_id(f"script|{relpath}|char|{idx}")
            payload["stable_doc_id"] = pid
            yield pid, payload, chunk


def _iter_derived_docs(
    *,
    out_root: Path,
    episode_prefixes: Sequence[str],
    season_prefixes: Sequence[str],
    topic_prefixes: Sequence[str],
    include_episode_cards: bool,
    include_season_cards: bool,
    include_topic_cards: bool,
    seasons_filter: Optional[Sequence[int]],
) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """Yield (point_id, payload, text) for derived card docs."""

    from derived.build_derived_cards_index import (  # type: ignore
        _auto_detect_episode_card_prefixes,
        _auto_detect_season_card_prefixes,
        _auto_detect_topic_card_prefixes,
        _iter_episode_card_files,
        _iter_season_card_files,
        _iter_topic_card_files,
        _render_episode_card,
        _render_season_card,
        _render_topic_card,
        _season_episode_from_id,
    )

    def _passes(season: Optional[int]) -> bool:
        if not seasons_filter:
            return True
        if season is None:
            return False
        return int(season) in {int(x) for x in seasons_filter}

    ep_prefs = list(episode_prefixes)
    ss_prefs = list(season_prefixes)
    tp_prefs = list(topic_prefixes)

    if include_episode_cards and not ep_prefs:
        ep_prefs = _auto_detect_episode_card_prefixes(out_root)
    if include_season_cards and not ss_prefs:
        ss_prefs = _auto_detect_season_card_prefixes(out_root)
    if include_topic_cards and not tp_prefs:
        tp_prefs = _auto_detect_topic_card_prefixes(out_root)

    if include_episode_cards:
        for pref in ep_prefs:
            for p in _iter_episode_card_files(out_root, pref):
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "EpisodeDerivedCardV1":
                    continue

                episode_id = str(card.get("episode_id") or "").strip().upper() or None
                season, episode = _season_episode_from_id(episode_id or "")
                if not _passes(season):
                    continue

                title = str(card.get("title") or "").strip() or None
                build_id = str(card.get("build_id") or "").strip() or None
                stable_build_tag = build_id or str(pref)

                text = _render_episode_card(card)
                rel = _as_posix_relpath(Path(p))

                meta: Dict[str, Any] = {
                    "doc_type": "derived",
                    "derived_type": "episode_card",
                    "episode_id": episode_id,
                    "season": season,
                    "episode": episode,
                    "title": title,
                    "build_id": build_id,
                    "episode_cards_build_prefix": str(pref),
                    "source": rel,
                    "source_file": rel,
                    "stable_doc_id": f"episode_card:{episode_id}:{stable_build_tag}",
                    "text": text,
                }
                pid = str(meta["stable_doc_id"])
                yield pid, meta, text

    if include_season_cards:
        for pref in ss_prefs:
            for p in _iter_season_card_files(out_root, pref):
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "SeasonDerivedCardV1":
                    continue

                season = int(card.get("season") or 0) or None
                if not _passes(season):
                    continue

                build_id = str(card.get("build_id") or "").strip() or None
                stable_build_tag = build_id or str(pref)

                text = _render_season_card(card)
                rel = _as_posix_relpath(Path(p))

                meta = {
                    "doc_type": "derived",
                    "derived_type": "season_card",
                    "season": season,
                    "build_id": build_id,
                    "season_cards_build_prefix": str(pref),
                    "source": rel,
                    "source_file": rel,
                    "stable_doc_id": f"season_card:{int(season):02d}:{stable_build_tag}",
                    "text": text,
                }
                pid = str(meta["stable_doc_id"])
                yield pid, meta, text

    if include_topic_cards:
        for pref in tp_prefs:
            for p in _iter_topic_card_files(out_root, pref):
                card = json.loads(p.read_text(encoding="utf-8"))
                if str(card.get("schema") or "") != "TopicCardV1":
                    continue

                topic_id = str(card.get("topic_id") or "").strip() or None
                topic_type = str(card.get("topic_type") or "").strip() or None
                entities = [str(x).strip() for x in (card.get("entity_names") or []) if str(x).strip()]
                episode_ids = [str(x).strip().upper() for x in (card.get("episode_ids") or []) if str(x).strip()]
                build_id = str(card.get("build_id") or "").strip() or None
                stable_build_tag = build_id or str(pref)

                # Season filter: keep if ANY episode is in an allowed season.
                if seasons_filter and episode_ids:
                    allowed = {int(x) for x in seasons_filter}
                    keep = False
                    for eid in episode_ids:
                        s, _e, _ = _parse_episode_from_relpath(eid.lower())  # not used; eid is canonical, but keep safe
                        try:
                            season = int(eid[1:3])
                        except Exception:
                            season = None
                        if season is not None and season in allowed:
                            keep = True
                            break
                    if not keep:
                        continue

                text = _render_topic_card(card)
                rel = _as_posix_relpath(Path(p))

                meta = {
                    "doc_type": "derived",
                    "derived_type": "topic_card",
                    "topic_id": topic_id,
                    "topic_type": topic_type,
                    "entity_names": entities,
                    "episode_ids": episode_ids,
                    "build_id": build_id,
                    "topic_cards_build_prefix": str(pref),
                    "source": rel,
                    "source_file": rel,
                    "stable_doc_id": f"topic_card:{topic_id}:{stable_build_tag}",
                    "text": text,
                }
                pid = str(meta["stable_doc_id"])
                yield pid, meta, text


def _batched(xs: Iterable[Tuple[str, Dict[str, Any], str]], batch_size: int) -> Iterator[List[Tuple[str, Dict[str, Any], str]]]:
    batch: List[Tuple[str, Dict[str, Any], str]] = []
    for item in xs:
        batch.append(item)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Path):
        return v.as_posix()
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _coerce_script_payload_from_chroma(
    *,
    doc_text: str,
    meta: Dict[str, Any],
    chroma_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """Return (stable_doc_id, payload) for a script chunk sourced from a Chroma index."""

    md: Dict[str, Any] = dict(meta or {})

    # Establish a stable-ish source relpath.
    relpath = str(md.get("source_relpath") or "").strip()
    if not relpath:
        src = str(md.get("source") or "").strip()
        if src:
            # If it looks absolute, try to make it repo-relative.
            try:
                relpath = _as_posix_relpath(Path(src))
            except Exception:
                relpath = src.replace("\\", "/")

    if relpath:
        md.setdefault("source", relpath)
        md.setdefault("source_relpath", relpath)

    # Ensure doc_type and episode_id exist.
    if not md.get("doc_type") and relpath:
        md["doc_type"] = _detect_doc_type_from_relpath(relpath)

    if not md.get("episode_id") and relpath:
        season, episode, eid = _parse_episode_from_relpath(relpath)
        if season is not None:
            md.setdefault("season", season)
        if episode is not None:
            md.setdefault("episode", episode)
        if eid:
            md["episode_id"] = eid

    chunk_type = str(md.get("chunk_type") or "char").strip() or "char"
    try:
        chunk_index = int(md.get("chunk_index"))
    except Exception:
        chunk_index = 0
        md["chunk_index"] = chunk_index

    stable_doc_id = _stable_id(f"script|{relpath}|{chunk_type}|{chunk_index}")
    payload: Dict[str, Any] = dict(md)
    payload["chunk_type"] = chunk_type
    payload["chunk_index"] = int(chunk_index)
    payload["stable_doc_id"] = stable_doc_id
    payload["chroma_id"] = str(chroma_id)
    payload["text"] = str(doc_text or "")

    return stable_doc_id, _jsonable(payload)


def migrate_scripts_from_chroma(
    *,
    client: Any,
    collection_name: str,
    chroma_persist_dir: Path,
    chroma_collection_name: Optional[str],
    recreate: bool,
    upsert_batch: int,
    verify: bool,
    verify_scroll: int,
) -> None:
    """Migrate an existing local Chroma script index into Qdrant without re-embedding."""

    from qdrant_client.models import PointStruct  # type: ignore

    try:
        import chromadb  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing chromadb dependency for Chroma->Qdrant migration. "
            "Install: pip install -U chromadb"
        ) from e

    persist = chroma_persist_dir.expanduser().resolve()
    if not persist.exists() or not persist.is_dir():
        raise SystemExit(f"Chroma persist dir not found: {persist}")

    ch_client = chromadb.PersistentClient(path=str(persist))
    col_name = (str(chroma_collection_name).strip() if chroma_collection_name else "")

    if not col_name:
        try:
            cols = ch_client.list_collections()
            names = [getattr(c, "name", "") for c in cols]
            names = [str(n).strip() for n in names if str(n).strip()]
            if not names:
                raise SystemExit(f"No Chroma collections found under: {persist}")
            col_name = names[0]
            print(f"Auto-selected Chroma collection: {col_name}")
        except Exception as e:
            raise SystemExit(f"Could not list collections in Chroma dir: {persist} ({type(e).__name__})") from e

    try:
        ch_col = ch_client.get_collection(name=str(col_name))
    except Exception as e:
        raise SystemExit(f"Could not open Chroma collection '{col_name}' in {persist}: {type(e).__name__}: {e}") from e

    total = int(ch_col.count() or 0)
    if total <= 0:
        print(f"No vectors found in Chroma collection '{col_name}' ({persist}); nothing to migrate.")
        return

    # Probe vector size.
    probe = ch_col.get(include=["embeddings"], limit=1)
    emb0: Any = None
    try:
        embs = probe.get("embeddings")
        if embs is None:
            emb0 = None
        elif isinstance(embs, list):
            emb0 = embs[0] if embs else None
        else:
            # Newer chromadb versions return numpy arrays.
            try:
                import numpy as np  # type: ignore

                if isinstance(embs, np.ndarray):
                    emb0 = embs[0] if len(embs) else None
            except Exception:
                # Fall back to sequence semantics.
                try:
                    emb0 = next(iter(embs))
                except Exception:
                    emb0 = None
    except Exception:
        emb0 = None

    if emb0 is None:
        raise SystemExit(
            f"Chroma collection '{col_name}' returned no embeddings; cannot migrate without re-embedding. "
            f"Persist dir: {persist}"
        )

    dim = len(emb0)
    _ensure_collection(client, collection_name=str(collection_name), vector_size=int(dim), recreate=bool(recreate))

    print(
        "=== Migrating scripts from Chroma -> Qdrant (no re-embedding) ===\n"
        f"Chroma persist: {persist}\n"
        f"Chroma collection: {col_name}\n"
        f"Qdrant collection: {collection_name}\n"
        f"Vectors: {total} | dim={dim} | batch={int(upsert_batch)}"
    )

    migrated = 0
    for offset in range(0, total, int(upsert_batch)):
        got = ch_col.get(
            # Note: in newer chromadb versions, `ids` is always returned and is not a valid include item.
            include=["documents", "metadatas", "embeddings"],
            limit=int(upsert_batch),
            offset=int(offset),
        )

        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        # Avoid truthiness checks: chromadb may return numpy arrays.
        embs = got.get("embeddings")
        if embs is None:
            embs = []

        points: List[Any] = []
        for cid, doc_text, meta, emb in zip(ids, docs, metas, embs):
            if emb is None:
                continue
            stable_doc_id, payload = _coerce_script_payload_from_chroma(
                doc_text=str(doc_text or ""),
                meta=(meta if isinstance(meta, dict) else {}),
                chroma_id=str(cid),
            )
            points.append(
                PointStruct(
                    id=_qdrant_point_id(stable_doc_id),
                    vector=[float(x) for x in emb],
                    payload=payload,
                )
            )

        if points:
            client.upsert(collection_name=str(collection_name), points=points)
            migrated += len(points)

        if migrated and migrated % (int(upsert_batch) * 20) == 0:
            print(f"  migrated: {migrated}/{total}")

    print(f"OK: migrated scripts total={migrated} into collection={collection_name}")

    if verify:
        _verify_collection(
            client,
            collection_name=str(collection_name),
            verify_scroll=int(verify_scroll),
        )


def _verify_collection(
    client: Any,
    *,
    collection_name: str,
    verify_scroll: int = 2,
) -> None:
    try:
        cnt = client.count(collection_name=str(collection_name), exact=True)
        print("Qdrant count:", getattr(cnt, "count", cnt))
    except Exception as e:
        print(f"WARN: could not count Qdrant collection: {type(e).__name__}: {e}")

    try:
        points, _next = client.scroll(
            collection_name=str(collection_name),
            limit=max(1, int(verify_scroll)),
            with_payload=True,
            with_vectors=False,
        )
        if points:
            sample = points[0]
            payload = getattr(sample, "payload", None)
            keys = sorted(list((payload or {}).keys()))
            print("Qdrant sample payload keys:", keys[:40])
            if "stable_doc_id" in (payload or {}):
                print("Qdrant sample stable_doc_id:", (payload or {}).get("stable_doc_id"))
            if "derived_type" in (payload or {}):
                print("Qdrant sample derived_type:", (payload or {}).get("derived_type"))
            if "episode_id" in (payload or {}):
                print("Qdrant sample episode_id:", (payload or {}).get("episode_id"))
    except Exception as e:
        print(f"WARN: could not scroll Qdrant collection: {type(e).__name__}: {e}")


def index_scripts(
    *,
    client: Any,
    embedder: Any,
    collection_name: str,
    docs_dir: Path,
    recreate: bool,
    use_metadata_headers: bool,
    chunk_size: int,
    chunk_overlap: int,
    upsert_batch: int,
) -> None:
    from qdrant_client.models import PointStruct  # type: ignore

    # Determine embedding dimensionality.
    dim = len(embedder.embed_query("dimension probe"))
    _ensure_collection(client, collection_name=str(collection_name), vector_size=int(dim), recreate=bool(recreate))

    stream = _iter_script_chunks(
        docs_dir=docs_dir,
        use_metadata_headers=bool(use_metadata_headers),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )

    total = 0
    for batch in _batched(stream, batch_size=int(upsert_batch)):
        texts = [payload["text"] for _pid, payload, _txt in batch]
        vectors = embedder.embed_documents(texts)

        points = []
        for (pid, payload, _txt), vec in zip(batch, vectors):
            points.append(PointStruct(id=_qdrant_point_id(pid), vector=[float(x) for x in vec], payload=payload))

        client.upsert(collection_name=str(collection_name), points=points)
        total += len(points)
        if total % (upsert_batch * 10) == 0:
            print(f"  scripts indexed: {total}")

    print(f"OK: scripts indexed total={total} into collection={collection_name}")


def index_derived(
    *,
    client: Any,
    embedder: Any,
    collection_name: str,
    out_root: Path,
    recreate: bool,
    episode_prefixes: Sequence[str],
    season_prefixes: Sequence[str],
    topic_prefixes: Sequence[str],
    include_episode_cards: bool,
    include_season_cards: bool,
    include_topic_cards: bool,
    seasons_filter: Optional[Sequence[int]],
    upsert_batch: int,
) -> None:
    from qdrant_client.models import PointStruct  # type: ignore

    dim = len(embedder.embed_query("dimension probe"))
    _ensure_collection(client, collection_name=str(collection_name), vector_size=int(dim), recreate=bool(recreate))

    stream = _iter_derived_docs(
        out_root=out_root,
        episode_prefixes=episode_prefixes,
        season_prefixes=season_prefixes,
        topic_prefixes=topic_prefixes,
        include_episode_cards=bool(include_episode_cards),
        include_season_cards=bool(include_season_cards),
        include_topic_cards=bool(include_topic_cards),
        seasons_filter=seasons_filter,
    )

    total = 0
    for batch in _batched(stream, batch_size=int(upsert_batch)):
        texts = [payload["text"] for _pid, payload, _txt in batch]
        vectors = embedder.embed_documents(texts)

        points = []
        for (pid, payload, _txt), vec in zip(batch, vectors):
            points.append(PointStruct(id=_qdrant_point_id(pid), vector=[float(x) for x in vec], payload=payload))

        client.upsert(collection_name=str(collection_name), points=points)
        total += len(points)
        if total % (upsert_batch * 10) == 0:
            print(f"  derived indexed: {total}")

    print(f"OK: derived indexed total={total} into collection={collection_name}")


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_int_csv(s: str) -> Optional[List[int]]:
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    if not items:
        return None
    out: List[int] = []
    for it in items:
        if not it.isdigit():
            raise ValueError(f"Invalid int in list: {it}")
        out.append(int(it))
    return out


def _load_derived_build_meta(path: Path) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"build meta not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"build meta must be a JSON object: {p}")
    return obj


def _maybe_apply_derived_build_meta(
    *,
    build_meta: Dict[str, Any],
    episode_prefixes: List[str],
    season_prefixes: List[str],
    topic_prefixes: List[str],
    include_episode_cards: bool,
    include_season_cards: bool,
    include_topic_cards: bool,
    seasons_filter: Optional[List[int]],
) -> Tuple[List[str], List[str], List[str], bool, bool, bool, Optional[List[int]]]:
    """Apply build-meta defaults to derived indexing config.

    Only fills values that are not already set explicitly via CLI.
    """

    def _get_list(key: str) -> List[str]:
        v = build_meta.get(key)
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    ep = list(episode_prefixes)
    ss = list(season_prefixes)
    tp = list(topic_prefixes)

    if not ep:
        ep = _get_list("episode_cards_build_prefixes")
    if not ss:
        ss = _get_list("season_cards_build_prefixes")
    if not tp:
        tp = _get_list("topic_cards_build_prefixes")

    ie = bool(include_episode_cards)
    is_ = bool(include_season_cards)
    it = bool(include_topic_cards)
    if build_meta.get("include_episode_cards") is False:
        ie = False
    if build_meta.get("include_season_cards") is False:
        is_ = False
    if build_meta.get("include_topic_cards") is False:
        it = False

    sf = seasons_filter
    if sf is None:
        raw = build_meta.get("seasons_filter")
        if isinstance(raw, list) and raw:
            out: List[int] = []
            for x in raw:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            sf = out if out else None

    return ep, ss, tp, ie, is_, it, sf


def main() -> None:
    _load_dotenv_if_available()

    p = argparse.ArgumentParser(description="Index The Office corpora into Qdrant (two collections).")
    p.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", ""), help="Qdrant URL")
    p.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY", ""), help="Qdrant API key")
    p.add_argument("--embed-model", default="text-embedding-3-small")

    p.add_argument(
        "--mode",
        choices=["scripts", "derived", "both", "scripts_from_chroma"],
        default="both",
        help="Which corpus to index.",
    )

    # Scripts options
    p.add_argument("--scripts-collection", default=os.environ.get("QDRANT_SCRIPTS_COLLECTION", "office_scripts"))
    p.add_argument("--docs-dir", default="ingestion/normalized_docs_txt")
    p.add_argument("--use-metadata-headers", action="store_true", help="Parse normalized-doc headers into payload")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)

    # Scripts-from-Chroma migration options (no embedding)
    p.add_argument(
        "--scripts-chroma-persist-dir",
        default="db/chroma_db_meta",
        help="Local Chroma persist dir to migrate from (no re-embedding)",
    )
    p.add_argument(
        "--scripts-chroma-collection",
        default="",
        help="Chroma collection name (optional; auto-pick first if omitted)",
    )
    p.add_argument("--verify", action="store_true", help="After indexing/migration, print Qdrant count + a sample payload")
    p.add_argument("--verify-scroll", type=int, default=2, help="How many points to scroll when verifying")

    # Derived options
    p.add_argument("--derived-collection", default=os.environ.get("QDRANT_DERIVED_COLLECTION", "office_derived_cards"))
    p.add_argument("--out-root", default="derived/artifacts")
    p.add_argument(
        "--derived-build-meta",
        default="",
        help=(
            "Optional path to a derived-cards _build_meta.json (written by derived/build_derived_cards_index.py). "
            "When provided, defaults to the recorded build prefixes so Qdrant indexing matches that build exactly."
        ),
    )
    p.add_argument(
        "--derived-build-meta-persist-dir",
        default="",
        help="Optional persist dir containing _build_meta.json (e.g. db/chroma_db_derived_cards).",
    )
    p.add_argument("--episode-prefixes", default="", help="CSV; if empty auto-detect")
    p.add_argument("--season-prefixes", default="", help="CSV; if empty auto-detect")
    p.add_argument("--topic-prefixes", default="", help="CSV; if empty auto-detect")

    p.add_argument("--no-episode-cards", action="store_true")
    p.add_argument("--no-season-cards", action="store_true")
    p.add_argument("--no-topic-cards", action="store_true")
    p.add_argument("--seasons", default="", help="Optional CSV season filter, e.g. 1,2,4")

    p.add_argument("--recreate", action="store_true", help="Drop/recreate the target collections")
    p.add_argument("--upsert-batch", type=int, default=64, help="Batch size for embedding+upsert")

    args = p.parse_args()

    if not str(args.qdrant_url).strip():
        raise SystemExit("Missing --qdrant-url (or set QDRANT_URL)")

    conn = QdrantConn(url=str(args.qdrant_url).strip(), api_key=(str(args.qdrant_api_key).strip() or None))
    client = _build_qdrant_client(conn)
    mode = str(args.mode)
    recreate = bool(args.recreate)
    upsert_batch = int(args.upsert_batch)

    needs_embedding = mode in {"scripts", "derived", "both"}
    embedder = None
    if needs_embedding:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Missing OPENAI_API_KEY (set it in .env or environment)")
        embedder = _build_embedder(embed_model=str(args.embed_model))

    if mode == "scripts_from_chroma":
        persist_dir = (
            Path(str(args.scripts_chroma_persist_dir))
            if Path(str(args.scripts_chroma_persist_dir)).is_absolute()
            else (_REPO_ROOT / str(args.scripts_chroma_persist_dir))
        ).resolve()
        migrate_scripts_from_chroma(
            client=client,
            collection_name=str(args.scripts_collection),
            chroma_persist_dir=persist_dir,
            chroma_collection_name=(str(args.scripts_chroma_collection).strip() or None),
            recreate=recreate,
            upsert_batch=upsert_batch,
            verify=bool(args.verify),
            verify_scroll=int(args.verify_scroll),
        )
        return

    if mode in {"scripts", "both"}:
        docs_dir = (Path(str(args.docs_dir)) if Path(str(args.docs_dir)).is_absolute() else (_REPO_ROOT / str(args.docs_dir))).resolve()
        if not docs_dir.exists():
            raise SystemExit(f"Docs dir not found: {docs_dir}")
        print(f"=== Indexing scripts into Qdrant: {args.scripts_collection} ===")
        index_scripts(
            client=client,
            embedder=embedder,
            collection_name=str(args.scripts_collection),
            docs_dir=docs_dir,
            recreate=recreate,
            use_metadata_headers=bool(args.use_metadata_headers),
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
            upsert_batch=upsert_batch,
        )

        if bool(args.verify):
            _verify_collection(
                client,
                collection_name=str(args.scripts_collection),
                verify_scroll=int(args.verify_scroll),
            )

    if mode in {"derived", "both"}:
        out_root = (Path(str(args.out_root)) if Path(str(args.out_root)).is_absolute() else (_REPO_ROOT / str(args.out_root))).resolve()
        if not out_root.exists():
            raise SystemExit(f"Derived out-root not found: {out_root}")

        seasons_filter = _parse_int_csv(str(args.seasons))

        episode_prefixes = _parse_csv_list(str(args.episode_prefixes))
        season_prefixes = _parse_csv_list(str(args.season_prefixes))
        topic_prefixes = _parse_csv_list(str(args.topic_prefixes))

        include_episode_cards = not bool(args.no_episode_cards)
        include_season_cards = not bool(args.no_season_cards)
        include_topic_cards = not bool(args.no_topic_cards)

        # Optional: load build-meta defaults to ensure we index exactly one known build.
        build_meta_path = str(args.derived_build_meta).strip()
        build_meta_persist = str(args.derived_build_meta_persist_dir).strip()
        if build_meta_persist and not build_meta_path:
            build_meta_path = str(Path(build_meta_persist) / "_build_meta.json")

        if build_meta_path:
            meta = _load_derived_build_meta(Path(build_meta_path))
            (
                episode_prefixes,
                season_prefixes,
                topic_prefixes,
                include_episode_cards,
                include_season_cards,
                include_topic_cards,
                seasons_filter,
            ) = _maybe_apply_derived_build_meta(
                build_meta=meta,
                episode_prefixes=episode_prefixes,
                season_prefixes=season_prefixes,
                topic_prefixes=topic_prefixes,
                include_episode_cards=include_episode_cards,
                include_season_cards=include_season_cards,
                include_topic_cards=include_topic_cards,
                seasons_filter=seasons_filter,
            )

        print(f"=== Indexing derived cards into Qdrant: {args.derived_collection} ===")
        index_derived(
            client=client,
            embedder=embedder,
            collection_name=str(args.derived_collection),
            out_root=out_root,
            recreate=recreate,
            episode_prefixes=episode_prefixes,
            season_prefixes=season_prefixes,
            topic_prefixes=topic_prefixes,
            include_episode_cards=include_episode_cards,
            include_season_cards=include_season_cards,
            include_topic_cards=include_topic_cards,
            seasons_filter=seasons_filter,
            upsert_batch=upsert_batch,
        )

        if bool(args.verify):
            _verify_collection(
                client,
                collection_name=str(args.derived_collection),
                verify_scroll=int(args.verify_scroll),
            )


if __name__ == "__main__":
    main()
