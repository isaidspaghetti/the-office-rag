from __future__ import annotations

import argparse
from pathlib import Path

import chromadb


def main() -> None:
    parser = argparse.ArgumentParser(description="Peek at a persisted Chroma collection")
    parser.add_argument("--persist-dir", default="db/chroma_db", help="Chroma persist dir")
    parser.add_argument("--limit", type=int, default=3, help="Number of records to print")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    print("Persist dir:", persist_dir.resolve())

    client = chromadb.PersistentClient(path=str(persist_dir))
    cols = client.list_collections()
    print("Collections:", [c.name for c in cols])
    if not cols:
        raise SystemExit("No collections found")

    col = client.get_collection(cols[0].name)
    print("Using collection:", col.name)

    res = col.get(limit=int(args.limit), include=["documents", "metadatas"])

    for i, (meta, doc, _id) in enumerate(zip(res["metadatas"], res["documents"], res["ids"]), start=1):
        meta = meta or {}
        doc = doc or ""
        print(f"\n--- CHUNK {i} ---")
        print("id:", _id)
        print("episode_id:", meta.get("episode_id"))
        print("doc_type:", meta.get("doc_type"))
        print("chunk_type:", meta.get("chunk_type"))
        print("chunk_index:", meta.get("chunk_index"))
        print("source:", meta.get("source"))
        print("meta_keys:", sorted(meta.keys()))
        print("preview:", doc[:200].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
