from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Allow importing `ingestion.*` when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.load_documents import load_documents

DEFAULT_DOCS_DIR = "ingestion/normalized_docs_txt"
DEFAULT_PERSIST_DIR = "db/chroma_db_derived"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _iter_summaries(docs: List[Document]) -> Iterable[Document]:
    for d in docs:
        if (d.metadata or {}).get("doc_type") == "summary":
            yield d


def _episode_id(doc: Document) -> Optional[str]:
    eid = (doc.metadata or {}).get("episode_id")
    return str(eid).strip().upper() if isinstance(eid, str) and eid.strip() else None


def _title(doc: Document) -> Optional[str]:
    t = (doc.metadata or {}).get("title")
    return str(t).strip() if isinstance(t, str) and t.strip() else None


def _mentions_name(text: str, name: str) -> bool:
    # Very simple matching; good enough for v1.
    t = (text or "").lower()
    n = (name or "").strip().lower()
    if not n:
        return False
    return n in t


def _pick_evidence_summaries(
    summaries: List[Document],
    *,
    subject: str,
    max_docs: int,
) -> List[Document]:
    hits = [d for d in summaries if _mentions_name(d.page_content or "", subject)]

    # Prefer earlier seasons first (stable ordering) if metadata exists.
    def _sort_key(d: Document) -> tuple:
        m = d.metadata or {}
        return (
            int(m.get("season") or 99),
            int(m.get("episode") or 999),
            str(m.get("source") or ""),
        )

    hits.sort(key=_sort_key)
    return hits[: int(max_docs)]


def _format_evidence_block(docs: List[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        eid = _episode_id(d) or "UNKNOWN"
        title = _title(d) or ""
        header = f"[{eid}] {title}".strip()
        body = (d.page_content or "").strip()
        lines.append(header)
        lines.append(body)
        lines.append("---")
    return "\n".join(lines)


@dataclass(frozen=True)
class DerivedBuildConfig:
    llm_model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = 700


def build_character_bio(
    *,
    llm: ChatOpenAI,
    subject: str,
    evidence_docs: List[Document],
) -> str:
    """Return markdown content for a character bio grounded in evidence_docs."""

    evidence = _format_evidence_block(evidence_docs)

    system = (
        "You are building a retrieval-time reference card for a RAG system. "
        "You must ONLY use the provided episode summaries as evidence. "
        "If something is not supported by the evidence, omit it or mark it as unknown. "
        "Cite episode IDs like S02E11 in-line for any non-trivial claim."
    )

    user = f"""Create a CHARACTER BIO reference card for: {subject}

Requirements:
- Output markdown.
- Sections:
  1) Quick bio (2-4 sentences)
  2) Key traits (bullets)
  3) Key relationships (bullets)
  4) Notable arcs / changes over time (bullets)
  5) Evidence coverage: list the episode IDs you relied on
- Every bullet should include at least one episode citation like (S02E11).

Episode summaries (evidence):
{evidence}
"""

    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return (getattr(msg, "content", None) or "").strip()


def build_relationship_timeline(
    *,
    llm: ChatOpenAI,
    subject: str,
    evidence_docs: List[Document],
) -> str:
    evidence = _format_evidence_block(evidence_docs)

    system = (
        "You are building a retrieval-time relationship timeline card for a RAG system. "
        "You must ONLY use the provided episode summaries as evidence. "
        "Cite episode IDs like S02E11 for each relationship/timeline claim."
    )

    user = f"""Create a RELATIONSHIP TIMELINE reference card for: {subject}

Requirements:
- Output markdown.
- Focus on romantic relationships.
- Provide a table with columns: Partner | Status/Type | How it ended | Key episode evidence
- Then a short notes section highlighting ambiguity/unknowns.
- Do not invent partners not supported by evidence.

Episode summaries (evidence):
{evidence}
"""

    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return (getattr(msg, "content", None) or "").strip()


def add_min_doc_identity_meta(doc: Document, *, chunk_type: str) -> Document:
    meta = dict(doc.metadata or {})
    meta.setdefault("chunk_type", chunk_type)
    meta.setdefault("chunk_index", 0)
    return Document(page_content=doc.page_content, metadata=meta)


def build_derived_index(
    *,
    docs_dir: Path,
    persist_dir: Path,
    collection_name: Optional[str],
    embed_model: str,
    reset: bool,
    include_summaries: bool,
    generate_character_bios: bool,
    characters: List[str],
    max_summary_docs_per_character: int,
    generate_relationship_timelines: bool,
    relationship_subjects: List[str],
    llm_model: str,
) -> None:
    load_dotenv()

    if reset and persist_dir.exists():
        print(f"--- Resetting derived persist dir: {persist_dir} ---")
        shutil.rmtree(persist_dir)

    docs = load_documents(docs_path=str(docs_dir), use_metadata=True, show_progress=True)
    summaries = list(_iter_summaries(docs))
    print(f"Loaded {len(summaries)} summary docs")

    out_docs: List[Document] = []

    if include_summaries:
        for s in summaries:
            meta = dict(s.metadata or {})
            meta.setdefault("doc_type", "summary")
            out_docs.append(
                add_min_doc_identity_meta(
                    Document(page_content=s.page_content, metadata=meta), chunk_type="summary_doc"
                )
            )

    created_at = utc_now_iso()
    build_id = f"derived_{created_at.replace(':', '-') }"

    llm: Optional[ChatOpenAI] = None
    if generate_character_bios or generate_relationship_timelines:
        llm = ChatOpenAI(model=llm_model, temperature=0.0)

    if generate_character_bios:
        assert llm is not None
        for subject in characters:
            t0 = time.time()
            evidence_docs = _pick_evidence_summaries(
                summaries,
                subject=subject,
                max_docs=int(max_summary_docs_per_character),
            )
            if not evidence_docs:
                print(f"[character_bio] No evidence found for {subject}; skipping")
                continue

            text = build_character_bio(llm=llm, subject=subject, evidence_docs=evidence_docs)
            evidence_eids = sorted({e for e in (_episode_id(d) for d in evidence_docs) if e})

            meta = {
                "doc_type": "derived",
                "derived_type": "character_bio",
                "subject": subject,
                "episode_ids": evidence_eids,
                "build_id": build_id,
                "created_at_utc": created_at,
                "chunk_type": "derived_doc",
                "chunk_index": 0,
                "source": f"derived://character_bio/{subject}",
            }
            out_docs.append(Document(page_content=text, metadata=meta))
            ms = int((time.time() - t0) * 1000)
            print(f"[character_bio] Built {subject} ({len(evidence_docs)} episodes) in {ms}ms")

    if generate_relationship_timelines:
        assert llm is not None
        for subject in relationship_subjects:
            t0 = time.time()
            evidence_docs = _pick_evidence_summaries(
                summaries,
                subject=subject,
                max_docs=int(max_summary_docs_per_character),
            )
            if not evidence_docs:
                print(f"[relationship_timeline] No evidence found for {subject}; skipping")
                continue

            text = build_relationship_timeline(
                llm=llm, subject=subject, evidence_docs=evidence_docs
            )
            evidence_eids = sorted({e for e in (_episode_id(d) for d in evidence_docs) if e})

            meta = {
                "doc_type": "derived",
                "derived_type": "relationship_timeline",
                "subject": subject,
                "episode_ids": evidence_eids,
                "build_id": build_id,
                "created_at_utc": created_at,
                "chunk_type": "derived_doc",
                "chunk_index": 0,
                "source": f"derived://relationship_timeline/{subject}",
            }
            out_docs.append(Document(page_content=text, metadata=meta))
            ms = int((time.time() - t0) * 1000)
            print(
                f"[relationship_timeline] Built {subject} ({len(evidence_docs)} episodes) in {ms}ms"
            )

    print(f"--- Writing derived index: {persist_dir} (docs={len(out_docs)}) ---")
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=embed_model)
    kwargs: Dict[str, Any] = {
        "documents": out_docs,
        "embedding": embeddings,
        "persist_directory": str(persist_dir),
        "collection_metadata": {"hnsw:space": "cosine"},
    }
    if collection_name:
        kwargs["collection_name"] = collection_name

    _ = Chroma.from_documents(**kwargs)
    print("--- Done ---")


def parse_csv_list(s: str) -> List[str]:
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


def main() -> None:
    p = argparse.ArgumentParser(description="Build a separate derived-corpus Chroma index.")
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="Path to normalized docs directory")
    p.add_argument(
        "--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist dir for derived corpus"
    )
    p.add_argument("--collection-name", default=None, help="Optional Chroma collection name")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    p.add_argument("--reset", action="store_true", help="Delete persist dir before rebuilding")

    p.add_argument(
        "--no-summaries",
        action="store_true",
        help="Do not include raw episode summaries in the derived index (usually you want them).",
    )

    p.add_argument(
        "--generate-character-bios",
        action="store_true",
        help="Use an LLM to generate character bio cards from summary evidence.",
    )
    p.add_argument(
        "--characters",
        default="Michael Scott,Dwight Schrute,Jim Halpert,Pam Beesly",
        help="Comma-separated character names",
    )

    p.add_argument(
        "--generate-relationship-timelines",
        action="store_true",
        help="Use an LLM to generate romantic relationship timeline cards from summary evidence.",
    )
    p.add_argument(
        "--relationship-subjects",
        default="Michael Scott",
        help="Comma-separated subjects for relationship timeline cards",
    )

    p.add_argument(
        "--max-summary-docs-per-subject",
        type=int,
        default=40,
        help="Max episode summaries to include as evidence per derived doc",
    )

    p.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL, help="LLM model for derived doc generation"
    )

    args = p.parse_args()

    build_derived_index(
        docs_dir=Path(args.docs_dir),
        persist_dir=Path(args.persist_dir),
        collection_name=(args.collection_name if args.collection_name else None),
        embed_model=str(args.embed_model),
        reset=bool(args.reset),
        include_summaries=not bool(args.no_summaries),
        generate_character_bios=bool(args.generate_character_bios),
        characters=parse_csv_list(args.characters),
        max_summary_docs_per_character=int(args.max_summary_docs_per_subject),
        generate_relationship_timelines=bool(args.generate_relationship_timelines),
        relationship_subjects=parse_csv_list(args.relationship_subjects),
        llm_model=str(args.llm_model),
    )


if __name__ == "__main__":
    main()
