from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_env() -> None:
    """Load .env without relying on python-dotenv's find_dotenv()."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def _payload_text(payload: Dict[str, Any]) -> str:
    for k in ("text", "page_content", "document"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="End-to-end smoke test: OpenAI embed → Qdrant search → OpenAI answer")
    p.add_argument(
        "--question",
        default="In which episode do Jim and Pam kiss?",
        help="Question to ask",
    )
    p.add_argument(
        "--collection",
        default=(os.environ.get("QDRANT_SCRIPTS_COLLECTION") or "office_scripts"),
        help="Qdrant collection name to query",
    )
    p.add_argument("--k", type=int, default=8, help="Top-k Qdrant hits")
    p.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model")
    p.add_argument("--llm-model", default=(os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"), help="OpenAI chat model")

    args = p.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not qdrant_url:
        raise SystemExit("QDRANT_URL is missing (set it in .env)")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY is missing (set it in .env)")

    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing qdrant-client. Install: pip install -U qdrant-client ({type(e).__name__})")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing langchain-openai. Install: pip install -U langchain-openai langchain-core "
            f"({type(e).__name__})"
        )

    question = str(args.question).strip()
    if not question:
        raise SystemExit("Question is empty")

    embedder = OpenAIEmbeddings(model=str(args.embed_model))
    qvec: List[float] = [float(x) for x in (embedder.embed_query(question) or [])]

    client = QdrantClient(
        url=str(qdrant_url),
        api_key=(os.environ.get("QDRANT_API_KEY") or None),
        timeout=20.0,
    )

    # qdrant-client API has changed across versions; newer versions use query_points().
    # We support both shapes here.
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=str(args.collection),
            query=qvec,
            limit=int(args.k),
            with_payload=True,
        )
        hits = list(getattr(resp, "points", None) or [])
    elif hasattr(client, "search"):
        hits = list(
            client.search(  # type: ignore[no-any-return]
                collection_name=str(args.collection),
                query_vector=qvec,
                limit=int(args.k),
                with_payload=True,
            )
        )
    else:
        raise SystemExit(
            "Unsupported qdrant-client API: expected QdrantClient.query_points() or QdrantClient.search(). "
            "Try upgrading: pip install -U qdrant-client"
        )

    print(f"collection={args.collection} hits={len(hits)}")

    context_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        payload = getattr(h, "payload", None) or {}
        score = getattr(h, "score", None)
        src = payload.get("source") or payload.get("source_relpath") or payload.get("source_file")
        eid = payload.get("episode_id")
        txt = _payload_text(payload)
        first_line = (txt.splitlines()[0].strip() if txt else "")
        score_s = f"{float(score):.4f}" if score is not None else "(none)"
        print(f"#{i} score={score_s} episode_id={eid} source={src} first_line={first_line[:90]!r}")
        if txt:
            context_parts.append(txt)

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a QA assistant for questions about the TV show The Office.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context does not contain the answer, say you don't know.\n"
        "When possible, cite episode identifiers present in the context (e.g., 'S02E06').\n"
        "When you make a specific claim, include at least one short exact quote from the context in double quotes."
    )

    llm = ChatOpenAI(model=str(args.llm_model), temperature=0.0)
    msg = llm.invoke(
        [
            ("system", system),
            ("human", f"Question: {question}\n\nContext:\n{context}"),
        ]
    )

    print("\n--- ANSWER ---")
    print(str(getattr(msg, "content", "") or ""))


if __name__ == "__main__":
    main()
