from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.fusion import rrf_fuse
from rag.query_expansion import QueryExpansionConfig, build_expander_llm, expand_queries


def build_db(*, persist_directory: str, embed_model: str) -> Chroma:
  embeddings = OpenAIEmbeddings(model=embed_model)
  return Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
  )


def main() -> None:
  load_dotenv()

  parser = argparse.ArgumentParser(description="Ad-hoc retrieval runner.")
  parser.add_argument("--persist-dir", default="db/chroma_db", help="Chroma persist dir")
  parser.add_argument("--embed-model", default="text-embedding-3-small", help="Embedding model")
  parser.add_argument("--query", default="In which episode do Jim and Pam kiss?", help="Search query")
  parser.add_argument("--k", type=int, default=12, help="Top-k results")

  parser.add_argument("--query-expansion", action="store_true", help="Enable query expansion + RRF fusion")
  parser.add_argument("--expand-n", type=int, default=5, help="Number of alternate queries")
  parser.add_argument("--expand-model", default="gpt-4.1-nano", help="Model for query expansion")
  parser.add_argument("--k-per-query", type=int, default=6, help="Docs per expanded query")
  parser.add_argument("--rrf-k0", type=int, default=60, help="RRF k0")
  parser.add_argument("--expand-cache", default="experiments/cache/query_expansion_cache.json", help="Expansion cache")

  args = parser.parse_args()

  db = build_db(persist_directory=args.persist_dir, embed_model=args.embed_model)

  if not args.query_expansion:
    docs = db.similarity_search(args.query, k=int(args.k))
    print(f"Query: {args.query}")
    print("\n--- Top Docs (similarity) ---")
    for i, doc in enumerate(docs, 1):
      src = doc.metadata.get("source")
      first_line = (doc.page_content.splitlines()[0] if doc.page_content else "").strip()
      print(f"\n#{i} source={src}")
      print(first_line)
    return

  expand_cfg = QueryExpansionConfig(
    enabled=True,
    n=int(args.expand_n),
    model=str(args.expand_model),
    temperature=0.0,
    cache_path=(None if not str(args.expand_cache).strip() else __import__("pathlib").Path(args.expand_cache)),
  )
  expander = build_expander_llm(expand_cfg)

  expanded = expand_queries(question=args.query, llm=expander, config=expand_cfg)
  print(f"Question: {args.query}")
  print("\n--- Expanded Queries ---")
  for q in expanded:
    print("-", q)

  per_query = {}
  for q in expanded:
    # Use relevance scores for more useful debugging output.
    per_query[q] = [(d, float(s)) for d, s in db.similarity_search_with_relevance_scores(q, k=int(args.k_per_query))]

  fused = rrf_fuse(per_query, k0=int(args.rrf_k0))

  print("\n--- Fused Top Docs (RRF) ---")
  for i, fd in enumerate(fused[: int(args.k)], 1):
    doc = fd.doc
    src = doc.metadata.get("source")
    first_line = (doc.page_content.splitlines()[0] if doc.page_content else "").strip()
    print(f"\n#{i} rrf={fd.fused_score:.6f} source={src}")
    print("retrieved_by:", "; ".join(fd.retrieved_by))
    print(first_line)


if __name__ == "__main__":
  main()