from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:
	from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
	load_dotenv = None  # type: ignore


# Allow importing repo modules when executed as a script.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

try:
	from qdrant_client import QdrantClient  # type: ignore
	from langchain_community.vectorstores import Qdrant as QdrantVS  # type: ignore

	_HAS_QDRANT = True
except Exception:  # pragma: no cover
	QdrantClient = None  # type: ignore
	QdrantVS = None  # type: ignore
	_HAS_QDRANT = False

from ingestion.chunk_documents import chunk_documents_by_scene
from ingestion.load_documents import load_documents

from derived.build_derived_cards_index import (  # re-use existing derived-card render + discovery
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
	parse_csv_list,
)


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_SCRIPTS_COLLECTION = "office_scripts"
DEFAULT_DERIVED_COLLECTION = "office_derived_cards"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
	s = os.environ.get(key)
	if s is None:
		return default
	s2 = str(s).strip()
	return s2 if s2 else default


def _resolve_under_repo(path_str: str) -> Path:
	p = Path(path_str)
	if not p.is_absolute():
		p = (_REPO_ROOT / p).resolve()
	return p


def _short_path(p: Path) -> str:
	try:
		return p.resolve().relative_to(_REPO_ROOT).as_posix()
	except Exception:
		return p.as_posix()


def _require_env(key: str) -> str:
	v = _env(key)
	if not v:
		raise EnvironmentError(f"Missing {key}. Set it in your environment or .env")
	return v


def _passes_season_filter(season: Optional[int], seasons_filter: Optional[Sequence[int]]) -> bool:
	if not seasons_filter:
		return True
	if season is None:
		return False
	return int(season) in {int(x) for x in seasons_filter}


@dataclass(frozen=True)
class IndexConfig:
	qdrant_url: str
	qdrant_api_key: Optional[str]
	scripts_collection: str
	derived_collection: str
	embed_model: str

	docs_path: Path
	derived_out_root: Path

	use_metadata_headers: bool
	window_size: int
	chunk_size: int
	chunk_overlap: int
	seasons: Optional[List[int]]

	recreate: bool
	dry_run: bool


def _load_and_chunk_scripts(cfg: IndexConfig) -> List[Document]:
	docs = load_documents(
		docs_path=str(cfg.docs_path),
		use_metadata=bool(cfg.use_metadata_headers),
		show_progress=True,
	)

	chunks = chunk_documents_by_scene(
		docs,
		window_size=int(cfg.window_size),
		fallback_chunk_size=int(cfg.chunk_size),
		fallback_chunk_overlap=int(cfg.chunk_overlap),
	)

	# Normalize absolute paths so deployed UI doesn't show local machine paths.
	normalized: List[Document] = []
	for d in chunks:
		meta = dict(d.metadata or {})
		src = meta.get("source")
		if src:
			try:
				meta["source"] = _short_path(Path(str(src)))
			except Exception:
				pass
		normalized.append(Document(page_content=d.page_content, metadata=meta))
	return normalized


def _load_derived_cards(cfg: IndexConfig) -> List[Document]:
	out_root = cfg.derived_out_root
	if not out_root.exists():
		raise FileNotFoundError(
			f"Derived artifacts root not found: {out_root}. "
			"Build derived artifacts under derived/artifacts/ first."
		)

	episode_prefixes = _auto_detect_episode_card_prefixes(out_root)
	season_prefixes = _auto_detect_season_card_prefixes(out_root)
	topic_prefixes = _auto_detect_topic_card_prefixes(out_root)

	docs: List[Document] = []

	for pref in episode_prefixes:
		for p in _iter_episode_card_files(out_root, pref):
			card = json.loads(p.read_text(encoding="utf-8"))
			if str(card.get("schema") or "") != "EpisodeDerivedCardV1":
				continue

			episode_id = str(card.get("episode_id") or "").strip().upper()
			season, episode = _season_episode_from_id(episode_id)
			if not _passes_season_filter(season, cfg.seasons):
				continue

			title = str(card.get("title") or "").strip()
			build_id = str(card.get("build_id") or "").strip()
			stable_build_tag = build_id or pref

			meta = {
				"doc_type": "derived",
				"derived_type": "episode_card",
				"episode_id": episode_id,
				"season": season,
				"episode": episode,
				"title": title,
				"build_id": build_id,
				"source_file": _short_path(p),
				"episode_cards_build_prefix": pref,
				"stable_doc_id": f"episode_card:{episode_id}:{stable_build_tag}",
			}
			docs.append(Document(page_content=_render_episode_card(card), metadata=meta))

	for pref in season_prefixes:
		for p in _iter_season_card_files(out_root, pref):
			card = json.loads(p.read_text(encoding="utf-8"))
			if str(card.get("schema") or "") != "SeasonDerivedCardV1":
				continue

			season = int(card.get("season") or 0) or None
			if not _passes_season_filter(season, cfg.seasons):
				continue

			build_id = str(card.get("build_id") or "").strip()
			stable_build_tag = build_id or pref

			meta = {
				"doc_type": "derived",
				"derived_type": "season_card",
				"season": season,
				"build_id": build_id,
				"source_file": _short_path(p),
				"season_cards_build_prefix": pref,
				"stable_doc_id": f"season_card:{int(season) if season is not None else 0}:{stable_build_tag}",
			}
			docs.append(Document(page_content=_render_season_card(card), metadata=meta))

	for pref in topic_prefixes:
		for p in _iter_topic_card_files(out_root, pref):
			card = json.loads(p.read_text(encoding="utf-8"))
			if str(card.get("schema") or "") != "TopicDerivedCardV1":
				continue

			# Topic cards may reference multiple episodes; keep as-is.
			build_id = str(card.get("build_id") or "").strip()
			stable_build_tag = build_id or pref
			topic_id = str(card.get("topic_id") or "").strip()

			episode_ids = [
				str(x).strip().upper() for x in (card.get("episode_ids") or []) if str(x).strip()
			]
			seasons: List[int] = []
			for eid in episode_ids:
				s, _ = _season_episode_from_id(eid)
				if s is not None:
					seasons.append(int(s))
			if cfg.seasons:
				# Keep the card if it overlaps the filter.
				if not ({int(x) for x in seasons} & {int(x) for x in cfg.seasons}):
					continue

			meta = {
				"doc_type": "derived",
				"derived_type": "topic_card",
				"topic_id": topic_id,
				"topic_type": str(card.get("topic_type") or "").strip(),
				"episode_ids": episode_ids,
				"seasons": sorted({int(x) for x in seasons}) if seasons else [],
				"build_id": build_id,
				"source_file": _short_path(p),
				"topic_cards_build_prefix": pref,
				"stable_doc_id": f"topic_card:{topic_id}:{stable_build_tag}" if topic_id else f"topic_card:{_short_path(p)}",
			}
			docs.append(Document(page_content=_render_topic_card(card), metadata=meta))

	return docs


def _delete_collection_if_exists(client: Any, collection: str) -> None:
	try:
		client.delete_collection(collection_name=str(collection))
	except Exception:
		# Qdrant raises if missing; that's fine.
		return


def _index_documents(
	*,
	client: Any,
	collection: str,
	embeddings: OpenAIEmbeddings,
	documents: List[Document],
	recreate: bool,
	dry_run: bool,
) -> None:
	print(f"\nCollection: {collection}")
	print(f"Documents: {len(documents)}")

	if dry_run:
		return

	if recreate:
		print("Recreate: deleting collection (if exists)…")
		_delete_collection_if_exists(client, collection)

	if not documents:
		print("No documents to index; skipping.")
		return

	# Prefer deterministic IDs when supported by the installed langchain-community version.
	ids = [str(d.metadata.get("stable_doc_id") or f"doc_{i}") for i, d in enumerate(documents)]

	try:
		QdrantVS.from_documents(
			documents=documents,
			embedding=embeddings,
			client=client,
			collection_name=str(collection),
			ids=ids,
		)
	except TypeError:
		# Older signature (no ids).
		QdrantVS.from_documents(
			documents=documents,
			embedding=embeddings,
			client=client,
			collection_name=str(collection),
		)

	print("Indexed successfully.")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Index scripts + derived cards into Qdrant for Streamlit Cloud.")
	p.add_argument(
		"--mode",
		choices=["scripts", "derived", "both"],
		default="both",
		help="Which collection(s) to index.",
	)

	p.add_argument("--qdrant-url", default=_env("QDRANT_URL"), help="QDRANT_URL override")
	p.add_argument("--qdrant-api-key", default=_env("QDRANT_API_KEY"), help="QDRANT_API_KEY override")
	p.add_argument(
		"--scripts-collection",
		default=_env("QDRANT_SCRIPTS_COLLECTION", DEFAULT_SCRIPTS_COLLECTION),
		help="Qdrant collection name for scripts.",
	)
	p.add_argument(
		"--derived-collection",
		default=_env("QDRANT_DERIVED_COLLECTION", DEFAULT_DERIVED_COLLECTION),
		help="Qdrant collection name for derived cards.",
	)
	p.add_argument("--embed-model", default=_env("EMBED_MODEL", DEFAULT_EMBED_MODEL))

	p.add_argument(
		"--docs-path",
		default="ingestion/normalized_docs_txt",
		help="Path to normalized script+summary txt corpus.",
	)
	p.add_argument(
		"--derived-out-root",
		default="derived/artifacts",
		help="Root folder for derived artifacts.",
	)
	p.add_argument(
		"--use-metadata-headers",
		action="store_true",
		help="Parse normalized TXT headers into metadata (recommended).",
	)

	p.add_argument("--window-size", type=int, default=1, help="Scene window size for script chunking.")
	p.add_argument("--chunk-size", type=int, default=1000, help="Fallback chunk size for non-scene docs.")
	p.add_argument("--chunk-overlap", type=int, default=150, help="Fallback chunk overlap.")

	p.add_argument(
		"--seasons",
		default=None,
		help="Optional season filter for derived cards (e.g. '1,2,3').",
	)

	p.add_argument("--recreate", action="store_true", help="Drop and recreate the collection(s).")
	p.add_argument("--dry-run", action="store_true", help="Load/prepare docs but do not upload.")
	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	if load_dotenv is not None:
		load_dotenv()

	args = _parse_args(argv)

	if not _HAS_QDRANT:
		raise RuntimeError(
			"Qdrant deps missing. Install `qdrant-client` and `langchain-community` (see requirements.txt)."
		)

	qdrant_url = str(args.qdrant_url).strip() if args.qdrant_url else _env("QDRANT_URL")
	if not args.dry_run:
		qdrant_url = qdrant_url or _require_env("QDRANT_URL")
		openai_key = _require_env("OPENAI_API_KEY")
		os.environ.setdefault("OPENAI_API_KEY", openai_key)

	seasons = parse_csv_list(str(args.seasons or "")) if args.seasons else []
	seasons_i: Optional[List[int]] = None
	if seasons:
		seasons_i = [int(x) for x in seasons if str(x).strip().isdigit()]

	cfg = IndexConfig(
		qdrant_url=str(qdrant_url or ""),
		qdrant_api_key=str(args.qdrant_api_key).strip() if args.qdrant_api_key else None,
		scripts_collection=str(args.scripts_collection),
		derived_collection=str(args.derived_collection),
		embed_model=str(args.embed_model),
		docs_path=_resolve_under_repo(str(args.docs_path)),
		derived_out_root=_resolve_under_repo(str(args.derived_out_root)),
		use_metadata_headers=bool(args.use_metadata_headers),
		window_size=int(args.window_size),
		chunk_size=int(args.chunk_size),
		chunk_overlap=int(args.chunk_overlap),
		seasons=seasons_i,
		recreate=bool(args.recreate),
		dry_run=bool(args.dry_run),
	)

	print("Qdrant indexing")
	print(f"- URL: {cfg.qdrant_url or '(not set)'}")
	print(f"- Embed model: {cfg.embed_model}")
	print(f"- Mode: {args.mode}")
	print(f"- Dry run: {cfg.dry_run}")
	print(f"- Recreate: {cfg.recreate}")

	if cfg.dry_run:
		if args.mode in {"scripts", "both"}:
			docs = _load_and_chunk_scripts(cfg)
			print(f"\nScripts prepared: {len(docs)} chunks")
		if args.mode in {"derived", "both"}:
			docs = _load_derived_cards(cfg)
			print(f"\nDerived prepared: {len(docs)} docs")
		return 0

	client = QdrantClient(url=str(cfg.qdrant_url), api_key=(cfg.qdrant_api_key or None))
	embeddings = OpenAIEmbeddings(model=str(cfg.embed_model))

	if args.mode in {"scripts", "both"}:
		docs = _load_and_chunk_scripts(cfg)
		_index_documents(
			client=client,
			collection=str(cfg.scripts_collection),
			embeddings=embeddings,
			documents=docs,
			recreate=cfg.recreate,
			dry_run=cfg.dry_run,
		)

	if args.mode in {"derived", "both"}:
		docs = _load_derived_cards(cfg)
		_index_documents(
			client=client,
			collection=str(cfg.derived_collection),
			embeddings=embeddings,
			documents=docs,
			recreate=cfg.recreate,
			dry_run=cfg.dry_run,
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
