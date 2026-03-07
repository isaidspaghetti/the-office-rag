from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_json(obj) + "\n", encoding="utf-8")


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _trim_for_substring(s: str, max_chars: int = 220) -> str:
    """Trim without adding characters (so it's still a substring of the original)."""
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip()


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        # 2026-03-04T22:25:40Z
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None


@dataclass(frozen=True)
class EpisodeCardRef:
    episode_id: str
    path: Path
    created_at_utc: Optional[datetime]
    build_id: str


def _iter_episode_card_paths(out_root: Path) -> Iterable[Path]:
    yield from out_root.rglob("episode_cards/*.json")


def _pick_latest_episode_cards(out_root: Path) -> Dict[str, EpisodeCardRef]:
    """Pick the newest EpisodeDerivedCardV1 per episode_id based on created_at_utc."""
    best: Dict[str, EpisodeCardRef] = {}

    for path in _iter_episode_card_paths(out_root):
        try:
            raw = _read_json(path)
        except Exception:
            continue
        if raw.get("schema") != "EpisodeDerivedCardV1":
            continue

        episode_id = str(raw.get("episode_id") or "").strip().upper()
        if not episode_id:
            continue

        created_at = _parse_dt(str(raw.get("created_at_utc") or ""))
        build_id = str(raw.get("build_id") or "")
        ref = EpisodeCardRef(
            episode_id=episode_id,
            path=path,
            created_at_utc=created_at,
            build_id=build_id,
        )

        prev = best.get(episode_id)
        if prev is None:
            best[episode_id] = ref
            continue

        # Prefer those with parsed created_at; fall back to lexical build_id.
        if prev.created_at_utc and created_at:
            if created_at > prev.created_at_utc:
                best[episode_id] = ref
            continue
        if (created_at is not None) and (prev.created_at_utc is None):
            best[episode_id] = ref
            continue
        if (created_at is None) and (prev.created_at_utc is not None):
            continue

        if ref.build_id > prev.build_id:
            best[episode_id] = ref

    return best


def _evidence_snippets(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect all EvidenceSpan-like objects found in known fields."""

    out: List[Dict[str, Any]] = []

    def _collect(x: Any) -> None:
        if isinstance(x, dict):
            # Evidence spans in our derived pipeline always have these keys.
            if {"episode_id", "source", "char_start", "char_end", "snippet"}.issubset(x.keys()):
                out.append(x)
            for v in x.values():
                _collect(v)
        elif isinstance(x, list):
            for v in x:
                _collect(v)

    _collect(card.get("main_threads"))
    _collect(card.get("character_highlights"))
    _collect(card.get("relationships"))
    return out


def _auto_character_names(episode_cards: Sequence[Dict[str, Any]], *, top_n: int = 40) -> List[str]:
    counts: Counter[str] = Counter()
    for card in episode_cards:
        for h in card.get("character_highlights") or []:
            name = str((h or {}).get("character") or "").strip()
            if name:
                counts[name] += 1
    return [name for (name, _) in counts.most_common(top_n)]


def _auto_relationship_pairs(episode_cards: Sequence[Dict[str, Any]], *, top_n: int = 80) -> List[Tuple[str, str]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for card in episode_cards:
        for rel in card.get("relationships") or []:
            pair = (rel or {}).get("pair")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a = str(pair[0] or "").strip()
            b = str(pair[1] or "").strip()
            if not a or not b:
                continue
            key = tuple(sorted([a, b]))  # type: ignore
            counts[(key[0], key[1])] += 1
    return [pair for (pair, _) in counts.most_common(top_n)]


def _select_supporting_quotes(
    evidence: Sequence[Dict[str, Any]],
    *,
    prefer_terms: Optional[Sequence[str]] = None,
    max_quotes: int = 3,
    max_chars: int = 220,
) -> List[Dict[str, Any]]:
    """Pick up to N evidence spans with short, high-signal snippets."""
    prefer_terms = [t for t in (prefer_terms or []) if str(t).strip()]
    prefer_terms_l = [str(t).lower() for t in prefer_terms]

    def _is_preferred(snippet: str) -> bool:
        if not prefer_terms_l:
            return True
        s = (snippet or "").lower()
        return any(t in s for t in prefer_terms_l)

    # Prefer snippets that mention the entity/object, look like dialogue, and are short.
    scored: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []
    for ev in evidence:
        snippet = str(ev.get("snippet") or "")
        if not snippet.strip():
            continue
        trimmed = _trim_for_substring(snippet, max_chars=max_chars)
        preferred = 1 if _is_preferred(trimmed) else 0
        looks_like_dialogue = 1 if ":" in trimmed else 0
        score = (
            -preferred,
            -looks_like_dialogue,
            len(trimmed),
            int(ev.get("char_start") or 0),
        )
        out_ev = dict(ev)
        out_ev["quote"] = trimmed
        scored.append((score, out_ev))

    scored.sort(key=lambda x: x[0])

    seen_quote = set()
    used_episodes: set[str] = set()
    picked: List[Dict[str, Any]] = []

    # Pass 1: diversify by episode when possible.
    for _, ev in scored:
        ep = str(ev.get("episode_id") or "")
        key = (ep, str(ev.get("quote")))
        if key in seen_quote:
            continue
        if ep and ep in used_episodes:
            continue
        seen_quote.add(key)
        if ep:
            used_episodes.add(ep)
        picked.append(ev)
        if len(picked) >= max_quotes:
            return picked

    # Pass 2: allow repeats.
    for _, ev in scored:
        ep = str(ev.get("episode_id") or "")
        key = (ep, str(ev.get("quote")))
        if key in seen_quote:
            continue
        seen_quote.add(key)
        picked.append(ev)
        if len(picked) >= max_quotes:
            break

    return picked


def build_character_cards(
    *,
    build_id: str,
    episode_cards: Sequence[Dict[str, Any]],
    character_names: Sequence[str],
    aliases_by_name: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"num_characters": 0}

    by_character: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for card in episode_cards:
        for h in card.get("character_highlights") or []:
            name = str((h or {}).get("character") or "").strip()
            if name:
                by_character[name].append({"episode_id": card.get("episode_id"), **(h or {})})

    for name in character_names:
        highlights = by_character.get(name, [])
        if not highlights:
            continue

        aliases = list((aliases_by_name or {}).get(name) or [])

        episode_ids = sorted({str(h.get("episode_id") or "").strip().upper() for h in highlights if h.get("episode_id")})

        key_facts: List[str] = []
        for h in highlights:
            wc = str(h.get("what_changes") or "").strip()
            if wc and wc not in key_facts:
                key_facts.append(wc)
            if len(key_facts) >= 12:
                break

        evidence: List[Dict[str, Any]] = []
        for h in highlights:
            for ev in (h.get("evidence") or []):
                if isinstance(ev, dict):
                    evidence.append(ev)

        supporting_quotes = _select_supporting_quotes(
            evidence,
            prefer_terms=[f"{name}:", name],
            max_quotes=3,
        )

        card = {
            "schema": "CharacterCardV1",
            "schema_version": 1,
            "build_id": build_id,
            "created_at_utc": utc_now_iso(),
            "doc_type": "derived",
            "derived_type": "character_card",
            "entity_names": [name],
            "character": {
                "name": name,
                "aliases": aliases,
            },
            "episode_ids": episode_ids,
            "key_facts": key_facts,
            "supporting_quotes": supporting_quotes,
        }
        cards.append(card)

    stats["num_characters"] = len(cards)
    return cards, stats


def build_relationship_cards(
    *,
    build_id: str,
    episode_cards: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[str, str]],
    labels_by_pair: Optional[Dict[Tuple[str, str], str]] = None,
    aliases_by_pair: Optional[Dict[Tuple[str, str], List[str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"num_relationships": 0}

    # Gather relationship entries keyed by sorted pair.
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for card in episode_cards:
        for rel in card.get("relationships") or []:
            pair = (rel or {}).get("pair")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a = str(pair[0] or "").strip()
            b = str(pair[1] or "").strip()
            if not a or not b:
                continue
            key = tuple(sorted([a, b]))  # type: ignore
            by_pair[(key[0], key[1])].append({"episode_id": card.get("episode_id"), **(rel or {})})

    for a, b in pairs:
        key_pair = tuple(sorted([a, b]))  # type: ignore
        entries = by_pair.get((key_pair[0], key_pair[1]), [])
        if not entries:
            continue

        label = str((labels_by_pair or {}).get((key_pair[0], key_pair[1])) or f"{a}–{b}")
        aliases = list((aliases_by_pair or {}).get((key_pair[0], key_pair[1])) or [])

        episode_ids = sorted({str(e.get("episode_id") or "").strip().upper() for e in entries if e.get("episode_id")})

        key_facts: List[str] = []
        for e in entries:
            status = str(e.get("status") or "").strip()
            if status and status not in key_facts:
                key_facts.append(status)
            if len(key_facts) >= 12:
                break

        evidence: List[Dict[str, Any]] = []
        for e in entries:
            for ev in (e.get("evidence") or []):
                if isinstance(ev, dict):
                    evidence.append(ev)

        supporting_quotes = _select_supporting_quotes(
            evidence,
            prefer_terms=[f"{a}:", f"{b}:", a, b],
            max_quotes=3,
        )

        card = {
            "schema": "RelationshipCardV1",
            "schema_version": 1,
            "build_id": build_id,
            "created_at_utc": utc_now_iso(),
            "doc_type": "derived",
            "derived_type": "relationship_card",
            "entity_names": [a, b],
            "relationship": {
                "entities": [a, b],
                "label": label,
                "aliases": aliases,
            },
            "episode_ids": episode_ids,
            "key_facts": key_facts,
            "supporting_quotes": supporting_quotes,
        }
        cards.append(card)

    stats["num_relationships"] = len(cards)
    return cards, stats


def _object_mentions_in_episode_card(obj_aliases: Sequence[str], card: Dict[str, Any]) -> bool:
    haystacks: List[str] = []
    haystacks.append(str(card.get("one_paragraph_synopsis") or ""))

    for t in card.get("main_threads") or []:
        haystacks.append(str((t or {}).get("thread") or ""))

    for tag in card.get("tags") or []:
        haystacks.append(str(tag or ""))

    joined = "\n".join([h for h in haystacks if h])
    joined_low = joined.lower()
    return any(a.lower() in joined_low for a in obj_aliases if a)


def build_plot_object_cards(
    *,
    build_id: str,
    episode_cards: Sequence[Dict[str, Any]],
    objects: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"num_objects": 0, "skipped_objects_no_matches": 0}

    for obj in objects:
        name = str(obj.get("name") or "").strip()
        aliases = [name] + [str(a).strip() for a in (obj.get("aliases") or []) if str(a).strip()]
        aliases = [a for a in aliases if a]
        if not name or not aliases:
            continue

        matched_cards: List[Dict[str, Any]] = []
        for card in episode_cards:
            if _object_mentions_in_episode_card(aliases, card):
                matched_cards.append(card)

        if not matched_cards:
            stats["skipped_objects_no_matches"] += 1
            continue

        episode_ids = sorted({str(c.get("episode_id") or "").strip().upper() for c in matched_cards if c.get("episode_id")})

        # Key facts: pull the best-matching thread/tag/synopsis sentence fragments (simple heuristic).
        key_facts: List[str] = []
        for c in matched_cards:
            syn = str(c.get("one_paragraph_synopsis") or "").strip()
            if syn:
                key_facts.append(syn)
            for t in c.get("main_threads") or []:
                thr = str((t or {}).get("thread") or "").strip()
                if thr:
                    key_facts.append(thr)
            if len(key_facts) >= 12:
                break
        # Dedupe while preserving order.
        seen_fact = set()
        key_facts_deduped: List[str] = []
        for f in key_facts:
            if f in seen_fact:
                continue
            seen_fact.add(f)
            key_facts_deduped.append(f)
        key_facts = key_facts_deduped[:12]

        # Evidence: reuse any evidence spans from the matched episode cards.
        evidence: List[Dict[str, Any]] = []
        for c in matched_cards:
            evidence.extend(_evidence_snippets(c))

        supporting_quotes = _select_supporting_quotes(
            evidence,
            prefer_terms=aliases,
            max_quotes=3,
        )

        card = {
            "schema": "PlotObjectCardV1",
            "schema_version": 1,
            "build_id": build_id,
            "created_at_utc": utc_now_iso(),
            "doc_type": "derived",
            "derived_type": "plot_object_card",
            "entity_names": [name],
            "plot_object": {
                "name": name,
                "aliases": [a for a in aliases if a != name],
            },
            "episode_ids": episode_ids,
            "key_facts": key_facts,
            "supporting_quotes": supporting_quotes,
        }
        cards.append(card)

    stats["num_objects"] = len(cards)
    return cards, stats


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return _read_json(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build entity-focused derived corpora (character/relationship/plot-object cards) from EpisodeDerivedCardV1 artifacts."
    )
    p.add_argument(
        "--episode-cards-root",
        default="derived/artifacts",
        help="Root directory containing derived episode_cards/**.json artifacts",
    )
    p.add_argument(
        "--out-root",
        default="derived/artifacts",
        help="Output root for the new corpora build directory",
    )
    p.add_argument(
        "--build-prefix",
        default=None,
        help="Output directory name under out-root (default: entitycards_v1_<UTC date>)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional config JSON (see derived/entity_corpora/config.example.json)",
    )
    p.add_argument(
        "--no-auto-characters",
        action="store_true",
        help="Disable auto character selection; requires config.characters",
    )
    p.add_argument(
        "--no-auto-relationships",
        action="store_true",
        help="Disable auto relationship selection; requires config.relationships",
    )
    p.add_argument(
        "--top-n-characters",
        type=int,
        default=40,
        help="When auto-picking characters, how many to include",
    )
    p.add_argument(
        "--top-n-relationships",
        type=int,
        default=80,
        help="When auto-picking relationships, how many pairs to include",
    )

    args = p.parse_args()

    episode_cards_root = Path(args.episode_cards_root)
    out_root = Path(args.out_root)

    build_prefix = args.build_prefix
    if not build_prefix:
        build_prefix = f"entitycards_v1_{datetime.now(timezone.utc).date().isoformat()}"

    build_dir = out_root / build_prefix

    cfg = _load_config(Path(args.config) if args.config else None)

    latest = _pick_latest_episode_cards(episode_cards_root)
    episode_cards: List[Dict[str, Any]] = []
    input_build_ids: List[str] = []
    for episode_id in sorted(latest.keys()):
        ref = latest[episode_id]
        raw = _read_json(ref.path)
        episode_cards.append(raw)
        if ref.build_id:
            input_build_ids.append(ref.build_id)

    if not episode_cards:
        raise SystemExit(
            f"No EpisodeDerivedCardV1 episode_cards found under {episode_cards_root}. "
            "Run the derived pipeline first (segment/map/reduce)."
        )

    build_id = build_prefix

    # Characters
    cfg_characters = cfg.get("characters") or []
    character_names: List[str] = []
    aliases_by_name: Dict[str, List[str]] = {}
    if cfg_characters:
        for c in cfg_characters:
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                if name:
                    character_names.append(name)
                    aliases = [str(a).strip() for a in (c.get("aliases") or []) if str(a).strip()]
                    if aliases:
                        aliases_by_name[name] = aliases
            elif isinstance(c, str) and c.strip():
                character_names.append(c.strip())
    elif not bool(args.no_auto_characters):
        character_names = _auto_character_names(episode_cards, top_n=int(args.top_n_characters))

    if not character_names:
        raise SystemExit(
            "No characters selected. Provide --config with characters or omit --no-auto-characters."
        )

    character_cards, char_stats = build_character_cards(
        build_id=build_id,
        episode_cards=episode_cards,
        character_names=character_names,
        aliases_by_name=aliases_by_name,
    )

    # Relationships
    cfg_relationships = cfg.get("relationships") or []
    pairs: List[Tuple[str, str]] = []
    labels_by_pair: Dict[Tuple[str, str], str] = {}
    aliases_by_pair: Dict[Tuple[str, str], List[str]] = {}
    if cfg_relationships:
        for r in cfg_relationships:
            if isinstance(r, dict):
                ents = r.get("entities")
                if isinstance(ents, list) and len(ents) == 2:
                    a = str(ents[0] or "").strip()
                    b = str(ents[1] or "").strip()
                    if a and b:
                        pairs.append((a, b))
                        key_pair = tuple(sorted([a, b]))  # type: ignore
                        label = str(r.get("label") or "").strip()
                        if label:
                            labels_by_pair[(key_pair[0], key_pair[1])] = label
                        aliases = [str(x).strip() for x in (r.get("aliases") or []) if str(x).strip()]
                        if aliases:
                            aliases_by_pair[(key_pair[0], key_pair[1])] = aliases
    elif not bool(args.no_auto_relationships):
        pairs = _auto_relationship_pairs(episode_cards, top_n=int(args.top_n_relationships))

    if not pairs:
        raise SystemExit(
            "No relationships selected. Provide --config with relationships or omit --no-auto-relationships."
        )

    relationship_cards, rel_stats = build_relationship_cards(
        build_id=build_id,
        episode_cards=episode_cards,
        pairs=pairs,
        labels_by_pair=labels_by_pair,
        aliases_by_pair=aliases_by_pair,
    )

    # Plot objects
    cfg_objects = cfg.get("plot_objects") or []
    objects: List[Dict[str, Any]] = []
    for o in cfg_objects:
        if isinstance(o, dict) and str(o.get("name") or "").strip():
            objects.append(o)

    plot_object_cards, obj_stats = build_plot_object_cards(
        build_id=build_id,
        episode_cards=episode_cards,
        objects=objects,
    )

    # Write outputs
    character_dir = build_dir / "character_cards"
    relationship_dir = build_dir / "relationship_cards"
    object_dir = build_dir / "plot_object_cards"

    for c in character_cards:
        slug = _slugify(str((c.get("character") or {}).get("name") or ""))
        _write_json(character_dir / f"{slug}.json", c)

    for r in relationship_cards:
        rel = r.get("relationship") or {}
        ents = rel.get("entities") or []
        label = "_".join([_slugify(str(x)) for x in ents])
        _write_json(relationship_dir / f"{label}.json", r)

    for o in plot_object_cards:
        slug = _slugify(str((o.get("plot_object") or {}).get("name") or ""))
        _write_json(object_dir / f"{slug}.json", o)

    manifest = {
        "schema": "EntityCorporaBuildManifestV1",
        "schema_version": 1,
        "build_id": build_id,
        "created_at_utc": utc_now_iso(),
        "episode_cards_root": str(episode_cards_root.as_posix()),
        "inputs": {
            "num_latest_episode_cards": len(episode_cards),
            "episode_card_build_ids": sorted(set(input_build_ids)),
        },
        "outputs": {
            "character_cards_dir": str(character_dir.as_posix()),
            "relationship_cards_dir": str(relationship_dir.as_posix()),
            "plot_object_cards_dir": str(object_dir.as_posix()),
        },
        "stats": {
            **char_stats,
            **rel_stats,
            **obj_stats,
        },
        "config_used": cfg,
    }
    _write_json(build_dir / "manifest.json", manifest)

    print(f"Wrote entity corpora build to: {build_dir}")
    print(_safe_json(manifest["stats"]))


if __name__ == "__main__":
    main()
