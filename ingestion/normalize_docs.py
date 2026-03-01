#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# PATHS
#
# This script is meant to be runnable from *any* working directory, e.g.:
#   python ingestion/normalize_docs.py
#   cd ingestion && python normalize_docs.py
#
# All paths are resolved relative to this file's directory.
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DOCS_DIR = SCRIPT_DIR / "raw_docs"

SCRIPTS_MIN_PATH = RAW_DOCS_DIR / "scripts.min.json"
SUMMARIES_PATH = RAW_DOCS_DIR / "episode_summaries.json"

OUTPUT_ROOT = SCRIPT_DIR / "normalized_docs_txt"
OUT_SCRIPTS = OUTPUT_ROOT / "scripts"
OUT_SUMMARIES = OUTPUT_ROOT / "summaries"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


# =============================================================================
# Helpers
# =============================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"\[([^\[\]]+?)\]")  # stage notes like [hangs up]


def strip_html(text: str) -> str:
    """Remove simple HTML tags and normalize whitespace (good enough for TVMaze <p> summaries)."""
    text = _TAG_RE.sub("", text or "")
    return " ".join(text.split()).strip()


def z2(n: int) -> str:
    return f"{n:02d}"


def safe_int(value: Any, field: str) -> int:
    """Coerce int or numeric string to int; raise ValueError otherwise."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s, 10)
    raise ValueError(f"Invalid {field}: {value!r}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_whitespace(s: str) -> str:
    return " ".join((s or "").split()).strip()


def safe_slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "untitled"


def extract_stage_and_dialogue(line: str) -> tuple[list[str], str]:
    """
    Extract stage direction notes and return (stage_notes, remaining_dialogue).

    Example:
      "[on the phone] Hello. [hangs up] Bye."
    returns:
      (["on the phone", "hangs up"], "Hello. Bye.")
    """
    if not line:
        return ([], "")

    stage_notes = [normalize_whitespace(m.group(1)) for m in _BRACKET_RE.finditer(line)]
    dialogue = normalize_whitespace(_BRACKET_RE.sub(" ", line))
    stage_notes = [n for n in stage_notes if n]
    return stage_notes, dialogue


def format_turn_option_a(speaker: str, raw_line: str) -> list[str]:
    """
    Option A:
    - If the line starts with a bracket note, fold the first note into the speaker label:
        Michael (on the phone): Hello...
    - Any remaining bracket notes become standalone lines:
        [STAGE] hangs up

    Returns list of output lines.
    """
    speaker = normalize_whitespace(speaker)
    raw_line = raw_line or ""
    stage_notes, dialogue = extract_stage_and_dialogue(raw_line)

    out: list[str] = []

    if not speaker:
        for note in stage_notes:
            out.append(f"[STAGE] {note}")
        if dialogue:
            out.append(dialogue)
        return out

    leading_stage = bool(stage_notes) and raw_line.lstrip().startswith("[")
    label = speaker
    remaining_notes = stage_notes

    if leading_stage:
        label = f"{speaker} ({stage_notes[0]})"
        remaining_notes = stage_notes[1:]

    if dialogue:
        out.append(f"{label}: {dialogue}")
    else:
        out.append(f"{label}:")

    for note in remaining_notes:
        out.append(f"[STAGE] {note}")

    return out


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    doc_type: str
    out_path: str
    season: int
    episode: int
    title: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "out_path": self.out_path,
            "season": self.season,
            "episode": self.episode,
            "title": self.title,
            "source_path": self.source_path,
        }


# =============================================================================
# Scripts: scripts.min.json -> ONE txt per EPISODE
# =============================================================================


def normalize_scripts_min_to_episode_txt() -> tuple[list[ManifestEntry], int]:
    """
    Input: ingestion/raw_docs/scripts.min.json
      - JSON LIST of episode objects:
        {id, season, episode, title, scenes: [ [ {speaker,line}, ... ], ... ] }

    Output: ingestion/normalized_docs_txt/scripts/season_XX/sXXeYY_<title>_script.txt
      - One file per episode, containing all scenes in order.
      - Stage directions handled with Option A.
    """
    if not SCRIPTS_MIN_PATH.exists():
        raise FileNotFoundError(
            "Scripts file not found. Expected: "
            f"{SCRIPTS_MIN_PATH} (do you have ingestion/raw_docs/scripts.min.json?)"
        )

    data = read_json(SCRIPTS_MIN_PATH)
    if not isinstance(data, list):
        raise ValueError(f"{SCRIPTS_MIN_PATH} must be a JSON list of episode objects")

    entries: list[ManifestEntry] = []
    written = 0

    for ep_obj in data:
        if not isinstance(ep_obj, dict):
            continue

        try:
            season = safe_int(ep_obj.get("season"), "season")
            episode = safe_int(ep_obj.get("episode"), "episode")
        except ValueError:
            continue

        title = ep_obj.get("title")
        title_str = title if isinstance(title, str) and title.strip() else "untitled"
        title_slug = safe_slug(title_str)
        episode_id = ep_obj.get("id") or ""

        scenes = ep_obj.get("scenes")
        if not isinstance(scenes, list):
            continue

        body_lines: list[str] = []
        speakers: set[str] = set()

        # Optional scene separators to preserve structure without separate files
        for scene_index, scene in enumerate(scenes):
            if not isinstance(scene, list):
                continue

            # Add a lightweight scene marker (helps retrieval + later debugging)
            body_lines.append(f"\n=== SCENE {scene_index:03d} ===\n")

            for turn in scene:
                if not isinstance(turn, dict):
                    continue

                speaker = str(turn.get("speaker") or "")
                line = str(turn.get("line") or "")

                if not normalize_whitespace(speaker) and not normalize_whitespace(line):
                    continue

                sp_norm = normalize_whitespace(speaker)
                if sp_norm:
                    speakers.add(sp_norm)

                body_lines.extend(format_turn_option_a(speaker, line))

        if not [ln for ln in body_lines if normalize_whitespace(ln)]:
            continue

        header_lines = [
            "SHOW: The Office",
            f"SEASON: {season}",
            f"EPISODE: {episode}",
            f"TITLE: {title_str}",
            f"EPISODE_ID: {episode_id}",
            f"SOURCE: {SCRIPTS_MIN_PATH}",
            f"SPEAKERS: {', '.join(sorted(speakers))}",
            "",
            "----",
            "",
        ]

        text = "\n".join(header_lines + body_lines)

        doc_id = f"scripts_s{z2(season)}e{z2(episode)}_episode"
        out_path = (
            OUT_SCRIPTS
            / f"season_{z2(season)}"
            / f"s{z2(season)}e{z2(episode)}_{title_slug}_script.txt"
        )

        write_text(out_path, text)
        written += 1

        entries.append(
            ManifestEntry(
                doc_id=doc_id,
                doc_type="script_episode_txt",
                out_path=str(out_path),
                season=season,
                episode=episode,
                title=title_str,
                source_path=str(SCRIPTS_MIN_PATH),
            )
        )

    return entries, written


# =============================================================================
# Summaries: episode_summaries.json -> ONE txt per EPISODE
# =============================================================================


def normalize_episode_summaries_to_episode_txt() -> tuple[list[ManifestEntry], int]:
    """
    Input: ingestion/raw_docs/episode_summaries.json (TVMaze-like list)
      - JSON LIST of episode objects:
        { season, number, name, summary (HTML), airdate, url, id, ... }

    Output: ingestion/normalized_docs_txt/summaries/season_XX/sXXeYY_<title>_summary.txt
      - One summary file per episode.
    """
    if not SUMMARIES_PATH.exists():
        raise FileNotFoundError(
            "Summaries file not found. Expected: "
            f"{SUMMARIES_PATH} (do you have ingestion/raw_docs/episode_summaries.json?)"
        )

    data = read_json(SUMMARIES_PATH)
    if not isinstance(data, list):
        raise ValueError(f"{SUMMARIES_PATH} must be a JSON list of episode objects")

    entries: list[ManifestEntry] = []
    written = 0

    for ep in data:
        if not isinstance(ep, dict):
            continue

        try:
            season = safe_int(ep.get("season"), "season")
            episode = safe_int(ep.get("number"), "number")
        except ValueError:
            continue

        title = ep.get("name")
        title_str = title if isinstance(title, str) and title.strip() else "untitled"
        title_slug = safe_slug(title_str)

        summary = strip_html(str(ep.get("summary") or ""))
        if not summary:
            summary = "(No summary available.)"

        header_lines = [
            "SHOW: The Office",
            f"SEASON: {season}",
            f"EPISODE: {episode}",
            f"TITLE: {title_str}",
            f"AIRDATE: {ep.get('airdate') or ''}",
            f"TVMAZE_ID: {ep.get('id') or ''}",
            f"TVMAZE_URL: {ep.get('url') or ''}",
            f"SOURCE: {SUMMARIES_PATH}",
            "",
            "----",
            "",
        ]

        text = "\n".join(header_lines + [summary])

        doc_id = f"summaries_s{z2(season)}e{z2(episode)}_summary"
        out_path = (
            OUT_SUMMARIES
            / f"season_{z2(season)}"
            / f"s{z2(season)}e{z2(episode)}_{title_slug}_summary.txt"
        )

        write_text(out_path, text)
        written += 1

        entries.append(
            ManifestEntry(
                doc_id=doc_id,
                doc_type="summary_episode_txt",
                out_path=str(out_path),
                season=season,
                episode=episode,
                title=title_str,
                source_path=str(SUMMARIES_PATH),
            )
        )

    return entries, written


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    script_entries, script_written = normalize_scripts_min_to_episode_txt()
    summary_entries, summary_written = normalize_episode_summaries_to_episode_txt()

    manifest = {
        "counts": {
            "script_episode_txt_docs": script_written,
            "summary_episode_txt_docs": summary_written,
            "total_docs": script_written + summary_written,
        },
        "inputs": {
            "scripts_min_path": str(SCRIPTS_MIN_PATH),
            "episode_summaries_path": str(SUMMARIES_PATH),
        },
        "outputs": {
            "output_root": str(OUTPUT_ROOT),
        },
        "documents": [e.to_dict() for e in (script_entries + summary_entries)],
    }

    write_json(MANIFEST_PATH, manifest)

    print("Normalization complete.")
    print(f"Wrote script EPISODE TXT docs:  {script_written}")
    print(f"Wrote summary EPISODE TXT docs: {summary_written}")
    print(f"Manifest:                       {MANIFEST_PATH}")
    print(f"Output root:                    {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()