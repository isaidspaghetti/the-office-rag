from __future__ import annotations

from typing import Iterable, Optional

from apps.dashboard.components.html import markdown_html, normalize_html


def hero_card(
    *,
    title: str,
    subtitles_html: Iterable[str] = (),
    highlight_html: Optional[str] = None,
    footer_html: Optional[str] = None,
    tone: str = "dark",
) -> None:
    """Render the top hero card.

    `tone`: "dark" (default) or "blue" (analysis methodology hero).
    """

    cls = "hero-card" if tone == "dark" else "hero-card--blue"

    parts = [f"<div class=\"{cls}\">", f"<div class=\"hero-title\">{title}</div>"]

    for s in subtitles_html:
        if not s:
            continue
        parts.append(f"<div class=\"hero-subtitle\">{s}</div>")

    if highlight_html:
        parts.append(f"<div class=\"highlight\">{highlight_html}</div>")

    if footer_html:
        parts.append(footer_html)

    parts.append("</div>")

    markdown_html(normalize_html("\n".join(parts)))
