from __future__ import annotations

from typing import Iterable, Optional

from apps.dashboard.components.html import markdown_html, normalize_html


def section_card(
    *,
    title: str,
    body_html: str,
    variant: str = "default",
    classes: Optional[str] = None,
    title_class: str = "section-title",
    body_class: Optional[str] = "body-text",
) -> None:
    """Render a section card.

    This is the primary building block for the legacy dashboard look.

    Args:
        title: Rendered as a div with `title_class`.
        body_html: Raw HTML fragment for the body.
        variant: "default" or "phase".
        title_class: Typically "section-title"; some legacy cards used "hero-title".
        body_class: If set, wraps body_html in a div of that class; if None, inserts body_html as-is.
    """

    extra_class = ""
    if variant == "phase":
        extra_class = " phase-card"

    extra_classes = str(classes or "").strip()
    if extra_classes:
        extra_class = f"{extra_class} {extra_classes}"

    if body_class:
        body_block = f"<div class=\"{body_class}\">{body_html}</div>"
    else:
        body_block = str(body_html)

    html = f"""
<div class=\"section-card{extra_class}\">
  <div class=\"{title_class}\">{title}</div>
  {body_block}
</div>
"""
    markdown_html(normalize_html(html))


def highlight_box(*, html: str) -> str:
    """Return HTML for a highlight block, intended to be nested inside cards."""

    return normalize_html(f"<div class=\"highlight\">{html}</div>")


def pills(*, labels: Iterable[str]) -> str:
    return "\n".join([f"<span class=\"pill\">{str(l)}</span>" for l in labels])


def muted(*, text: str) -> str:
    return normalize_html(f"<div class=\"muted\">{text}</div>")


def raw_card(*, inner_html: str, classes: Optional[str] = None) -> None:
    """Escape hatch for legacy HTML chunks.

    Use sparingly; prefer `section_card`/`hero_card` building blocks.
    """

    cls = str(classes or "section-card")
    html = f"<div class=\"{cls}\">{inner_html}</div>"
    markdown_html(normalize_html(html))
