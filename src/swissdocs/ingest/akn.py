"""Parse Fedlex Akoma Ntoso consolidations into articles.

Amendment notes (``authorialNote``) are dropped so retrieval sees the operative
text rather than the legislative history.
"""

import re
from xml.etree import ElementTree as ET

from swissdocs.models import ParsedArticle
from swissdocs.text import clean_block, clean_inline

AKN = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

_STRUCTURAL = {
    "book",
    "tome",
    "part",
    "title",
    "chapter",
    "section",
    "subsection",
    "level",
}
_SKIP = {"authorialNote"}
_REPEALED = re.compile(
    r"^(abrog|aufgehoben|repealed|abrogato)\w*\b",
    re.IGNORECASE,
)
_ARTICLE_NUM = re.compile(r"\d+[a-z]*(?:-\d+[a-z]*)?", re.IGNORECASE)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _extract(el: ET.Element) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        name = _local(node.tag)
        if name in _SKIP:
            return
        if name in {"item", "listIntroduction", "paragraph"}:
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return clean_inline(" ".join(parts))


def _article_label(num_el: ET.Element | None, eId: str) -> str:
    """Keep letter suffixes that Fedlex stores outside the bold number (Art. 1*a*)."""
    if num_el is not None:
        raw = _extract(num_el)
        raw = raw.replace("Art.", " ")
        raw = raw.replace("à", "-").replace("–", "-")
        raw = re.sub(r"(\d)\s+([a-z]+)", r"\1\2", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*-\s*", "-", raw)
        match = _ARTICLE_NUM.search(raw)
        if match:
            return match.group(0)
    fallback = eId.removeprefix("art_").replace("_", "")
    return fallback or eId


def _paragraph_text(paragraph: ET.Element) -> str:
    number = ""
    pieces: list[str] = []
    for child in list(paragraph):
        name = _local(child.tag)
        if name in _SKIP:
            continue
        if name == "num":
            number = _extract(child).strip()
            continue
        pieces.append(_extract(child))
    body = clean_inline(" ".join(pieces))
    if number and body:
        return f"{number} {body}"
    return body or number


def _version_date(root: ET.Element) -> str:
    for el in root.iter(f"{{{AKN}}}FRBRdate"):
        name = el.attrib.get("name", "")
        if name.endswith("dateApplicability") and el.attrib.get("date"):
            return el.attrib["date"]
    return ""


def _names(root: ET.Element) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for el in root.iter(f"{{{AKN}}}FRBRname"):
        lang = el.attrib.get(XML_LANG, "")
        if not lang:
            continue
        found[lang] = (el.attrib.get("shortForm", ""), el.attrib.get("value", ""))
    return found


def _sr_number(root: ET.Element) -> str:
    for el in root.iter(f"{{{AKN}}}FRBRnumber"):
        if el.attrib.get("value"):
            return el.attrib["value"]
    return ""


def _heading_path(article: ET.Element, parents: dict[ET.Element, ET.Element]) -> list[str]:
    path: list[str] = []
    node: ET.Element | None = parents.get(article)
    while node is not None:
        if _local(node.tag) in _STRUCTURAL:
            heading = next((child for child in list(node) if _local(child.tag) == "heading"), None)
            if heading is not None:
                label = _extract(heading)
                if label:
                    path.append(label)
        node = parents.get(node)
    path.reverse()
    return path


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    parents: dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parents[child] = parent
    return parents


def parse_akn(
    xml_bytes: bytes,
    *,
    language: str,
    law_sr: str,
    law_abbr: str,
    law_title: str,
    eli: str,
    source_url: str,
    binding: bool,
) -> list[ParsedArticle]:
    root = ET.fromstring(xml_bytes)
    parents = _parent_map(root)
    version = _version_date(root) or "unknown"
    names = _names(root)
    short, title = names.get(language, ("", ""))
    abbr = short or law_abbr
    full_title = title or law_title
    declared_sr = _sr_number(root)
    sr = declared_sr or law_sr

    articles: list[ParsedArticle] = []
    for article in root.iter(f"{{{AKN}}}article"):
        eId = article.attrib.get("eId", "")
        num_el = next((child for child in list(article) if _local(child.tag) == "num"), None)
        heading_el = next(
            (child for child in list(article) if _local(child.tag) == "heading"),
            None,
        )
        paragraphs = [
            _paragraph_text(child) for child in list(article) if _local(child.tag) == "paragraph"
        ]
        text = clean_block("\n".join(part for part in paragraphs if part))
        if not text:
            text = clean_block(_extract(article))
        without_number = re.sub(r"^\d+\w*\s+", "", text)
        if not text or _REPEALED.match(text) or _REPEALED.match(without_number):
            continue
        if len(text) < 40:
            continue
        label = _article_label(num_el, eId)
        path = _heading_path(article, parents)
        title = _extract(heading_el) if heading_el is not None else ""
        if not title and path:
            title = " — ".join(path[-2:])
        articles.append(
            ParsedArticle(
                law_sr=sr,
                law_abbr=abbr,
                law_title=full_title,
                language=language,
                article=label,
                eId=eId or f"art_{label}",
                article_title=title,
                text=text,
                version_date=version,
                binding=binding,
                source_url=source_url,
                eli=eli,
                heading_path=path,
            )
        )
    return articles
