"""Resolve and download the latest applicable Fedlex consolidation."""

import logging
import time
from dataclasses import dataclass
from datetime import date

import httpx

from swissdocs.config import LANG_URI, LawSpec

logger = logging.getLogger(__name__)

USER_AGENT = "swissdocs-rag/0.1 (educational portfolio; public Fedlex texts only)"
SPARQL_URL = "https://fedlex.data.admin.ch/sparqlendpoint"


@dataclass(frozen=True)
class Manifestation:
    version_date: str
    xml_url: str
    pdf_url: str | None


def human_url(eli: str, language: str) -> str:
    return f"https://www.fedlex.admin.ch/eli/{eli}/{language}"


def choose_files(
    rows: list[dict[str, str]],
    today: date,
) -> Manifestation:
    """Pick the newest consolidation on or before ``today`` that has XML.

    Fedlex also publishes future consolidations (a version date after today).
    Those are excluded so the corpus reflects law that is already applicable.
    """
    eligible = []
    for row in rows:
        try:
            version = date.fromisoformat(row["date"])
        except ValueError:
            continue
        if version <= today:
            eligible.append(row)
    if not eligible:
        raise LookupError("No applicable Fedlex consolidation")
    latest = max(row["date"] for row in eligible)
    same = [row for row in eligible if row["date"] == latest]
    xml_url = next((row["url"] for row in same if row["format"] == "xml"), None)
    pdf_url = next((row["url"] for row in same if row["format"] in {"pdf-a", "pdf"}), None)
    if xml_url is None:
        raise LookupError(f"No XML manifestation for {latest}")
    return Manifestation(version_date=latest, xml_url=xml_url, pdf_url=pdf_url)


def _sparql(client: httpx.Client, query: str) -> list[dict[str, str]]:
    response = client.post(
        SPARQL_URL,
        content=query.encode("utf-8"),
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": USER_AGENT,
        },
    )
    response.raise_for_status()
    rows: list[dict[str, str]] = []
    for binding in response.json()["results"]["bindings"]:
        rows.append(
            {
                "date": binding["date"]["value"],
                "format": binding["format"]["value"].rstrip("/").split("/")[-1],
                "url": binding["url"]["value"],
            }
        )
    return rows


def manifestations_query(law: LawSpec, language: str) -> str:
    lang_uri = LANG_URI[language]
    eli = f"https://fedlex.data.admin.ch/eli/{law.eli}"
    return f"""
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT ?date ?format ?url WHERE {{
  ?work jolux:isMemberOf <{eli}> ;
        jolux:dateApplicability ?date ;
        jolux:isRealizedBy ?expression .
  ?expression jolux:language <{lang_uri}> ;
              jolux:isEmbodiedBy ?manifestation .
  ?manifestation jolux:userFormat ?format ;
                 jolux:isExemplifiedBy ?url .
}}
"""


def resolve_manifestation(
    law: LawSpec,
    language: str,
    *,
    client: httpx.Client,
    today: date | None = None,
) -> Manifestation:
    rows = _sparql(client, manifestations_query(law, language))
    return choose_files(rows, today or date.today())


def download_bytes(client: httpx.Client, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.get(url, timeout=180, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("download failed (%s) attempt %s: %s", url, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}") from last_error


def polite_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, read=180.0),
        follow_redirects=True,
    )
