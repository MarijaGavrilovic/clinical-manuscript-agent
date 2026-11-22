"""Discovery is metadata-only; only validated original full text is evidence."""
import re
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from models import Paper

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF = "https://api.crossref.org/works"

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=2, connect=2, read=2, backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"])
)))


def get(url, params=None, *, as_json=True):
    response = _session.get(
        url, params=params, timeout=(6, 22),
        headers={"User-Agent": "ClinicalEvidenceAgent/1.0 (academic research)"}
    )
    response.raise_for_status()
    return response.json() if as_json else response.content


def normalize_doi(doi: str) -> str:
    x = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if x.startswith(prefix):
            x = x[len(prefix):]
    return x.strip()


def author_matches(requested: str, actual: str) -> bool:
    """Exact first and last names; initials are intentionally not accepted."""
    a = re.findall(r"[\w'-]+", requested.casefold())
    b = re.findall(r"[\w'-]+", actual.casefold())
    return len(a) >= 2 and len(b) >= 2 and a[0] == b[0] and a[-1] == b[-1]


def _text(node):
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def search_pubmed(author: str, topic: str, limit: int = 30) -> list[Paper]:
    terms = []
    names = author.strip().split()
    if len(names) >= 2:
        # Broad discovery then strict full-name filtering using PubMed metadata.
        terms.append(f'"{names[-1]} {names[0][0]}"[Author]')
    elif len(names) == 1:
        terms.append(f'"{names[0]}"[Author]')
    if topic.strip():
        # Words separately rather than an overly restrictive phrase.
        words = re.findall(r"[\w-]+", topic.strip())
        terms.append("(" + " AND ".join(f'{word}[Title/Abstract]' for word in words) + ")")
    if not terms:
        return []

    ids = get(f"{NCBI}/esearch.fcgi", {
        "db": "pubmed", "term": " AND ".join(terms),
        "retmode": "json", "retmax": min(max(limit * 5, 30), 200), "sort": "pub_date"
    }).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    root = ET.fromstring(get(f"{NCBI}/efetch.fcgi", {
        "db": "pubmed", "id": ",".join(ids), "retmode": "xml"
    }, as_json=False))
    papers = []
    for entry in root.findall(".//PubmedArticle"):
        citation = entry.find("MedlineCitation")
        if citation is None:
            continue
        authors = []
        for node in citation.findall(".//Article/AuthorList/Author"):
            first = node.findtext("ForeName") or node.findtext("Initials") or ""
            last = node.findtext("LastName") or ""
            if first and last:
                authors.append(f"{first} {last}")
        if len(names) >= 2 and not any(author_matches(author, a) for a in authors):
            continue
        # Single-token author searches remain broad by design.
        pmid = _text(citation.find("PMID"))
        title = _text(citation.find(".//ArticleTitle"))
        if not title:
            continue
        year = citation.findtext(".//PubDate/Year") or ""
        if not year:
            raw_date = citation.findtext(".//PubDate/MedlineDate") or ""
            m = re.search(r"\b(?:19|20)\d{2}\b", raw_date)
            year = m.group(0) if m else ""
        doi = pmcid = ""
        for item in entry.findall(".//PubmedData/ArticleIdList/ArticleId"):
            kind = item.get("IdType", "")
            if kind == "doi":
                doi = _text(item)
            elif kind == "pmc":
                pmcid = _text(item)
        papers.append(Paper(uid=f"PUBMED:{pmid}", title=title, authors=authors,
                            year=year, doi=normalize_doi(doi), pmid=pmid,
                            pmcid=pmcid, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            provider="PubMed"))
        if len(papers) >= limit:
            break
    return papers


def search_crossref(author: str, topic: str, limit: int = 30) -> list[Paper]:
    params = {"rows": min(max(limit * 5, 50), 250)}
    if author.strip():
        params["query.author"] = author.strip()
    if topic.strip():
        params["query.bibliographic"] = topic.strip()
    if not author.strip() and not topic.strip():
        return []
    items = get(CROSSREF, params).get("message", {}).get("items", [])
    papers = []
    for item in items:
        authors = [" ".join(filter(None, [a.get("given", ""), a.get("family", "")]))
                   for a in item.get("author", [])]
        if len(author.split()) >= 2 and not any(author_matches(author, a) for a in authors):
            continue
        titles = item.get("title") or []
        if not titles:
            continue
        doi = normalize_doi(item.get("DOI", ""))
        date_parts = item.get("published", {}).get("date-parts", [[]])
        year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
        papers.append(Paper(uid=f"CROSSREF:{doi or item.get('URL','')}",
                            title=titles[0], authors=authors, year=year,
                            doi=doi, url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                            provider="Crossref"))
        if len(papers) >= limit:
            break
    return papers


def search_publications(author: str, topic: str, limit: int = 30):
    """Returns (papers, warnings), preserving successful results if another provider fails."""
    warnings = []
    results = []
    for label, search_fn in (("PubMed", search_pubmed), ("Crossref", search_crossref)):
        try:
            results.extend(search_fn(author, topic, limit))
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            warnings.append(f"{label} unavailable: {type(exc).__name__}: {exc}")

    unique = {}
    for p in results:
        key = f"doi:{p.doi}" if p.doi else re.sub(r"\W+", "", p.title.casefold())
        # Prefer PubMed records with PMCID over Crossref-only metadata.
        if key not in unique or (p.pmcid and not unique[key].pmcid):
            unique[key] = p
    papers = list(unique.values())
    papers.sort(key=lambda p: p.year, reverse=True)
    return papers[:limit], warnings
