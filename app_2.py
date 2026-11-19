
import streamlit as st
import pandas as pd
import requests
import json
import re
import io
import time
import unicodedata
import xml.etree.ElementTree as ET

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Clinical Literature Evidence Agent",
    layout="wide"
)

st.title("📚 Clinical Literature Evidence Matrix Agent")
st.caption(
    "Scientific Publication Search • Full-Text Extraction • "
    "Evidence Validation • Structured Clinical Benchmarks"
)

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CROSSREF = "https://api.crossref.org/works"

# ============================================================
# NETWORK SETTINGS
# ============================================================

session = requests.Session()

retry_strategy = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)


def get_request(url, params=None, timeout=15):
    response = session.get(
        url,
        params=params,
        timeout=(5, timeout),
        headers={
            "User-Agent": "ClinicalEvidenceResearch/1.0"
        }
    )
    response.raise_for_status()
    return response


# ============================================================
# API KEY
# ============================================================

try:
    saved_api_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    saved_api_key = ""

with st.sidebar:

    st.header("Configuration")

    if saved_api_key:
        api_key = saved_api_key
        st.success("OpenAI API key loaded")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password"
        )

    model = st.selectbox(
        "Extraction model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )

    max_papers = st.slider(
        "Maximum papers to discover",
        min_value=5,
        max_value=100,
        value=30
    )

    st.info(
        "Searches public scientific databases. "
        "Only accessible original full text or uploaded "
        "PDFs can be used for evidence extraction."
    )


# ============================================================
# AUTHOR AND TOPIC SEARCH
# ============================================================

def search_europe_pmc(author, topic, limit):

    query_parts = []

    if author.strip():
        safe_author = author.replace('"', '').strip()
        query_parts.append(f'AUTH:"{safe_author}"')

    if topic.strip():
        query_parts.append(f"({topic.strip()})")

    if not query_parts:
        return []

    params = {
        "query": " AND ".join(query_parts),
        "format": "json",
        "resultType": "lite",
        "pageSize": min(limit, 100),
        "sort": "FIRST_PDATE_D desc"
    }

    response = get_request(
        f"{EPMC}/search",
        params=params
    )

    records = response.json().get(
        "resultList", {}
    ).get("result", [])

    papers = []

    for record in records:

        doi = record.get("doi", "")

        papers.append({
            "id": f"EPMC-{record.get('source')}-{record.get('id')}",
            "title": record.get("title", ""),
            "authors": record.get("authorString", ""),
            "year": record.get("pubYear", ""),
            "doi": doi,
            "pmcid": record.get("pmcid", ""),
            "url": (
                f"https://doi.org/{doi}"
                if doi else ""
            ),
            "provider": "Europe PMC"
        })

    return papers


def search_crossref(author, topic, limit):

    params = {
        "rows": min(limit * 3, 100),
        "select": "DOI,title,author,published,URL"
    }

    if author.strip():
        params["query.author"] = author.strip()

    if topic.strip():
        params["query.bibliographic"] = topic.strip()

    response = get_request(
        CROSSREF,
        params=params
    )

    records = response.json().get(
        "message", {}
    ).get("items", [])

    papers = []

    requested_tokens = set(
        re.findall(r"\w+", author.casefold())
    )

    for record in records:

        author_names = []

        for a in record.get("author", []):
            name = " ".join(
                filter(None, [
                    a.get("given", ""),
                    a.get("family", "")
                ])
            )
            author_names.append(name)

        # Check that all name tokens match one author,
        # not different coauthors combined.
        if requested_tokens:

            matched = any(
                requested_tokens.issubset(
                    set(re.findall(r"\w+", name.casefold()))
                )
                for name in author_names
            )

            if not matched:
                continue

        title_list = record.get("title", [])

        if not title_list:
            continue

        doi = record.get("DOI", "")

        date_parts = record.get(
            "published", {}
        ).get("date-parts", [[]])

        year = (
            date_parts[0][0]
            if date_parts and date_parts[0]
            else ""
        )

        papers.append({
            "id": f"CROSSREF-{doi}",
            "title": title_list[0],
            "authors": ", ".join(author_names[:10]),
            "year": year,
            "doi": doi,
            "pmcid": "",
            "url": record.get("URL", ""),
            "provider": "Crossref"
        })

        if len(papers) >= limit:
            break

    return papers


def search_literature(author, topic, limit):

    errors = []

    try:
        papers = search_europe_pmc(
            author, topic, limit
        )

        if papers:
            return papers, "Europe PMC", ""

    except requests.RequestException as e:
        errors.append(
            f"Europe PMC: {type(e).__name__}"
        )

    # Automatic fallback
    try:
        papers = search_crossref(
            author, topic, limit
        )

        return (
            papers,
            "Crossref fallback",
            "; ".join(errors)
        )

    except requests.RequestException as e:

        errors.append(
            f"Crossref: {type(e).__name__}"
        )

        return [], "Unavailable", "; ".join(errors)


# ============================================================
# RETRIEVE ORIGINAL FULL TEXT
# ============================================================

def lookup_pmcid(doi):

    if not doi:
        return ""

    safe_doi = doi.replace('"', '').strip()

    response = get_request(
        f"{EPMC}/search",
        params={
            "query": f'DOI:"{safe_doi}"',
            "format": "json",
            "pageSize": 5
        }
    )

    records = response.json().get(
        "resultList", {}
    ).get("result", [])

    for record in records:

        if record.get("doi", "").lower() == doi.lower():
            return record.get("pmcid", "")

    return ""


def xml_to_text(xml_content):

    root = ET.fromstring(xml_content)

    blocks = []

    # Preserve some table context through row extraction.
    for element in root.iter():

        tag = element.tag.split("}")[-1]

        if tag == "p":

            text = " ".join(element.itertext())
            text = " ".join(text.split())

            if text:
                blocks.append(text)

        elif tag == "tr":

            cells = []

            for cell in element:

                cell_tag = cell.tag.split("}")[-1]

                if cell_tag in ("th", "td"):

                    value = " ".join(
                        " ".join(cell.itertext()).split()
                    )

                    cells.append(value)

            if cells:
                blocks.append(
                    "TABLE ROW: " + " | ".join(cells)
                )

    return "\n".join(
        f"[Section {i+1}] {block}"
        for i, block in enumerate(blocks)
    )


def retrieve_full_text(paper):

    pmcid = paper.get("pmcid", "")

    if not pmcid:
        pmcid = lookup_pmcid(
            paper.get("doi", "")
        )

    if not pmcid:
        return None

    response = get_request(
        f"{EPMC}/{pmcid}/fullTextXML",
        timeout=25
    )

    text = xml_to_text(response.text)

    if len(text) < 500:
        return None

    return {
        "title": paper["title"],
        "doi": paper.get("doi", ""),
        "source": "Europe PMC original full text",
        "text": text
    }


def read_pdf(uploaded_file):

    reader = PdfReader(
        io.BytesIO(uploaded_file.getvalue())
    )

    pages = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        pages.append(
            f"[Page {i+1}]\n{text}"
        )

    return {
        "title": uploaded_file.name,
        "doi": "",
        "source": "Uploaded original PDF",
        "text": "\n\n".join(pages)
    }


# ============================================================
# STRUCTURED AI EXTRACTION
# ============================================================

class MetricFinding(BaseModel):

    metric: str
    value: str
    statistic_type: str
    population: str
    source_label: str
    evidence_quote: str
    explanation: str


class ExtractionResult(BaseModel):

    findings: list[MetricFinding]


METRICS = [
    "Cohort Size",
    "Mean Maternal Age",
    "Median Maternal Age",
    "Monosomy X PPV",
    "Autosomal Trisomy PPV",
    "Initial No-Call Rate"
]


def split_text(text, max_chars=14000):

    # Keep sections together where possible.
    lines = text.splitlines()

    chunks = []
    current = ""

    for line in lines:

        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i + max_chars])

            continue

        if len(current) + len(line) + 1 > max_chars:
            chunks.append(current)
            current = ""

        current += line + "\n"

    if current.strip():
        chunks.append(current)

    return chunks


def extract_metrics(client, model, text, metrics):

    prompt = f"""
You are a senior clinical evidence extractor.

Extract ONLY requested metrics from the ORIGINAL
publication text provided below.

STRICT RULES:

1. Never use external knowledge.
2. Never guess missing measurements.
3. Extract each metric independently.
4. Distinguish MEAN from MEDIAN.
5. Distinguish full cohort from subgroups.
6. Distinguish individual disease PPV
   from pooled PPV.
7. Do not substitute related statistics.
8. Use exact continuous quotations from the text.
9. Record the source section or page label.
10. If a metric is missing, return an empty value.
11. Do not calculate or impute missing values.
12. Report every candidate measurement separately
    if there are different cohorts or estimates.
13. Never call a median a mean.
14. Never confuse initial and final no-call rates.
15. Never fabricate an evidence quotation.

Return structured findings.

REQUESTED METRICS:
{json.dumps(metrics)}

ORIGINAL SOURCE TEXT:
{text}
"""

    response = client.beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract scientific measurements "
                    "using only supplied source text."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=ExtractionResult
    )

    result = response.choices[0].message.parsed

    return result.findings


# ============================================================
# INDEPENDENT EVIDENCE VALIDATION
# ============================================================

def normalize(text):

    text = unicodedata.normalize(
        "NFKC", str(text)
    )

    return re.sub(r"\s+", " ", text).strip()


def parse_number(value):

    value = str(value).replace(",", "").strip()

    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)

    return None


def validate_finding(finding, source_text):

    value = finding.value.strip()
    quote = finding.evidence_quote.strip()

    if not value:
        return "NOT_FOUND_IN_PROCESSED_TEXT"

    if not quote:
        return "REVIEW: Missing evidence quotation"

    if normalize(quote) not in normalize(source_text):
        return "REVIEW: Quote not found in source"

    number = parse_number(value)

    if number is None:
        return "REVIEW: Value is not a single number"

    quoted_numbers = re.findall(
        r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?(?![\w.])",
        quote
    )

    quoted_values = [
        parse_number(n)
        for n in quoted_numbers
    ]

    if number not in quoted_values:
        return "REVIEW: Number absent from quote"

    metric = finding.metric.lower()
    stat_type = finding.statistic_type.lower()

    if "mean" in metric and stat_type != "mean":
        return "REVIEW: Mean/median mismatch"

    if "median" in metric and stat_type != "median":
        return "REVIEW: Mean/median mismatch"

    if "ppv" in metric or "rate" in metric:

        if number < 0 or number > 100:
            return "REVIEW: Invalid percentage"

    return "QUOTE_MATCHED_REVIEW_CONTEXT"


# ============================================================
# SESSION STATE
# ============================================================

if "papers" not in st.session_state:
    st.session_state.papers = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# ============================================================
# SECTION 1: SEARCH SCIENTIFIC LITERATURE
# ============================================================

st.subheader("1. Search Scientific Publications")

col1, col2 = st.columns(2)

with col1:

    author = st.text_input(
        "Author name (optional)",
        value="Julia Wynn",
        placeholder="First name Last name"
    )

with col2:

    topic = st.text_input(
        "Research topic (optional)",
        placeholder="e.g. prenatal screening, NIPT"
    )

if st.button(
    "🔎 Search Publications",
    type="primary"
):

    if not author.strip() and not topic.strip():

        st.warning(
            "Enter an author name or research topic."
        )

    else:

        with st.spinner(
            "Searching scientific databases..."
        ):

            papers, provider, warning = search_literature(
                author,
                topic,
                max_papers
            )

            st.session_state.papers = papers

        if warning:
            st.warning(warning)

        if papers:
            st.success(
                f"Found {len(papers)} publications "
                f"using {provider}."
            )
        else:
            st.warning("No publications found.")


# ============================================================
# SELECT PUBLICATIONS
# ============================================================

papers = st.session_state.papers

if papers:

    paper_df = pd.DataFrame(papers)

    st.dataframe(
        paper_df[
            ["title", "authors", "year", "doi", "url"]
        ],
        use_container_width=True,
        hide_index=True
    )

    paper_choices = {
        f"{i+1}. {p['title']} ({p['year']})": i
        for i, p in enumerate(papers)
    }

    selected_labels = st.multiselect(
        "Select publications to analyze",
        options=list(paper_choices.keys())
    )

    if st.button("📄 Retrieve Selected Full Texts"):

        selected_indices = [
            paper_choices[label]
            for label in selected_labels
        ]

        retrieved = []

        for index in selected_indices:

            paper = papers[index]

            with st.spinner(
                f"Retrieving: {paper['title']}"
            ):

                try:

                    document = retrieve_full_text(paper)

                    if document:
                        retrieved.append(document)

                    else:
                        st.warning(
                            "Full text not available: "
                            + paper["title"]
                        )

                except Exception as e:

                    st.warning(
                        f"Retrieval failed for "
                        f"{paper['title']}: {e}"
                    )

        st.session_state.documents = retrieved

        if retrieved:
            st.success(
                f"Retrieved {len(retrieved)} "
                "full-text publications."
            )


# ============================================================
# SECTION 2: OPTIONAL PDF UPLOAD
# ============================================================

st.subheader("2. Optional PDF Upload")

uploaded_files = st.file_uploader(
    "Upload original publications if full text is unavailable online",
    type=["pdf"],
    accept_multiple_files=True
)

uploaded_documents = []

for uploaded_file in uploaded_files or []:

    try:
        document = read_pdf(uploaded_file)

        uploaded_documents.append(document)

    except Exception as e:

        st.warning(
            f"Could not process {uploaded_file.name}: {e}"
        )


all_documents = (
    st.session_state.documents
    + uploaded_documents
)

if all_documents:

    st.success(
        f"{len(all_documents)} full-text documents "
        "available for extraction."
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Title": d["title"],
                "Source": d["source"],
                "Characters": len(d["text"])
            }
            for d in all_documents
        ]),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# SECTION 3: CLINICAL METRIC EXTRACTION
# ============================================================

st.subheader("3. Extract Clinical Evidence")

selected_metrics = st.multiselect(
    "Select clinical parameters",
    options=METRICS,
    default=METRICS
)

custom_metrics = st.text_area(
    "Additional parameters (optional; one per line)",
    placeholder=(
        "Sensitivity for Trisomy 21\n"
        "Specificity for Trisomy 21\n"
        "Fetal Fraction"
    )
)

additional = [
    m.strip()
    for m in custom_metrics.splitlines()
    if m.strip()
]

all_metrics = list(
    dict.fromkeys(selected_metrics + additional)
)


# ============================================================
# RUN EXTRACTION
# ============================================================

if st.button(
    "📊 Extract Structured Evidence Matrix",
    type="primary"
):

    if not api_key:

        st.error("Enter your OpenAI API key.")

    elif not all_documents:

        st.error(
            "Search and retrieve a publication, "
            "or upload an original PDF."
        )

    elif not all_metrics:

        st.error("Select at least one metric.")

    else:

        client = OpenAI(api_key=api_key)

        audit_rows = []

        with st.spinner(
            "Extracting and validating clinical evidence..."
        ):

            for document in all_documents:

                chunks = split_text(
                    document["text"]
                )

                study_rows = []
                failed_chunks = 0

                for chunk in chunks:

                    try:

                        findings = extract_metrics(
                            client,
                            model,
                            chunk,
                            all_metrics
                        )

                        for finding in findings:

                            if finding.metric not in all_metrics:
                                continue

                            if not finding.value.strip():
                                continue

                            status = validate_finding(
                                finding,
                                chunk
                            )

                            row = {
                                "Study": document["title"],
                                "DOI": document["doi"],
                                "Metric": finding.metric,
                                "Value": finding.value,
                                "Statistic": finding.statistic_type,
                                "Population": finding.population,
                                "Source Label": finding.source_label,
                                "Evidence Quote": finding.evidence_quote,
                                "Validation": status,
                                "Source": document["source"]
                            }

                            study_rows.append(row)

                    except Exception as e:

                        failed_chunks += 1

                        st.warning(
                            f"Extraction error in "
                            f"{document['title']}: {e}"
                        )

                # Keep all competing values instead of
                # selecting one silently.
                audit_rows.extend(study_rows)

                found_metrics = {
                    r["Metric"]
                    for r in study_rows
                }

                for metric in all_metrics:

                    if metric not in found_metrics:

                        audit_rows.append({
                            "Study": document["title"],
                            "DOI": document["doi"],
                            "Metric": metric,
                            "Value": "",
                            "Statistic": "",
                            "Population": "",
                            "Source Label": "",
                            "Evidence Quote": "",
                            "Validation": (
                                "INCOMPLETE_EXTRACTION"
                                if failed_chunks
                                else "NOT_FOUND_IN_PROCESSED_TEXT"
                            ),
                            "Source": document["source"]
                        })

        # ====================================================
        # DISPLAY EVIDENCE AUDIT
        # ====================================================

        df = pd.DataFrame(audit_rows)

        if df.empty:

            st.warning(
                "No extraction results were returned."
            )

        else:

            st.subheader("4. Detailed Evidence Audit")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # ================================================
            # BUILD CONSERVATIVE COMPARISON MATRIX
            # ================================================

            summary_rows = []

            for (study, metric), group in df.groupby(
                ["Study", "Metric"],
                sort=False
            ):

                matched = group[
                    group["Validation"]
                    == "QUOTE_MATCHED_REVIEW_CONTEXT"
                ]

                if len(group) == 1 and len(matched) == 1:

                    result = matched.iloc[0]["Value"]

                elif len(matched) > 0:

                    result = "Multiple candidates - review"

                elif (
                    len(group) == 1
                    and group.iloc[0]["Validation"]
                    == "NOT_FOUND_IN_PROCESSED_TEXT"
                ):

                    result = "Not found in extracted text"

                else:

                    result = "Needs review"

                summary_rows.append({
                    "Study": study,
                    "Metric": metric,
                    "Result": result
                })

            summary_df = pd.DataFrame(summary_rows)

            matrix = summary_df.pivot(
                index="Study",
                columns="Metric",
                values="Result"
            )

            st.subheader("5. Clinical Evidence Matrix")

            st.dataframe(
                matrix,
                use_container_width=True
            )

            # ================================================
            # EXPORT RESULTS
            # ================================================

            st.download_button(
                "📥 Download Evidence Matrix CSV",
                data=matrix.to_csv().encode("utf-8-sig"),
                file_name="clinical_evidence_matrix.csv",
                mime="text/csv"
            )

            st.download_button(
                "📥 Download Detailed Evidence Audit CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="clinical_evidence_audit.csv",
                mime="text/csv"
            )

            st.info(
                "QUOTE_MATCHED_REVIEW_CONTEXT means the "
                "number and quotation match the extracted "
                "source text. It does not guarantee that "
                "the correct cohort, statistic, endpoint, "
                "or table column was selected. "
                "Human scientific review is still required."
            )

st.divider()

st.caption(
    "Research limitations: Only accessible full text and "
    "uploaded PDFs are analyzed. Crossref is used for "
    "publication discovery, not as clinical evidence. "
    "Some PDF tables may be extracted incorrectly. "
    "Absent metrics are not assumed absent from the "
    "original publication."
)
