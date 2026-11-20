
import streamlit as st
import pandas as pd
import requests
import re
import io
import time
import unicodedata
import xml.etree.ElementTree as ET

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Clinical Literature Evidence Agent",
    layout="wide"
)

st.title("📚 Clinical Literature Evidence Agent")

st.caption(
    "Scientific Literature Discovery | "
    "Original Full-Text Extraction | "
    "Clinical Evidence Validation"
)

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF = "https://api.crossref.org/works"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# ============================================================
# 2. NETWORK CONFIGURATION
# ============================================================

session = requests.Session()

retry = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)


def api_get(url, params=None, timeout=20):
    response = session.get(
        url,
        params=params,
        timeout=(7, timeout),
        headers={
            "User-Agent": "ClinicalEvidenceAgent/1.0"
        }
    )
    response.raise_for_status()
    return response


def xml_text(element):
    if element is None:
        return ""
    return " ".join(
        " ".join(element.itertext()).split()
    )


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 3. OPENAI CONFIGURATION
# ============================================================

try:
    saved_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    saved_key = ""

with st.sidebar:

    st.header("Settings")

    if saved_key:
        api_key = saved_key
        st.success("OpenAI API key loaded")
