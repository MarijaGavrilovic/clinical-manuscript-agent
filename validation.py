"""Deterministic text and number checks; not a clinical correctness oracle."""
import re
import unicodedata
from models import Finding, ExtractedRecord, Document

NUMBER_PATTERN = re.compile(r'(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).replace('−', '-').split())


def numeric_tokens(value):
    return [m.group().replace(',', '').replace('−', '-') for m in NUMBER_PATTERN.finditer(str(value))]


def validate_finding(finding: Finding, source_chunk: str) -> str:
    quote = finding.evidence_quote or ''
    if not quote or normalize(quote) not in normalize(source_chunk):
        return 'REVIEW_QUOTE_NOT_IN_SOURCE'
    label = finding.source_label or ''
    if not label or label not in source_chunk:
        return 'REVIEW_SOURCE_LABEL_MISSING'
    vals = numeric_tokens(finding.value)
    if not vals:
        return 'REVIEW_VALUE_NOT_NUMERIC'
    if not all(v in numeric_tokens(quote) for v in vals):
        return 'REVIEW_NUMBER_NOT_IN_QUOTE'
    # Detect obvious mean vs median errors; this does not infer table headings.
    low_quote = quote.lower()
    stat = finding.statistic.casefold().strip()
    if stat == 'mean' and 'median' in low_quote and 'mean' not in low_quote:
        return 'REVIEW_MEAN_MEDIAN_CONFLICT'
    if stat == 'median' and 'mean' in low_quote and 'median' not in low_quote:
        return 'REVIEW_MEAN_MEDIAN_CONFLICT'
    return 'QUOTE_AND_NUMBER_MATCHED_REVIEW_CONTEXT'


def validate_document(document: Document) -> bool:
    """Only validated PMC papers pass unattended identity gate.
    User PDFs can be reviewed via the UI but are not automatically identity-verified.
    """
    return document.identity_status == 'DOI_AND_TITLE_MATCHED'


def deduplicate(records: list[ExtractedRecord]):
    """Remove identical records; keep distinct cohorts, estimates and sources."""
    seen, result = set(), []
    for record in records:
        f = record.finding
        key = (
            record.paper_uid,
            normalize(f.variable).casefold(),
            normalize(f.statistic).casefold(),
            normalize(f.value),
            normalize(f.population).casefold(),
            normalize(f.table_column).casefold(),
            normalize(f.source_label),
        )
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result
