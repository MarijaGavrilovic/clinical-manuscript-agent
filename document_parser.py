"""Read scientific full texts while retaining evidence and publication identity."""
import hashlib
import re
import xml.etree.ElementTree as ET
import fitz
import requests
from literature_search import get, NCBI, normalize_doi
from models import Paper, Document


def _tag(element):
    return element.tag.rsplit('}', 1)[-1]


def _clean(value):
    return ' '.join(str(value).split())


def _text(node):
    return _clean(' '.join(node.itertext())) if node is not None else ''


def validate_publication(expected_doi: str, actual_doi: str) -> bool:
    """Reject if either DOI is absent or the identifiers differ."""
    return bool(normalize_doi(expected_doi) and normalize_doi(actual_doi)
                and normalize_doi(expected_doi) == normalize_doi(actual_doi))


def _article_metadata(root):
    meta = root.find('.//article-meta')
    if meta is None:
        raise ValueError('Full-text XML has no article-meta; cannot check identity')
    doi = ''
    for child in meta.findall('article-id'):
        if child.get('pub-id-type') == 'doi':
            doi = _text(child)
            break
    title = _text(meta.find('./title-group/article-title'))
    return normalize_doi(doi), title


def _table_lines(table, label):
    """Keep cells on a row together. Reject overconfidence about merged headers."""
    title = _text(table.find('label')) or label
    caption = _text(table.find('caption'))
    output = [f'[Table {label}] TITLE: {title}' + (f' CAPTION: {caption}' if caption else '')]
    # Include every heading/body row and label each row distinctly.
    tr_elements = [e for e in table.iter() if _tag(e) == 'tr']
    for index, tr in enumerate(tr_elements, 1):
        cells = []
        for cell in tr:
            if _tag(cell) not in ('td', 'th'):
                continue
            value = _text(cell)
            colspan = cell.attrib.get('colspan', '1')
            rowspan = cell.attrib.get('rowspan', '1')
            role = 'HEADER' if _tag(cell) == 'th' else 'CELL'
            cells.append(f'{role}(colspan={colspan},rowspan={rowspan}): {value}')
        if cells:
            output.append(f'[Table {label} Row {index}] ' + ' | '.join(cells))
    return output


def parse_jats_xml(xml_content: bytes | str, paper: Paper) -> Document:
    root = ET.fromstring(xml_content)
    actual_doi, actual_title = _article_metadata(root)
    if not validate_publication(paper.doi, actual_doi):
        raise ValueError(f'PUBLICATION IDENTITY REJECTED: selected DOI={paper.doi or "missing"}, '
                         f'full-text DOI={actual_doi or "missing"}; title={actual_title}')
    # Optional independent name check for DOI collisions or bad metadata.
    if paper.title and actual_title:
        normalize_title = lambda s: re.sub(r'\W+', '', s.casefold())
        if normalize_title(paper.title) != normalize_title(actual_title):
            raise ValueError(f'PUBLICATION TITLE MISMATCH: {paper.title!r} != {actual_title!r}')

    blocks = []
    body = root.find('.//body')
    if body is None:
        raise ValueError('XML lacks full-text body')

    table_index = 0
    section_index = 0
    # Walk in document order, only emitting paragraphs and tables once.
    def walk(node):
        nonlocal table_index, section_index
        tag = _tag(node)
        if tag == 'table-wrap':
            table_index += 1
            blocks.extend(_table_lines(node, str(table_index)))
            return
        if tag == 'sec':
            section_index += 1
            heading = _text(node.find('title'))
            blocks.append(f'[Section {section_index}] HEADING: {heading}')
            # children are individually labeled below
        if tag == 'p':
            content = _text(node)
            if content:
                blocks.append(f'[Section {section_index}] {content}')
            return
        for child in node:
            if _tag(child) != 'title':
                walk(child)

    walk(body)
    text = '\n\n'.join(blocks)
    if len(text) < 200:
        raise ValueError('Retrieved XML has insufficient extracted Results text')
    return Document(paper=paper, text=text, source_kind='PMC full-text XML',
                    identity_status='DOI_AND_TITLE_MATCHED')


def _pmcid_lookup_by_doi(doi):
    if not doi:
        return ''
    root = ET.fromstring(get(f'{NCBI}/esearch.fcgi', {
        'db': 'pmc', 'term': f'{doi}[DOI]', 'retmode': 'xml', 'retmax': 5
    }, as_json=False))
    for node in root.findall('.//IdList/Id'):
        if node.text and node.text.isdigit():
            return f'PMC{node.text}'
    return ''


def retrieve_original_fulltext(paper: Paper) -> Document:
    """Fetch PMC original XML; fail CLOSED on DOI/title mismatch."""
    if not paper.doi:
        raise ValueError('Publication DOI unavailable: cannot verify full-text identity')
    pmcid = paper.pmcid or _pmcid_lookup_by_doi(paper.doi)
    if not re.fullmatch(r'PMC\d+', pmcid or ''):
        raise ValueError('No public PMC full text found; upload a publisher PDF instead')
    xml_data = get(f'{NCBI}/efetch.fcgi', {
        'db': 'pmc', 'id': pmcid.removeprefix('PMC'), 'retmode': 'xml'
    }, as_json=False)
    return parse_jats_xml(xml_data, paper)


def read_uploaded_pdf(data: bytes, filename: str) -> Document:
    """User-provided PDFs have an unverified publication identity unless audited manually."""
    blocks = []
    with fitz.open(stream=data, filetype='pdf') as doc:
        for number, page in enumerate(doc, 1):
            text = _clean(page.get_text('text', sort=True))
            if text:
                blocks.append(f'[Page {number}] {text}')
            try:
                tables = page.find_tables()
                for table_index, table in enumerate(tables.tables, 1):
                    rows = table.extract()
                    for row_index, row in enumerate(rows, 1):
                        vals = [_clean(cell or '') for cell in row]
                        blocks.append(f'[Page {number} Table {table_index} Row {row_index}] ' + ' | '.join(vals))
            except (ValueError, RuntimeError, AttributeError):
                # PDF text can still be used; a failed table parse must be reviewable.
                blocks.append(f'[Page {number}] WARNING: automatic table recognition unavailable')
    if not blocks:
        raise ValueError('No extractable text; scanned PDFs need an OCR workflow')
    paper = Paper(uid='PDF:' + hashlib.sha256(data).hexdigest()[:16],
                  title=filename, authors=[], provider='Uploaded PDF')
    return Document(paper=paper, text='\n\n'.join(blocks), source_kind='Uploaded PDF',
                    identity_status='UNVERIFIED_UPLOADED_PDF',
                    warning='Confirm uploaded PDF title/DOI manually before clinical use')
