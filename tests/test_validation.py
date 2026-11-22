import pytest
from literature_search import author_matches, normalize_doi
from document_parser import parse_jats_xml, validate_publication
from extraction import chunk_document
from validation import validate_finding, deduplicate
from models import Paper, Finding, ExtractedRecord


def test_author_exact():
    assert author_matches('Julia Wynn', 'Julia Wynn')
    assert not author_matches('Julia Wynn', 'Joe Wynn')
    assert not author_matches('Julia Wynn', 'J Wynn')


def test_doi_identity():
    assert validate_publication('https://doi.org/10.1007/s10803-024-06273-x',
                                '10.1007/s10803-024-06273-x')
    assert not validate_publication('10.1007/s10803-024-06273-x',
                                    '10.1038/s41588-022-01148-2')
    assert not validate_publication('10.1007/s10803-024-06273-x', '')


def xml_fixture(doi):
    return f'''<article><front><article-meta>
       <article-id pub-id-type="doi">{doi}</article-id>
       <title-group><article-title>Sample Paper</article-title></title-group>
       </article-meta></front><body><sec><title>Results</title>
       <p>There were 847 participants in the study and 148 in group A.</p>
       <table-wrap><label>Table 1a</label><caption><title>Characteristics</title></caption>
       <table><thead><tr><th>Statistic</th><th>Total</th><th>Group A</th></tr></thead>
       <tbody><tr><td>Age mean (SD)</td><td>44.1 (8.1)</td><td>44.6 (9.0)</td></tr>
       </tbody></table></table-wrap></sec></body></article>'''


def test_reject_wrong_paper_before_parsing():
    paper = Paper(uid='x', title='Sample Paper', authors=[], doi='10.1007/s10803-024-06273-x')
    with pytest.raises(ValueError, match='PUBLICATION IDENTITY REJECTED'):
        parse_jats_xml(xml_fixture('10.1038/s41588-022-01148-2'), paper)


def test_table_1a_retains_row_and_headers():
    paper = Paper(uid='x', title='Sample Paper', authors=[], doi='10.1007/s10803-024-06273-x')
    doc = parse_jats_xml(xml_fixture(paper.doi), paper)
    assert 'Table 1a' in doc.text
    assert 'Total' in doc.text
    assert 'Age mean (SD)' in doc.text
    assert '44.1 (8.1)' in doc.text


def test_quote_validation():
    chunk = '[Table 1 Row 2] CELL: Age mean (SD) | CELL: 44.1 (8.1)'
    f = Finding(variable='Parental age', statistic='mean', value='44.1', unit='years',
                population='Total', table_column='Total', source_label='[Table 1 Row 2]',
                evidence_quote='CELL: Age mean (SD) | CELL: 44.1 (8.1)')
    assert validate_finding(f, chunk) == 'QUOTE_AND_NUMBER_MATCHED_REVIEW_CONTEXT'
    f.value = '34.3'
    assert validate_finding(f, chunk) == 'REVIEW_NUMBER_NOT_IN_QUOTE'


def test_all_chunks_included():
    content = '\n\n'.join(f'[Section {i}] Result was {i}.' for i in range(1, 30))
    chunks = chunk_document(content, max_chars=45)
    assert '[Section 29]' in chunks[-1]
    assert len(chunks) > 12
