"""Run: streamlit run app.py"""
import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from models import ExtractedRecord
from literature_search import search_publications
from document_parser import retrieve_original_fulltext, read_uploaded_pdf
from extraction import chunk_document, extract_candidates
from validation import validate_finding, validate_document, deduplicate

st.set_page_config(page_title='Clinical Evidence Agent', layout='wide')
st.title('Clinical Literature Evidence Agent')
st.caption('Find publications • Confirm source identity • Extract all numerical Results • Audit every finding')

try:
    secret_key = st.secrets.get('OPENAI_API_KEY', '')
except Exception:
    secret_key = ''

with st.sidebar:
    st.header('Configuration')
    key = secret_key or os.getenv('OPENAI_API_KEY', '')
    if not key:
        key = st.text_input('OpenAI API key', type='password')
    else:
        st.success('OpenAI key configured')
    model = st.text_input('OpenAI model (structured outputs)', value='gpt-4o-mini')
    st.caption('No arbitrary chunk cap. Every chunk is attempted; long papers increase API usage.')

if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'extracted_records' not in st.session_state:
    st.session_state.extracted_records = []
if 'audit_rows' not in st.session_state:
    st.session_state.audit_rows = []

st.subheader('1. Find publications (optional author and topic)')
c1, c2 = st.columns(2)
with c1:
    author = st.text_input('Author full name (optional)', value='Julia Wynn')
with c2:
    topic = st.text_input('Topic (optional)', value='')
limit = st.slider('Maximum results', 5, 100, 30)
if st.button('Search publications'):
    if not author.strip() and not topic.strip():
        st.warning('Enter an author or topic.')
    else:
        with st.spinner('Searching PubMed and Crossref...'):
            papers, warnings = search_publications(author, topic, limit)
        st.session_state.papers = papers
        st.session_state.documents = {}
        for warning in warnings:
            st.warning(warning)
        st.success(f'{len(papers)} metadata matches found') if papers else st.warning('No results found')

papers = st.session_state.papers
if papers:
    st.dataframe(pd.DataFrame([{
        'Title': p.title, 'Authors': ', '.join(p.authors), 'Year': p.year,
        'DOI': p.doi, 'PMCID': p.pmcid, 'Source': p.provider, 'URL': p.url
    } for p in papers]), use_container_width=True, hide_index=True)
    chosen_ids = st.multiselect(
        'Select publications for full-text retrieval',
        [p.uid for p in papers],
        format_func=lambda uid: next((f'{p.year} — {p.title}' for p in papers if p.uid == uid), uid)
    )
    if st.button('Retrieve selected original full texts'):
        docs = {}
        for paper in papers:
            if paper.uid not in chosen_ids:
                continue
            try:
                doc = retrieve_original_fulltext(paper)
                docs[paper.uid] = doc
                st.success(f'Identity checked: {paper.title}')
            except Exception as exc:
                st.warning(f'{paper.title}: {exc}')
        st.session_state.documents = docs

st.subheader('2. Upload publisher PDFs (optional)')
uploads = st.file_uploader('PDFs if the online full text is unavailable', type=['pdf'], accept_multiple_files=True)
manual = {}
for upload in uploads or []:
    try:
        doc = read_uploaded_pdf(upload.getvalue(), upload.name)
        manual[doc.paper.uid] = doc
    except Exception as exc:
        st.error(f'{upload.name}: {exc}')

documents = list({**st.session_state.documents, **manual}.values())
if documents:
    st.dataframe(pd.DataFrame([{
        'Publication': d.paper.title, 'Source': d.source_kind,
        'Identity': d.identity_status, 'Characters': len(d.text), 'Warning': d.warning
    } for d in documents]), hide_index=True, use_container_width=True)

st.subheader('3. Extract ALL numerical results')
st.caption('Extraction includes baseline and Results tables, cohort statistics, effects, and statistical tests. '
           'Quote matching is not proof that a result belongs to the correct table column.')
allow_pdf = st.checkbox('I have checked the title and identity of uploaded PDFs', value=False)
if st.button('Extract numerical findings', type='primary'):
    if not documents:
        st.error('Retrieve at least one full text or upload a PDF.')
    elif not key:
        st.error('Configure an OpenAI API key.')
    else:
        client = OpenAI(api_key=key)
        all_records, audit = [], []
        progress = st.progress(0)
        total_chunks = sum(len(chunk_document(d.text)) for d in documents)
        completed = 0
        for doc in documents:
            identity_ok = validate_document(doc)
            if not identity_ok and not (doc.source_kind == 'Uploaded PDF' and allow_pdf):
                audit.append({
                    'Publication ID': doc.paper.uid, 'Study': doc.paper.title,
                    'DOI': doc.paper.doi, 'Identity': doc.identity_status,
                    'Chunks': 0, 'Failed chunks': 0, 'Findings': 0,
                    'Quote matches': 0, 'Status': 'SKIPPED_UNVERIFIED_IDENTITY',
                    'Errors': 'Confirm uploaded PDF identity to extract'
                })
                completed += len(chunk_document(doc.text))
                progress.progress(min(completed / max(total_chunks, 1), 1))
                continue

            chunks = chunk_document(doc.text)
            errors = []
            per_paper = []
            for index, chunk in enumerate(chunks, 1):
                try:
                    findings = extract_candidates(client, model, chunk)
                    for f in findings:
                        per_paper.append(ExtractedRecord(
                            paper_uid=doc.paper.uid, paper_title=doc.paper.title,
                            doi=doc.paper.doi, finding=f,
                            validation=validate_finding(f, chunk), chunk_index=index
                        ))
                except Exception as exc:
                    errors.append(f'Chunk {index}: {type(exc).__name__}: {exc}')
                completed += 1
                progress.progress(min(completed / max(total_chunks, 1), 1))
            per_paper = deduplicate(per_paper)
            all_records.extend(per_paper)
            matches = sum(r.validation == 'QUOTE_AND_NUMBER_MATCHED_REVIEW_CONTEXT' for r in per_paper)
            audit.append({
                'Publication ID': doc.paper.uid, 'Study': doc.paper.title,
                'DOI': doc.paper.doi, 'Identity': doc.identity_status,
                'Chunks': len(chunks), 'Failed chunks': len(errors),
                'Findings': len(per_paper), 'Quote matches': matches,
                'Status': 'INCOMPLETE_EXTRACTION' if errors else 'EXTRACTED_REVIEW_REQUIRED',
                'Errors': ' | '.join(errors)
            })
            if errors:
                st.warning(f'{doc.paper.title}: {len(errors)} chunks failed; examine audit errors.')

        st.session_state.extracted_records = all_records
        st.session_state.audit_rows = audit

if st.session_state.extracted_records or st.session_state.audit_rows:
    st.subheader('4. Detailed numerical findings (one row per number)')
    detail = pd.DataFrame([r.to_dict() for r in st.session_state.extracted_records])
    if not detail.empty:
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.download_button('Download all numerical findings CSV',
                           detail.to_csv(index=False).encode('utf-8-sig'),
                           file_name='all_numerical_results.csv', mime='text/csv')
    else:
        st.info('No findings; check identity or extraction errors.')
    st.subheader('5. Evidence audit (one row per publication)')
    audit_df = pd.DataFrame(st.session_state.audit_rows)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
    st.download_button('Download publication audit CSV',
                       audit_df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='publication_audit.csv', mime='text/csv')

st.divider()
st.caption('Limitations: only public PMC full texts or uploaded PDFs are analyzed; exact identity checks '
           'apply to retrieved XML. PDF identity requires user review. Complex tables, merged headers, '
           'OCR scans, and statistical context may require manual checking. All findings are candidates, '
           'not clinically verified conclusions.')
