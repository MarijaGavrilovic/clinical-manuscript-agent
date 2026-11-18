import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

st.set_page_config(page_title="Clinical Manuscript Verification Agent", layout="wide")

st.title("🧬 Clinical Manuscript Table-to-Text Verification Agent")
st.caption("Deterministic Statistical Engine + LLM Clinical Writer + Automated Zero-Hallucination Auditor")

# Safely check if secret exists without crashing if secrets.toml is empty
try:
    secret_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    secret_key = ""

with st.sidebar:
    st.header("Configuration")
    if secret_key:
        st.success("✅ API Key loaded securely from server")
        api_key = secret_key
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        
    journal_style = st.selectbox(
        "Target Journal Style",
        list(("Genetics in Medicine / AJOG", "Annals of Gynecology & Obstetrics", "NEJM / JAMA Concise"))
    )
    st.markdown("---")
    st.info("Demo pre-loaded with synthetic cohort counts modeled on published NIPT validation metrics.")

# Pre-loaded Synthetic Cohort Data
default_data = pd.DataFrame({
    "Condition": list(("Trisomy 21", "Trisomy 18", "Trisomy 13", "Monosomy X")),
    "True_Positive": list((65, 40, 12, 22)),
    "False_Positive": list((6, 2, 3, 25)),
    "True_Negative": list((1600, 1640, 1670, 1630)),
    "False_Negative": list((1, 1, 0, 2))
})

st.subheader("1. Clinical Validation Cohort Input (Editable)")
edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# AGENT 1: Deterministic Python Statistical Engine
def compute_clinical_metrics(df):
    results = list()
    for _, row in df.iterrows():
        if pd.isna(row["Condition"]) or str(row["Condition"]).strip() == "":
            continue
        tp = float(row["True_Positive"] or 0)
        fp = float(row["False_Positive"] or 0)
        tn = float(row["True_Negative"] or 0)
        fn = float(row["False_Negative"] or 0)
        
        sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
        ppv = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        npv = (tn / (tn + fn)) * 100 if (tn + fn) > 0 else 0.0
        
        n_pos = tp + fp
        ppv_p = ppv / 100.0
        se_ppv = np.sqrt((ppv_p * (1.0 - ppv_p)) / n_pos) * 100 if n_pos > 0 else 0.0
        ppv_low = max(0.0, round(ppv - 1.96 * se_ppv, 1))
        ppv_high = min(100.0, round(ppv + 1.96 * se_ppv, 1))
        
        results.append({
            "Condition": str(row["Condition"]),
            "Total_Screened": int(tp + fp + tn + fn),
            "Affected_Cases": int(tp + fn),
            "Sensitivity_Pct": round(sens, 1),
            "Specificity_Pct": round(spec, 1),
            "PPV_Pct": round(ppv, 1),
            "PPV_95_CI": f"{ppv_low}%-{ppv_high}%",
            "NPV_Pct": round(npv, 2)
        })
    return pd.DataFrame(results)

metrics_df = compute_clinical_metrics(edited_df)

st.subheader("2. Agent 1 Output: Locked Deterministic Statistics")
st.dataframe(metrics_df, use_container_width=True)

# Run Multi-Agent Pipeline
if st.button("🚀 Run Writer & Numerical Auditor Agents", type="primary"):
    if not api_key:
        st.error("Please add your OpenAI API key in Streamlit Secrets or the left sidebar.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            locked_json = metrics_df.to_json(orient="records", indent=2)
            
            with st.spinner("Agent 2 drafting publication-ready Results section..."):
                writer_prompt = f"""
                You are a senior clinical biostatistician writing the Results section for {journal_style}.
                Using ONLY the exact numbers in the locked JSON below, write two concise, publication-grade paragraphs summarizing the assay's clinical performance across conditions.
                Do not invent or round numbers differently than provided.
                
                LOCKED DATA:
                {locked_json}
                """
                writer_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    messages=list(({"role": "user", "content": writer_prompt},))
                )
                draft_text = writer_resp.choices[0].message.content

            with st.spinner("Agent 3 auditing every number against source table..."):
                auditor_prompt = f"""
                You are a strict Clinical QA Numerical Auditor.
                Compare the DRAFT TEXT against the LOCKED DATA JSON.
                Extract every numerical claim in the draft text and verify whether it strictly matches the source JSON.
                Return a clean markdown checklist showing each claim and status ([VERIFIED] or [DISCREPANCY]).
                
                LOCKED DATA:
                {locked_json}
                
                DRAFT TEXT:
                {draft_text}
                """
                auditor_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=list(({"role": "user", "content": auditor_prompt},))
                )
                audit_report = auditor_resp.choices[0].message.content

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("3. Agent 2 Output: Manuscript Results Draft")
                st.write(draft_text)
            with col2:
                st.subheader("4. Agent 3 Output: Zero-Hallucination Audit Log")
                st.markdown(audit_report)
        except Exception as e:
            st.error(f"API Error: {e}")
