# app.py (snippet - integrate into your existing app)
import streamlit as st
import pandas as pd
from ingest_resumes import load_resumes_from_folder, extract_text_from_uploaded_file
from backend.parser import parse_job_description, extract_skills_from_text
from backend.scorer import compute_scores
import io

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("📄 Resume Screening Agent")

with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX/TXT)", accept_multiple_files=True)
    jd_text = st.text_area("Job Description", height=200, help="Paste job description here")
    sample_data = st.checkbox("Use sample resumes", value=False)
    run_btn = st.button("Run Screening")

# show sample resumes option
if sample_data:
    resumes = load_resumes_from_folder("sample_resumes")
else:
    resumes = []
    if uploaded_files:
        for f in uploaded_files:
            text = extract_text_from_uploaded_file(f)
            resumes.append({"name": f.name, "text": text})

if run_btn:
    with st.spinner("Computing scores..."):
        results = compute_scores(jd_text, resumes)
    st.success("Done — results below")

    # Show top 3 as cards
    top3 = results[:3]
    st.markdown("### Top matches")
    cols = st.columns(len(top3) if top3 else 1)
    for c, r in zip(cols, top3):
        with c:
            st.header(r.get("name", "Unknown"))
            st.metric("Score", f"{r.get('score',0):.3f}")
            st.write(r.get("text", "")[:300])
            st.write("**Matched skills:**", ", ".join(r.get("matched_skills", [])))

    # Show full results table and CSV export
    df = pd.DataFrame(results)
    st.markdown("### All results")
    st.dataframe(df[["name","score"] + [col for col in df.columns if col not in ('name','score','text')]], use_container_width=True)

    # CSV export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "screening_results.csv", "text/csv")
