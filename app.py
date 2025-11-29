# app.py
import streamlit as st
import os
import pandas as pd
from ingest_resumes import load_resumes_from_folder
from backend.parser import parse_job_description
from backend.scorer import compute_scores

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("📄 Resume Screening Agent — Rank resumes by Job Description")

# Sidebar: upload resumes
with st.sidebar:
    st.header("Upload & Manage Resumes")
    uploaded = st.file_uploader("Upload resumes (pdf/docx/txt)", accept_multiple_files=True)
    if st.button("Save uploaded resumes"):
        os.makedirs("data/resumes", exist_ok=True)
        if not uploaded:
            st.warning("No files selected to upload.")
        else:
            for f in uploaded:
                dest = os.path.join("data", "resumes", f.name)
                with open(dest, "wb") as out:
                    out.write(f.read())
            st.success(f"Saved {len(uploaded)} file(s) to data/resumes/")
    if st.button("Clear data/resumes/ (delete all)"):
        folder = "data/resumes"
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, fname))
                except:
                    pass
            st.success("Cleared data/resumes/")
        else:
            st.info("No resumes folder found.")

st.markdown("## Paste Job Description (JD)")
jd_text = st.text_area("Paste full job description here", height=220)

# Weight sliders (allow recruiter tuning)
st.markdown("## Scoring Weights (adjust to tune ranking)")
colw1, colw2, colw3, colw4 = st.columns(4)
with colw1:
    w_sim = st.slider("Semantic similarity (50%)", 0.0, 1.0, 0.50, step=0.05)
with colw2:
    w_kw = st.slider("Keyword match (25%)", 0.0, 1.0, 0.25, step=0.05)
with colw3:
    w_years = st.slider("Experience years (15%)", 0.0, 1.0, 0.15, step=0.05)
with colw4:
    w_title = st.slider("Title match (10%)", 0.0, 1.0, 0.10, step=0.05)

# Normalize weights automatically (so sum to 1)
total = w_sim + w_kw + w_years + w_title
if total == 0:
    st.warning("All weights are zero — adjust sliders.")
else:
    w_sim /= total
    w_kw /= total
    w_years /= total
    w_title /= total

st.caption(f"Normalized weights — sim: {w_sim:.2f}, kw: {w_kw:.2f}, years: {w_years:.2f}, title: {w_title:.2f}")

# Load resumes (from folder)
st.markdown("## Resumes")
if st.button("Load resumes from data/resumes/"):
    resumes = load_resumes_from_folder()
    st.success(f"Loaded {len(resumes)} resume(s).")
else:
    resumes = load_resumes_from_folder()

if not resumes:
    st.info("No resumes found. Upload files in the sidebar or drop files into data/resumes/ and click 'Load resumes'.")
else:
    st.write(f"Found {len(resumes)} resumes in data/resumes/")

# Rank button
if st.button("Rank resumes") or (jd_text.strip() and st.session_state.get("auto_rank") is None):
    if not jd_text.strip():
        st.error("Please paste a Job Description (JD) first.")
    elif not resumes:
        st.error("No resumes to rank. Upload resumes and try again.")
    else:
        st.info("Computing embeddings and scores — this may take a few seconds.")
        jd = parse_job_description(jd_text)
        # Use compute_scores but provide our weights: we'll call compute_scores and then re-weight
        results = compute_scores(jd, resumes)

        # Re-weight final_score by sliders:
        # compute component breakdown if available (some keys must exist)
        for r in results:
            # Extract raw components (some are similarity_norm, keyword_score, years_score, title_score)
            sim = r.get("similarity_norm", max(0.0, (r.get("similarity", 0.0) + 1) / 2))
            kw = r.get("keyword_score", 0.0)
            yrs = r.get("years_score", 0.0)
            title = r.get("title_score", 0.0)
            r["final_score"] = float(w_sim * sim + w_kw * kw + w_years * yrs + w_title * title)

        # Sort again by new final_score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        # Display DataFrame
        df = pd.DataFrame(results)
        display_cols = ["path", "final_score", "similarity", "similarity_norm", "keyword_score", "years", "years_score", "title_score"]
        # ensure all cols present
        for c in display_cols:
            if c not in df.columns:
                df[c] = None
        st.subheader("Ranking results")
        st.dataframe(df[display_cols].sort_values("final_score", ascending=False), height=420)

        # Download CSV
        csv = df[display_cols].to_csv(index=False)
        st.download_button("Download ranking CSV", csv, file_name="resume_ranking.csv")

        # Show top candidate details
        top = results[0]
        st.markdown("### Top Candidate")
        st.write("Path:", top.get("path"))
        st.write("Final score:", round(top.get("final_score", 0), 4))
        st.write("Similarity (raw):", round(top.get("similarity", 0), 4))
        st.write("Keyword match:", round(top.get("keyword_score", 0), 4))
        st.write("Years:", top.get("years", 0), "→ score:", round(top.get("years_score", 0), 4))
        st.write("Title score:", round(top.get("title_score", 0), 4))
        st.write("---")
        st.success("Ranking complete.")
