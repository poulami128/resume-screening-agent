# Resume Screening Agent

**Category:** People & HR — Resume Screening Agent  
**Built for:** AI Agent Development Challenge  
**Author:** Your Name Here  
**Demo:** https://<your-username>-resume-screening-agent.streamlit.app  
**Repo:** https://github.com/poulami128/resume-screening-agent

---

## Overview
Resume Screening Agent is a Streamlit app that ranks candidate resumes against a job description using embeddings and similarity scoring. It helps recruiters shortlist candidates faster and highlights matched / missing skills.

Key capabilities:
- Upload resumes (PDF / DOCX / TXT) or use sample folder
- Parse job description and extract key skills
- Compute embeddings (OpenAI → sentence-transformers → TF-IDF fallback)
- Rank resumes by cosine similarity and show short explanations
- Export ranked results as CSV

---

## Demo Link
Open the live demo here:

https://resume-screening-agent-jwcznsj2yiokfmlhqcuxrs.streamlit.app/



---

## Features
- Job description parsing and requirement extraction
- Embedding-based ranking with robust fallback
- Resume upload and automated text extraction (PDF/DOCX)
- Skill matching and a short explanation per resume
- CSV export of results

---

## Architecture (short)
1. Streamlit UI collects JD + resumes.  
2. Preprocessing extracts text (PDF/DOCX/TXT).  
3. Embeddings wrapper chooses OpenAI (if key), then HF, then TF-IDF fallback.  
4. Cosine similarity computes score per resume.  
5. Results displayed with explanations and CSV export.

Mermaid diagram included in README (or see diagram.png).

---

## How to run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows:
.venv\Scripts\activate
# mac/linux:
source .venv/bin/activate

pip install -r requirements.txt
# Optional: add OpenAI API key
export OPENAI_API_KEY="sk-..."

streamlit run app.py

