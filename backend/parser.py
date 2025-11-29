# backend/parser.py
import os
from docx import Document
from pypdf import PdfReader
import re
from collections import Counter

# ----------------------------
# RESUME TEXT EXTRACTION
# ----------------------------

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except:
            page_text = None
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

def extract_resume_text(path: str) -> str:
    """Reads resume (PDF, DOCX, TXT) and returns plain text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    if ext == ".txt":
        return extract_text_from_txt(path)

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


# ----------------------------
# JOB DESCRIPTION PARSING
# ----------------------------

def parse_job_description(jd_text: str):
    """
    Breaks down JD into:
    - required years
    - keyword list
    - raw JD text
    """

    jd_lower = jd_text.lower()

    # Extract years like "2 years", "3+ yrs", etc.
    years = None
    match = re.search(r"(\d+)\+?\s*(?:years|yrs)", jd_lower)
    if match:
        years = int(match.group(1))

    # Extract keywords (simple most frequent words)
    stopwords = set([
        "the","and","for","with","you","are","job","role","our","your","this",
        "we","to","in","of","a","an","on","as","is","be","or","from","will"
    ])

    words = [w.strip(".,()") for w in jd_lower.split()]
    clean_words = [w for w in words if w.isalpha() and w not in stopwords and len(w) > 2]

    keyword_list = [w for w,_ in Counter(clean_words).most_common(40)]

    return {
        "years": years,
        "keywords": keyword_list,
        "raw": jd_text
    }
