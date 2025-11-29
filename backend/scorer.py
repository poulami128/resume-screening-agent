# backend/scorer.py
from typing import List, Dict
import re
from .embeddings_store import get_embeddings_for_texts, cosine_sim

def extract_years_of_experience(resume_text: str) -> int:
    """
    Naive extraction: find all occurrences of "N years" or "N yrs" and return the max.
    """
    m = re.findall(r"(\d+)\+?\s*(?:years|yrs)", resume_text.lower())
    if not m:
        return 0
    nums = [int(x) for x in m]
    return max(nums)

def keyword_match_score(resume_text: str, keywords: List[str]) -> float:
    """
    Fraction of keywords present in resume_text.
    keywords: list of strings (already lowercased)
    """
    r = resume_text.lower()
    total = len(keywords)
    if total == 0:
        return 0.0
    found = sum(1 for k in keywords if k.lower() in r)
    return found / total

def title_match_score(resume_text: str, jd_text: str) -> float:
    """
    Simple title match: check overlap of significant words from JD in resume.
    """
    jd_words = [w for w in re.findall(r"\w+", jd_text.lower()) if len(w) > 3]
    r = resume_text.lower()
    matched = sum(1 for w in jd_words[:20] if w in r)
    # normalize and cap at 1.0
    return min(matched / 10.0, 1.0)

def compute_scores(jd: Dict, resumes: List[Dict]) -> List[Dict]:
    """
    jd: {"years": int or None, "keywords": List[str], "raw": jd_text}
    resumes: list of {"path": path_str, "text": resume_text}
    Returns: list of dicts with score breakdown, sorted by final_score desc.
    """
    jd_text = jd["raw"]
    resume_texts = [r["text"] for r in resumes]

    # 1. Create embeddings for JD + resumes
    texts_for_emb = [jd_text] + resume_texts
    emb_vectors = get_embeddings_for_texts(texts_for_emb)
    jd_vec = emb_vectors[0]
    resume_vecs = emb_vectors[1:]

    results = []
    for idx, r in enumerate(resumes):
        sim = cosine_sim(jd_vec, resume_vecs[idx])  # semantic similarity in [ -1, 1 ]
        # normalize sim to [0,1] in case negative values appear (rare)
        sim_norm = max(0.0, (sim + 1) / 2) if sim < 0.0 else sim

        years = extract_years_of_experience(r["text"])
        # years_score: if jd has no years requirement -> full credit (1.0)
        if jd.get("years") is None:
            years_score = 1.0
        else:
            req = max(1, int(jd.get("years", 0)))
            years_score = min(1.0, years / req) if req > 0 else 1.0

        kw_score = keyword_match_score(r["text"], jd.get("keywords", []))
        title_score = title_match_score(r["text"], jd_text)

        # FINAL weighted score (tunable)
        final = 0.50 * sim_norm + 0.25 * kw_score + 0.15 * years_score + 0.10 * title_score

        results.append({
            "path": r.get("path"),
            "similarity": float(sim),
            "similarity_norm": float(sim_norm),
            "keyword_score": float(kw_score),
            "years": int(years),
            "years_score": float(years_score),
            "title_score": float(title_score),
            "final_score": float(final)
        })

    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results_sorted
