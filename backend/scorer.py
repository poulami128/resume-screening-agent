# replace the compute_scores function in backend/scorer.py with this exact code

import logging
from typing import List, Dict
from backend.embeddings_store import get_embeddings_for_texts
from backend.parser import extract_skills_from_text

logger = logging.getLogger(__name__)

def compute_scores(jd_text: str, resumes: List[Dict]) -> List[Dict]:
    """
    Compute similarity scores between a job description (jd_text) and a list of resumes.
    Defensive & robust: coerces None -> "", ensures each resume has 'text', logs problems,
    uses embeddings backend and attaches matched_skills from parser.extract_skills_from_text.
    """
    # Coerce and validate JD text
    if jd_text is None:
        logger.warning("compute_scores received jd_text=None — treating as empty string.")
        jd_text = ""
    elif not isinstance(jd_text, str):
        logger.warning("compute_scores received jd_text of type %s — coercing to str.", type(jd_text))
        jd_text = str(jd_text)

    # Prepare safe resume list and texts for embeddings
    safe_resumes = []
    texts_for_emb = [jd_text]
    for idx, r in enumerate(resumes or []):
        if not isinstance(r, dict):
            logger.warning("resume at index %d is not a dict: %s — treating as empty.", idx, type(r))
            safe = {"name": f"resume_{idx}", "text": ""}
            safe_resumes.append(safe)
            texts_for_emb.append("")
            continue
        txt = r.get("text", "")
        if txt is None:
            logger.warning("resume at index %d missing 'text' (None) — using empty string.", idx)
            txt = ""
        elif not isinstance(txt, str):
            logger.warning("resume at index %d has non-str 'text' of type %s — coercing to str.", idx, type(txt))
            txt = str(txt)
        safe = dict(r)  # shallow copy so we can add score later
        safe["text"] = txt
        safe_resumes.append(safe)
        texts_for_emb.append(txt)

    # Get embeddings (safe call)
    try:
        vectors = get_embeddings_for_texts(texts_for_emb)
    except Exception as e:
        logger.exception("Failed to compute embeddings in compute_scores: %s", e)
        # Return resumes with zero scores and empty matched_skills so UI remains responsive
        for r in safe_resumes:
            r["score"] = 0.0
            r["matched_skills"] = []
        return safe_resumes

    # Validate embeddings shape
    if not isinstance(vectors, (list, tuple)) or len(vectors) < 1:
        logger.error("Embeddings returned unexpected value or empty: %s", type(vectors))
        for r in safe_resumes:
            r["score"] = 0.0
            r["matched_skills"] = []
        return safe_resumes

    jd_vec = vectors[0]
    resume_vectors = vectors[1:]

    # Safety check: matching lengths
    if len(resume_vectors) != len(safe_resumes):
        logger.error("Mismatch: %d resumes but %d resume vectors", len(safe_resumes), len(resume_vectors))
        for r in safe_resumes:
            r["score"] = 0.0
            r["matched_skills"] = []
        return safe_resumes

    # cosine similarity helper
    def cosine(a, b):
        try:
            import math
            if a is None or b is None:
                return 0.0
            if len(a) != len(b) or len(a) == 0:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

    # Precompute JD skills set
    jd_skills = set(extract_skills_from_text(jd_text.lower()))

    # Build results
    results = []
    for r_meta, vec in zip(safe_resumes, resume_vectors):
        try:
            score = float(cosine(jd_vec, vec))
        except Exception:
            score = 0.0

        resume_text = r_meta.get("text", "").lower()
        resume_skills = set(extract_skills_from_text(resume_text))
        matched = list(jd_skills.intersection(resume_skills))

        r_out = dict(r_meta)
        r_out["score"] = score
        r_out["matched_skills"] = matched
        results.append(r_out)

    # Sort results by score descending
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results
