# inside backend/scorer.py — replace the compute_scores function with this

import logging
from typing import List, Dict
from backend.embeddings_store import get_embeddings_for_texts

logger = logging.getLogger(__name__)

def compute_scores(jd_text: str, resumes: List[Dict]) -> List[Dict]:
    """
    Compute similarity scores between a job description (jd_text) and a list of resumes.
    Defensive: coerces None -> "", ensures each resume has 'text', logs problems,
    and returns an empty list if embeddings fail.
    """
    # Coerce jd_text to string and guard
    if jd_text is None:
        logger.warning("compute_scores received jd_text=None — treating as empty string.")
        jd_text = ""
    else:
        # in case jd_text is not a string (e.g. dict), coerce to str
        if not isinstance(jd_text, str):
            logger.warning("compute_scores received jd_text of type %s — coercing to str.", type(jd_text))
            jd_text = str(jd_text)

    # Prepare resume texts (defensive)
    safe_resumes = []
    texts_for_emb = [jd_text]
    for idx, r in enumerate(resumes or []):
        if not isinstance(r, dict):
            logger.warning("resume at index %d is not a dict: %s — skipping.", idx, type(r))
            safe_resumes.append({"score": 0.0, "text": ""})
            texts_for_emb.append("")
            continue
        txt = r.get("text")
        if txt is None:
            logger.warning("resume at index %d missing 'text' field — using empty string.", idx)
            txt = ""
        elif not isinstance(txt, str):
            logger.warning("resume at index %d has non-str 'text' of type %s — coercing to str.", idx, type(txt))
            txt = str(txt)
        texts_for_emb.append(txt)
        # keep original metadata to attach score later
        safe_resumes.append({**r, "text": txt})

    # Get embeddings (this is where previous crashes happened)
    try:
        vectors = get_embeddings_for_texts(texts_for_emb)
    except Exception as e:
        logger.exception("Failed to compute embeddings in compute_scores: %s", e)
        # return resumes with zero scores so app continues
        for r in safe_resumes:
            r["score"] = 0.0
        return safe_resumes

    # Validate embeddings
    if not isinstance(vectors, (list, tuple)) or len(vectors) < 1:
        logger.error("Embeddings returned unexpected value: %s", type(vectors))
        for r in safe_resumes:
            r["score"] = 0.0
        return safe_resumes

    jd_vec = vectors[0]
    resume_vectors = vectors[1:]

    # If dimension mismatch or lengths differ, bail gracefully
    if len(resume_vectors) != len(safe_resumes):
        logger.error("Mismatch: %d resumes but %d resume vectors", len(safe_resumes), len(resume_vectors))
        for r in safe_resumes:
            r["score"] = 0.0
        return safe_resumes

    # compute cosine similarity (defensive)
    def cosine(a, b):
        try:
            import math
            # ensure same length
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

    from backend.parser import extract_skills_from_text

jd_skills = set(extract_skills_from_text(jd_text.lower()))

results = []
for r_meta, vec in zip(safe_resumes, resume_vectors):
    try:
        score = float(cosine(jd_vec, vec))
    except Exception:
        score = 0.0

    # Determine matched skills
    resume_text = r_meta.get("text", "").lower()
    resume_skills = set(extract_skills_from_text(resume_text))
    matched = list(jd_skills.intersection(resume_skills))

    r = { **r_meta }
    r["score"] = score
    r["matched_skills"] = matched

    results.append(r)


    # Optionally sort by score descending
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results
