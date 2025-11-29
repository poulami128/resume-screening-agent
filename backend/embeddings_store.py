# backend/embeddings_store.py
"""
Resilient embeddings loader:
 - Try OpenAI
 - Try sentence-transformers
 - Try TF-IDF (scikit-learn)
 - If all fail, return zero vectors (app stays up; logs contain details)
This version logs stack traces to Streamlit logs for diagnosis.
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === OpenAI ===
openai = None
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    try:
        import openai as _openai  # type: ignore
        openai = _openai
        openai.api_key = OPENAI_KEY
        logger.info("OpenAI library loaded and key configured.")
    except Exception as e:
        logger.exception("OpenAI import/config failed: %s", e)
        openai = None

# === sentence-transformers ===
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _has_st = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    _has_st = False
    logger.info("sentence-transformers not available: %s", e)

_ST_MODEL: Optional["SentenceTransformer"] = None

def _init_local_model(name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    global _ST_MODEL
    if _ST_MODEL is None:
        if not _has_st:
            raise RuntimeError("sentence-transformers not installed.")
        logger.info("Loading sentence-transformers model: %s", name)
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL

# === OpenAI wrapper ===
def _openai_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    if openai is None:
        raise RuntimeError("OpenAI not configured.")
    if not texts:
        return []
    vectors = []
    batch_size = int(os.getenv("EMB_BATCH_SIZE", "16"))
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = openai.Embeddings.create(model=model_name, input=batch)
        except Exception as e:
            logger.exception("OpenAI embeddings request failed for batch starting %d: %s", i, e)
            raise
        # defensive parsing
        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            logger.error("OpenAI response has no data: %s", resp)
            raise RuntimeError("OpenAI response missing data")
        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if emb is None:
                logger.error("OpenAI item missing embedding: %s", item)
                raise RuntimeError("OpenAI item missing embedding")
            vectors.append(list(emb))
    return vectors

# === HF wrapper ===
def _hf_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    model = _init_local_model(model_name)
    if not texts:
        return []
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [list(vec) for vec in arr]

# === TF-IDF fallback ===
def _tfidf_embeddings(texts: List[str], max_features: int = 512) -> List[List[float]]:
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        logger.exception("scikit-learn import failed for TF-IDF fallback: %s", e)
        raise
    vect = TfidfVectorizer(max_features=max_features)
    X = vect.fit_transform(texts)  # sparse matrix
    dense = []
    # convert sparse rows safely
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel().tolist()
        dense.append(row)
    logger.info("TF-IDF produced shape (%d, %d)", X.shape[0], len(dense[0]) if dense else 0)
    return dense

# === Public API ===
def get_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """
    Robust embeddings: OpenAI -> HF -> TF-IDF -> zero-vectors fallback.
    All exceptions are logged in detail so you can inspect Streamlit logs.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings.")
    if len(texts) == 0:
        return []

    # 1) OpenAI
    if openai is not None:
        try:
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            logger.info("Attempting OpenAI embeddings with model %s", model_name)
            return _openai_embeddings(texts, model_name)
        except Exception:
            logger.exception("OpenAI embeddings failed; will try next backend.")

    # 2) sentence-transformers
    if _has_st:
        try:
            hf_model = os.getenv("HF_EMB_MODEL", "all-MiniLM-L6-v2")
            logger.info("Attempting sentence-transformers embeddings with model %s", hf_model)
            return _hf_embeddings(texts, hf_model)
        except Exception:
            logger.exception("sentence-transformers embeddings failed; will try TF-IDF fallback.")

    # 3) TF-IDF fallback
    try:
        logger.info("Attempting TF-IDF fallback embeddings.")
        tfidf_max = int(os.getenv("TFIDF_MAX_FEATURES", "512"))
        return _tfidf_embeddings(texts, max_features=tfidf_max)
    except Exception:
        logger.exception("TF-IDF fallback failed.")

    # 4) Last-resort: return zero vectors (keeps app running).
    logger.error("All embedding backends failed. Returning zero vectors to prevent app crash.")
    # choose a default dimension
    dim = int(os.getenv("FALLBACK_VECTOR_DIM", "128"))
    zero_vec = [0.0] * dim
    return [zero_vec for _ in texts]
