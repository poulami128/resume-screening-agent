# backend/embeddings_store.py
"""
Embeddings utility with three tiers:
 1) OpenAI (if OPENAI_API_KEY present and openai installed)
 2) sentence-transformers (if installed)
 3) TF-IDF fallback using scikit-learn (guaranteed if scikit-learn in env)
This ensures the app won't crash just because OpenAI/HF are unavailable.
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Try OpenAI import (optional) ===
openai = None
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    try:
        import openai as _openai  # type: ignore
        openai = _openai
        openai.api_key = OPENAI_KEY
        logger.info("OpenAI detected and configured.")
    except Exception:
        logger.exception("OpenAI import/config failed; continuing without OpenAI.")

# === Try sentence-transformers import (optional) ===
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _has_st = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _has_st = False

_ST_MODEL: Optional[SentenceTransformer] = None

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
        raise RuntimeError("OpenAI not available.")
    if not texts:
        return []

    vectors: List[List[float]] = []
    batch_size = int(os.getenv("EMB_BATCH_SIZE", "16"))
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = openai.Embeddings.create(model=model_name, input=batch)
        except Exception as e:
            logger.exception("OpenAI embeddings call failed: %s", e)
            raise RuntimeError("OpenAI embeddings call failed: " + str(e))

        # Defensive checks
        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            logger.error("OpenAI response missing 'data': %s", resp)
            raise RuntimeError("OpenAI returned unexpected response format.")

        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if emb is None:
                logger.error("OpenAI response item missing embedding: %s", item)
                raise RuntimeError("OpenAI response item missing embedding.")
            vectors.append(list(emb))
    return vectors

# === HF wrapper ===
def _hf_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    model = _init_local_model(model_name)
    if not texts:
        return []
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [list(vec) for vec in arr]

# === TF-IDF fallback (guaranteed if scikit-learn present) ===
def _tfidf_embeddings(texts: List[str], max_features: int = 512) -> List[List[float]]:
    """
    Lightweight fallback: convert texts to TF-IDF vectors (dense lists).
    Not semantic but keeps the app functional without extra deps.
    """
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        logger.exception("scikit-learn not available for TF-IDF fallback: %s", e)
        raise RuntimeError("TF-IDF fallback requires scikit-learn.") from e

    vect = TfidfVectorizer(max_features=max_features)
    X = vect.fit_transform(texts)  # sparse matrix
    # Convert each row to dense list
    dense_list = []
    for row_idx in range(X.shape[0]):
        row = X[row_idx].toarray().ravel().tolist()
        dense_list.append(row)
    logger.info("TF-IDF fallback produced vectors shape: (%d, %d)", X.shape[0], len(dense_list[0]) if dense_list else 0)
    return dense_list

# === Public API ===
def get_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for the input texts using:
      OpenAI -> sentence-transformers -> TF-IDF fallback.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings.")
    if len(texts) == 0:
        return []

    # 1) OpenAI if configured
    if openai is not None:
        try:
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            return _openai_embeddings(texts, model_name)
        except Exception:
            logger.exception("OpenAI embeddings failed; attempting next backend.")

    # 2) sentence-transformers if installed
    if _has_st:
        try:
            hf_model = os.getenv("HF_EMB_MODEL", "all-MiniLM-L6-v2")
            return _hf_embeddings(texts, hf_model)
        except Exception:
            logger.exception("sentence-transformers embeddings failed; attempting TF-IDF fallback.")

    # 3) TF-IDF fallback
    try:
        return _tfidf_embeddings(texts, max_features=int(os.getenv("TFIDF_MAX_FEATURES", "512")))
    except Exception as e:
        logger.exception("TF-IDF fallback failed: %s", e)
        raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers.") from e
