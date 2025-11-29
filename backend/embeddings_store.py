# backend/embeddings_store.py
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Detect OpenAI key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Try to import openai if key present
openai = None
if OPENAI_KEY:
    try:
        import openai as _openai
        openai = _openai
        openai.api_key = OPENAI_KEY
        logger.info("OpenAI package available; will attempt to use OpenAI embeddings.")
    except Exception as e:
        import traceback
        logger.exception("Failed to import or configure openai: %s", e)
        openai = None

# Local fallback: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except Exception:
    SentenceTransformer = None
    _has_st = False

_ST_MODEL: Optional[SentenceTransformer] = None

def _init_local_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        if not _has_st:
            raise RuntimeError(
                "sentence-transformers not installed. Either set OPENAI_API_KEY or install sentence-transformers."
            )
        logger.info("Loading local SentenceTransformer model: %s", name)
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL

def _openai_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Call OpenAI embeddings endpoint and return list of vectors.
    Defensive: checks response shape and raises clear errors if format unexpected.
    """
    if openai is None:
        raise RuntimeError("OpenAI not available in this environment.")

    if not texts:
        return []

    logger.info("Requesting OpenAI embeddings (model=%s) for %d texts", model_name, len(texts))
    vectors: List[List[float]] = []
    batch_size = int(os.getenv("EMB_BATCH_SIZE", "16"))

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = openai.Embeddings.create(model=model_name, input=batch)
        except Exception as e:
            logger.exception("OpenAI embeddings call failed for batch starting at %d: %s", i, e)
            raise RuntimeError("OpenAI embedding request failed: " + str(e))

        # Defensive checks: resp should be a dict-like with 'data'
        if not isinstance(resp, dict) and not hasattr(resp, "get"):
            # try to coerce if it's an object with .data
            try:
                resp_dict = dict(resp)
                resp = resp_dict
            except Exception:
                logger.error("OpenAI returned unexpected response type: %s", type(resp))
                raise RuntimeError("OpenAI returned unexpected response type")

        data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
        if not data:
            logger.error("OpenAI response missing 'data' or returned empty. Full response: %s", resp)
            raise RuntimeError("OpenAI embeddings response missing 'data' or empty")

        for item in data:
            # item should be dict-like and contain 'embedding'
            embedding = None
            if isinstance(item, dict):
                embedding = item.get("embedding")
            else:
                embedding = getattr(item, "embedding", None)

            if embedding is None:
                logger.error("OpenAI response item missing 'embedding': %s", item)
                raise RuntimeError("OpenAI returned item without 'embedding' field")
            vectors.append(list(embedding))

    logger.info("Received %d embeddings from OpenAI", len(vectors))
    return vectors

def _hf_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    if not _has_st:
        raise RuntimeError("sentence-transformers not installed.")
    model = _init_local_model(model_name)
    if not texts:
        return []
    logger.info("Computing local embeddings for %d texts with %s", len(texts), model_name)
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [list(vec) for vec in arr]

def get_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for each text in `texts`.
    Prefer OpenAI (if OPENAI_API_KEY present and import succeeded), otherwise fall back to sentence-transformers.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings.")
    if len(texts) == 0:
        return []

    # 1) Try OpenAI if available
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if openai is not None:
        try:
            return _openai_embeddings(texts, model_name)
        except Exception as e:
            logger.exception("OpenAI embeddings failed; will attempt fallback. Error: %s", e)

    # 2) Fallback to local sentence-transformers
    hf_model = os.getenv("HF_EMB_MODEL", "all-MiniLM-L6-v2")
    if _has_st:
        return _hf_embeddings(texts, hf_model)

    raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers.")
