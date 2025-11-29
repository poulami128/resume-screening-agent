# backend/embeddings_store.py
import os
import logging
from typing import List

log = logging.getLogger(__name__)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Try direct OpenAI usage first (no langchain dependency)
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
   import traceback
except Exception as e:
    # log full traceback + repr to help debugging in cloud logs
    log.error("OpenAI embeddings request failed: %s", repr(e))
    log.error("Traceback:\n%s", traceback.format_exc())
    raise RuntimeError("OpenAI embedding failed: " + repr(e))

else:
    openai = None

# Local fallback: sentence-transformers (only if installed)
try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except Exception:
    SentenceTransformer = None
    _has_st = False

_ST_MODEL = None
def _init_local_model(name="all-MiniLM-L6-v2"):
    global _ST_MODEL
    if _ST_MODEL is None:
        if not _has_st:
            raise RuntimeError("sentence-transformers not installed. Set OPENAI_API_KEY or install sentence-transformers locally.")
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL

def get_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for each text.
    Prefer OpenAI (if OPENAI_API_KEY present), else fallback to local sentence-transformers.
    """
    if openai is not None:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        vectors = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = openai.Embedding.create(input=batch, model=model)
                for item in resp["data"]:
                    vectors.append(item["embedding"])
            except Exception as e:
                log.error("OpenAI embedding call failed: %s", e)
                raise RuntimeError("OpenAI embedding failed: " + str(e))
        return vectors

    # fallback to local
    if _has_st:
        model = _init_local_model()
        arr = model.encode(texts, show_progress_bar=False)
        # SentenceTransformer returns numpy array; convert to lists
        return [list(v) for v in arr]

    raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers.")
