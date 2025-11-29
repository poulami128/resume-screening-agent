# backend/embeddings_store.py
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # if present, used for OpenAI embeddings

def get_embeddings_for_texts(texts):
    """
    Input: texts -> list[str]
    Output: list of numpy arrays (one vector per text)

    Behavior:
    - If OPENAI_API_KEY is set, use OpenAI embeddings via LangChain's OpenAIEmbeddings.
    - Otherwise, use sentence-transformers local model "all-MiniLM-L6-v2".
    """
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    # If OpenAI key present, use OpenAI via LangChain
    if OPENAI_API_KEY:
        try:
            from langchain.embeddings import OpenAIEmbeddings
        except Exception as e:
            raise RuntimeError("langchain is required for OpenAI embeddings. Install via `pip install langchain`") from e

        embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectors = embedder.embed_documents(texts)
        return [np.array(v) for v in vectors]

    # Fallback: use sentence-transformers local model
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for local embeddings. Install via `pip install sentence-transformers`") from e

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts, show_progress_bar=True)
    return [np.array(v) for v in vectors]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D numpy arrays (vectors)."""
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
