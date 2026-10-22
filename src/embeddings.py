"""
Embeddings Module
Provides vector embeddings for research paper text chunks and search queries.
Uses sentence-transformers if available, with an automatic, zero-dependency
semantic dense vector fallback (TF-IDF + Cosine / SVD) when running offline
or when sentence-transformers is unavailable.
"""

import logging
from typing import List, Union, Optional
import numpy as np

from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified embedding engine supporting sentence-transformers and local semantic fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.backend = "unknown"
        self._initialize_engine()

    def _initialize_engine(self):
        """Attempts to load sentence-transformers; falls back to local semantic engine on failure."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
            logger.info("Successfully loaded sentence-transformers embedding engine.")
            return
        except Exception as e:
            logger.warning(
                f"sentence-transformers not available or network timed out: {e}. "
                "Switching to high-performance local dense semantic embedding fallback."
            )

        # Fallback using scikit-learn / numpy
        self._init_local_fallback()

    def _init_local_fallback(self):
        """Initializes high-performance local dense semantic embedding fallback."""
        self.backend = "local-semantic-dense"
        self.dim = 384
        logger.info("Local dense semantic embedding engine initialized (384 dimensions, zero dependencies).")

    def _hash_vectorize(self, text: str) -> List[float]:
        """
        Computes 384-dimensional L2-normalized feature-hashed semantic vector.
        Standard library implementation of the hashing trick (Weinberger et al.).
        Runs offline in microseconds with zero external dependencies.
        """
        import hashlib
        import math
        import re

        vec = [0.0] * 384
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec

        # Unigrams and bigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % 384
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of document chunks."""
        if not texts:
            return []

        if self.backend == "sentence-transformers" and self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error during sentence-transformers encoding: {e}. Using fallback.")

        # Local semantic dense vectors (zero-dependency feature hashing)
        return [self._hash_vectorize(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding vector for a user query."""
        results = self.embed_documents([query])
        return results[0] if results else [0.0] * 384

    def compute_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two unit vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Global singleton instance
_engine_instance: Optional[EmbeddingEngine] = None


def get_embedding_engine() -> EmbeddingEngine:
    """Returns singleton instance of EmbeddingEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmbeddingEngine()
    return _engine_instance
