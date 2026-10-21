"""
Semantic embedding and neural similarity matcher.

Computes semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2)
with lazy loading, in-memory caching, CPU execution, and batch processing.
Excludes personally identifiable information (names, emails, phones) before embedding.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.cache_manager import CacheManager
from src.config import DEFAULT_SEMANTIC_MODEL, SEMANTIC_EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Global in-memory cache for loaded model instance to avoid reloading on Streamlit reruns
_MODEL_CACHE: Dict[str, Any] = {}


def strip_identifiers_for_embedding(
    text: str,
    candidate_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> str:
    """
    Remove direct candidate identifiers (name, email, phone, URLs) from document
    text before computing semantic embeddings to ensure privacy and prevent bias.

    Args:
        text: Raw document text.
        candidate_name: Optional candidate name to redact.
        email: Optional email to redact.
        phone: Optional phone number to redact.

    Returns:
        Cleaned text focused strictly on skills, experience, and domain content.
    """
    if not text:
        return ""

    cleaned = text

    # Strip explicit email and phone if provided
    if email and email != "N/A":
        cleaned = re.sub(re.escape(email), " ", cleaned, flags=re.IGNORECASE)
    if phone and phone != "N/A":
        cleaned = re.sub(re.escape(phone), " ", cleaned)

    # Strip any remaining email patterns and URLs
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", " ", cleaned)

    # Strip candidate name tokens if provided and sufficiently specific
    if candidate_name and candidate_name != "Candidate" and candidate_name != "Unknown Candidate":
        for part in candidate_name.split():
            if len(part) > 2:
                cleaned = re.sub(rf"\b{re.escape(part)}\b", " ", cleaned, flags=re.IGNORECASE)

    # Normalize whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


class SemanticMatcher:
    """
    Computes dense semantic embeddings and cosine similarities for resumes and job descriptions.
    Supports lazy model loading, local caching, and CPU batch execution.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        self.cache_manager = CacheManager() if use_cache else None
        self._model = None
        self._is_available = self._check_package_availability()

    @staticmethod
    def _check_package_availability() -> bool:
        """Check if sentence-transformers and torch are installed."""
        try:
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """True if sentence-transformers library is present in environment."""
        return self._is_available

    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Lazily load the SentenceTransformer model onto designated device.
        Uses in-memory cache to prevent reloading across Streamlit reruns.

        Returns:
            Tuple of (success_boolean, error_or_none_message).
        """
        if not self._is_available:
            return False, "sentence-transformers is not installed in the environment."

        global _MODEL_CACHE
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return True, None

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading semantic embedding model '{self.model_name}' on {self.device}...")
            model = SentenceTransformer(self.model_name, device=self.device)
            _MODEL_CACHE[self.model_name] = model
            self._model = model
            return True, None
        except Exception as e:
            err = f"Failed to load embedding model '{self.model_name}': {str(e)}"
            logger.warning(err)
            return False, err

    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate L2-normalized dense embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            1D numpy array of embedding, or None if model unavailable.
        """
        if not text or not text.strip():
            return None

        # Check local disk cache
        text_hash = CacheManager.compute_text_hash(text)
        if self.cache_manager:
            cached_emb = self.cache_manager.get_cached_embedding(text_hash, self.model_name)
            if cached_emb is not None:
                return cached_emb

        success, _ = self.load_model()
        if not success or self._model is None:
            return None

        try:
            emb = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if self.cache_manager:
                self.cache_manager.store_cached_embedding(text_hash, self.model_name, emb)
            return emb
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def compute_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Batch compute L2-normalized embeddings for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            List of 1D numpy arrays (or None for empty texts).
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        texts_to_compute: List[Tuple[int, str]] = []

        for idx, t in enumerate(texts):
            if not t or not t.strip():
                continue
            text_hash = CacheManager.compute_text_hash(t)
            if self.cache_manager:
                cached = self.cache_manager.get_cached_embedding(text_hash, self.model_name)
                if cached is not None:
                    results[idx] = cached
                    continue
            texts_to_compute.append((idx, t))

        if not texts_to_compute:
            return results

        success, _ = self.load_model()
        if not success or self._model is None:
            return results

        # Process in batches
        batch_size = SEMANTIC_EMBEDDING_BATCH_SIZE
        for i in range(0, len(texts_to_compute), batch_size):
            chunk = texts_to_compute[i : i + batch_size]
            chunk_indices = [item[0] for item in chunk]
            chunk_texts = [item[1] for item in chunk]

            try:
                embeddings = self._model.encode(
                    chunk_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for orig_idx, emb, txt in zip(chunk_indices, embeddings, chunk_texts):
                    results[orig_idx] = emb
                    if self.cache_manager:
                        t_hash = CacheManager.compute_text_hash(txt)
                        self.cache_manager.store_cached_embedding(t_hash, self.model_name, emb)
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")

        return results

    def compute_batch_similarities(
        self,
        jd_text: str,
        resume_texts: List[str],
    ) -> Tuple[List[float], bool, Optional[str]]:
        """
        Compute pairwise cosine similarities between a job description and resume texts.

        Args:
            jd_text: Job description string.
            resume_texts: List of candidate resume texts.

        Returns:
            Tuple of (similarity_percentages, fallback_used_flag, optional_warning_msg).
        """
        num_resumes = len(resume_texts)
        if not jd_text.strip() or num_resumes == 0:
            return [0.0] * num_resumes, False, None

        if not self._is_available:
            return (
                [0.0] * num_resumes,
                True,
                "Semantic similarity unavailable (sentence-transformers not installed). Fallback used.",
            )

        success, err = self.load_model()
        if not success:
            return (
                [0.0] * num_resumes,
                True,
                f"Semantic model could not be loaded ({err}). Fallback used.",
            )

        jd_emb = self.compute_embedding(jd_text)
        if jd_emb is None:
            return [0.0] * num_resumes, True, "Failed to compute embedding for Job Description."

        resume_embs = self.compute_embeddings_batch(resume_texts)

        similarities: List[float] = []
        for emb in resume_embs:
            if emb is None:
                similarities.append(0.0)
            else:
                # Dot product of normalized embeddings = cosine similarity
                cos_sim = float(np.dot(jd_emb, emb))
                # Clip to [0, 1] and scale to percentage
                pct = min(100.0, max(0.0, round(cos_sim * 100.0, 2)))
                similarities.append(pct)

        return similarities, False, None
