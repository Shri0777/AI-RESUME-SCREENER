"""
Local disk cache manager for parsed resume texts and semantic embeddings.

Improves batch processing efficiency for large datasets (1,000+ resumes)
by storing parsed document text and vector embeddings keyed by content hash.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)

PARSER_CACHE_VERSION = "v1.1"


class CacheManager:
    """Manages content-addressed local disk caching for text extraction and embeddings."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.text_cache_dir = self.cache_dir / "text_cache"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create cache directories if they do not exist."""
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(file_input: Union[str, Path, bytes]) -> str:
        """Compute SHA-256 hash of a file path or raw bytes."""
        hasher = hashlib.sha256()
        if isinstance(file_input, bytes):
            hasher.update(file_input)
        elif isinstance(file_input, (str, Path)):
            with open(file_input, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute SHA-256 hash of text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_cached_text(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve parsed document text and metadata if cached."""
        cache_file = self.text_cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("cache_version") == PARSER_CACHE_VERSION:
                    return data
            except Exception as e:
                logger.warning(f"Error reading text cache for {file_hash}: {e}")
        return None

    def store_cached_text(self, file_hash: str, parsed_record_dict: Dict[str, Any]) -> None:
        """Save parsed record metadata and extracted text to local cache."""
        cache_file = self.text_cache_dir / f"{file_hash}.json"
        try:
            record_copy = dict(parsed_record_dict)
            record_copy["cache_version"] = PARSER_CACHE_VERSION
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record_copy, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing text cache for {file_hash}: {e}")

    def get_cached_embedding(self, text_hash: str, model_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding array from cache if available."""
        safe_model_id = hashlib.md5(model_id.encode("utf-8")).hexdigest()[:8]
        cache_file = self.embedding_cache_dir / f"{text_hash}_{safe_model_id}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Error reading embedding cache: {e}")
        return None

    def store_cached_embedding(self, text_hash: str, model_id: str, embedding: np.ndarray) -> None:
        """Save embedding array to local cache."""
        safe_model_id = hashlib.md5(model_id.encode("utf-8")).hexdigest()[:8]
        cache_file = self.embedding_cache_dir / f"{text_hash}_{safe_model_id}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Error writing embedding cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached files."""
        for p in self.text_cache_dir.glob("*.json"):
            try:
                p.unlink()
            except OSError:
                pass
        for p in self.embedding_cache_dir.glob("*.npy"):
            try:
                p.unlink()
            except OSError:
                pass
