"""
NLTK-based text preprocessing and cleaning pipeline.

Implements lowercase normalization, URL/email removal, punctuation and numeric
filtering (preserving technical skill identifiers), tokenization, stop-word removal,
WordNet lemmatization, and text statistics generation with reliable fallbacks.
"""

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Fallback stop words in case NLTK corpus download is unavailable
FALLBACK_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}

_NLTK_RESOURCES_CHECKED = False


def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available, attempting download once if missing.
    Catches errors gracefully so the application continues with fallback mechanisms.
    """
    global _NLTK_RESOURCES_CHECKED
    if _NLTK_RESOURCES_CHECKED:
        return

    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                logger.info(f"Downloading missing NLTK resource '{pkg}'...")
                nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(
                    f"Could not download NLTK resource '{pkg}': {e}. Fallback will be used."
                )

    _NLTK_RESOURCES_CHECKED = True


class TextCleaner:
    """
    Modular text cleaner and preprocessor for resumes and job descriptions.
    """

    def __init__(self) -> None:
        ensure_nltk_resources()
        self._stopwords = self._load_stopwords()
        self._lemmatizer = self._load_lemmatizer()

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from NLTK, falling back to built-in set if unavailable."""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"NLTK stopwords unavailable ({e}), using built-in fallback set.")
            return FALLBACK_STOPWORDS

    def _load_lemmatizer(self):
        """Load WordNet lemmatizer from NLTK, falling back to dummy if unavailable."""
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            # Test that it works
            lemmatizer.lemmatize("testing")
            return lemmatizer
        except Exception as e:
            logger.warning(f"WordNet lemmatizer unavailable ({e}), using fallback.")
            return None

    def clean(self, text: Optional[str]) -> str:
        """
        Execute full cleaning pipeline and return single normalized clean text string.

        Args:
            text: Input raw text string or None.

        Returns:
            Normalized, cleaned text string suitable for TF-IDF vectorization.
        """
        tokens = self.tokenize_and_clean(text)
        return " ".join(tokens)

    def tokenize_and_clean(self, text: Optional[str]) -> List[str]:
        """
        Execute the step-by-step cleaning and tokenization pipeline:
        1. Handle empty / None text
        2. Convert to lowercase
        3. Remove URLs and email addresses
        4. Normalize key technical terms to preserve them through tokenization
        5. Remove unwanted punctuation and special characters
        6. Tokenize with NLTK (or fallback regex)
        7. Remove English stop words and standalone numeric tokens
        8. Lemmatize tokens
        9. Return clean token list

        Args:
            text: Input raw text string.

        Returns:
            List of cleaned, lemmatized tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Lowercase text
        processed = text.lower()

        # 2. Remove URLs
        processed = re.sub(r"https?://\S+|www\.\S+", " ", processed)

        # 3. Remove email addresses
        processed = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", processed)

        # 4. Preserve critical tech terms before punctuation removal
        # E.g. 'c++' -> 'cplusplus', 'c#' -> 'csharp', '.net' -> 'dotnet', 'node.js' -> 'nodejs'
        # 'scikit-learn' -> 'scikit_learn'
        processed = re.sub(r"(?<![a-z0-9_])c\+\+(?![a-z0-9_])", " cplusplus ", processed)
        processed = re.sub(r"(?<![a-z0-9_])c\#(?![a-z0-9_])", " csharp ", processed)
        processed = re.sub(r"(?<![a-z0-9_])\.net(?![a-z0-9_])", " dotnet ", processed)
        processed = re.sub(r"(?<![a-z0-9_])node\.js(?![a-z0-9_])", " nodejs ", processed)
        processed = re.sub(r"(?<![a-z0-9_])scikit-learn(?![a-z0-9_])", " scikitlearn ", processed)

        # 5. Replace non-alphanumeric characters with spaces (preserve alphanumeric and internal hyphens/underscores if desired)
        processed = re.sub(r"[^a-z0-9_\s]", " ", processed)

        # 6. Tokenize
        raw_tokens = self._tokenize(processed)

        # 7. Filter stop words, standalone numbers, single characters (except meaningful ones)
        filtered_tokens: List[str] = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # Remove standalone digits
            if token.isdigit():
                continue
            # Remove stop words
            if token in self._stopwords:
                continue
            # Skip single-character tokens (unless they carry significance)
            if len(token) == 1 and token not in {"r", "c"}:
                continue

            # 8. Lemmatize
            lemmatized = self._lemmatize_token(token)
            if lemmatized:
                filtered_tokens.append(lemmatized)

        return filtered_tokens

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize with safe regex fallback."""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            return re.findall(r"\b[a-z0-9_]+\b", text)

    def _lemmatize_token(self, token: str) -> str:
        """Lemmatize a single token using WordNetLemmatizer if available."""
        if self._lemmatizer is not None:
            try:
                # Lemmatize as noun, then verb if appropriate
                lemma = self._lemmatizer.lemmatize(token, pos="n")
                return self._lemmatizer.lemmatize(lemma, pos="v")
            except Exception:
                return token
        return token

    def get_statistics(self, text: Optional[str]) -> Dict[str, int]:
        """
        Calculate descriptive text statistics for UI display.

        Args:
            text: Raw or processed text string.

        Returns:
            Dictionary with character count, word count, cleaned word count,
            and vocabulary size.
        """
        if not text or not isinstance(text, str):
            return {
                "char_count": 0,
                "word_count": 0,
                "cleaned_word_count": 0,
                "vocab_size": 0,
            }

        raw_words = text.split()
        cleaned_tokens = self.tokenize_and_clean(text)

        return {
            "char_count": len(text),
            "word_count": len(raw_words),
            "cleaned_word_count": len(cleaned_tokens),
            "vocab_size": len(set(cleaned_tokens)),
        }
