"""
Unit tests for TextCleaner module.
"""

import pytest
from src.text_cleaner import TextCleaner, ensure_nltk_resources


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_clean_empty_and_none(cleaner):
    """Test cleaning empty string and None inputs."""
    assert cleaner.clean("") == ""
    assert cleaner.clean(None) == ""
    assert cleaner.clean("   \t\n  ") == ""
    assert cleaner.tokenize_and_clean("") == []
    assert cleaner.tokenize_and_clean(None) == []


def test_clean_mixed_case(cleaner):
    """Test that input is normalized to lowercase."""
    text = "Machine LEARNING and Deep Neural NETWORKS"
    result = cleaner.clean(text)
    assert "machine" in result
    assert "learn" in result
    assert "network" in result
    assert result == result.lower()


def test_clean_urls_and_emails(cleaner):
    """Test that URLs and emails are cleanly removed from text."""
    text = (
        "Check my portfolio at https://example.com/alice and www.mycv.io. "
        "Contact me via alice@example.com or user.name+tag@sub.domain.org."
    )
    result = cleaner.clean(text)
    assert "https" not in result
    assert "example.com" not in result
    assert "www.mycv.io" not in result
    assert "alice@example.com" not in result
    assert "user.name" not in result


def test_clean_numeric_filtering(cleaner):
    """Test that standalone digits are filtered while alphanumeric tokens remain."""
    text = "Completed 100 projects in 2024 using Python3 and k8s with 5.5 score"
    tokens = cleaner.tokenize_and_clean(text)
    assert "100" not in tokens
    assert "2024" not in tokens
    assert "python3" in tokens
    assert "k8s" in tokens


def test_clean_technical_term_preservation(cleaner):
    """Test that C++, C#, .NET, Node.js, and scikit-learn are preserved safely."""
    text = "Proficient in C++, C#, .NET, Node.js, and scikit-learn."
    cleaned = cleaner.clean(text)
    # They are preserved through normalized tokens
    assert "cplusplus" in cleaned or "cpp" in cleaned
    assert "csharp" in cleaned
    assert "dotnet" in cleaned
    assert "nodejs" in cleaned
    assert "scikitlearn" in cleaned


def test_text_statistics(cleaner):
    """Test text statistics calculation."""
    text = "Python developer with machine learning experience."
    stats = cleaner.get_statistics(text)

    assert stats["char_count"] == len(text)
    assert stats["word_count"] == 6
    assert stats["cleaned_word_count"] > 0
    assert stats["vocab_size"] <= stats["cleaned_word_count"]

    # Empty stats
    empty_stats = cleaner.get_statistics("")
    assert empty_stats["char_count"] == 0
    assert empty_stats["word_count"] == 0
    assert empty_stats["cleaned_word_count"] == 0
    assert empty_stats["vocab_size"] == 0


def test_ensure_nltk_resources_idempotent():
    """Verify ensure_nltk_resources can be called multiple times without error."""
    ensure_nltk_resources()
    ensure_nltk_resources()
