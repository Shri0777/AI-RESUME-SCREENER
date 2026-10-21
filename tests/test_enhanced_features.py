"""
Unit tests for enhanced ranking strategies, semantic fallback,
hierarchical skill matching, privacy safeguards, and evaluation metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.cache_manager import CacheManager
from src.config import STRATEGY_BASELINE, STRATEGY_ENHANCED
from src.ranking_engine import RankingEngine
from src.ranking_evaluation import (
    dcg_at_k,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.semantic_matcher import SemanticMatcher, strip_identifiers_for_embedding
from src.skill_extractor import SkillExtractor


@pytest.fixture
def baseline_engine():
    return RankingEngine(strategy=STRATEGY_BASELINE)


@pytest.fixture
def enhanced_engine():
    return RankingEngine(
        strategy=STRATEGY_ENHANCED,
        lexical_weight=0.40,
        semantic_weight=0.35,
        enhanced_skill_weight=0.25,
    )


def test_enhanced_weight_validation():
    """Verify that enhanced three-part weights must sum to 1.0 and be non-negative."""
    # Valid
    RankingEngine(
        strategy=STRATEGY_ENHANCED,
        lexical_weight=0.40,
        semantic_weight=0.30,
        enhanced_skill_weight=0.30,
    )

    # Negative weight
    with pytest.raises(ValueError):
        RankingEngine(
            strategy=STRATEGY_ENHANCED,
            lexical_weight=-0.1,
            semantic_weight=0.6,
            enhanced_skill_weight=0.5,
        )

    # Does not sum to 1.0
    with pytest.raises(ValueError):
        RankingEngine(
            strategy=STRATEGY_ENHANCED,
            lexical_weight=0.5,
            semantic_weight=0.5,
            enhanced_skill_weight=0.5,
        )


def test_semantic_fallback_graceful(enhanced_engine):
    """
    Test that when semantic model is unavailable or in fallback,
    enhanced ranking falls back predictably without fake scores.
    """
    jd = "Senior Python engineer with Docker and AWS experience."
    candidates = [
        {
            "candidate_name": "Test Candidate",
            "filename": "cand.pdf",
            "email": "cand@test.com",
            "phone": "555-0101",
            "extracted_text": "Experienced Python engineer with Docker and AWS.",
            "parse_status": "success",
            "warnings": [],
        }
    ]

    df = enhanced_engine.rank_candidates(jd, candidates, strategy=STRATEGY_ENHANCED)
    assert not df.empty
    row = df.iloc[0]

    # Scores must be bounded [0, 100]
    assert 0.0 <= row["final_score"] <= 100.0
    assert 0.0 <= row["similarity_score"] <= 100.0
    assert 0.0 <= row["skill_match_score"] <= 100.0

    # If semantic matcher is in fallback mode, strategy label reflects fallback
    if not enhanced_engine.semantic_matcher.is_available:
        assert "fallback" in row["strategy_used"]
        assert row["semantic_similarity_score"] == 0.0


def test_candidate_outranking_benchmark(baseline_engine, enhanced_engine):
    """
    A candidate matching job requirements must outrank an unrelated candidate
    in both baseline and enhanced modes.
    """
    jd = "Data Scientist with Python, SQL, and Machine Learning."
    candidates = [
        {
            "candidate_name": "Irrelevant Candidate",
            "filename": "irr.pdf",
            "email": "irr@example.com",
            "phone": "555-1111",
            "extracted_text": "Chef specialized in French pastry and bakery management.",
            "parse_status": "success",
            "warnings": [],
        },
        {
            "candidate_name": "Relevant Candidate",
            "filename": "rel.pdf",
            "email": "rel@example.com",
            "phone": "555-2222",
            "extracted_text": "Data Scientist with Python, SQL, and Machine Learning expertise.",
            "parse_status": "success",
            "warnings": [],
        },
    ]

    for engine, strat in [(baseline_engine, STRATEGY_BASELINE), (enhanced_engine, STRATEGY_ENHANCED)]:
        df = engine.rank_candidates(jd, candidates, strategy=strat)
        assert df.iloc[0]["candidate_name"] == "Relevant Candidate"
        assert df.iloc[1]["candidate_name"] == "Irrelevant Candidate"
        assert df.iloc[0]["final_score"] > df.iloc[1]["final_score"]


def test_strip_identifiers_for_privacy():
    """Ensure names, emails, phone numbers, and URLs are stripped from embedding text."""
    raw = (
        "John Doe\n"
        "Email: john.doe@example.com | Phone: (555) 019-2831\n"
        "Portfolio: https://johndoe.dev\n"
        "Senior Software Engineer specializing in Python, Docker, and Kubernetes."
    )
    sanitized = strip_identifiers_for_embedding(
        raw, candidate_name="John Doe", email="john.doe@example.com", phone="(555) 019-2831"
    )

    assert "john.doe@example.com" not in sanitized
    assert "555" not in sanitized
    assert "https://" not in sanitized
    assert "Python" in sanitized
    assert "Docker" in sanitized
    assert "Kubernetes" in sanitized


def test_required_vs_preferred_skill_sections():
    """Test heuristic parsing of mandatory vs. preferred skills."""
    extractor = SkillExtractor()
    jd_text = (
        "Role: Senior Data Scientist\n\n"
        "Required Qualifications:\n"
        "- Python\n"
        "- SQL\n"
        "- Machine Learning\n\n"
        "Preferred Skills:\n"
        "- Docker\n"
        "- Kubernetes\n"
        "- PyTorch\n"
    )

    req, pref, is_inferred = extractor.extract_skills_from_jd_sections(jd_text)
    assert is_inferred is True
    assert "Python" in req
    assert "SQL" in req
    assert "Machine Learning" in req

    assert "Docker" in pref
    assert "Kubernetes" in pref
    assert "PyTorch" in pref

    # Disjointness check
    assert len(set(req) & set(pref)) == 0


def test_hierarchical_skill_score_weighting():
    """
    Verify that mandatory skills carry greater weight than preferred skills,
    preventing 10 preferred skills from outweighing core missing requirements.
    """
    extractor = SkillExtractor()

    # Candidate 1: Has 100% of mandatory (4/4), but 0% of preferred (0/5)
    score_cand1, req1, pref1, _ = extractor.calculate_hierarchical_skill_score(
        matched_req_count=4,
        total_req_count=4,
        matched_pref_count=0,
        total_pref_count=5,
        req_weight=0.80,
        pref_weight=0.20,
    )

    # Candidate 2: Has 25% of mandatory (1/4), but 100% of preferred (5/5)
    score_cand2, req2, pref2, _ = extractor.calculate_hierarchical_skill_score(
        matched_req_count=1,
        total_req_count=4,
        matched_pref_count=5,
        total_pref_count=5,
        req_weight=0.80,
        pref_weight=0.20,
    )

    assert req1 == 100.0
    assert pref1 == 0.0
    assert score_cand1 == 80.0

    assert req2 == 25.0
    assert pref2 == 100.0
    # 0.80 * 25 + 0.20 * 100 = 20 + 20 = 40.0
    assert score_cand2 == 40.0

    # Candidate 1 must score significantly higher because mandatory skills dominate
    assert score_cand1 > score_cand2


def test_ranking_evaluation_metrics():
    """Verify calculation of IR ranking metrics and edge cases."""
    ranked = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant = {"doc_A", "doc_C"}

    # Precision@K
    assert precision_at_k(ranked, relevant, k=1) == 1.0  # doc_A is relevant
    assert precision_at_k(ranked, relevant, k=2) == 0.5  # 1/2
    assert precision_at_k(ranked, relevant, k=3) == round(2 / 3, 4)

    # Recall@K
    assert recall_at_k(ranked, relevant, k=1) == 0.5   # 1 of 2 total
    assert recall_at_k(ranked, relevant, k=3) == 1.0   # both retrieved in top 3

    # MRR
    assert reciprocal_rank(ranked, relevant) == 1.0  # first relevant is at rank 1
    assert reciprocal_rank(["doc_B", "doc_A"], relevant) == 0.5  # first hit at rank 2
    assert reciprocal_rank(["doc_B", "doc_D"], relevant) == 0.0  # no relevant hit

    # NDCG@K
    graded = {"doc_A": 3.0, "doc_B": 0.0, "doc_C": 2.0, "doc_D": 0.0, "doc_E": 1.0}
    ndcg3 = ndcg_at_k(ranked, graded, k=3)
    assert 0.0 <= ndcg3 <= 1.0
    assert ndcg3 > 0.80  # highly relevant doc_A is rank 1

    # Empty edge cases
    assert precision_at_k([], relevant, k=3) == 0.0
    assert recall_at_k([], relevant, k=3) == 0.0
    assert ndcg_at_k([], graded, k=3) == 0.0
    assert ndcg_at_k(ranked, {}, k=3) == 0.0

    # Pairwise accuracy
    acc, conc, tot = pairwise_accuracy(ranked, graded)
    assert 0.0 <= acc <= 1.0
    assert tot > 0


def test_cache_manager(tmp_path):
    """Test disk caching of parsed records and embeddings."""
    cache = CacheManager(cache_dir=tmp_path)
    file_hash = "abc123hash"
    record = {"filename": "test.pdf", "extracted_text": "Sample text", "parse_status": "success"}

    # Not cached initially
    assert cache.get_cached_text(file_hash) is None

    # Store and retrieve
    cache.store_cached_text(file_hash, record)
    cached = cache.get_cached_text(file_hash)
    assert cached is not None
    assert cached["filename"] == "test.pdf"

    # Embedding cache
    text_hash = "txthash456"
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cache.get_cached_embedding(text_hash, "model_v1") is None
    cache.store_cached_embedding(text_hash, "model_v1", emb)
    retrieved_emb = cache.get_cached_embedding(text_hash, "model_v1")
    assert retrieved_emb is not None
    np.testing.assert_allclose(emb, retrieved_emb)
