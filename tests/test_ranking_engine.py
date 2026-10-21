"""
Unit tests for RankingEngine module.
"""

import pandas as pd
import pytest
from src.ranking_engine import RankingEngine
from src.utils import DATAFRAME_COLUMNS


@pytest.fixture
def ranking_engine():
    return RankingEngine(similarity_weight=0.70, skill_weight=0.30)


def test_ranking_weights_validation():
    """Test validation of weights constructor parameters."""
    # Valid
    RankingEngine(similarity_weight=0.5, skill_weight=0.5)
    RankingEngine(similarity_weight=1.0, skill_weight=0.0)

    # Invalid range
    with pytest.raises(ValueError):
        RankingEngine(similarity_weight=-0.1, skill_weight=1.1)

    # Do not sum to 1.0
    with pytest.raises(ValueError):
        RankingEngine(similarity_weight=0.8, skill_weight=0.5)


def test_ranking_output_schema_and_order(ranking_engine):
    """Test DataFrame column schema, descending sorting, and rank indexing."""
    jd = "Seeking a Data Scientist proficient in Python, SQL, and Machine Learning."
    candidates = [
        {
            "candidate_name": "Low Match Candidate",
            "filename": "low.pdf",
            "email": "low@example.com",
            "phone": "555-0001",
            "extracted_text": "Experienced in React and CSS.",
            "parse_status": "success",
            "warnings": [],
        },
        {
            "candidate_name": "High Match Candidate",
            "filename": "high.pdf",
            "email": "high@example.com",
            "phone": "555-0002",
            "extracted_text": "Expert in Python, SQL, and Machine Learning.",
            "parse_status": "success",
            "warnings": [],
        },
    ]

    df = ranking_engine.rank_candidates(jd, candidates)

    # Verify columns match standard specification
    for col in DATAFRAME_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) == 2
    assert df.iloc[0]["candidate_name"] == "High Match Candidate"
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[1]["candidate_name"] == "Low Match Candidate"
    assert df.iloc[1]["rank"] == 2

    # Scores must be descending
    assert df.iloc[0]["final_score"] >= df.iloc[1]["final_score"]


def test_final_score_formula_verification(ranking_engine):
    """Verify that final_score = 0.70 * similarity + 0.30 * skill_match."""
    jd = "Python and SQL developer required."
    candidates = [
        {
            "candidate_name": "Test Candidate",
            "filename": "test.pdf",
            "email": "test@example.com",
            "phone": "555-0003",
            "extracted_text": "I specialize in Python and SQL database management.",
            "parse_status": "success",
            "warnings": [],
        }
    ]

    df = ranking_engine.rank_candidates(jd, candidates)
    row = df.iloc[0]

    expected_score = round(
        0.70 * row["similarity_score"] + 0.30 * row["skill_match_score"], 2
    )
    assert abs(row["final_score"] - expected_score) <= 0.05
    assert 0.0 <= row["final_score"] <= 100.0


def test_score_bounds_and_no_nan(ranking_engine):
    """Ensure all scores are within [0, 100] and no NaN values exist."""
    jd = "Senior Python engineer with Docker and Kubernetes."
    candidates = [
        {
            "candidate_name": "Normal Candidate",
            "filename": "cand1.pdf",
            "email": "c1@test.com",
            "phone": "555-0010",
            "extracted_text": "Python engineer with Docker expertise.",
            "parse_status": "success",
            "warnings": [],
        },
        {
            "candidate_name": "Empty Candidate",
            "filename": "empty.pdf",
            "email": "empty@test.com",
            "phone": "555-0011",
            "extracted_text": "",
            "parse_status": "warning",
            "warnings": ["No text"],
        },
    ]

    df = ranking_engine.rank_candidates(jd, candidates)

    for score_col in ["similarity_score", "skill_match_score", "final_score"]:
        assert not df[score_col].isna().any(), f"NaN found in {score_col}"
        assert (df[score_col] >= 0.0).all()
        assert (df[score_col] <= 100.0).all()


def test_edge_case_empty_candidate_list(ranking_engine):
    """Verify handling of empty candidate input."""
    df = ranking_engine.rank_candidates("Job Description", [])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == DATAFRAME_COLUMNS


def test_edge_case_empty_job_description(ranking_engine):
    """Verify handling when job description is empty string."""
    candidates = [
        {
            "candidate_name": "Cand",
            "filename": "cand.pdf",
            "email": "c@test.com",
            "phone": "123",
            "extracted_text": "Python and SQL",
            "parse_status": "success",
            "warnings": [],
        }
    ]
    df = ranking_engine.rank_candidates("", candidates)
    assert not df.empty
    assert df.iloc[0]["similarity_score"] == 0.0
    assert df.iloc[0]["final_score"] == 0.0
    assert "Job description is empty" in df.iloc[0]["warnings"]


def test_edge_case_empty_vocabulary(ranking_engine):
    """Verify handling when documents contain only stop words or symbols."""
    jd = "and the of to in a is"
    candidates = [
        {
            "candidate_name": "Stopword Only",
            "filename": "stop.pdf",
            "email": "stop@test.com",
            "phone": "123",
            "extracted_text": "that with as for it on",
            "parse_status": "success",
            "warnings": [],
        }
    ]
    # Should not crash on empty vocabulary
    df = ranking_engine.rank_candidates(jd, candidates)
    assert not df.empty
    assert df.iloc[0]["similarity_score"] == 0.0
