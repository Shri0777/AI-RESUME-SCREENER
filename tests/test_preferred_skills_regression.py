"""
Regression tests for preferred_skills schema consistency, empty results,
safe column handling, and scoring policy preservation.
"""

import pandas as pd
import pytest

from src.config import STRATEGY_BASELINE, STRATEGY_ENHANCED
from src.ranking_engine import RankingEngine
from src.report_generator import ReportGenerator, prepare_export_dataframe
from src.utils import DATAFRAME_COLUMNS
from src.visualizer import (
    plot_candidate_ranking,
    plot_candidate_skill_gap,
    plot_score_breakdown,
    plot_score_distribution,
    plot_skill_frequency,
)


@pytest.fixture
def baseline_engine():
    return RankingEngine(strategy=STRATEGY_BASELINE)


@pytest.fixture
def enhanced_engine():
    return RankingEngine(strategy=STRATEGY_ENHANCED)


@pytest.fixture
def sample_candidates():
    return [
        {
            "candidate_name": "Alice Johnson",
            "filename": "alice.pdf",
            "email": "alice@example.com",
            "phone": "555-0100",
            "extracted_text": (
                "Senior Data Scientist with experience in Python, SQL, Machine Learning, and Docker."
            ),
            "parse_status": "success",
            "warnings": [],
        },
        {
            "candidate_name": "Bob Smith",
            "filename": "bob.pdf",
            "email": "bob@example.com",
            "phone": "555-0200",
            "extracted_text": (
                "Data Analyst proficient in Python and SQL database queries."
            ),
            "parse_status": "success",
            "warnings": [],
        },
    ]


@pytest.fixture
def sample_jd_with_sections():
    return (
        "Role: Senior Data Scientist\n\n"
        "Required Skills:\n"
        "- Python\n"
        "- SQL\n"
        "- Machine Learning\n\n"
        "Preferred Skills:\n"
        "- Docker\n"
        "- Kubernetes\n"
    )


def test_baseline_ranking_includes_preferred_skills_column(
    baseline_engine, sample_jd_with_sections, sample_candidates
):
    """
    Ensure baseline mode returns the unified 21-column schema and specifically includes
    'preferred_skills', resolving the KeyError in app.py.
    """
    df = baseline_engine.rank_candidates(sample_jd_with_sections, sample_candidates)

    assert not df.empty
    assert "preferred_skills" in df.columns
    assert "required_skills" in df.columns
    assert "matched_preferred_skills" in df.columns
    assert "missing_preferred_skills" in df.columns
    assert "preferred_skill_match_score" in df.columns
    assert "required_skill_match_score" in df.columns

    # Accessing df.iloc[0]["preferred_skills"] directly MUST NOT raise KeyError
    preferred = df.iloc[0]["preferred_skills"]
    assert isinstance(preferred, list)
    assert "Docker" in preferred
    assert "Kubernetes" in preferred

    # Check candidate-specific preferred match
    alice_row = df[df["candidate_name"] == "Alice Johnson"].iloc[0]
    assert "Docker" in alice_row["matched_preferred_skills"]
    assert "Kubernetes" in alice_row["missing_preferred_skills"]
    assert alice_row["preferred_skill_match_score"] == 50.0

    bob_row = df[df["candidate_name"] == "Bob Smith"].iloc[0]
    assert bob_row["matched_preferred_skills"] == []
    assert "Docker" in bob_row["missing_preferred_skills"]
    assert "Kubernetes" in bob_row["missing_preferred_skills"]
    assert bob_row["preferred_skill_match_score"] == 0.0


def test_enhanced_ranking_includes_preferred_skills_column(
    enhanced_engine, sample_jd_with_sections, sample_candidates
):
    """
    Ensure enhanced mode also returns 'preferred_skills' and unified schema.
    """
    df = enhanced_engine.rank_candidates(sample_jd_with_sections, sample_candidates)

    assert not df.empty
    assert "preferred_skills" in df.columns
    assert df.iloc[0]["preferred_skills"] == ["Docker", "Kubernetes"]

    alice_row = df[df["candidate_name"] == "Alice Johnson"].iloc[0]
    assert alice_row["preferred_skill_match_score"] == 50.0


def test_baseline_scoring_policy_unaltered_by_preferred_skills(
    baseline_engine, sample_jd_with_sections, sample_candidates
):
    """
    Verify that in baseline mode, final_score is strictly 0.70 * similarity + 0.30 * skill_match,
    where skill_match is based exclusively on required skills (baseline policy preservation).
    """
    df = baseline_engine.rank_candidates(sample_jd_with_sections, sample_candidates)

    for _, row in df.iterrows():
        expected_final = round(
            0.70 * row["similarity_score"] + 0.30 * row["skill_match_score"], 2
        )
        assert abs(row["final_score"] - expected_final) <= 0.05
        # In baseline mode, skill_match_score must equal required_skill_match_score
        assert row["skill_match_score"] == row["required_skill_match_score"]


def test_empty_results_handling(baseline_engine, enhanced_engine):
    """
    Verify that ranking with empty candidate list returns an empty DataFrame
    containing all unified columns, including 'preferred_skills'.
    """
    for engine in [baseline_engine, enhanced_engine]:
        df = engine.rank_candidates("Seeking Python developer", [])
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "preferred_skills" in df.columns
        assert "required_skills" in df.columns
        assert list(df.columns) == DATAFRAME_COLUMNS


def test_empty_job_description_handling(baseline_engine, sample_candidates):
    """
    Verify ranking when job description is empty.
    """
    df = baseline_engine.rank_candidates("", sample_candidates)
    assert not df.empty
    assert "preferred_skills" in df.columns
    assert df.iloc[0]["preferred_skills"] == []
    assert df.iloc[0]["required_skills"] == []
    assert df.iloc[0]["preferred_skill_match_score"] == 0.0
    assert df.iloc[0]["final_score"] == 0.0


def test_no_preferred_skills_in_jd(baseline_engine, sample_candidates):
    """
    Verify behavior when job description has no preferred skills section.
    """
    jd_plain = "Data Scientist with Python, SQL, and Machine Learning."
    df = baseline_engine.rank_candidates(jd_plain, sample_candidates)

    assert not df.empty
    assert "preferred_skills" in df.columns
    assert df.iloc[0]["preferred_skills"] == []
    assert df.iloc[0]["matched_preferred_skills"] == []
    assert df.iloc[0]["missing_preferred_skills"] == []
    assert df.iloc[0]["preferred_skill_match_score"] == 0.0


def test_legacy_dataframe_missing_preferred_skills_defensive_handling():
    """
    Verify that UI and reporting helper functions handle DataFrames that lack
    'preferred_skills' or other optional columns without throwing KeyError.
    """
    legacy_df = pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate_name": "Legacy Candidate",
                "filename": "legacy.pdf",
                "email": "legacy@example.com",
                "phone": "555-9999",
                "similarity_score": 75.0,
                "skill_match_score": 80.0,
                "final_score": 76.5,
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["Machine Learning"],
                "detected_skills": ["Python", "SQL", "Git"],
                "required_skills": ["Python", "SQL", "Machine Learning"],
                "parse_status": "success",
                "warnings": "None",
                # 'preferred_skills', 'semantic_similarity_score', etc. omitted intentionally
            }
        ]
    )

    # 1. Export DataFrame preparation
    formatted = prepare_export_dataframe(legacy_df)
    assert not formatted.empty
    assert "Candidate Name" in formatted.columns

    # 2. Report generation
    rg = ReportGenerator()
    csv_bytes = rg.export_csv(legacy_df)
    assert len(csv_bytes) > 0

    excel_bytes = rg.export_excel(legacy_df)
    assert len(excel_bytes) > 0

    pdf_bytes = rg.export_pdf(legacy_df)
    assert len(pdf_bytes) > 0

    # 3. Visualizations
    fig_rank = plot_candidate_ranking(legacy_df)
    assert fig_rank is not None

    fig_breakdown = plot_score_breakdown(legacy_df)
    assert fig_breakdown is not None

    fig_freq = plot_skill_frequency(legacy_df)
    assert fig_freq is not None

    fig_dist = plot_score_distribution(legacy_df)
    assert fig_dist is not None

    fig_gap = plot_candidate_skill_gap(legacy_df.iloc[0])
    assert fig_gap is not None
