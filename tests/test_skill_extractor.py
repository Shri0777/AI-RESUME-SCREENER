"""
Unit tests for SkillExtractor module.
"""

import pytest
from src.skill_extractor import SkillExtractor


@pytest.fixture
def extractor():
    return SkillExtractor()


def test_extract_canonical_skills(extractor):
    """Test extraction of canonical skills."""
    text = "Candidate has strong proficiency in Python, PostgreSQL, and Docker."
    skills = extractor.extract_skills(text)
    assert "Python" in skills
    assert "PostgreSQL" in skills
    assert "Docker" in skills


def test_extract_skill_variations_and_aliases(extractor):
    """Test alias normalization to canonical names."""
    text = "Hands-on experience with ML, DL, k8s, postgres, sklearn, nodejs, and aws."
    skills = extractor.extract_skills(text)

    assert "Machine Learning" in skills
    assert "Deep Learning" in skills
    assert "Kubernetes" in skills
    assert "PostgreSQL" in skills
    assert "Scikit-learn" in skills
    assert "Node.js" in skills
    assert "AWS" in skills


def test_deduplication(extractor):
    """Test that multiple mentions of a skill or its aliases produce a single entry."""
    text = "Machine Learning, ML, Machine Learning, ml expert."
    skills = extractor.extract_skills(text)
    assert skills.count("Machine Learning") == 1
    assert len(skills) == 1


def test_false_positive_java_in_javascript(extractor):
    """Ensure Java is NOT detected solely because JavaScript is present."""
    text = "Experienced in JavaScript, TypeScript, and React development."
    skills = extractor.extract_skills(text)
    assert "JavaScript" in skills
    assert "Java" not in skills, "False positive: Java detected in JavaScript!"


def test_false_positive_c_in_cpp(extractor):
    """Ensure C is NOT detected solely because C++ or C# is present."""
    text = "Software Engineer specialized in C++ and C#."
    skills = extractor.extract_skills(text)
    assert "C++" in skills
    assert "C#" in skills
    assert "C" not in skills, "False positive: C detected in C++ or C#!"


def test_single_letter_r_boundary_and_case(extractor):
    """Ensure R is detected accurately and not matched inside common words."""
    # Matches uppercase standalone R
    text_with_r = "Statistical modeling with R, Python, and SQL."
    skills_r = extractor.extract_skills(text_with_r)
    assert "R" in skills_r

    # Should not match 'R' inside 'Resume', 'Required', or 'User'
    text_without_r = "Reviewing the user resume for required skills."
    skills_no_r = extractor.extract_skills(text_without_r)
    assert "R" not in skills_no_r


def test_match_skills(extractor):
    """Test match_skills separating matched and missing sets."""
    candidate_skills = ["Python", "SQL", "Docker", "Pandas"]
    required_skills = ["Python", "SQL", "Kubernetes", "AWS"]

    matched, missing = extractor.match_skills(candidate_skills, required_skills)
    assert matched == ["Python", "SQL"]
    assert missing == ["AWS", "Kubernetes"]


def test_skill_match_score_scenarios(extractor):
    """Test skill score calculation for 100%, partial, and 0% match."""
    # 100% match
    score, warn = extractor.calculate_skill_match_score(4, 4)
    assert score == 100.0
    assert warn is None

    # 50% match
    score, warn = extractor.calculate_skill_match_score(2, 4)
    assert score == 50.0
    assert warn is None

    # 0% match
    score, warn = extractor.calculate_skill_match_score(0, 5)
    assert score == 0.0
    assert warn is None

    # Capped at 100%
    score, warn = extractor.calculate_skill_match_score(6, 5)
    assert score == 100.0


def test_skill_match_score_no_required_skills(extractor):
    """Test edge case where JD has 0 detected skills."""
    score, warning = extractor.calculate_skill_match_score(0, 0)
    assert score == 0.0
    assert warning is not None
    assert "No required skills detected" in warning


def test_categorize_skills(extractor):
    """Test skill categorization helper."""
    skills = ["Python", "PostgreSQL", "Docker", "Machine Learning"]
    categorized = extractor.categorize_skills(skills)

    assert "Programming Languages" in categorized
    assert "Python" in categorized["Programming Languages"]
    assert "Databases & Query Tools" in categorized
    assert "PostgreSQL" in categorized["Databases & Query Tools"]
    assert "Cloud Platforms & DevOps" in categorized
    assert "Docker" in categorized["Cloud Platforms & DevOps"]
    assert "Machine Learning & Deep Learning" in categorized
    assert "Machine Learning" in categorized["Machine Learning & Deep Learning"]
