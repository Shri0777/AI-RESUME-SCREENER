"""
Central configuration settings and parameters for the AI Resume Screener.

Contains default scoring weights, strategy definitions, batch sizes,
model identifiers, and paths to ensure settings are not scattered across modules.
"""

import os
from pathlib import Path
from typing import Dict, List

# Base directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EVALUATION_DIR: Path = REPORTS_DIR / "evaluation"
LOCAL_AUDIT_DIR: Path = REPORTS_DIR / "local_audit"

# Default path to the 2,484 Kaggle resume dataset
DEFAULT_DATASET_DIR: Path = Path(
    os.environ.get("RESUME_DATASET_DIR", "/home/shri/Downloads/resume-dataset")
)

# Ranking Strategies
STRATEGY_BASELINE = "baseline"
STRATEGY_ENHANCED = "enhanced"

# Baseline Scoring Weights (Must sum to 1.0)
DEFAULT_BASELINE_SIMILARITY_WEIGHT: float = 0.70
DEFAULT_BASELINE_SKILL_WEIGHT: float = 0.30

# Enhanced Hybrid Scoring Weights (Must sum to 1.0)
DEFAULT_ENHANCED_LEXICAL_WEIGHT: float = 0.40
DEFAULT_ENHANCED_SEMANTIC_WEIGHT: float = 0.35
DEFAULT_ENHANCED_SKILL_WEIGHT: float = 0.25

# Hierarchical Skill Match Weights (within skill match component)
DEFAULT_REQUIRED_SKILL_WEIGHT: float = 0.80
DEFAULT_PREFERRED_SKILL_WEIGHT: float = 0.20

# Semantic Similarity Settings
DEFAULT_SEMANTIC_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_EMBEDDING_BATCH_SIZE: int = 32
ENABLE_SEMANTIC_CACHE: bool = True

# Batch Processing & Ingestion Limits
DEFAULT_BATCH_SIZE: int = 50
MAX_RECOMMENDED_BATCH_SIZE: int = 1000

# Extended DataFrame Column Specification
ENHANCED_DATAFRAME_COLUMNS: List[str] = [
    "rank",
    "candidate_name",
    "filename",
    "email",
    "phone",
    "strategy_used",
    "similarity_score",
    "semantic_similarity_score",
    "skill_match_score",
    "required_skill_match_score",
    "preferred_skill_match_score",
    "final_score",
    "matched_skills",
    "missing_skills",
    "matched_preferred_skills",
    "missing_preferred_skills",
    "detected_skills",
    "required_skills",
    "preferred_skills",
    "parse_status",
    "warnings",
]
