"""
Candidate ranking and scoring engine.

Computes hybrid candidate match scores using lexical TF-IDF cosine similarity,
optional semantic dense embeddings (SentenceTransformers), and hierarchical
skill coverage (mandatory vs. preferred).

Strategies:
    - Baseline: 0.70 * TF-IDF Similarity + 0.30 * Skill Match Score
    - Enhanced: w_lexical * TF-IDF + w_semantic * Semantic + w_skill * Skill Match
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DEFAULT_BASELINE_SIMILARITY_WEIGHT,
    DEFAULT_BASELINE_SKILL_WEIGHT,
    DEFAULT_ENHANCED_LEXICAL_WEIGHT,
    DEFAULT_ENHANCED_SEMANTIC_WEIGHT,
    DEFAULT_ENHANCED_SKILL_WEIGHT,
    DEFAULT_PREFERRED_SKILL_WEIGHT,
    DEFAULT_REQUIRED_SKILL_WEIGHT,
    ENHANCED_DATAFRAME_COLUMNS,
    STRATEGY_BASELINE,
    STRATEGY_ENHANCED,
)
from src.pdf_parser import ParsedCandidateRecord
from src.semantic_matcher import SemanticMatcher, strip_identifiers_for_embedding
from src.skill_extractor import SkillExtractor
from src.text_cleaner import TextCleaner
from src.utils import DATAFRAME_COLUMNS

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Ranks candidate resumes against a job description using either the baseline
    lexical TF-IDF + rule-based skill model or an enhanced hybrid semantic model.
    """

    def __init__(
        self,
        strategy: str = STRATEGY_BASELINE,
        similarity_weight: float = DEFAULT_BASELINE_SIMILARITY_WEIGHT,
        skill_weight: float = DEFAULT_BASELINE_SKILL_WEIGHT,
        lexical_weight: float = DEFAULT_ENHANCED_LEXICAL_WEIGHT,
        semantic_weight: float = DEFAULT_ENHANCED_SEMANTIC_WEIGHT,
        enhanced_skill_weight: float = DEFAULT_ENHANCED_SKILL_WEIGHT,
        required_skill_weight: float = DEFAULT_REQUIRED_SKILL_WEIGHT,
        preferred_skill_weight: float = DEFAULT_PREFERRED_SKILL_WEIGHT,
    ) -> None:
        self.strategy = strategy

        # Baseline weights
        self._validate_weights(similarity_weight, skill_weight, "Baseline (sim + skill)")
        self.similarity_weight = similarity_weight
        self.skill_weight = skill_weight

        # Enhanced hybrid weights
        self._validate_enhanced_weights(
            lexical_weight, semantic_weight, enhanced_skill_weight
        )
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.enhanced_skill_weight = enhanced_skill_weight

        # Hierarchical skill weights
        self.required_skill_weight = required_skill_weight
        self.preferred_skill_weight = preferred_skill_weight

        self.text_cleaner = TextCleaner()
        self.skill_extractor = SkillExtractor()
        self.semantic_matcher = SemanticMatcher()

    @staticmethod
    def _validate_weights(w1: float, w2: float, label: str) -> None:
        """Validate that two weights are in [0, 1] and sum to 1.0."""
        if not (0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0):
            raise ValueError(f"{label} weights must be between 0.0 and 1.0. Given: {w1}, {w2}")
        if abs((w1 + w2) - 1.0) > 1e-4:
            raise ValueError(f"{label} weights must sum to 1.0. Given: {w1} + {w2} = {w1 + w2}")

    @staticmethod
    def _validate_enhanced_weights(w_lex: float, w_sem: float, w_skl: float) -> None:
        """Validate that enhanced three-part weights are in [0, 1] and sum to 1.0."""
        for w, name in [(w_lex, "lexical"), (w_sem, "semantic"), (w_skl, "skill")]:
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"Enhanced {name} weight must be between 0.0 and 1.0. Given: {w}")
        total = w_lex + w_sem + w_skl
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Enhanced weights must sum to 1.0. Given: {w_lex} + {w_sem} + {w_skl} = {total}"
            )

    def rank_candidates(
        self,
        job_description: str,
        candidates: List[Union[ParsedCandidateRecord, Dict[str, Any]]],
        strategy: Optional[str] = None,
        explicit_required_skills: Optional[List[str]] = None,
        explicit_preferred_skills: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process a batch of candidates against a job description and generate
        a ranked DataFrame.

        Args:
            job_description: Raw job description text.
            candidates: List of ParsedCandidateRecord or dict records.
            strategy: Optional strategy override ('baseline' or 'enhanced').
            explicit_required_skills: Optional list of mandatory required skills.
            explicit_preferred_skills: Optional list of preferred/nice-to-have skills.

        Returns:
            Pandas DataFrame sorted by 'final_score' descending with standard and enhanced columns.
        """
        active_strategy = strategy or self.strategy

        cols_to_return = (
            DATAFRAME_COLUMNS
            if active_strategy == STRATEGY_BASELINE
            else ENHANCED_DATAFRAME_COLUMNS
        )

        if not candidates:
            return pd.DataFrame(columns=cols_to_return)

        normalized_candidates: List[Dict[str, Any]] = []
        for cand in candidates:
            if isinstance(cand, ParsedCandidateRecord):
                normalized_candidates.append(cand.to_dict())
            elif isinstance(cand, dict):
                normalized_candidates.append(cand)
            else:
                logger.warning(f"Skipping candidate record with invalid type: {type(cand)}")

        if not normalized_candidates:
            return pd.DataFrame(columns=cols_to_return)

        jd_is_empty = not job_description or not job_description.strip()

        # Skill extraction from JD
        required_skills, preferred_skills, _ = (
            self.skill_extractor.extract_skills_from_jd_sections(
                job_description,
                explicit_required=explicit_required_skills,
                explicit_preferred=explicit_preferred_skills,
            )
        )

        # Clean candidate texts without personal identifiers
        sanitized_resumes: List[str] = []
        cleaned_resumes: List[str] = []
        for c in normalized_candidates:
            raw = c.get("extracted_text", "")
            sanitized = strip_identifiers_for_embedding(
                raw,
                candidate_name=c.get("candidate_name"),
                email=c.get("email"),
                phone=c.get("phone"),
            )
            sanitized_resumes.append(sanitized)
            cleaned_resumes.append(self.text_cleaner.clean(sanitized))

        # 1. Compute TF-IDF Lexical Similarities
        jd_clean = self.text_cleaner.clean(job_description)
        use_enhanced_tfidf = active_strategy == STRATEGY_ENHANCED
        tfidf_similarities = self._compute_tfidf_similarities(
            jd_clean, cleaned_resumes, jd_is_empty, enhanced=use_enhanced_tfidf
        )

        # 2. Compute Semantic Similarities (if enhanced strategy requested)
        semantic_similarities: List[float] = [0.0] * len(normalized_candidates)
        semantic_fallback_used = False
        semantic_warning = None

        if active_strategy == STRATEGY_ENHANCED and not jd_is_empty:
            (
                semantic_similarities,
                semantic_fallback_used,
                semantic_warning,
            ) = self.semantic_matcher.compute_batch_similarities(
                job_description, sanitized_resumes
            )

        strategy_label = (
            STRATEGY_BASELINE
            if active_strategy == STRATEGY_BASELINE
            else (
                "enhanced (fallback: lexical-only)"
                if semantic_fallback_used
                else STRATEGY_ENHANCED
            )
        )

        results: List[Dict[str, Any]] = []

        for idx, cand in enumerate(normalized_candidates):
            raw_text = cand.get("extracted_text", "")
            cand_warnings = list(cand.get("warnings", []))

            if semantic_warning and active_strategy == STRATEGY_ENHANCED:
                cand_warnings.append(semantic_warning)

            # Candidate skill extraction
            detected_skills = self.skill_extractor.extract_skills(raw_text)

            # Match required and preferred
            matched_req, missing_req = self.skill_extractor.match_skills(
                detected_skills, required_skills
            )
            matched_pref, missing_pref = self.skill_extractor.match_skills(
                detected_skills, preferred_skills
            )

            # Skill score calculation
            if active_strategy == STRATEGY_BASELINE or not preferred_skills:
                # Classic baseline score policy
                skill_score, skill_warn = (
                    self.skill_extractor.calculate_skill_match_score(
                        len(matched_req), len(required_skills)
                    )
                )
                req_score = skill_score
                pref_score = 0.0
            else:
                # Hierarchical score
                (
                    skill_score,
                    req_score,
                    pref_score,
                    skill_warn,
                ) = self.skill_extractor.calculate_hierarchical_skill_score(
                    len(matched_req),
                    len(required_skills),
                    len(matched_pref),
                    len(preferred_skills),
                    req_weight=self.required_skill_weight,
                    pref_weight=self.preferred_skill_weight,
                )

            if skill_warn:
                cand_warnings.append(skill_warn)

            sim_score = tfidf_similarities[idx]
            sem_score = semantic_similarities[idx]

            if not raw_text.strip():
                sim_score = 0.0
                sem_score = 0.0
                cand_warnings.append("Document has empty or unreadable text; similarity set to 0.0%.")

            if jd_is_empty:
                sim_score = 0.0
                sem_score = 0.0
                cand_warnings.append("Job description is empty; similarity set to 0.0%.")

            # Final Score Calculation
            if active_strategy == STRATEGY_BASELINE:
                final_score = (self.similarity_weight * sim_score) + (
                    self.skill_weight * skill_score
                )
            else:
                if semantic_fallback_used:
                    # Rebalance lexical and skill weights to sum to 1.0
                    sub_total = self.lexical_weight + self.enhanced_skill_weight
                    adj_lex = self.lexical_weight / sub_total if sub_total > 0 else 0.70
                    adj_skl = self.enhanced_skill_weight / sub_total if sub_total > 0 else 0.30
                    final_score = (adj_lex * sim_score) + (adj_skl * skill_score)
                else:
                    final_score = (
                        (self.lexical_weight * sim_score)
                        + (self.semantic_weight * sem_score)
                        + (self.enhanced_skill_weight * skill_score)
                    )

            final_score = min(100.0, max(0.0, round(final_score, 2)))

            record = {
                "rank": 0,
                "candidate_name": cand.get("candidate_name", "Unknown Candidate"),
                "filename": cand.get("filename", "unknown.pdf"),
                "email": cand.get("email", "N/A"),
                "phone": cand.get("phone", "N/A"),
                "strategy_used": strategy_label,
                "similarity_score": round(sim_score, 2),
                "semantic_similarity_score": round(sem_score, 2) if active_strategy == STRATEGY_ENHANCED else 0.0,
                "skill_match_score": round(skill_score, 2),
                "required_skill_match_score": round(req_score, 2),
                "preferred_skill_match_score": round(pref_score, 2),
                "final_score": final_score,
                "matched_skills": matched_req,
                "missing_skills": missing_req,
                "matched_preferred_skills": matched_pref,
                "missing_preferred_skills": missing_pref,
                "detected_skills": detected_skills,
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "parse_status": cand.get("parse_status", "success"),
                "warnings": "; ".join(cand_warnings) if cand_warnings else "None",
            }
            results.append(record)

        df = pd.DataFrame(results)
        df = df.sort_values(
            by=["final_score", "skill_match_score", "similarity_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        df["rank"] = df.index + 1
        return df[cols_to_return]

    def _compute_tfidf_similarities(
        self,
        jd_clean: str,
        cleaned_resumes: List[str],
        jd_is_empty: bool,
        enhanced: bool = False,
    ) -> List[float]:
        """Compute TF-IDF cosine similarities between job description and resume documents."""
        num_resumes = len(cleaned_resumes)
        if jd_is_empty or not jd_clean.strip():
            return [0.0] * num_resumes

        if not any(res.strip() for res in cleaned_resumes):
            return [0.0] * num_resumes

        all_docs = [jd_clean] + cleaned_resumes

        try:
            if enhanced:
                # Enhanced TF-IDF with unigrams + bigrams and sublinear term frequency
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\-\+\#\.]+\b",
                )
            else:
                # Baseline standard unigram vectorizer
                vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

            tfidf_matrix = vectorizer.fit_transform(all_docs)
            jd_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]

            sim_matrix = cosine_similarity(jd_vector, resume_vectors)
            sim_values = sim_matrix[0]

            percentages: List[float] = []
            for val in sim_values:
                if np.isnan(val) or np.isinf(val) or val < 0.0:
                    percentages.append(0.0)
                else:
                    pct = min(100.0, max(0.0, float(val) * 100.0))
                    percentages.append(round(pct, 2))
            return percentages

        except ValueError as e:
            logger.warning(f"TfidfVectorizer empty vocabulary: {e}")
            return [0.0] * num_resumes
        except Exception as e:
            logger.error(f"Error in TF-IDF similarity calculation: {e}")
            return [0.0] * num_resumes
