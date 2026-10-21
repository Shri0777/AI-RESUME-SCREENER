"""
Skill extraction, synonym normalization, and skill gap calculation.

Uses boundary-aware pattern matching to extract canonical skills from text,
maps acronyms and variants to canonical entries, calculates match/missing sets,
and computes skill match percentages.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from src.utils import (
    ALL_CANONICAL_SKILLS,
    CASE_SENSITIVE_SKILLS,
    SKILL_VARIATIONS,
    SKILLS_DATABASE,
)

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts canonical skills from candidate resumes and job descriptions
    using boundary-aware regex matching and alias normalization.
    """

    def __init__(self) -> None:
        self.category_map: Dict[str, str] = {}
        for category, skills in SKILLS_DATABASE.items():
            for skill in skills:
                self.category_map[skill] = category

        # Build compiled regex matchers for each canonical skill and variant
        self._compiled_patterns = self._build_compiled_patterns()

    def _build_compiled_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Build compiled regular expression patterns sorted by length descending.
        Sorting ensures longer multi-word phrases match accurately.
        """
        # Map of search_term -> canonical_skill
        term_map: Dict[str, str] = {}

        # 1. Canonical skill names
        for skill in ALL_CANONICAL_SKILLS:
            term_map[skill] = skill

        # 2. Aliases and variations
        for alias, canonical in SKILL_VARIATIONS.items():
            term_map[alias] = canonical

        # Sort terms by length descending
        sorted_terms = sorted(term_map.keys(), key=lambda t: len(t), reverse=True)

        compiled: List[Tuple[re.Pattern, str]] = []
        for term in sorted_terms:
            canonical = term_map[term]
            is_case_sensitive = term in CASE_SENSITIVE_SKILLS

            # Safe boundary construction
            escaped = re.escape(term)

            if term in {"C", "C++"}:
                pattern_str = rf"(?<![A-Za-z0-9_+\#]){escaped}(?![A-Za-z0-9_+\#])"
            elif term.startswith("."):
                # e.g. .net
                pattern_str = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
            else:
                pattern_str = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"

            flags = 0 if is_case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern_str, flags)
                compiled.append((regex, canonical))
            except re.error as e:
                logger.error(f"Regex compilation error for term '{term}': {e}")

        return compiled

    def extract_skills(self, text: Optional[str]) -> List[str]:
        """
        Extract deduplicated, sorted list of canonical skills from text.

        Args:
            text: Raw or preprocessed document text.

        Returns:
            Sorted list of canonical skill names detected in text.
        """
        if not text or not isinstance(text, str):
            return []

        detected: Set[str] = set()

        for pattern, canonical in self._compiled_patterns:
            if pattern.search(text):
                detected.add(canonical)

        return sorted(list(detected))

    def match_skills(
        self, candidate_skills: List[str], required_skills: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Compare candidate skills against required skills.

        Args:
            candidate_skills: List of skills detected in candidate resume.
            required_skills: List of required skills from job description.

        Returns:
            Tuple of (matched_skills, missing_skills), both sorted.
        """
        cand_set = set(candidate_skills)
        req_set = set(required_skills)

        matched = sorted(list(cand_set & req_set))
        missing = sorted(list(req_set - cand_set))

        return matched, missing

    def calculate_skill_match_score(
        self, matched_count: int, required_count: int
    ) -> Tuple[float, Optional[str]]:
        """
        Calculate skill match score as (matched / required) * 100 capped at 100.

        Args:
            matched_count: Number of matched required skills.
            required_count: Number of required skills.

        Returns:
            Tuple of (score_in_percentage, warning_or_none).
        """
        if required_count <= 0:
            warning = (
                "No required skills detected in job description. "
                "Skill match score is set to 0.0% (neutral policy)."
            )
            return 0.0, warning

        ratio = (matched_count / required_count) * 100.0
        score = min(100.0, max(0.0, round(ratio, 2)))
        return score, None

    def calculate_hierarchical_skill_score(
        self,
        matched_req_count: int,
        total_req_count: int,
        matched_pref_count: int = 0,
        total_pref_count: int = 0,
        req_weight: float = 0.80,
        pref_weight: float = 0.20,
    ) -> Tuple[float, float, float, Optional[str]]:
        """
        Calculate hierarchical skill coverage with explicit weighting for mandatory
        vs. preferred skills. Prevents preferred skills from dominating core requirements.

        Args:
            matched_req_count: Matched mandatory required skills.
            total_req_count: Total mandatory required skills.
            matched_pref_count: Matched preferred/nice-to-have skills.
            total_pref_count: Total preferred/nice-to-have skills.
            req_weight: Weight given to mandatory skills (default 0.80).
            pref_weight: Weight given to preferred skills (default 0.20).

        Returns:
            Tuple of (combined_score, required_score, preferred_score, warning_or_none).
        """
        if total_req_count <= 0 and total_pref_count <= 0:
            return (
                0.0,
                0.0,
                0.0,
                "No skills detected in job description. Skill score set to 0.0% (neutral policy).",
            )

        # Required score
        if total_req_count > 0:
            req_score = min(100.0, max(0.0, round((matched_req_count / total_req_count) * 100.0, 2)))
            warn = None
        else:
            req_score = 0.0
            warn = "No mandatory required skills detected; evaluated on preferred skills only."

        # Preferred score
        if total_pref_count > 0:
            pref_score = min(100.0, max(0.0, round((matched_pref_count / total_pref_count) * 100.0, 2)))
            # Combine weighted
            combined = round((req_weight * req_score) + (pref_weight * pref_score), 2)
        else:
            pref_score = 0.0
            combined = req_score

        combined = min(100.0, max(0.0, combined))
        return combined, req_score, pref_score, warn

    def extract_skills_from_jd_sections(
        self,
        job_description: Optional[str],
        explicit_required: Optional[List[str]] = None,
        explicit_preferred: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], bool]:
        """
        Extract required and preferred skills from explicit user inputs or
        heuristically from job description section headers.

        Args:
            job_description: Raw job description text.
            explicit_required: Optional list of explicit required skills.
            explicit_preferred: Optional list of explicit preferred skills.

        Returns:
            Tuple of (required_skills, preferred_skills, is_inferred).
        """
        # If user explicitly provided lists, canonicalize and return
        if explicit_required is not None or explicit_preferred is not None:
            req = []
            if explicit_required:
                for s in explicit_required:
                    c = self.extract_skills(s)
                    req.extend(c)
            pref = []
            if explicit_preferred:
                for s in explicit_preferred:
                    c = self.extract_skills(s)
                    pref.extend(c)
            # Ensure disjoint: if a skill is in both, required takes precedence
            req_sorted = sorted(list(set(req)))
            pref_sorted = sorted(list(set(pref) - set(req_sorted)))
            return req_sorted, pref_sorted, False

        if not job_description or not job_description.strip():
            return [], [], True

        # Heuristic section splitting
        lines = job_description.splitlines()
        req_lines: List[str] = []
        pref_lines: List[str] = []
        other_lines: List[str] = []

        current_section = "other"  # 'required', 'preferred', 'other'

        req_pattern = re.compile(
            r"(?i)\b(?:required|requirements|mandatory|must\s*have|qualifications|basic\s*qualifications)\b"
        )
        pref_pattern = re.compile(
            r"(?i)\b(?:preferred|nice\s*to\s*have|desired|bonus|plus|good\s*to\s*have|additional\s*skills)\b"
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line looks like a section header
            if len(line_str) < 60:
                if pref_pattern.search(line_str):
                    current_section = "preferred"
                    continue
                elif req_pattern.search(line_str):
                    current_section = "required"
                    continue

            if current_section == "required":
                req_lines.append(line_str)
            elif current_section == "preferred":
                pref_lines.append(line_str)
            else:
                other_lines.append(line_str)

        req_skills = self.extract_skills("\n".join(req_lines))
        pref_skills = self.extract_skills("\n".join(pref_lines))

        # If no preferred section was detected, check general text
        if not pref_skills and not req_skills:
            all_skills = self.extract_skills(job_description)
            return all_skills, [], True

        # If required section had nothing but other sections had skills, merge other into required
        if not req_skills and other_lines:
            other_skills = self.extract_skills("\n".join(other_lines))
            req_skills = other_skills

        # Ensure disjoint
        pref_skills = sorted(list(set(pref_skills) - set(req_skills)))

        return req_skills, pref_skills, True

    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Organize a list of skills by their designated categories.

        Args:
            skills: List of canonical skill names.

        Returns:
            Dictionary mapping category names to lists of detected skills.
        """
        result: Dict[str, List[str]] = {cat: [] for cat in SKILLS_DATABASE.keys()}
        result["Other"] = []

        for skill in skills:
            cat = self.category_map.get(skill, "Other")
            result[cat].append(skill)

        # Remove categories with no skills for cleaner reporting
        return {cat: sorted(skill_list) for cat, skill_list in result.items() if skill_list}
