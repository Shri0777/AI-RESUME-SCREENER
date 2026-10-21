"""
PDF parsing and candidate metadata extraction module.

Extracts text from PDF documents using pdfplumber, extracts candidate metadata
(name, email, phone) heuristically, and handles corrupt or image-only PDFs
gracefully without crashing batch processing.
"""

import io
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pdfplumber

logger = logging.getLogger(__name__)

# Heuristic exclusion terms for candidate name detection
RESUME_HEADER_TERMS = {
    "resume",
    "curriculum vitae",
    "curriculum-vitae",
    "cv",
    "biodata",
    "bio-data",
    "profile",
    "summary",
    "professional summary",
    "executive summary",
    "contact",
    "contact information",
    "personal details",
    "objective",
    "career objective",
    "experience",
    "work experience",
    "education",
    "skills",
    "technical skills",
    "portfolio",
    "page",
}


@dataclass
class ParsedCandidateRecord:
    """Structured record representing a parsed PDF document."""

    filename: str
    candidate_name: str
    is_name_inferred: bool
    email: str
    phone: str
    extracted_text: str
    char_count: int
    word_count: int
    parse_status: str  # "success", "warning", "failed"
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


class PDFParser:
    """
    Parser for extracting text and metadata from PDF resumes.
    Supports single or batch parsing of file paths and file-like objects (e.g. Streamlit UploadedFile).
    """

    def __init__(self) -> None:
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        self.phone_pattern = re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        )

    def parse_file(
        self, file_input: Union[str, Path, Any], filename: Optional[str] = None
    ) -> ParsedCandidateRecord:
        """
        Parse a single PDF file input (path or file-like object).

        Args:
            file_input: File path, BytesIO, or Streamlit UploadedFile object.
            filename: Optional filename override if file_input is a buffer.

        Returns:
            ParsedCandidateRecord containing extracted text and metadata.
        """
        resolved_filename = self._resolve_filename(file_input, filename)
        warnings: List[str] = []

        try:
            stream = self._get_stream(file_input)
            extracted_text = self._extract_text_from_stream(stream)
        except Exception as e:
            logger.warning(f"Error parsing PDF file {resolved_filename}: {e}")
            return ParsedCandidateRecord(
                filename=resolved_filename,
                candidate_name=self._fallback_name_from_filename(resolved_filename),
                is_name_inferred=True,
                email="N/A",
                phone="N/A",
                extracted_text="",
                char_count=0,
                word_count=0,
                parse_status="failed",
                warnings=[f"Failed to read PDF document: {str(e)}"],
            )

        char_count = len(extracted_text)
        word_count = len(extracted_text.split())

        if not extracted_text.strip():
            warnings.append(
                "No selectable text found. Document may be scanned, image-only, or empty."
            )
            parse_status = "warning"
        else:
            parse_status = "success"

        candidate_name, is_inferred = self.extract_candidate_name(
            extracted_text, resolved_filename
        )
        email = self.extract_email(extracted_text)
        phone = self.extract_phone(extracted_text)

        return ParsedCandidateRecord(
            filename=resolved_filename,
            candidate_name=candidate_name,
            is_name_inferred=is_inferred,
            email=email,
            phone=phone,
            extracted_text=extracted_text,
            char_count=char_count,
            word_count=word_count,
            parse_status=parse_status,
            warnings=warnings,
        )

    def parse_multiple(
        self, files: List[Union[str, Path, Any]]
    ) -> List[ParsedCandidateRecord]:
        """
        Parse a batch of PDF documents. Processing continues even if individual files fail.

        Args:
            files: List of file paths or file-like objects.

        Returns:
            List of ParsedCandidateRecord objects for all submitted files.
        """
        records: List[ParsedCandidateRecord] = []
        for f in files:
            record = self.parse_file(f)
            records.append(record)
        return records

    def _resolve_filename(
        self, file_input: Union[str, Path, Any], filename: Optional[str]
    ) -> str:
        """Resolve a human-readable filename from various file input types."""
        if filename:
            return os.path.basename(filename)
        if isinstance(file_input, (str, Path)):
            return os.path.basename(str(file_input))
        if hasattr(file_input, "name"):
            return os.path.basename(str(file_input.name))
        return "uploaded_resume.pdf"

    def _get_stream(self, file_input: Union[str, Path, Any]) -> io.BytesIO:
        """Convert input to an in-memory BytesIO stream for safe repeated reads."""
        if isinstance(file_input, (str, Path)):
            with open(file_input, "rb") as f:
                return io.BytesIO(f.read())
        if hasattr(file_input, "getvalue"):
            return io.BytesIO(file_input.getvalue())
        if hasattr(file_input, "read"):
            current_pos = file_input.tell() if hasattr(file_input, "tell") else 0
            data = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(current_pos)
            return io.BytesIO(data)
        raise ValueError("Unsupported file input format.")

    def _extract_text_from_stream(self, stream: io.BytesIO) -> str:
        """Read all pages from a PDF stream using pdfplumber."""
        text_pages: List[str] = []
        with pdfplumber.open(stream) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text.strip())
                except Exception as e:
                    logger.warning(f"Failed extracting text from page {page_idx + 1}: {e}")
        return "\n\n".join(text_pages).strip()

    def extract_candidate_name(
        self, text: str, filename: str
    ) -> tuple[str, bool]:
        """
        Extract candidate name heuristically from the beginning of resume text.
        Falls back to filename stem if no high-confidence name line is found.

        Args:
            text: Extracted raw text.
            filename: Document filename.

        Returns:
            Tuple of (candidate_name, is_inferred).
        """
        if not text or not text.strip():
            return self._fallback_name_from_filename(filename), True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Examine the top 8 non-empty lines
        candidate_lines = lines[:8]

        for line in candidate_lines:
            lower = line.lower()

            # Skip common section headings or label terms
            if lower in RESUME_HEADER_TERMS:
                continue
            if any(term in lower for term in ["curriculum vitae", "resume of", "portfolio of"]):
                continue

            # Skip lines with contact information or links
            if "@" in line or "http" in lower or "www." in lower or "linkedin" in lower or "github" in lower:
                continue

            # Skip lines that contain phone numbers or excessive digits
            digit_count = sum(c.isdigit() for c in line)
            if digit_count > 2:
                continue

            # Check if line looks like a valid name: 2-4 words, primarily alphabetic
            words = line.split()
            if 1 <= len(words) <= 4:
                # Remove punctuation from words (e.g. initials like 'J.')
                cleaned_words = [re.sub(r"[^\w\s-]", "", w) for w in words]
                if all(w.isalpha() or "-" in w for w in cleaned_words if w):
                    # Filter out short abbreviations or stray symbols
                    total_len = sum(len(w) for w in cleaned_words)
                    if 3 <= total_len <= 40:
                        candidate_name = " ".join(cleaned_words).title()
                        return candidate_name, False

        # If no reliable line matched, infer name from filename stem
        return self._fallback_name_from_filename(filename), True

    def _fallback_name_from_filename(self, filename: str) -> str:
        """Derive a plausible name from the filename stem."""
        stem = Path(filename).stem
        cleaned = re.sub(r"[_\-]+", " ", stem).strip()
        # Remove common prefixes like 'resume', 'cv', 'sample'
        tokens = [
            t for t in cleaned.split()
            if t.lower() not in {"resume", "cv", "sample", "draft", "v1", "v2", "final"}
        ]
        if tokens:
            return " ".join(tokens).title()
        return "Candidate"

    def extract_email(self, text: str) -> str:
        """Extract first unique email address from text, or 'N/A' if none found."""
        if not text:
            return "N/A"
        matches = self.email_pattern.findall(text)
        if matches:
            # Return first email normalized to lowercase
            return matches[0].strip().lower()
        return "N/A"

    def extract_phone(self, text: str) -> str:
        """Extract first normalized phone candidate from text, or 'N/A' if none found."""
        if not text:
            return "N/A"
        matches = self.phone_pattern.findall(text)
        for m in matches:
            cleaned = m.strip()
            # Require at least 7 digits to prevent matching small year or date ranges
            digits = re.sub(r"\D", "", cleaned)
            if 7 <= len(digits) <= 15:
                return cleaned
        return "N/A"
