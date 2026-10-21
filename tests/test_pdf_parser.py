"""
Unit tests for PDFParser module.
"""

import io
from pathlib import Path
import pytest
from src.pdf_parser import PDFParser, ParsedCandidateRecord


@pytest.fixture
def parser():
    return PDFParser()


def test_parse_valid_sample_pdf(parser):
    """Test parsing one of the generated synthetic resumes."""
    sample_path = Path("data/resumes/sample_alice_johnson.pdf")
    if not sample_path.exists():
        pytest.skip("Sample resume PDF not generated yet.")

    record = parser.parse_file(sample_path)
    assert isinstance(record, ParsedCandidateRecord)
    assert record.parse_status == "success"
    assert record.candidate_name == "Alice Johnson"
    assert "alice.johnson@example.com" in record.email
    assert record.phone != "N/A"
    assert record.word_count > 50
    assert "Machine Learning" in record.extracted_text


def test_parse_corrupt_pdf_gracefully(parser, tmp_path):
    """Test parser handles completely corrupt/non-PDF files without crashing."""
    corrupt_file = tmp_path / "corrupt_resume.pdf"
    corrupt_file.write_bytes(b"This is not a valid PDF file content at all.")

    record = parser.parse_file(corrupt_file)
    assert record.parse_status == "failed"
    assert record.word_count == 0
    assert record.is_name_inferred is True
    assert record.candidate_name == "Corrupt"
    assert len(record.warnings) > 0
    assert "Failed to read PDF" in record.warnings[0]


def test_parse_empty_stream(parser):
    """Test parser handles zero-byte buffer."""
    empty_stream = io.BytesIO(b"")
    record = parser.parse_file(empty_stream, filename="empty_upload.pdf")
    assert record.parse_status == "failed"
    assert record.candidate_name == "Empty Upload"


def test_fallback_name_inference(parser):
    """Test candidate name heuristic and filename fallback."""
    # Text with explicit name
    text_with_name = "Jane Doe\nSoftware Engineer\njane@example.com\n"
    name, inferred = parser.extract_candidate_name(text_with_name, "jane_doe_cv.pdf")
    assert name == "Jane Doe"
    assert inferred is False

    # Text with only generic headings
    text_headings_only = "CURRICULUM VITAE\nRESUME\nCONTACT INFORMATION\n"
    name_inf, inferred2 = parser.extract_candidate_name(text_headings_only, "robert_smith_2024.pdf")
    assert name_inf == "Robert Smith 2024"
    assert inferred2 is True


def test_batch_parsing_with_mixed_valid_and_corrupt(parser, tmp_path):
    """Test parse_multiple continues batch processing even if one file is corrupt."""
    corrupt_file = tmp_path / "broken.pdf"
    corrupt_file.write_bytes(b"Corrupted bytes")

    sample_path = Path("data/resumes/sample_bob_smith.pdf")
    files_to_parse = [corrupt_file]
    if sample_path.exists():
        files_to_parse.append(sample_path)

    records = parser.parse_multiple(files_to_parse)
    assert len(records) == len(files_to_parse)
    assert records[0].parse_status == "failed"

    if len(files_to_parse) > 1:
        assert records[1].parse_status == "success"
        assert records[1].candidate_name == "Bob Smith"


def test_email_and_phone_extraction(parser):
    """Test regex extraction of email and phone."""
    text = (
        "Contact: John Doe\n"
        "Email: john.doe_123@sub.domain.co.uk\n"
        "Phone: +1 (555) 234-5678\n"
        "Work: 555.876.5432\n"
    )
    email = parser.extract_email(text)
    assert email == "john.doe_123@sub.domain.co.uk"

    phone = parser.extract_phone(text)
    assert "555" in phone
