"""
Unit tests for ReportGenerator module.
"""

import io
from pathlib import Path
import openpyxl
import pandas as pd
import pytest
from src.report_generator import ReportGenerator, prepare_export_dataframe


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate_name": "Alice Johnson",
                "filename": "alice.pdf",
                "email": "alice@example.com",
                "phone": "(555) 019-2831",
                "similarity_score": 62.50,
                "skill_match_score": 80.00,
                "final_score": 67.75,
                "matched_skills": ["Python", "SQL", "Machine Learning"],
                "missing_skills": ["AWS"],
                "detected_skills": ["Python", "SQL", "Machine Learning", "Docker"],
                "required_skills": ["Python", "SQL", "Machine Learning", "AWS"],
                "parse_status": "success",
                "warnings": "None",
            },
            {
                "rank": 2,
                "candidate_name": "Bob Smith",
                "filename": "bob.pdf",
                "email": "bob@example.com",
                "phone": "(555) 014-9923",
                "similarity_score": 50.00,
                "skill_match_score": 50.00,
                "final_score": 50.00,
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["Machine Learning", "AWS"],
                "detected_skills": ["Python", "SQL"],
                "required_skills": ["Python", "SQL", "Machine Learning", "AWS"],
                "parse_status": "success",
                "warnings": "None",
            },
        ]
    )


@pytest.fixture
def report_generator():
    return ReportGenerator()


def test_prepare_export_dataframe(sample_dataframe):
    """Test dataframe formatting converting lists to comma-separated strings."""
    formatted = prepare_export_dataframe(sample_dataframe)
    assert isinstance(formatted["Matched Required Skills"].iloc[0], str)
    assert "Python, SQL, Machine Learning" == formatted["Matched Required Skills"].iloc[0]
    assert "Rank" in formatted.columns
    assert "Final Score (%)" in formatted.columns


def test_export_csv(report_generator, sample_dataframe):
    """Test CSV byte generation."""
    csv_bytes = report_generator.export_csv(sample_dataframe)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0
    decoded = csv_bytes.decode("utf-8")
    assert "Alice Johnson" in decoded
    assert "Bob Smith" in decoded
    assert "Final Score (%)" in decoded


def test_export_excel(report_generator, sample_dataframe):
    """Test Excel (.xlsx) generation and workbook structure."""
    excel_bytes = report_generator.export_excel(sample_dataframe)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert "Candidate Rankings" in wb.sheetnames
    sheet = wb["Candidate Rankings"]
    assert sheet.cell(row=1, column=1).value == "Rank"
    assert sheet.cell(row=2, column=2).value == "Alice Johnson"


def test_export_pdf(report_generator, sample_dataframe):
    """Test PDF generation with ReportLab."""
    pdf_bytes = report_generator.export_pdf(sample_dataframe, job_title="Senior Data Scientist")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # PDF magic header
    assert pdf_bytes.startswith(b"%PDF-")


def test_save_report_to_disk(report_generator, sample_dataframe, tmp_path):
    """Test saving reports to a designated disk directory."""
    csv_bytes = report_generator.export_csv(sample_dataframe)
    success, saved_path = report_generator.save_report(
        csv_bytes, "test_ranking.csv", custom_dir=str(tmp_path)
    )
    assert success is True
    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0


def test_empty_exports(report_generator):
    """Test exports return empty bytes when input is empty."""
    empty_df = pd.DataFrame()
    assert report_generator.export_csv(empty_df) == b""
    assert report_generator.export_excel(empty_df) == b""
    assert report_generator.export_pdf(empty_df) == b""

    success, msg = report_generator.save_report(b"", "empty.csv")
    assert success is False
    assert "empty report" in msg.lower()
