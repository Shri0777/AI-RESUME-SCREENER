#!/usr/bin/env python3
"""
Audit script for the 2,484-resume PDF dataset.

Safely audits PDF parse success, text length distributions, and category coverage
without logging or persisting candidate names, contact details, or raw resume text.
Outputs privacy-minimized summary statistics to reports/local_audit/.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to sys.path so src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_manager import CacheManager
from src.config import DEFAULT_DATASET_DIR, LOCAL_AUDIT_DIR
from src.pdf_parser import PDFParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dataset_audit")


def find_dataset_pdfs(dataset_dir: Path, sample_per_category: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Locate PDF files under dataset_dir/data/data/<CATEGORY>/*.pdf.
    Optionally samples up to sample_per_category PDFs from each category.
    """
    pdf_entries: List[Dict[str, Any]] = []

    candidate_paths = [
        dataset_dir / "data" / "data",
        dataset_dir / "data",
        dataset_dir,
    ]

    base_search = None
    for p in candidate_paths:
        if p.exists() and any(p.glob("*/*.pdf")):
            base_search = p
            break

    if not base_search:
        # Fallback recursive search
        for pdf_path in sorted(dataset_dir.rglob("*.pdf")):
            category = pdf_path.parent.name
            pdf_entries.append({"path": pdf_path, "category": category})
        if sample_per_category:
            by_cat = defaultdict(list)
            for entry in pdf_entries:
                if len(by_cat[entry["category"]]) < sample_per_category:
                    by_cat[entry["category"]].append(entry)
            pdf_entries = [e for group in by_cat.values() for e in group]
        return pdf_entries

    for cat_dir in sorted(base_search.iterdir()):
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            category_name = cat_dir.name
            cat_pdfs = sorted(list(cat_dir.glob("*.pdf")))
            if sample_per_category and sample_per_category > 0:
                cat_pdfs = cat_pdfs[:sample_per_category]
            for pdf_file in cat_pdfs:
                pdf_entries.append({"path": pdf_file, "category": category_name})

    return pdf_entries


def load_csv_metadata(dataset_dir: Path) -> Dict[str, str]:
    """
    Load mapping of resume ID -> Category from Resume/Resume.csv if present.
    """
    csv_candidates = [
        dataset_dir / "Resume" / "Resume.csv",
        dataset_dir / "Resume.csv",
    ]
    id_to_category: Dict[str, str] = {}
    for csv_path in csv_candidates:
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        resume_id = row.get("ID", "").strip()
                        cat = row.get("Category", "").strip()
                        if resume_id:
                            id_to_category[resume_id] = cat
                logger.info(f"Loaded {len(id_to_category)} records from {csv_path.name}")
                break
            except Exception as e:
                logger.warning(f"Could not read {csv_path}: {e}")
    return id_to_category


def run_audit(
    dataset_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    sample_per_category: Optional[int] = None,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Execute privacy-safe audit across the PDF dataset.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning for PDF resumes in: {dataset_dir}")
    all_pdfs = find_dataset_pdfs(dataset_dir, sample_per_category=sample_per_category)
    total_found = len(all_pdfs)
    logger.info(f"Discovered {total_found} total PDF files across categories.")

    if total_found == 0:
        logger.warning("No PDF files found to audit.")
        return {"total_found": 0}

    # Load CSV mapping for cross-referencing
    csv_mapping = load_csv_metadata(dataset_dir)

    # Apply limit if specified
    audit_target = all_pdfs[:limit] if limit and limit > 0 else all_pdfs
    target_count = len(audit_target)
    logger.info(f"Proceeding to audit {target_count} files (batch size: {batch_size})...")

    parser = PDFParser()
    cache = CacheManager()

    # Metrics accumulators
    status_counts = Counter()
    category_counts = Counter()
    category_success = Counter()
    category_chars = defaultdict(list)
    category_words = defaultdict(list)
    char_lengths: List[int] = []
    word_lengths: List[int] = []
    file_hashes: Set[str] = set()
    duplicate_hash_count = 0
    scanned_image_count = 0

    audit_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(audit_target, start=1):
        pdf_path: Path = item["path"]
        category: str = item["category"]
        category_counts[category] += 1

        try:
            file_hash = CacheManager.compute_file_hash(pdf_path)
            if file_hash in file_hashes:
                duplicate_hash_count += 1
            file_hashes.add(file_hash)

            record = parser.parse_file(pdf_path)
            parse_status = record.parse_status
            status_counts[parse_status] += 1

            if parse_status == "success":
                category_success[category] += 1

            if any("scanned" in w.lower() or "image-only" in w.lower() for w in record.warnings):
                scanned_image_count += 1

            char_lengths.append(record.char_count)
            word_lengths.append(record.word_count)
            category_chars[category].append(record.char_count)
            category_words[category].append(record.word_count)

            # Privacy-safe audit record (no raw text, no names, no emails, no phones)
            file_stem = pdf_path.stem
            matched_csv_cat = csv_mapping.get(file_stem, "Unmapped")

            audit_records.append(
                {
                    "file_id": file_stem,
                    "file_hash_prefix": file_hash[:12],
                    "category": category,
                    "csv_category_match": matched_csv_cat == category if matched_csv_cat != "Unmapped" else None,
                    "char_count": record.char_count,
                    "word_count": record.word_count,
                    "status": parse_status,
                    "warnings_count": len(record.warnings),
                }
            )

        except Exception as e:
            status_counts["failed"] += 1
            logger.warning(f"Error processing file index {idx} ({pdf_path.name}): {e}")

        # Batch progress feedback
        if idx % batch_size == 0 or idx == target_count:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info(
                f"Audited {idx}/{target_count} ({idx / target_count * 100:.1f}%) | "
                f"Success: {status_counts['success']} | "
                f"Warnings: {status_counts['warning']} | "
                f"Failed: {status_counts['failed']} | "
                f"Speed: {rate:.1f} files/sec"
            )

    elapsed_total = round(time.time() - start_time, 2)

    # Compute Aggregate Statistics
    avg_chars = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    avg_words = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    min_chars = min(char_lengths) if char_lengths else 0
    max_chars = max(char_lengths) if char_lengths else 0

    summary = {
        "dataset_directory": str(dataset_dir),
        "total_files_discovered": total_found,
        "files_audited": target_count,
        "elapsed_seconds": elapsed_total,
        "processing_rate_files_per_sec": round(target_count / elapsed_total, 2) if elapsed_total > 0 else 0,
        "parse_success_count": status_counts["success"],
        "parse_warning_count": status_counts["warning"],
        "parse_failed_count": status_counts["failed"],
        "scanned_or_image_only_count": scanned_image_count,
        "duplicate_hash_count": duplicate_hash_count,
        "unique_categories_count": len(category_counts),
        "text_statistics": {
            "mean_char_count": round(avg_chars, 1),
            "mean_word_count": round(avg_words, 1),
            "min_char_count": min_chars,
            "max_char_count": max_chars,
        },
        "category_breakdown": {},
    }

    for cat, total in sorted(category_counts.items()):
        succ = category_success[cat]
        c_chars = category_chars[cat]
        summary["category_breakdown"][cat] = {
            "total_files": total,
            "success_count": succ,
            "success_rate_pct": round((succ / total) * 100, 2) if total > 0 else 0,
            "mean_char_count": round(sum(c_chars) / len(c_chars), 1) if c_chars else 0,
        }

    # Save summary JSON
    summary_path = output_dir / "audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save category CSV
    csv_out_path = output_dir / "category_metrics.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Total Files", "Parsed Successfully", "Success Rate (%)", "Mean Characters"])
        for cat, data in summary["category_breakdown"].items():
            writer.writerow([
                cat,
                data["total_files"],
                data["success_count"],
                f"{data['success_rate_pct']:.2f}",
                data["mean_char_count"],
            ])

    # Save Markdown Report
    md_out_path = output_dir / "audit_report.md"
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Audit Report: Resume PDF Corpus\n\n")
        f.write(f"- **Audited Directory:** `{dataset_dir}`\n")
        f.write(f"- **Files Audited:** {target_count} / {total_found}\n")
        f.write(f"- **Execution Time:** {elapsed_total}s ({summary['processing_rate_files_per_sec']} files/sec)\n")
        f.write(f"- **Parse Success:** {status_counts['success']} ({status_counts['success'] / target_count * 100:.1f}%)\n")
        f.write(f"- **Parse Warnings (Scanned/Empty):** {status_counts['warning']}\n")
        f.write(f"- **Parse Failures:** {status_counts['failed']}\n")
        f.write(f"- **Duplicate Files (by SHA-256):** {duplicate_hash_count}\n")
        f.write(f"- **Mean Word Count:** {round(avg_words, 1)} words\n\n")
        f.write("## Category Performance Summary\n\n")
        f.write("| Category | Total Files | Succeeded | Success Rate | Avg Chars |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cat, data in summary["category_breakdown"].items():
            f.write(
                f"| {cat} | {data['total_files']} | {data['success_count']} | "
                f"{data['success_rate_pct']:.1f}% | {data['mean_char_count']} |\n"
            )

    logger.info(f"Audit completed. Summary saved to: {summary_path}")
    logger.info(f"Category metrics saved to: {csv_out_path}")
    logger.info(f"Markdown report saved to: {md_out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit resume PDF dataset without leaking personal data.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to downloaded resume dataset directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDF files to audit (omit for full batch).",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=None,
        help="Sample up to N resumes from each category for cross-category coverage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Progress logging batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_AUDIT_DIR,
        help="Directory to save audit summaries and reports.",
    )

    args = parser.parse_args()
    run_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        sample_per_category=args.sample_per_category,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
