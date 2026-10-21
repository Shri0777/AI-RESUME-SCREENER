#!/usr/bin/env python3
"""
Ranking Quality Evaluation Harness.

Benchmarks the Baseline ranker (TF-IDF + Rule-based Skill Match) against the
Enhanced ranker (Hierarchical Skills + Enhanced TF-IDF + Optional Semantic Similarity)
across validation and held-out test query sets.

Calculates:
- Precision@K (P@1, P@3)
- Recall@K (R@3)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@3, NDCG@5)
- Pairwise ordering accuracy

Outputs machine-readable JSON and Markdown reports to reports/evaluation/.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_DIR, STRATEGY_BASELINE, STRATEGY_ENHANCED
from src.pdf_parser import PDFParser
from src.ranking_engine import RankingEngine
from src.ranking_evaluation import (
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ranking")

# Synthetic Benchmark Candidates
BENCHMARK_CANDIDATES = [
    {
        "candidate_id": "alice_johnson",
        "candidate_name": "Alice Johnson",
        "filename": "sample_alice_johnson.pdf",
        "email": "alice@example.com",
        "phone": "555-0101",
        "extracted_text": (
            "Alice Johnson. Lead Data Scientist and Machine Learning Specialist. "
            "Expertise in Python, SQL, PostgreSQL, Machine Learning, Deep Learning, Statistics, "
            "Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Docker, Kubernetes, AWS, Model Evaluation."
        ),
        "parse_status": "success",
        "warnings": [],
    },
    {
        "candidate_id": "bob_smith",
        "candidate_name": "Bob Smith",
        "filename": "sample_bob_smith.pdf",
        "email": "bob@example.com",
        "phone": "555-0102",
        "extracted_text": (
            "Bob Smith. Data Scientist & Analytics Practitioner. "
            "Skilled in Python, SQL, PostgreSQL, Machine Learning, Pandas, NumPy, Scikit-learn, "
            "Exploratory Data Analysis, Model Evaluation, Docker, Git."
        ),
        "parse_status": "success",
        "warnings": [],
    },
    {
        "candidate_id": "carol_davis",
        "candidate_name": "Carol Davis",
        "filename": "sample_carol_davis.pdf",
        "email": "carol@example.com",
        "phone": "555-0103",
        "extracted_text": (
            "Carol Davis. Business Intelligence and Data Analyst. "
            "Proficient in SQL, Python, MySQL, Data Analysis, Exploratory Data Analysis, "
            "Data Cleaning, Tableau, Power BI, Excel, Statistics."
        ),
        "parse_status": "success",
        "warnings": [],
    },
    {
        "candidate_id": "david_wilson",
        "candidate_name": "David Wilson",
        "filename": "sample_david_wilson.pdf",
        "email": "david@example.com",
        "phone": "555-0104",
        "extracted_text": (
            "David Wilson. Senior Backend Software Engineer. "
            "Experienced in Java, C++, Go, REST API, Microservices, Docker, Kubernetes, AWS, "
            "PostgreSQL, Redis, CI/CD, Git, Linux."
        ),
        "parse_status": "success",
        "warnings": [],
    },
    {
        "candidate_id": "emma_brown",
        "candidate_name": "Emma Brown",
        "filename": "sample_emma_brown.pdf",
        "email": "emma@example.com",
        "phone": "555-0105",
        "extracted_text": (
            "Emma Brown. Senior Frontend & Web Developer. "
            "Expert in JavaScript, TypeScript, React, HTML, CSS, Next.js, Node.js, Express.js, "
            "Tailwind CSS, REST API, Git."
        ),
        "parse_status": "success",
        "warnings": [],
    },
    {
        "candidate_id": "frank_white",
        "candidate_name": "Frank White",
        "filename": "sample_frank_white.pdf",
        "email": "frank@example.com",
        "phone": "555-0106",
        "extracted_text": (
            "Frank White. Cloud Infrastructure and DevOps Architect. "
            "Specialized in AWS, GCP, Azure, Docker, Kubernetes, Terraform, Ansible, Jenkins, "
            "CI/CD, Linux, Shell Scripting, Git, Python."
        ),
        "parse_status": "success",
        "warnings": [],
    },
]

# Benchmark Queries divided into Validation and Held-Out Test Splits
BENCHMARK_QUERIES = [
    # --- VALIDATION SPLIT (used to calibrate scoring weights) ---
    {
        "query_id": "VAL_Q1_DATA_SCIENCE",
        "split": "validation",
        "role_title": "Senior Data Scientist",
        "job_description": (
            "Looking for Senior Data Scientist with deep expertise in Python, SQL, "
            "Machine Learning, Deep Learning, Statistics, Scikit-learn, TensorFlow, "
            "and cloud deployment on AWS with Docker."
        ),
        # 0 = irrelevant, 1 = weak, 2 = relevant, 3 = highly relevant
        "graded_relevance": {
            "alice_johnson": 3.0,
            "bob_smith": 2.0,
            "carol_davis": 1.0,
            "david_wilson": 0.0,
            "emma_brown": 0.0,
            "frank_white": 0.0,
        },
    },
    {
        "query_id": "VAL_Q2_FRONTEND_WEB",
        "split": "validation",
        "role_title": "Senior Frontend Engineer",
        "job_description": (
            "Seeking Senior Frontend Engineer with extensive experience in JavaScript, "
            "TypeScript, React, Next.js, HTML, CSS, Tailwind CSS, and REST API integration."
        ),
        "graded_relevance": {
            "emma_brown": 3.0,
            "david_wilson": 1.0,  # some REST API
            "alice_johnson": 0.0,
            "bob_smith": 0.0,
            "carol_davis": 0.0,
            "frank_white": 0.0,
        },
    },
    # --- HELD-OUT TEST SPLIT (unseen during weight tuning) ---
    {
        "query_id": "TEST_Q1_ML_ENGINEER",
        "split": "test",
        "role_title": "Machine Learning Engineer",
        "job_description": (
            "Job Title: Machine Learning Engineer.\n"
            "Required Skills: Python, Machine Learning, Scikit-learn, Docker, Model Evaluation, Statistics.\n"
            "Preferred Skills: Deep Learning, PyTorch, Kubernetes, AWS."
        ),
        "graded_relevance": {
            "alice_johnson": 3.0,
            "bob_smith": 2.0,
            "david_wilson": 1.0,  # knows Docker/Kubernetes/AWS
            "carol_davis": 0.0,
            "emma_brown": 0.0,
            "frank_white": 1.0,  # knows Docker/Kubernetes/AWS/Python
        },
    },
    {
        "query_id": "TEST_Q2_DEVOPS_CLOUD",
        "split": "test",
        "role_title": "DevOps / Cloud Platform Engineer",
        "job_description": (
            "Job Title: Cloud Platform Engineer.\n"
            "Required Skills: Docker, Kubernetes, AWS, CI/CD, Terraform, Linux.\n"
            "Preferred Skills: Python, Jenkins, Ansible, PostgreSQL."
        ),
        "graded_relevance": {
            "frank_white": 3.0,
            "david_wilson": 2.0,  # strong Docker, K8s, AWS, Linux, CI/CD
            "alice_johnson": 1.0,  # knows Docker, K8s, AWS, Linux
            "bob_smith": 0.0,
            "carol_davis": 0.0,
            "emma_brown": 0.0,
        },
    },
    {
        "query_id": "TEST_Q3_BI_ANALYST",
        "split": "test",
        "role_title": "Data Analyst / Business Intelligence",
        "job_description": (
            "Job Title: BI Analyst.\n"
            "Required Skills: SQL, Tableau, Power BI, Statistics, Data Analysis, Exploratory Data Analysis.\n"
            "Preferred Skills: Python, Pandas, MySQL, Excel."
        ),
        "graded_relevance": {
            "carol_davis": 3.0,
            "bob_smith": 2.0,  # knows SQL, Statistics, Data Analysis, Python, Pandas
            "alice_johnson": 1.0,  # knows SQL, Statistics, Python, Pandas
            "david_wilson": 0.0,
            "emma_brown": 0.0,
            "frank_white": 0.0,
        },
    },
]


def evaluate_engine_on_query(
    engine: RankingEngine,
    strategy: str,
    query_dict: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate ranking for a single query against the candidate pool.
    """
    jd = query_dict["job_description"]
    graded_rel = query_dict["graded_relevance"]
    # Binary relevance threshold: grade >= 2 is relevant
    relevant_set = {cid for cid, score in graded_rel.items() if score >= 2.0}

    # Run ranking
    df = engine.rank_candidates(jd, candidate_pool, strategy=strategy)

    # Map candidate filename/name back to candidate_id
    cand_name_to_id = {c["candidate_name"]: c["candidate_id"] for c in candidate_pool}
    ranked_ids = [cand_name_to_id.get(name, name) for name in df["candidate_name"]]

    p1 = precision_at_k(ranked_ids, relevant_set, k=1)
    p3 = precision_at_k(ranked_ids, relevant_set, k=3)
    r3 = recall_at_k(ranked_ids, relevant_set, k=3)
    rr = reciprocal_rank(ranked_ids, relevant_set)
    ndcg3 = ndcg_at_k(ranked_ids, graded_rel, k=3)
    ndcg5 = ndcg_at_k(ranked_ids, graded_rel, k=5)
    pw_acc, conc, tot_pairs = pairwise_accuracy(ranked_ids, graded_rel)

    return {
        "query_id": query_dict["query_id"],
        "split": query_dict["split"],
        "role_title": query_dict["role_title"],
        "strategy": strategy,
        "ranked_candidates": ranked_ids,
        "precision@1": p1,
        "precision@3": p3,
        "recall@3": r3,
        "mrr": rr,
        "ndcg@3": ndcg3,
        "ndcg@5": ndcg5,
        "pairwise_accuracy": pw_acc,
        "concordant_pairs": conc,
        "total_pairs": tot_pairs,
    }


def run_benchmark(
    output_dir: Path = EVALUATION_DIR,
) -> Dict[str, Any]:
    """
    Run full benchmark comparison across Baseline and Enhanced strategies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = RankingEngine()

    strategies = [STRATEGY_BASELINE, STRATEGY_ENHANCED]
    results_by_strategy: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}

    logger.info("Executing ranking evaluation harness across benchmark queries...")

    for query in BENCHMARK_QUERIES:
        for strat in strategies:
            res = evaluate_engine_on_query(engine, strat, query, BENCHMARK_CANDIDATES)
            results_by_strategy[strat].append(res)

    # Compute Aggregate Metrics per Strategy per Split
    summary_report: Dict[str, Any] = {
        "label_source": "synthetic_graded_judgements_0_to_3",
        "candidate_pool_size": len(BENCHMARK_CANDIDATES),
        "total_queries": len(BENCHMARK_QUERIES),
        "validation_queries_count": sum(1 for q in BENCHMARK_QUERIES if q["split"] == "validation"),
        "test_queries_count": sum(1 for q in BENCHMARK_QUERIES if q["split"] == "test"),
        "metrics_by_strategy": {},
    }

    splits = ["validation", "test", "overall"]

    for strat in strategies:
        strat_results = results_by_strategy[strat]
        summary_report["metrics_by_strategy"][strat] = {}

        for sp in splits:
            if sp == "overall":
                subset = strat_results
            else:
                subset = [r for r in strat_results if r["split"] == sp]

            if not subset:
                continue

            n = len(subset)
            avg_p1 = sum(r["precision@1"] for r in subset) / n
            avg_p3 = sum(r["precision@3"] for r in subset) / n
            avg_r3 = sum(r["recall@3"] for r in subset) / n
            avg_mrr = sum(r["mrr"] for r in subset) / n
            avg_ndcg3 = sum(r["ndcg@3"] for r in subset) / n
            avg_ndcg5 = sum(r["ndcg@5"] for r in subset) / n

            tot_conc = sum(r["concordant_pairs"] for r in subset)
            tot_p = sum(r["total_pairs"] for r in subset)
            overall_pw = tot_conc / tot_p if tot_p > 0 else 1.0

            summary_report["metrics_by_strategy"][strat][sp] = {
                "query_count": n,
                "mean_precision@1": round(avg_p1, 4),
                "mean_precision@3": round(avg_p3, 4),
                "mean_recall@3": round(avg_r3, 4),
                "mrr": round(avg_mrr, 4),
                "mean_ndcg@3": round(avg_ndcg3, 4),
                "mean_ndcg@5": round(avg_ndcg5, 4),
                "pairwise_accuracy": round(overall_pw, 4),
            }

    # Save JSON results
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_report,
                "detailed_query_results": results_by_strategy,
            },
            f,
            indent=2,
        )

    # Save Markdown report
    md_path = output_dir / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ranking Evaluation Report: Baseline vs. Enhanced Ranker\n\n")
        f.write(
            "> **Methodology Note:** Benchmark evaluates ranking quality across structured queries "
            "with relevance graded from 0 (irrelevant) to 3 (highly relevant). "
            "Queries are partitioned into a Validation set (used for parameter tuning) and an unseen "
            "held-out Test set. Metrics reflect synthetic decision-support benchmarks and are not "
            "guarantees of real-world hiring outcomes.\n\n"
        )
        f.write(f"- **Candidate Pool Size:** {len(BENCHMARK_CANDIDATES)} synthetic profiles\n")
        f.write(f"- **Validation Queries:** {summary_report['validation_queries_count']}\n")
        f.write(f"- **Held-Out Test Queries:** {summary_report['test_queries_count']}\n")
        f.write(f"- **Label Source:** Synthetic graded relevance judgements (0–3 scale)\n\n")

        for sp in ["test", "validation", "overall"]:
            f.write(f"## {sp.capitalize()} Split Performance\n\n")
            f.write(
                "| Metric | Baseline (TF-IDF + Skills) | Enhanced Strategy | Relative Delta |\n"
                "| :--- | :---: | :---: | :---: |\n"
            )

            b_met = summary_report["metrics_by_strategy"][STRATEGY_BASELINE].get(sp, {})
            e_met = summary_report["metrics_by_strategy"][STRATEGY_ENHANCED].get(sp, {})

            metric_keys = [
                ("mean_precision@1", "Precision@1"),
                ("mean_precision@3", "Precision@3"),
                ("mean_recall@3", "Recall@3"),
                ("mrr", "Mean Reciprocal Rank (MRR)"),
                ("mean_ndcg@3", "NDCG@3"),
                ("mean_ndcg@5", "NDCG@5"),
                ("pairwise_accuracy", "Pairwise Accuracy"),
            ]

            for key, label in metric_keys:
                b_val = b_met.get(key, 0.0)
                e_val = e_met.get(key, 0.0)
                delta = e_val - b_val
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                f.write(f"| {label} | {b_val:.4f} | {e_val:.4f} | {delta_str} |\n")
            f.write("\n")

    logger.info(f"Evaluation finished. Machine-readable output: {json_path}")
    logger.info(f"Markdown report generated: {md_path}")

    # Print summary table to console
    test_base = summary_report["metrics_by_strategy"][STRATEGY_BASELINE]["test"]
    test_enh = summary_report["metrics_by_strategy"][STRATEGY_ENHANCED]["test"]
    print("\n" + "=" * 65)
    print("HELD-OUT TEST SPLIT EVALUATION RESULTS (N=3 Queries)")
    print("=" * 65)
    print(f"{'Metric':<25} {'Baseline':<12} {'Enhanced':<12} {'Delta':<10}")
    print("-" * 65)
    for key, label in [
        ("mean_precision@1", "Precision@1"),
        ("mean_precision@3", "Precision@3"),
        ("mean_recall@3", "Recall@3"),
        ("mrr", "MRR"),
        ("mean_ndcg@3", "NDCG@3"),
        ("mean_ndcg@5", "NDCG@5"),
        ("pairwise_accuracy", "Pairwise Acc"),
    ]:
        b = test_base[key]
        e = test_enh[key]
        d = e - b
        d_str = f"+{d:.4f}" if d > 0 else f"{d:.4f}"
        print(f"{label:<25} {b:<12.4f} {e:<12.4f} {d_str:<10}")
    print("=" * 65 + "\n")

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs. enhanced ranking quality.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_DIR,
        help="Target folder for evaluation reports and JSON data.",
    )
    args = parser.parse_args()
    run_benchmark(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
