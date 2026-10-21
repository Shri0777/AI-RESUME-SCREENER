#!/usr/bin/env python3
"""
Synthetic Sample Data Generation Script.

Generates five deterministic synthetic PDF resumes with diverse technical skill profiles
for demonstration and automated testing of the AI-Powered Resume Screening System.
All data is purely synthetic and does not represent real persons.
"""

import os
from pathlib import Path
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

# Project root path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESUMES_DIR = PROJECT_ROOT / "data" / "resumes"
JOB_DESC_DIR = PROJECT_ROOT / "data" / "job_descriptions"

CANDIDATES_DATA = [
    {
        "filename": "sample_alice_johnson.pdf",
        "name": "Alice Johnson",
        "title": "Senior Data Scientist & Machine Learning Specialist",
        "contact": "alice.johnson@example.com | (555) 019-2831 | San Francisco, CA | github.com/alice-demo",
        "summary": (
            "Innovative Senior Data Scientist with 6+ years of experience building end-to-end "
            "machine learning and predictive modeling pipelines. Deep background in statistical analysis, "
            "feature engineering, and cloud deployment of ML models on AWS."
        ),
        "skills": [
            "Programming Languages: Python, SQL, R, Bash",
            "Data Science & ML: Machine Learning, Deep Learning, Statistics, Exploratory Data Analysis, "
            "Feature Engineering, Model Evaluation, A/B Testing",
            "Libraries & Frameworks: Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, SciPy",
            "Databases & Cloud: PostgreSQL, SQL, Redis, AWS, Docker, Kubernetes",
            "Tools & Practices: Git, JIRA, CI/CD, Linux, Agile",
        ],
        "experience": [
            (
                "Lead Data Scientist — NexaGen Analytics (2021 – Present)",
                "Architected customer churn and lifetime-value predictive models in Python and Scikit-learn, "
                "improving model evaluation precision by 24%. Led deployment using Docker containers on AWS ECS.",
            ),
            (
                "Data Scientist — DataCore Solutions (2018 – 2021)",
                "Conducted statistical analysis and exploratory data analysis across 10M+ transaction records using "
                "PostgreSQL and Pandas. Implemented deep learning NLP models using TensorFlow and PyTorch.",
            ),
        ],
        "education": "M.S. in Computer Science (Machine Learning Focus) — Stanford University (2018)",
    },
    {
        "filename": "sample_bob_smith.pdf",
        "name": "Bob Smith",
        "title": "Data Scientist & Analytics Practitioner",
        "contact": "bob.smith@example.com | (555) 014-9923 | Austin, TX | github.com/bob-demo",
        "summary": (
            "Data Scientist with 4 years of practical experience in predictive modeling, exploratory data analysis, "
            "and statistical programming in Python and SQL. Experienced in containerizing models with Docker."
        ),
        "skills": [
            "Programming Languages: Python, SQL, Bash",
            "Data Science & ML: Machine Learning, Statistics, Data Analysis, Exploratory Data Analysis, "
            "Model Evaluation, Feature Engineering",
            "Libraries & Frameworks: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn",
            "Databases & Cloud: PostgreSQL, SQL, MySQL, Docker, GCP",
            "Tools & Practices: Git, JIRA, Unit Testing, Linux",
        ],
        "experience": [
            (
                "Data Scientist — MetricFlow Systems (2021 – Present)",
                "Built regression and classification models using Scikit-learn, Pandas, and NumPy. "
                "Conducted A/B testing and statistical significance evaluations for product launches.",
            ),
            (
                "Junior Data Analyst — InsightPulse (2019 – 2021)",
                "Queried PostgreSQL databases and performed data cleaning and feature engineering in Python. "
                "Maintained automated ETL workflows and model evaluation reporting.",
            ),
        ],
        "education": "B.S. in Statistics and Data Science — University of Texas at Austin (2019)",
    },
    {
        "filename": "sample_carol_davis.pdf",
        "name": "Carol Davis",
        "title": "Business Intelligence & Data Analyst",
        "contact": "carol.davis@example.com | (555) 018-7744 | Chicago, IL | linkedin.com/in/carol-demo",
        "summary": (
            "Detail-oriented Data Analyst with 4+ years of experience transforming complex business data into "
            "actionable insights. Highly proficient in SQL database queries, exploratory data analysis, and dashboarding."
        ),
        "skills": [
            "Programming & Query: SQL, Python, MySQL, SQLite",
            "Analytics & BI: Data Analysis, Exploratory Data Analysis, Data Cleaning, Data Visualization, "
            "Statistics, Business Intelligence",
            "Tools & Libraries: Pandas, NumPy, Tableau, Power BI, Excel",
            "Collaboration: Git, JIRA, Agile, Scrum",
        ],
        "experience": [
            (
                "Senior BI Analyst — Apex Financial Services (2021 – Present)",
                "Engineered complex SQL queries across relational databases to deliver executive reports. "
                "Automated data cleaning and summary statistics pipelines using Python and Pandas.",
            ),
            (
                "Data Reporting Analyst — Beacon Analytics (2019 – 2021)",
                "Created interactive dashboards in Power BI and Tableau. Conducted exploratory data analysis "
                "to pinpoint operational bottlenecks.",
            ),
        ],
        "education": "B.S. in Information Systems & Analytics — University of Illinois (2019)",
    },
    {
        "filename": "sample_david_wilson.pdf",
        "name": "David Wilson",
        "title": "Senior Backend Software Engineer",
        "contact": "david.wilson@example.com | (555) 012-3488 | Seattle, WA | github.com/david-demo",
        "summary": (
            "Backend Systems Engineer with 5+ years of experience designing scalable microservices, REST APIs, "
            "and distributed infrastructure. Expert in Java, C++, Go, and container orchestration."
        ),
        "skills": [
            "Programming Languages: Java, C++, Go, Python, C#",
            "Backend & Frameworks: REST API, FastAPI, Spring Boot, Microservices",
            "Databases: PostgreSQL, MySQL, Redis, Cassandra",
            "Cloud & DevOps: Docker, Kubernetes, AWS, CI/CD, Jenkins, Linux, Terraform",
            "Tools: Git, JIRA, PyTest, Unit Testing",
        ],
        "experience": [
            (
                "Senior Backend Engineer — CloudGrid Networks (2020 – Present)",
                "Built high-throughput distributed microservices in Java and Go handling 50k requests/sec. "
                "Orchestrated deployments with Docker and Kubernetes on AWS.",
            ),
            (
                "Software Engineer — CoreLogic Software (2018 – 2020)",
                "Developed REST APIs in Python (FastAPI) and C++. Integrated Redis caching layers and "
                "managed PostgreSQL database schemas.",
            ),
        ],
        "education": "B.S. in Computer Engineering — University of Washington (2018)",
    },
    {
        "filename": "sample_emma_brown.pdf",
        "name": "Emma Brown",
        "title": "Senior Frontend & Web Developer",
        "contact": "emma.brown@example.com | (555) 016-5521 | New York, NY | github.com/emma-demo",
        "summary": (
            "Frontend Engineer with 4 years of experience building modern, responsive, user-friendly web "
            "applications. Deep expertise in JavaScript, TypeScript, React, and component-driven UI architecture."
        ),
        "skills": [
            "Programming Languages: JavaScript, TypeScript, HTML, CSS",
            "Web Frameworks: React, Next.js, Vue.js, Node.js, Express.js",
            "Styling & UI: Tailwind CSS, Bootstrap, Responsive Design",
            "APIs & Integration: REST API, GraphQL",
            "Tools & Collaboration: Git, JIRA, Agile, Scrum, Unit Testing",
        ],
        "experience": [
            (
                "Lead Frontend Developer — PixelForge Media (2021 – Present)",
                "Architected accessible single-page web applications using React, TypeScript, and Next.js. "
                "Optimized web vital scores and integrated backend REST APIs.",
            ),
            (
                "UI Engineer — StudioWeb Design (2019 – 2021)",
                "Developed modular UI component libraries with React and Tailwind CSS. Implemented automated "
                "unit testing and version control workflows in Git.",
            ),
        ],
        "education": "B.A. in Digital Media & Computer Science — New York University (2019)",
    },
]


def create_resume_pdf(data: Dict, output_path: Path) -> None:
    """Generate a clean single-page PDF resume using ReportLab."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    disclaimer_style = ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#DC2626"),
        alignment=1,  # Center
        fontName="Helvetica-Bold",
        spaceAfter=6,
    )

    name_style = ParagraphStyle(
        name="CandidateName",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=2,
    )

    title_style = ParagraphStyle(
        name="CandidateTitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2563EB"),
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )

    contact_style = ParagraphStyle(
        name="ContactInfo",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=8,
    )

    section_header = ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        name="ResumeBody",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=4,
    )

    exp_title = ParagraphStyle(
        name="ExpTitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
    )

    elements = []

    # Synthetic Demo Disclaimer
    elements.append(
        Paragraph("SYNTHETIC DEMO RESUME — FOR TESTING & DEMONSTRATION PURPOSES ONLY", disclaimer_style)
    )

    # Name and Title
    elements.append(Paragraph(data["name"], name_style))
    elements.append(Paragraph(data["title"], title_style))
    elements.append(Paragraph(data["contact"], contact_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"), spaceAfter=6))

    # Professional Summary
    elements.append(Paragraph("PROFESSIONAL SUMMARY", section_header))
    elements.append(Paragraph(data["summary"], body_style))

    # Technical Skills
    elements.append(Paragraph("TECHNICAL SKILLS", section_header))
    for skill_line in data["skills"]:
        elements.append(Paragraph(f"• {skill_line}", body_style))

    # Professional Experience
    elements.append(Paragraph("PROFESSIONAL EXPERIENCE", section_header))
    for role, desc in data["experience"]:
        elements.append(Paragraph(role, exp_title))
        elements.append(Paragraph(desc, body_style))
        elements.append(Spacer(1, 2))

    # Education
    elements.append(Paragraph("EDUCATION", section_header))
    elements.append(Paragraph(data["education"], body_style))

    doc.build(elements)


def generate_all_samples() -> None:
    """Generate all five synthetic resumes and ensure directory existence."""
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DESC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic resumes in {RESUMES_DIR}...")
    for candidate in CANDIDATES_DATA:
        file_path = RESUMES_DIR / candidate["filename"]
        create_resume_pdf(candidate, file_path)
        print(f"  ✓ Created: {file_path.name}")

    print("\nAll synthetic demo resumes generated successfully.")


if __name__ == "__main__":
    generate_all_samples()
