"""
AI-Powered Resume Screening and Candidate Ranking System.

A transparent, local decision-support Streamlit application that evaluates
candidate resumes against a job description using lexical TF-IDF cosine similarity,
optional dense semantic embeddings (SentenceTransformers), and hierarchical skill matching.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import List, Optional

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_BASELINE_SIMILARITY_WEIGHT,
    DEFAULT_BASELINE_SKILL_WEIGHT,
    DEFAULT_ENHANCED_LEXICAL_WEIGHT,
    DEFAULT_ENHANCED_SEMANTIC_WEIGHT,
    DEFAULT_ENHANCED_SKILL_WEIGHT,
    EVALUATION_DIR,
    LOCAL_AUDIT_DIR,
    STRATEGY_BASELINE,
    STRATEGY_ENHANCED,
)
from src.pdf_parser import PDFParser
from src.ranking_engine import RankingEngine
from src.report_generator import ReportGenerator
from src.semantic_matcher import SemanticMatcher
from src.skill_extractor import SkillExtractor
from src.visualizer import (
    plot_candidate_ranking,
    plot_candidate_skill_gap,
    plot_score_breakdown,
    plot_score_distribution,
    plot_skill_frequency,
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener & Candidate Ranking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "parsing_records" not in st.session_state:
    st.session_state.parsing_records = []
if "required_skills" not in st.session_state:
    st.session_state.required_skills = []
if "preferred_skills" not in st.session_state:
    st.session_state.preferred_skills = []
if "report_save_message" not in st.session_state:
    st.session_state.report_save_message = None


def load_sample_jd() -> str:
    """Load sample job description text from file."""
    for path in [
        Path("sample_job_description.txt"),
        Path("data/job_descriptions/sample_job_description.txt"),
    ]:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return "Sample job description file not found."


semantic_matcher = SemanticMatcher()
is_semantic_available = semantic_matcher.is_available

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ System Configuration")
    st.markdown(
        """
        **AI-Powered Resume Screening System**
        Local decision-support tool providing transparent, objective candidate evaluation.
        """
    )
    st.markdown("---")

    st.subheader("🎯 Ranking Strategy")
    strategy_options = ["Baseline (TF-IDF 70% + Skills 30%)", "Enhanced Hybrid (Lexical + Semantic + Skills)"]
    selected_strategy_label = st.radio(
        "Select Ranking Strategy",
        strategy_options,
        index=0,
        help="Baseline uses classical lexical TF-IDF. Enhanced incorporates dense semantic embeddings and hierarchical skills.",
    )
    strategy_key = STRATEGY_BASELINE if "Baseline" in selected_strategy_label else STRATEGY_ENHANCED

    if strategy_key == STRATEGY_ENHANCED:
        if is_semantic_available:
            st.success("🟢 Semantic Embeddings: Ready (SentenceTransformers)")
        else:
            st.warning("🟡 Semantic Embeddings: Fallback (library not installed; runs enhanced lexical)")

    st.markdown("---")
    st.subheader("⚖️ Scoring Weights")

    if strategy_key == STRATEGY_BASELINE:
        sim_weight = st.slider(
            "TF-IDF Similarity Weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_BASELINE_SIMILARITY_WEIGHT,
            step=0.05,
            help="Weight assigned to document lexical similarity.",
        )
        skill_weight = round(1.0 - sim_weight, 2)
        st.info(f"**Skill Match Weight:** `{skill_weight:.2f}` (Auto-balanced to 1.0)")
        w_lex = sim_weight
        w_sem = 0.0
        w_skl = skill_weight
    else:
        st.caption("Configure weights for 3-component hybrid model (must sum to 1.0):")
        w_lex = st.slider("Lexical (TF-IDF) Weight", 0.0, 1.0, DEFAULT_ENHANCED_LEXICAL_WEIGHT, 0.05)
        w_sem = st.slider("Semantic Similarity Weight", 0.0, 1.0, DEFAULT_ENHANCED_SEMANTIC_WEIGHT, 0.05)
        w_skl = round(max(0.0, 1.0 - (w_lex + w_sem)), 2)
        st.info(f"**Skill Match Weight:** `{w_skl:.2f}` (Auto-balanced to 1.0)")

    st.markdown("---")
    st.subheader("🔒 Privacy & Compliance")
    st.caption(
        "• All processing is performed locally in-memory.\n"
        "• Names, emails, and phone numbers are excluded from text embeddings.\n"
        "• No candidate data is transmitted externally.\n"
        "• Ensure you have authorization to screen submitted resumes."
    )

    st.markdown("---")
    st.subheader("🧪 Demo Utilities")
    if st.button("Load Demo Sample Resumes", use_container_width=True):
        sample_pdfs = sorted(list(Path("data/resumes").glob("*.pdf")))
        if sample_pdfs:
            st.session_state.job_description = load_sample_jd()
            with st.spinner("Processing 5 synthetic demo resumes..."):
                parser = PDFParser()
                records = parser.parse_multiple(sample_pdfs)
                engine = RankingEngine(
                    strategy=strategy_key,
                    similarity_weight=w_lex,
                    skill_weight=w_skl if strategy_key == STRATEGY_BASELINE else w_skl,
                    lexical_weight=w_lex,
                    semantic_weight=w_sem,
                    enhanced_skill_weight=w_skl,
                )
                df = engine.rank_candidates(st.session_state.job_description, records)
                # Keep UI-only columns consistent across baseline and enhanced results.
                # The baseline ranking schema intentionally omits semantic score fields.
                if "semantic_similarity_score" not in df.columns:
                    df["semantic_similarity_score"] = 0.0

                st.session_state.processed_df = df
                st.session_state.parsing_records = records
                (
                    st.session_state.required_skills,
                    st.session_state.preferred_skills,
                    _,
                ) = SkillExtractor().extract_skills_from_jd_sections(
                    st.session_state.job_description
                )
            st.success("Loaded 5 synthetic resumes & Senior Data Scientist JD!")
            st.rerun()
        else:
            st.warning("Sample PDFs not found. Run scripts/generate_sample_data.py first.")


# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📤 1. Upload & Process",
        "🏆 2. Candidate Rankings",
        "📊 3. Analysis & Charts",
        "📑 4. Reports & Export",
        "📈 5. Benchmark & Evaluation",
    ]
)

# ==========================================
# TAB 1: UPLOAD & PROCESS
# ==========================================
with tab1:
    st.header("Upload Resumes and Specify Job Description")
    st.markdown(
        f"Active Strategy: **{selected_strategy_label}** | Evaluates candidate resumes with full transparency."
    )

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("1. Job Description")
        col_jd_btn, col_jd_clr = st.columns([2, 1])
        with col_jd_btn:
            if st.button("Load Sample Senior Data Scientist JD"):
                st.session_state.job_description = load_sample_jd()
                st.rerun()
        with col_jd_clr:
            if st.button("Clear JD"):
                st.session_state.job_description = ""
                st.rerun()

        jd_text = st.text_area(
            "Paste Job Description Here",
            value=st.session_state.job_description,
            height=300,
            placeholder="Paste role requirements, responsibilities, and qualifications here...",
        )
        st.session_state.job_description = jd_text

        with st.expander("🛠️ Advanced: Explicit Skills Override (Optional)"):
            st.caption("Leave blank to automatically infer required and preferred skills from the JD text.")
            explicit_req_input = st.text_input(
                "Mandatory / Required Skills (comma-separated)",
                placeholder="e.g. Python, SQL, Machine Learning",
            )
            explicit_pref_input = st.text_input(
                "Preferred / Nice-to-Have Skills (comma-separated)",
                placeholder="e.g. AWS, Docker, PyTorch",
            )

    with col2:
        st.subheader("2. Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF documents. Scanned image-only PDFs will be flagged.",
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} PDF file(s) uploaded and ready for processing.")
            with st.expander("Uploaded Files List"):
                for uf in uploaded_files:
                    st.write(f"• **{uf.name}** ({uf.size / 1024:.1f} KB)")
        else:
            st.info("No files uploaded. Upload PDFs or click 'Load Demo Sample Resumes' in the sidebar.")

    st.markdown("---")
    process_btn = st.button("🚀 Process & Rank Candidates", type="primary", use_container_width=True)

    if process_btn:
        if not jd_text.strip():
            st.error("Please provide a Job Description before processing resumes.")
        elif not uploaded_files:
            st.error("Please upload at least one candidate resume PDF.")
        else:
            progress_bar = st.progress(0, text="Starting batch PDF extraction...")
            start_t = time.time()

            parser = PDFParser()
            parsed_records = []
            total_files = len(uploaded_files)

            for i, uf in enumerate(uploaded_files):
                rec = parser.parse_file(uf)
                parsed_records.append(rec)
                progress_bar.progress((i + 1) / total_files, text=f"Parsed {i + 1}/{total_files} resumes...")

            progress_bar.progress(1.0, text="Computing similarity and hybrid candidate ranking...")

            explicit_req = [s.strip() for s in explicit_req_input.split(",") if s.strip()] if explicit_req_input else None
            explicit_pref = [s.strip() for s in explicit_pref_input.split(",") if s.strip()] if explicit_pref_input else None

            engine = RankingEngine(
                strategy=strategy_key,
                similarity_weight=w_lex,
                skill_weight=w_skl if strategy_key == STRATEGY_BASELINE else w_skl,
                lexical_weight=w_lex,
                semantic_weight=w_sem,
                enhanced_skill_weight=w_skl,
            )
            df = engine.rank_candidates(
                jd_text,
                parsed_records,
                strategy=strategy_key,
                explicit_required_skills=explicit_req,
                explicit_preferred_skills=explicit_pref,
            )

            elapsed = round(time.time() - start_t, 2)
            # The baseline engine returns a legacy schema without a semantic score.
            # Add a neutral UI-only value so shared tabs can render both strategies.
            if "semantic_similarity_score" not in df.columns:
                df["semantic_similarity_score"] = 0.0

            st.session_state.processed_df = df
            st.session_state.parsing_records = parsed_records
            (
                st.session_state.required_skills,
                st.session_state.preferred_skills,
                _,
            ) = SkillExtractor().extract_skills_from_jd_sections(
                jd_text,
                explicit_required=explicit_req,
                explicit_preferred=explicit_pref,
            )

            st.success(f"Processing complete in {elapsed}s! Successfully evaluated {len(df)} candidate(s).")

    if st.session_state.parsing_records:
        success_count = sum(1 for r in st.session_state.parsing_records if r.parse_status == "success")
        warning_count = sum(1 for r in st.session_state.parsing_records if r.parse_status == "warning")
        failed_count = sum(1 for r in st.session_state.parsing_records if r.parse_status == "failed")

        st.markdown("### 📋 Batch Parsing Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Submitted", len(st.session_state.parsing_records))
        m2.metric("Parsed Cleanly", success_count)
        m3.metric("Warnings (Scanned/Empty)", warning_count)
        m4.metric("Parse Failures", failed_count)

        with st.expander("View Document Details & Warnings"):
            summary_rows = []
            for r in st.session_state.parsing_records:
                summary_rows.append(
                    {
                        "Filename": r.filename,
                        "Candidate Name": r.candidate_name,
                        "Name Inferred": "Yes" if r.is_name_inferred else "No",
                        "Email": r.email,
                        "Phone": r.phone,
                        "Word Count": r.word_count,
                        "Status": r.parse_status,
                        "Warnings / Notes": "; ".join(r.warnings) if r.warnings else "None",
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# ==========================================
# TAB 2: CANDIDATE RANKINGS
# ==========================================
with tab2:
    st.header("🏆 Candidate Screening Leaderboard")

    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.info("No candidates have been ranked yet. Please upload and process resumes in Tab 1.")
    else:
        df = st.session_state.processed_df
        active_strat_used = df.iloc[0].get("strategy_used", strategy_key)

        st.caption(
            f"**Applied Strategy:** `{active_strat_used}` | "
            f"**Weights:** Lexical={w_lex:.2f}, Semantic={w_sem:.2f}, Skills={w_skl:.2f}"
        )

        avg_score = df["final_score"].mean()
        top_cand = df.iloc[0]["candidate_name"]
        top_score = df.iloc[0]["final_score"]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Candidates Ranked", len(df))
        col_m2.metric("Top Ranked Candidate", top_cand)
        col_m3.metric("Highest Final Score", f"{top_score:.1f}%")
        col_m4.metric("Average Final Score", f"{avg_score:.1f}%")

        display_df = df.copy()
        display_df["matched_skills_str"] = display_df["matched_skills"].apply(
            lambda s: ", ".join(s) if s else "None"
        )
        display_df["missing_skills_str"] = display_df["missing_skills"].apply(
            lambda s: ", ".join(s) if s else "None"
        )

        # Leaderboard table
        table_cols = [
            "rank",
            "candidate_name",
            "final_score",
            "similarity_score",
            "semantic_similarity_score",
            "skill_match_score",
            "matched_skills_str",
            "missing_skills_str",
            "parse_status",
        ]

        rename_map = {
            "rank": "Rank",
            "candidate_name": "Candidate Name",
            "final_score": "Final Score (%)",
            "similarity_score": "TF-IDF (%)",
            "semantic_similarity_score": "Semantic (%)",
            "skill_match_score": "Skill Match (%)",
            "matched_skills_str": "Matched Required",
            "missing_skills_str": "Missing Required",
            "parse_status": "Status",
        }

        st.dataframe(
            display_df[table_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("🔍 Detailed Candidate Evaluation")
        selected_name = st.selectbox("Select Candidate for Deep Dive", options=df["candidate_name"].tolist())
        selected_row = df[df["candidate_name"] == selected_name].iloc[0]

        d_col1, d_col2 = st.columns([1, 1], gap="medium")

        with d_col1:
            st.markdown(f"### **{selected_row['candidate_name']}** (Rank #{selected_row['rank']})")
            st.markdown(f"**Filename:** `{selected_row['filename']}`")
            st.markdown(f"**Contact:** Email: `{selected_row['email']}` | Phone: `{selected_row['phone']}`")
            st.markdown(f"**Parse Status:** `{selected_row['parse_status']}`")
            if selected_row["warnings"] != "None":
                st.warning(f"**Warnings:** {selected_row['warnings']}")

            st.markdown("#### Score Components")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Final Score", f"{selected_row['final_score']:.2f}%")
            sc2.metric("TF-IDF Score", f"{selected_row['similarity_score']:.2f}%")
            sc3.metric("Semantic Score", f"{selected_row['semantic_similarity_score']:.2f}%")
            sc4.metric("Skill Score", f"{selected_row['skill_match_score']:.2f}%")

        with d_col2:
            st.markdown("#### Skills Analysis")
            matched_req = selected_row["matched_skills"]
            missing_req = selected_row["missing_skills"]
            matched_pref = selected_row.get("matched_preferred_skills", [])
            missing_pref = selected_row.get("missing_preferred_skills", [])

            st.markdown(f"**✅ Matched Mandatory Skills ({len(matched_req)}):**")
            if matched_req:
                st.success(", ".join(matched_req))
            else:
                st.info("No mandatory skills matched.")

            st.markdown(f"**❌ Missing Mandatory Skills ({len(missing_req)}):**")
            if missing_req:
                st.error(", ".join(missing_req))
            else:
                st.success("All mandatory skills matched!")

            if matched_pref or missing_pref:
                st.markdown(f"**🌟 Preferred Skills Coverage:** Matched: `{len(matched_pref)}` | Missing: `{len(missing_pref)}`")
                if matched_pref:
                    st.caption(f"Matched Preferred: {', '.join(matched_pref)}")


# ==========================================
# TAB 3: ANALYSIS & CHARTS
# ==========================================
with tab3:
    st.header("📊 Screening Analytics & Visualizations")

    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.info("No candidate data to visualize. Process resumes in Tab 1 to view charts.")
    else:
        df = st.session_state.processed_df
        required_skills = st.session_state.required_skills

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Resumes Processed", len(df))
        k2.metric("Mean Final Score", f"{df['final_score'].mean():.2f}%")
        k3.metric("Required Skills in JD", len(required_skills))
        all_detected_skills = [s for sublist in df["detected_skills"] for s in sublist]
        k4.metric("Unique Skills Across Batch", len(set(all_detected_skills)))

        st.markdown("---")

        c_left, c_right = st.columns([1, 1], gap="medium")
        with c_left:
            st.plotly_chart(plot_candidate_ranking(df), use_container_width=True)
        with c_right:
            st.plotly_chart(plot_score_breakdown(df), use_container_width=True)

        c_left2, c_right2 = st.columns([1, 1], gap="medium")
        with c_left2:
            st.plotly_chart(plot_skill_frequency(df, top_n=12), use_container_width=True)
        with c_right2:
            st.plotly_chart(plot_score_distribution(df), use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Candidate Skill Gap Donut Chart")
        selected_for_chart = st.selectbox(
            "Select Candidate for Gap Analysis",
            options=df["candidate_name"].tolist(),
            key="donut_select",
        )
        cand_chart_row = df[df["candidate_name"] == selected_for_chart].iloc[0]
        st.plotly_chart(plot_candidate_skill_gap(cand_chart_row), use_container_width=True)


# ==========================================
# TAB 4: REPORTS & EXPORT
# ==========================================
with tab4:
    st.header("📑 Reports & Data Export")
    st.markdown(
        "Generate and download structured reports in CSV, formatted Excel (.xlsx), "
        "and publication-ready PDF formats."
    )

    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.warning("No candidate ranking data available. Process resumes first before exporting.")
    else:
        df = st.session_state.processed_df
        rg = ReportGenerator()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"candidate_screening_report_{timestamp}"

        csv_bytes = rg.export_csv(df)
        excel_bytes = rg.export_excel(df)
        pdf_bytes = rg.export_pdf(df, job_title="Candidate Screening Evaluation")

        st.subheader("📥 Direct File Downloads")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_bytes,
                file_name=f"{base_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="📊 Download Excel (.xlsx) Report",
                data=excel_bytes,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with d3:
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{base_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("---")
        if st.button("Save All Formats to 'reports/' Directory"):
            saved_paths = []
            for ext, data in [("csv", csv_bytes), ("xlsx", excel_bytes), ("pdf", pdf_bytes)]:
                fname = f"{base_name}.{ext}"
                ok, path = rg.save_report(data, fname)
                if ok:
                    saved_paths.append(path)

            if len(saved_paths) == 3:
                st.session_state.report_save_message = (
                    f"Successfully saved 3 reports to disk:\n"
                    + "\n".join(f"• `{p}`" for p in saved_paths)
                )

        if st.session_state.report_save_message:
            st.success(st.session_state.report_save_message)

        st.markdown("---")
        st.subheader("👀 Export Preview")
        st.dataframe(df, use_container_width=True)


# ==========================================
# TAB 5: BENCHMARK & MODEL EVALUATION
# ==========================================
with tab5:
    st.header("📈 Model Evaluation & Dataset Audit")
    st.markdown(
        "Evaluate information retrieval ranking quality across benchmark queries and inspect "
        "the 2,484-resume PDF dataset audit metrics."
    )

    ev_col1, ev_col2 = st.columns([1, 1], gap="medium")

    with ev_col1:
        st.subheader("1. Ranking Quality Benchmark (IR Metrics)")
        st.caption(
            "Compares Baseline (TF-IDF + Skills) vs. Enhanced Strategy on held-out test queries."
        )

        eval_json_path = EVALUATION_DIR / "evaluation_results.json"
        if eval_json_path.exists():
            try:
                with open(eval_json_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                summary_m = eval_data.get("summary", {}).get("metrics_by_strategy", {})

                test_base = summary_m.get(STRATEGY_BASELINE, {}).get("test", {})
                test_enh = summary_m.get(STRATEGY_ENHANCED, {}).get("test", {})

                metric_display = [
                    {"Metric": "Precision@1", "Baseline": f"{test_base.get('mean_precision@1', 0):.4f}", "Enhanced": f"{test_enh.get('mean_precision@1', 0):.4f}"},
                    {"Metric": "Precision@3", "Baseline": f"{test_base.get('mean_precision@3', 0):.4f}", "Enhanced": f"{test_enh.get('mean_precision@3', 0):.4f}"},
                    {"Metric": "Recall@3", "Baseline": f"{test_base.get('mean_recall@3', 0):.4f}", "Enhanced": f"{test_enh.get('mean_recall@3', 0):.4f}"},
                    {"Metric": "Mean Reciprocal Rank (MRR)", "Baseline": f"{test_base.get('mrr', 0):.4f}", "Enhanced": f"{test_enh.get('mrr', 0):.4f}"},
                    {"Metric": "NDCG@3", "Baseline": f"{test_base.get('mean_ndcg@3', 0):.4f}", "Enhanced": f"{test_enh.get('mean_ndcg@3', 0):.4f}"},
                    {"Metric": "Pairwise Accuracy", "Baseline": f"{test_base.get('pairwise_accuracy', 0):.4f}", "Enhanced": f"{test_enh.get('pairwise_accuracy', 0):.4f}"},
                ]
                st.dataframe(pd.DataFrame(metric_display), use_container_width=True, hide_index=True)
                st.caption(f"Source: Synthetic graded relevance judgements (0-3 scale). N={test_base.get('query_count', 0)} test queries.")
            except Exception as e:
                st.warning(f"Could not load evaluation data: {e}")
        else:
            st.info("Evaluation results not found. Run scripts/evaluate_ranking.py to generate benchmarks.")

    with ev_col2:
        st.subheader("2. Dataset Audit Summary (2,484 PDFs)")
        st.caption("Cross-category audit summary from Kaggle Resume Dataset.")

        audit_json_path = LOCAL_AUDIT_DIR / "audit_summary.json"
        if audit_json_path.exists():
            try:
                with open(audit_json_path, "r", encoding="utf-8") as f:
                    audit_data = json.load(f)

                am1, am2, am3 = st.columns(3)
                am1.metric("Total In Dataset", audit_data.get("total_files_discovered", 2484))
                am2.metric("Audited Sample", audit_data.get("files_audited", 0))
                am3.metric("Parse Success Rate", f"{(audit_data.get('parse_success_count', 0) / max(1, audit_data.get('files_audited', 1))) * 100:.1f}%")

                cat_breakdown = audit_data.get("category_breakdown", {})
                if cat_breakdown:
                    cat_rows = []
                    for cname, cinfo in list(cat_breakdown.items())[:8]:
                        cat_rows.append({
                            "Category": cname,
                            "Audited": cinfo.get("total_files", 0),
                            "Success Rate": f"{cinfo.get('success_rate_pct', 100):.1f}%",
                            "Avg Chars": int(cinfo.get("mean_char_count", 0)),
                        })
                    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load audit summary: {e}")
        else:
            st.info("Audit summary not found. Run scripts/audit_resume_dataset.py to audit the dataset.")
