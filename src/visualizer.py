"""
Plotly visualization components for candidate screening analytics.

Generates interactive charts for:
1. Candidate final score ranking
2. Similarity vs. Skill Match component breakdown
3. Skill frequency distribution
4. Final score distribution
5. Candidate skill gap donut chart
"""

from collections import Counter
from typing import Any, Dict, List, Union

import pandas as pd
import plotly.graph_objects as go


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a clear informational message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6B7280"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig


def plot_candidate_ranking(df: pd.DataFrame) -> go.Figure:
    """
    Generate horizontal bar chart of candidates ranked by final score.

    Args:
        df: DataFrame containing 'candidate_name', 'final_score', and 'rank'.

    Returns:
        Plotly Figure.
    """
    if df is None or df.empty or "final_score" not in df.columns:
        return _empty_figure("No candidate ranking data available.")

    # Sort ascending for horizontal bar chart display (highest on top)
    plot_df = df.sort_values(by="final_score", ascending=True)

    names = (
        plot_df["candidate_name"].tolist()
        if "candidate_name" in plot_df.columns
        else [f"Candidate {i + 1}" for i in range(len(plot_df))]
    )
    scores = plot_df["final_score"].tolist()
    ranks = (
        plot_df["rank"].tolist()
        if "rank" in plot_df.columns
        else list(range(1, len(plot_df) + 1))
    )

    hover_text = [
        f"<b>Rank {r}: {name}</b><br>Final Score: {score:.2f}%"
        for r, name, score in zip(ranks, names, scores)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=names,
                orientation="h",
                text=[f"{s:.1f}%" for s in scores],
                textposition="auto",
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(
                    color=scores,
                    colorscale="Blues",
                    cmin=0,
                    cmax=100,
                    line=dict(color="#1E3A8A", width=1),
                ),
            )
        ]
    )

    fig.update_layout(
        title="<b>Candidate Final Score Ranking</b>",
        xaxis=dict(title="Final Score (%)", range=[0, 105], dtick=20),
        yaxis=dict(title="Candidate", autorange="reversed" if False else True),
        height=max(380, len(df) * 45),
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_score_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Generate grouped comparison bar chart of similarity score vs skill match score.

    Args:
        df: DataFrame containing 'candidate_name', 'similarity_score', 'skill_match_score'.

    Returns:
        Plotly Figure.
    """
    if df is None or df.empty:
        return _empty_figure("No score data available for comparison.")

    # Sort descending by final score for consistent order
    plot_df = (
        df.sort_values(by="final_score", ascending=False)
        if "final_score" in df.columns
        else df.copy()
    )
    names = (
        plot_df["candidate_name"].tolist()
        if "candidate_name" in plot_df.columns
        else [f"Candidate {i + 1}" for i in range(len(plot_df))]
    )
    sim_scores = (
        plot_df["similarity_score"].tolist()
        if "similarity_score" in plot_df.columns
        else [0.0] * len(plot_df)
    )
    skill_scores = (
        plot_df["skill_match_score"].tolist()
        if "skill_match_score" in plot_df.columns
        else [0.0] * len(plot_df)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="TF-IDF Similarity",
            x=names,
            y=sim_scores,
            marker_color="#3B82F6",
            text=[f"{s:.1f}%" for s in sim_scores],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>TF-IDF Similarity: %{y:.2f}%<extra></extra>",
        )
    )

    # Optional Semantic Similarity bar if enhanced mode is active
    if "semantic_similarity_score" in plot_df.columns and plot_df["semantic_similarity_score"].sum() > 0:
        sem_scores = plot_df["semantic_similarity_score"].tolist()
        fig.add_trace(
            go.Bar(
                name="Semantic Similarity",
                x=names,
                y=sem_scores,
                marker_color="#8B5CF6",
                text=[f"{s:.1f}%" for s in sem_scores],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Semantic Similarity: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Bar(
            name="Skill Match Score",
            x=names,
            y=skill_scores,
            marker_color="#10B981",
            text=[f"{s:.1f}%" for s in skill_scores],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Skill Match Score: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        title="<b>Score Component Breakdown (Similarity vs. Skill Match)</b>",
        xaxis=dict(title="Candidate", tickangle=-25),
        yaxis=dict(title="Score (%)", range=[0, 105], dtick=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=20, r=20, t=60, b=60),
        template="plotly_white",
    )
    return fig


def plot_skill_frequency(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Generate bar chart of most frequently detected skills across uploaded resumes.

    Args:
        df: DataFrame containing 'detected_skills' column.
        top_n: Number of top skills to display (default 15).

    Returns:
        Plotly Figure.
    """
    if df is None or df.empty or "detected_skills" not in df.columns:
        return _empty_figure("No detected skills data available.")

    all_detected: List[str] = []
    for skill_list in df["detected_skills"]:
        if isinstance(skill_list, list):
            all_detected.extend(skill_list)

    if not all_detected:
        return _empty_figure("No technical skills were detected in uploaded resumes.")

    counts = Counter(all_detected).most_common(top_n)
    # Reverse so top skill is at the top of horizontal chart
    counts.reverse()

    skills = [c[0] for c in counts]
    frequencies = [c[1] for c in counts]

    fig = go.Figure(
        data=[
            go.Bar(
                x=frequencies,
                y=skills,
                orientation="h",
                text=frequencies,
                textposition="auto",
                marker=dict(
                    color=frequencies,
                    colorscale="Teal",
                    line=dict(color="#047857", width=1),
                ),
                hovertemplate="<b>%{y}</b><br>Occurrences: %{x} candidates<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"<b>Top {len(skills)} Most Frequently Detected Skills</b>",
        xaxis=dict(title="Number of Resumes", dtick=1),
        yaxis=dict(title="Skill"),
        height=max(380, len(skills) * 26),
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Generate distribution histogram of candidate final scores.

    Args:
        df: DataFrame containing 'final_score'.

    Returns:
        Plotly Figure.
    """
    if df is None or df.empty or "final_score" not in df.columns:
        return _empty_figure("No score data available for distribution.")

    scores = df["final_score"].dropna().tolist()
    if not scores:
        return _empty_figure("No valid score records found.")

    fig = go.Figure(
        data=[
            go.Histogram(
                x=scores,
                xbins=dict(start=0, end=100, size=10),
                marker=dict(
                    color="#6366F1",
                    line=dict(color="#4338CA", width=1),
                ),
                hovertemplate="Score Range: %{x}%<br>Candidate Count: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="<b>Distribution of Candidate Final Scores</b>",
        xaxis=dict(title="Final Score Range (%)", range=[0, 105], dtick=10),
        yaxis=dict(title="Number of Candidates", dtick=1),
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_candidate_skill_gap(
    candidate_row: Union[pd.Series, Dict[str, Any]]
) -> go.Figure:
    """
    Generate donut chart of matched vs missing required skills for a single candidate.

    Args:
        candidate_row: Series or Dict with 'candidate_name', 'matched_skills', 'missing_skills', 'skill_match_score'.

    Returns:
        Plotly Figure.
    """
    if candidate_row is None:
        return _empty_figure("No candidate selected.")

    if isinstance(candidate_row, pd.Series):
        cand_dict = candidate_row.to_dict()
    else:
        cand_dict = candidate_row

    name = cand_dict.get("candidate_name", "Candidate")
    matched = cand_dict.get("matched_skills", [])
    missing = cand_dict.get("missing_skills", [])
    skill_score = cand_dict.get("skill_match_score", 0.0)

    matched_count = len(matched) if isinstance(matched, list) else 0
    missing_count = len(missing) if isinstance(missing, list) else 0

    if matched_count == 0 and missing_count == 0:
        return _empty_figure(
            f"No required skills detected in job description for {name}."
        )

    matched_text = "<br>".join(matched[:8]) + ("..." if len(matched) > 8 else "")
    missing_text = "<br>".join(missing[:8]) + ("..." if len(missing) > 8 else "")

    labels = ["Matched Required Skills", "Missing Required Skills"]
    values = [matched_count, missing_count]
    colors = ["#10B981", "#EF4444"]
    hover_details = [
        f"<b>Matched ({matched_count})</b><br>{matched_text if matched_text else 'None'}",
        f"<b>Missing ({missing_count})</b><br>{missing_text if missing_text else 'None'}",
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+value",
                hoverinfo="text",
                hovertext=hover_details,
            )
        ]
    )

    fig.update_layout(
        title=f"<b>Required Skills Coverage: {name}</b>",
        annotations=[
            dict(
                text=f"<b>{skill_score:.1f}%</b><br>Match",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig
