"""
Information-retrieval ranking evaluation metrics and benchmark harness.

Implements standard ranking evaluation metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Pairwise ordering accuracy
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple


def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@K: fraction of retrieved items in top-K that are relevant.

    Args:
        ranked_ids: Ordered list of candidate IDs produced by ranker.
        relevant_ids: Set of relevant candidate IDs (binary relevance).
        k: Cutoff rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    matched = sum(1 for cid in top_k if cid in relevant_ids)
    return round(matched / k, 4)


def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K: fraction of all relevant items retrieved in top-K.

    Args:
        ranked_ids: Ordered list of candidate IDs produced by ranker.
        relevant_ids: Set of relevant candidate IDs.
        k: Cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not relevant_ids or k <= 0 or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    matched = sum(1 for cid in top_k if cid in relevant_ids)
    return round(matched / len(relevant_ids), 4)


def reciprocal_rank(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the first relevant item.

    Args:
        ranked_ids: Ordered list of candidate IDs produced by ranker.
        relevant_ids: Set of relevant candidate IDs.

    Returns:
        1/rank of first hit, or 0.0 if no relevant items retrieved.
    """
    for rank_idx, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return round(1.0 / rank_idx, 4)
    return 0.0


def dcg_at_k(ranked_ids: List[str], graded_relevance: Dict[str, float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain (DCG@K) using exponential gain: (2^rel - 1) / log2(i + 1).

    Args:
        ranked_ids: Ordered candidate IDs.
        graded_relevance: Mapping of candidate_id -> relevance grade (e.g. 0 to 3).
        k: Cutoff rank.

    Returns:
        DCG score.
    """
    dcg = 0.0
    top_k = ranked_ids[:k]
    for i, cid in enumerate(top_k, start=1):
        rel = graded_relevance.get(cid, 0.0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1.0)
        dcg += gain / discount
    return dcg


def ndcg_at_k(ranked_ids: List[str], graded_relevance: Dict[str, float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        ranked_ids: Ordered candidate IDs.
        graded_relevance: Mapping of candidate_id -> relevance grade (0-3 scale).
        k: Cutoff rank.

    Returns:
        NDCG score in [0.0, 1.0].
    """
    if k <= 0 or not ranked_ids or not graded_relevance:
        return 0.0

    actual_dcg = dcg_at_k(ranked_ids, graded_relevance, k)

    # Ideal ranking: candidates sorted by true graded relevance descending
    ideal_order = sorted(
        graded_relevance.keys(),
        key=lambda cid: graded_relevance[cid],
        reverse=True,
    )
    idcg = dcg_at_k(ideal_order, graded_relevance, k)

    if idcg <= 0.0:
        return 0.0

    return round(min(1.0, actual_dcg / idcg), 4)


def pairwise_accuracy(
    ranked_ids: List[str],
    graded_relevance: Dict[str, float],
) -> Tuple[float, int, int]:
    """
    Compute pairwise ordering accuracy across all pairs with different relevance levels.

    Args:
        ranked_ids: Ordered candidate IDs.
        graded_relevance: Mapping of candidate_id -> relevance grade.

    Returns:
        Tuple of (accuracy, concordant_pairs_count, total_pairs_count).
    """
    pos_map = {cid: idx for idx, cid in enumerate(ranked_ids)}
    candidates = [cid for cid in ranked_ids if cid in graded_relevance]

    total_pairs = 0
    concordant = 0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            r1 = graded_relevance[c1]
            r2 = graded_relevance[c2]

            if r1 == r2:
                continue

            total_pairs += 1
            # If r1 > r2, c1 should be ranked before c2 (pos_map[c1] < pos_map[c2])
            if (r1 > r2 and pos_map[c1] < pos_map[c2]) or (r2 > r1 and pos_map[c2] < pos_map[c1]):
                concordant += 1

    if total_pairs == 0:
        return 1.0, 0, 0

    return round(concordant / total_pairs, 4), concordant, total_pairs
