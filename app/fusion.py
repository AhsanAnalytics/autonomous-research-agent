def reciprocal_rank_fusion(result_lists, k: int = 60):
    """Fuse multiple ranked lists of ids into one, by RANK (not raw score).

    Each id earns 1 / (k + rank + 1) points per list it appears in (rank is
    0-based), and results are sorted by total points, highest first. This is how
    keyword (BM25) and vector rankings get combined into one hybrid ranking.
    """
    scores = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)