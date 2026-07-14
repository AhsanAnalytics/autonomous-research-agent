import sys
from sentence_transformers import CrossEncoder
from hybrid_search import keyword_ids, vector_ids, rrf, by_id

# A small, free, local cross-encoder trained to score query/passage relevance.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_candidates(query, n=8):
    """Retrieve WIDE with hybrid fusion: a big candidate pool to rerank."""
    fused = rrf([keyword_ids(query, n), vector_ids(query, n)])
    return fused[:n]


def rerank(query, candidate_ids, top_k=3):
    """Score each candidate against the query TOGETHER, keep the best."""
    pairs = [(query, by_id[cid].page_content) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "what are the rules about taking time off"
    candidates = hybrid_candidates(query)
    print(f"\nQuery: {query}")
    print("hybrid candidates (before rerank):", candidates, "\n")

    for rank, (cid, score) in enumerate(rerank(query, candidates), start=1):
        c = by_id[cid]
        preview = c.page_content.replace("\n", " ").strip()
        print(f"[{rank}] rerank_score={score:.2f}  chunk_id={cid}  ({c.metadata.get('source','?')})")
        print(f"    {preview[:160]}\n")
