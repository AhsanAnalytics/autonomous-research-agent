from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from app.rag import load_and_chunk, get_embeddings, PERSIST_DIR, COLLECTION

# Load models/indexes ONCE at import; reused across every call (fast after warm-up).
_chunks = load_and_chunk()
_by_id = {c.metadata["chunk_id"]: c for c in _chunks}
_bm25 = BM25Okapi([c.page_content.lower().split() for c in _chunks])
_vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _rrf(result_lists, k: int = 60):
    scores = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(query: str, k: int = 4, candidates: int = 8):
    """Hybrid (BM25 + vector) retrieval with cross-encoder reranking.
    Returns the top-k chunk Documents, most relevant first. This is the whole
    M4 stack behind one call."""
    kw_scores = _bm25.get_scores(query.lower().split())
    kw_order = sorted(range(len(_chunks)), key=lambda i: kw_scores[i], reverse=True)
    kw_ids = [_chunks[i].metadata["chunk_id"] for i in kw_order[:candidates]]

    vec_ids = [h.metadata["chunk_id"] for h in _vectordb.similarity_search(query, k=candidates)]

    fused = _rrf([kw_ids, vec_ids])[:candidates]

    pairs = [(query, _by_id[cid].page_content) for cid in fused]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)
    return [_by_id[cid] for cid, _ in ranked[:k]]
