import sys
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from app.rag import load_and_chunk, get_embeddings, PERSIST_DIR, COLLECTION

TOP_N = 5  # candidates each method contributes to the fusion

chunks = load_and_chunk()
by_id = {c.metadata["chunk_id"]: c for c in chunks}

# Keyword index (BM25) over the chunk texts.
tokenized = [c.page_content.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

# Vector store handle (the persisted Chroma DB from ingest.py).
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
)


def rrf(result_lists, k: int = 60):
    """Fuse ranked lists of chunk_ids by RANK, not raw score."""
    scores = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def keyword_ids(query, n=TOP_N):
    s = bm25.get_scores(query.lower().split())
    order = sorted(range(len(chunks)), key=lambda i: s[i], reverse=True)
    return [chunks[i].metadata["chunk_id"] for i in order[:n]]


def vector_ids(query, n=TOP_N):
    hits = vectordb.similarity_search(query, k=n)
    return [h.metadata["chunk_id"] for h in hits]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "how do I fix the DHCP error on my router"
    kw = keyword_ids(query)
    vec = vector_ids(query)
    fused = rrf([kw, vec])

    print(f"\nQuery: {query}\n")
    print("keyword ranking:", kw)
    print("vector  ranking:", vec)
    print("fused   ranking:", fused[:5], "\n")

    for rank, cid in enumerate(fused[:3], start=1):
        c = by_id[cid]
        preview = c.page_content.replace("\n", " ").strip()
        print(f"[{rank}] chunk_id={cid}  ({c.metadata.get('source','?')})")
        print(f"    {preview[:160]}\n")
