import sys
from rank_bm25 import BM25Okapi
from app.rag import load_and_chunk

# Build the keyword index from the SAME chunks the vector DB uses.
chunks = load_and_chunk()
corpus = [c.page_content for c in chunks]

# BM25 works on tokens (words). A simple lowercase + split is enough here.
tokenized_corpus = [text.lower().split() for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

question = " ".join(sys.argv[1:]) or "ERR_4521"
print(f"\nQuestion: {question}\n")

# Score every chunk against the query words, then take the top 3.
scores = bm25.get_scores(question.lower().split())
ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

for rank, i in enumerate(ranked[:3], start=1):
    src = chunks[i].metadata.get("source", "?")
    preview = chunks[i].page_content.replace("\n", " ").strip()
    print(f"[{rank}] score={scores[i]:.2f}  ({src})  chunk_id={chunks[i].metadata['chunk_id']}")
    print(f"    {preview[:160]}\n")
