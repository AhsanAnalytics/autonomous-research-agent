import sys
from langchain_chroma import Chroma
from app.rag import get_embeddings, PERSIST_DIR, COLLECTION

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
)

# A query that could match BOTH documents ("register" appears in each context).
query = " ".join(sys.argv[1:]) or "how do I register"

print(f"\nQuery: {query}")

# 1. Unfiltered: search the whole knowledge base.
print("\n-- no filter (all documents) --")
for h in vectordb.similarity_search(query, k=2):
    print(f"   ({h.metadata['source']}) {h.page_content[:90].strip()}")

# 2. Filtered: restrict to the product specs ONLY, then rank by similarity.
print("\n-- filtered to product_specs.md only --")
hits = vectordb.similarity_search(
    query,
    k=2,
    filter={"source": "data\\product_specs.md"},   # hard filter on metadata
)
for h in hits:
    print(f"   ({h.metadata['source']}) {h.page_content[:90].strip()}")
