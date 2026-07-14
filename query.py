import sys
from langchain_chroma import Chroma
from app.rag import get_embeddings, PERSIST_DIR, COLLECTION

# Open the EXISTING vector database (note: we do NOT re-index here).
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    embedding_function=get_embeddings(),
)

# Turn the DB into a retriever that returns the top 3 closest chunks.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Take the question from the command line, or use a default.
question = " ".join(sys.argv[1:]) or "How many days off do I get?"
print(f"\nQuestion: {question}\n")

hits = retriever.invoke(question)
for i, h in enumerate(hits, start=1):
    source = h.metadata.get("source", "?")
    preview = h.page_content.replace("\n", " ").strip()
    print(f"[{i}] ({source})")
    print(f"    {preview[:200]}...\n")
