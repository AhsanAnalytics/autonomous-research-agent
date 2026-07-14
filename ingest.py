from langchain_chroma import Chroma
from app.rag import get_embeddings, load_and_chunk, PERSIST_DIR, COLLECTION

chunks = load_and_chunk()
sources = sorted({c.metadata.get("source", "?") for c in chunks})
print(f"Prepared {len(chunks)} chunks from {len(sources)} document(s): {sources}")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    ids=[c.metadata["chunk_id"] for c in chunks],
)
print(f"Indexed {len(chunks)} chunks into {PERSIST_DIR}/. Done.")
