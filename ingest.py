from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.rag import get_embeddings, PERSIST_DIR, COLLECTION

# 1. LOAD: read the source document into memory.
docs = TextLoader("data/company_handbook.md", encoding="utf-8").load()
print(f"Loaded {len(docs)} document(s).")

# 2. CHUNK: split into small, overlapping passages.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # ~a paragraph; small enough to be specific
    chunk_overlap=80,      # repeat a little across boundaries so nothing is lost
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer to break on natural seams
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# 3 + 4. EMBED each chunk and STORE the vectors in Chroma (saved to disk).
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
)
print(f"Indexed {len(chunks)} chunks into {PERSIST_DIR}/. Done.")
