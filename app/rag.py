import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Central RAG config: everything that must stay consistent across scripts.
DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION = "knowledge"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def get_embeddings():
    """The embedding model. Must be identical for indexing and querying."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_chunk():
    """Load every .md file in data/, split into small chunks, tag each with a
    stable chunk_id and source. The indexer AND the keyword/hybrid search all
    import this, so they always operate over the exact same chunks."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        docs.extend(TextLoader(path, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # smaller than M3 so each section is its own chunk
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Deterministic order -> stable id, so keyword and vector hits can be matched.
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = str(i)
    return chunks
