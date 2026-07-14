import os
from langchain_huggingface import HuggingFaceEmbeddings

# Central RAG config. The index script and the query script both import
# from here, so they are guaranteed to use the SAME settings.
PERSIST_DIR = "chroma_db"        # folder where the vector DB lives on disk
COLLECTION = "knowledge"          # name of the collection inside it
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def get_embeddings():
    """The embedding model that turns text into meaning-vectors.

    Must be identical for indexing and querying, so both import it here.
    normalize_embeddings=True makes similarity scores behave well.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
