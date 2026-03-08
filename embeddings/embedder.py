from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_documents(docs):
    """
    Generate embeddings for a list of documents
    """
    embeddings = model.encode(
        docs,
        show_progress_bar=True
    )
    return embeddings


def embed_query(query):
    """
    Generate embedding for a single query
    """
    return model.encode([query])[0]