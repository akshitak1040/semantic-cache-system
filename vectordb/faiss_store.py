import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension, use_hnsw=False):

        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.documents = []


    def add_documents(self, embeddings, docs):

        self.index.add(np.array(embeddings).astype("float32"))

        self.documents.extend(docs)


    def search(self, query_embedding, k=5):

        D, I = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )

        results = [self.documents[i] for i in I[0]]

        return results