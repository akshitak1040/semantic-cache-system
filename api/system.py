import os
import numpy as np
import pickle

from preprocessing.load_dataset import load_documents
from embeddings.embedder import embed_documents
from vectordb.faiss_store import VectorStore
from clustering.fuzzy_cluster import FuzzyClusterer


class SemanticSystem:

    def __init__(self):

        self.docs = None
        self.embeddings = None
        self.vector_store = None
        self.clusterer = None

        self.embedding_file = "data/embeddings.npy"
        self.docs_file = "data/docs.pkl"


    def initialize(self):

        print("Initializing semantic system...")

        if os.path.exists(self.embedding_file) and os.path.exists(self.docs_file):

            print("Loading cached embeddings...")

            self.embeddings = np.load(self.embedding_file)

            with open(self.docs_file, "rb") as f:
                self.docs = pickle.load(f)

        else:

            print("Loading dataset...")

            self.docs = load_documents()

            print("Generating embeddings...")

            self.embeddings = embed_documents(self.docs)

            os.makedirs("data", exist_ok=True)

            np.save(self.embedding_file, self.embeddings)

            with open(self.docs_file, "wb") as f:
                pickle.dump(self.docs, f)

            print("Embeddings saved to disk.")

        print("Building vector database...")

        dimension = len(self.embeddings[0])

        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(self.embeddings, self.docs)

        print("Training fuzzy clustering...")

        self.clusterer = FuzzyClusterer()

        self.clusterer.fit(self.embeddings)

        print("System ready.")