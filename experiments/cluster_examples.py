import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import embed_query
from api.system import SemanticSystem


system = SemanticSystem()
system.initialize()


docs = system.docs[:200]


for i in range(5):

    doc = random.choice(docs)

    embedding = embed_query(doc[:120])

    probs = system.clusterer.get_cluster_distribution(embedding)

    ranked = sorted(
        list(enumerate(probs)),
        key=lambda x: x[1],
        reverse=True
    )[:3]


    print("\nDocument snippet:")
    print(doc[:150])

    print("Cluster distribution:")

    for cluster, score in ranked:
        print("Cluster", cluster, ":", round(score, 3))