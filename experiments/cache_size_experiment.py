import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import embed_query
from cache.semantic_cache import SemanticCache
from api.system import SemanticSystem


system = SemanticSystem()
system.initialize()


queries = system.docs[:500]

cache_sizes = [50, 100, 200, 500]


for size in cache_sizes:

    print("\nTesting cache size:", size)

    cache = SemanticCache(threshold=0.8, max_size=size)

    hits = 0
    total = 200

    for i in range(total):

        doc = random.choice(queries)
        query = doc[:120]

        embedding = embed_query(query)

        probs = system.clusterer.get_cluster_distribution(embedding)

        cluster_id = probs.argmax()

        hit, entry, sim = cache.lookup(embedding, cluster_id)

        if hit:
            hits += 1
        else:
            results = system.vector_store.search(embedding)
            cache.add(query, embedding, results[0], cluster_id)

    print("Hit rate:", hits / total)