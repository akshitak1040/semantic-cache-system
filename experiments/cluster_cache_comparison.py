import sys
import os
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import embed_query
from cache.semantic_cache import SemanticCache
from api.system import SemanticSystem


print("Initializing system...")

system = SemanticSystem()
system.initialize()

queries = system.docs[:500]


NUM_QUERIES = 200


# Flat cache (no clustering)
flat_cache = SemanticCache(threshold=0.8)

# Cluster-aware cache
cluster_cache = SemanticCache(threshold=0.8)


flat_hits = 0
cluster_hits = 0


flat_start = time.time()

for i in range(NUM_QUERIES):

    doc = random.choice(queries)
    query = doc[:120]

    embedding = embed_query(query)

    # FLAT CACHE LOOKUP
    hit, entry, sim = flat_cache.lookup(embedding, 0)

    if hit:
        flat_hits += 1
    else:
        results = system.vector_store.search(embedding)
        flat_cache.add(query, embedding, results[0], 0)

flat_end = time.time()


cluster_start = time.time()

for i in range(NUM_QUERIES):

    doc = random.choice(queries)
    query = doc[:120]

    embedding = embed_query(query)

    probs = system.clusterer.get_cluster_distribution(embedding)

    cluster_id = probs.argmax()

    hit, entry, sim = cluster_cache.lookup(embedding, cluster_id)

    if hit:
        cluster_hits += 1
    else:
        results = system.vector_store.search(embedding)
        cluster_cache.add(query, embedding, results[0], cluster_id)

cluster_end = time.time()


print("\nFlat Cache Results")
print("Hit rate:", flat_hits / NUM_QUERIES)
print("Time:", flat_end - flat_start)

print("\nCluster Cache Results")
print("Hit rate:", cluster_hits / NUM_QUERIES)
print("Time:", cluster_end - cluster_start)