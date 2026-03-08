import sys
import os
import time
import random
import csv

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import embed_query
from cache.semantic_cache import SemanticCache
from api.system import SemanticSystem


print("Initializing semantic system...")

system = SemanticSystem()
system.initialize()

print("System ready.\n")


# Use real dataset documents as query source
query_pool = system.docs[:500]


# Similarity thresholds to test
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# Number of queries to simulate
NUM_QUERIES = 200


results = []

for threshold in thresholds:

    print("Testing threshold:", threshold)

    cache = SemanticCache(threshold=threshold)

    hits = 0
    total = 0

    start_time = time.time()

    for i in range(NUM_QUERIES):

        # Simulate realistic query
        doc = random.choice(query_pool)
        query = doc[:120]

        embedding = embed_query(query)

        probs = system.clusterer.get_cluster_distribution(embedding)

        cluster_id = probs.argmax()

        hit, entry, sim = cache.lookup(embedding, cluster_id)

        if hit:

            hits += 1

        else:

            results_search = system.vector_store.search(embedding)

            cache.add(query, embedding, results_search[0], cluster_id)

        total += 1

    end_time = time.time()

    hit_rate = hits / total
    avg_latency = (end_time - start_time) / total

    print("Hit rate:", hit_rate)
    print("Average latency:", avg_latency, "\n")

    results.append({
        "threshold": threshold,
        "hit_rate": hit_rate,
        "avg_latency": avg_latency
    })


# Save results to CSV
os.makedirs("experiments/results", exist_ok=True)

csv_file = "experiments/results/cache_benchmark.csv"

with open(csv_file, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=["threshold", "hit_rate", "avg_latency"])

    writer.writeheader()

    for r in results:
        writer.writerow(r)


print("Benchmark complete.")
print("Results saved to:", csv_file)