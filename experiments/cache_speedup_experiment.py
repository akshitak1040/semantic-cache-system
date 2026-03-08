import time
import numpy as np

def benchmark_speedup(queries, search_fn, cache_fn):
    faiss_times = []
    cache_times = []

    for q in queries:

        start = time.time()
        search_fn(q)
        faiss_times.append(time.time() - start)

        start = time.time()
        cache_fn(q)
        cache_times.append(time.time() - start)

    avg_faiss = np.mean(faiss_times)
    avg_cache = np.mean(cache_times)

    speedup = avg_faiss / avg_cache

    print("Average FAISS latency:", avg_faiss)
    print("Average Cache latency:", avg_cache)
    print("Speedup:", speedup)