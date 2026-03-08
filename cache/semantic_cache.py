from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85, max_size=500):

        self.threshold = threshold
        self.max_size = max_size

        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0


    def lookup(self, query_embedding, cluster_id):

        if cluster_id not in self.cache:

            self.miss_count += 1
            return False, None, 0


        cluster_cache = self.cache[cluster_id]

        best_sim = 0
        best_entry = None


        for query, entry in cluster_cache.items():

            sim = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > best_sim:

                best_sim = sim
                best_entry = entry


        if best_sim > self.threshold:

            self.hit_count += 1
            return True, best_entry, best_sim


        self.miss_count += 1
        return False, None, best_sim


    def add(self, query, embedding, result, cluster_id):

        if cluster_id not in self.cache:

            self.cache[cluster_id] = OrderedDict()


        cluster_cache = self.cache[cluster_id]


        if len(cluster_cache) >= self.max_size:

            cluster_cache.popitem(last=False)


        cluster_cache[query] = {

            "embedding": embedding,
            "result": result
        }


    def stats(self):

        total = self.hit_count + self.miss_count

        rate = self.hit_count / total if total > 0 else 0


        return {

            "total_entries": sum(len(v) for v in self.cache.values()),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": rate
        }


    def clear(self):

        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0