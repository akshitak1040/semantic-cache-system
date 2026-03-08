from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from embeddings.embedder import embed_query
from cache.semantic_cache import SemanticCache
from api.system import SemanticSystem


# Initialize FastAPI app
app = FastAPI()


# Initialize core system and cache
system = SemanticSystem()
cache = SemanticCache()


# Request schema
class QueryRequest(BaseModel):
    query: str


# Startup event to initialize dataset, embeddings, FAISS, clustering
@app.on_event("startup")
def startup_event():

    print("Starting semantic search system...")

    system.initialize()

    print("System initialization complete.")


# Query endpoint
@app.post("/query")
def query_endpoint(req: QueryRequest):

    query = req.query

    # Generate query embedding
    embedding = embed_query(query)

    # Get fuzzy cluster probabilities
    probs = system.clusterer.get_cluster_distribution(embedding)

    # Select top 2 clusters
    top_clusters = np.argsort(probs)[-2:][::-1]

    cluster_id = int(top_clusters[0])

    hit = False
    entry = None
    sim = 0


    # Check cache in top clusters
    for cid in top_clusters:

        hit, entry, sim = cache.lookup(embedding, int(cid))

        if hit:
            cluster_id = int(cid)
            break


    # If cache hit
    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": query,
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": cluster_id
        }


    # Cache miss → perform FAISS search
    results = system.vector_store.search(embedding)

    result = results[0]


    # Store result in cache
    cache.add(query, embedding, result, cluster_id)


    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(sim),
        "result": result,
        "dominant_cluster": cluster_id
    }


# Cache statistics endpoint
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# Clear cache endpoint
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}