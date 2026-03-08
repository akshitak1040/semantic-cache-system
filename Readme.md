# Semantic Search System with Cluster-Aware Semantic Cache

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**.  
The system combines **transformer-based embeddings, fuzzy clustering, vector search, and a custom semantic cache** to efficiently retrieve relevant documents while avoiding redundant computation.

The project demonstrates how semantic similarity can be leveraged to build a cache that recognizes queries even when phrased differently.

## System Architecture

The system processes queries through several stages:

User Query  
↓  
SentenceTransformer Embedding  
↓  
Fuzzy Clustering (Gaussian Mixture Model)  
↓  
Cluster-Aware Semantic Cache  
↓  
Cache Hit → Return Result  
Cache Miss → FAISS Vector Search  
↓  
Store Result in Cache

## Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents** across **20 discussion categories** such as:

- computer graphics
- space science
- politics
- sports
- religion

The dataset provides a challenging testbed because documents often overlap across topics.

## Core Components
### Embeddings

Documents and queries are embedded using:

sentence-transformers/all-MiniLM-L6-v2

This model produces 384-dimensional semantic vectors that capture contextual similarity between texts.

### Vector Database

The system uses **FAISS** for efficient nearest neighbor search over document embeddings.

Index used:
IndexFlatL2
### Fuzzy Clustering

Documents are clustered using **Gaussian Mixture Models (GMM)**.

Unlike hard clustering, GMM assigns **probability distributions** across clusters.

Example:

Politics → 0.52  
Firearms → 0.41  
Other → 0.07
### Semantic Cache

A custom semantic cache was implemented without using Redis or other caching libraries.

Features:

- cosine similarity matching
- cluster-aware lookup
- LRU eviction policy

## Experimental Results
![Hit Rate](experiments/results/hit_rate_plot.png)
The best cache hit rate occurs around a similarity threshold of 0.80.

![Latency](experiments/results/latency_plot.png)

![Clusters](experiments/results/cluster_map.png)

## API Endpoints
POST /query
{
 "query": "techniques for realistic 3D rendering"
}
{
 "query": "...",
 "cache_hit": true,
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
GET /cache/stats
DELETE /cache

## Running the Project
pip install -r requirements.txt
uvicorn api.main:app --reload
http://localhost:8000/docs

docker build -t semantic-cache .
docker run -p 8000:8000 semantic-cache
http://localhost:8000/docs

## Future Work

Possible improvements include:

- using approximate FAISS indexes such as HNSW
- adaptive similarity thresholds per cluster
- online cluster updates as new documents are added