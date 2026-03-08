# Semantic Search System with Cluster-Aware Semantic Cache

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**.

The system combines **transformer-based embeddings, fuzzy clustering, vector search, and a custom semantic cache** to efficiently retrieve relevant documents while avoiding redundant computation.

The project demonstrates how semantic similarity can be leveraged to build a cache that recognizes queries even when phrased differently.

---

# Quick Start

## Run Locally

Install dependencies:
# Semantic Search System with Cluster-Aware Semantic Cache

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**.

The system combines **transformer-based embeddings, fuzzy clustering, vector search, and a custom semantic cache** to efficiently retrieve relevant documents while avoiding redundant computation.

The project demonstrates how semantic similarity can be leveraged to build a cache that recognizes queries even when phrased differently.

---

# Quick Start

## Run Locally

Install dependencies:
# Semantic Search System with Cluster-Aware Semantic Cache

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**.

The system combines **transformer-based embeddings, fuzzy clustering, vector search, and a custom semantic cache** to efficiently retrieve relevant documents while avoiding redundant computation.

The project demonstrates how semantic similarity can be leveraged to build a cache that recognizes queries even when phrased differently.

---

# Quick Start

## Run Locally

Install dependencies:
pip install -r requirements.txt

Start the FastAPI server:

uvicorn api.main:app --reload

Open the API documentation:
http://localhost:8000/docs

---

## Run with Docker

Build the container:
docker build -t semantic-cache .

Run the container:
docker run -p 8000:8000 semantic-cache

Open:
http://localhost:8000/docs


---

# System Architecture

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


Instead of matching queries exactly, the system detects **semantic similarity between queries** and reuses cached results.

---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents** across **20 discussion categories** such as:

- computer graphics
- space science
- politics
- sports
- religion

The dataset provides a challenging testbed because documents often overlap across topics.

---

# Core Components

## Embeddings

Documents and queries are embedded using:

Instead of matching queries exactly, the system detects **semantic similarity between queries** and reuses cached results.

---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents** across **20 discussion categories** such as:

- computer graphics
- space science
- politics
- sports
- religion

The dataset provides a challenging testbed because documents often overlap across topics.

---

# Core Components

## Embeddings

Documents and queries are embedded using:
sentence-transformers/all-MiniLM-L6-v2

This model produces **384-dimensional semantic vectors** that capture contextual similarity between texts.

---

## Vector Database

The system uses **FAISS (Facebook AI Similarity Search)** for efficient nearest neighbor search over document embeddings.

Index used:
IndexFlatL2

FAISS allows fast similarity search across thousands of document embeddings.

---

## Fuzzy Clustering

Documents are clustered using **Gaussian Mixture Models (GMM)**.

Unlike hard clustering methods (such as KMeans), GMM assigns **probability distributions across clusters**, allowing documents to belong to multiple topics.

Example cluster membership:
Politics → 0.52
Firearms → 0.41
Other → 0.07

This reflects the real-world nature of discussions where topics often overlap.

---

## Semantic Cache

A custom semantic cache was implemented **without using Redis or external caching libraries**.

Features:

- cosine similarity matching
- cluster-aware lookup
- LRU eviction policy
- reuse of semantically similar query results

This allows the system to recognize **similar queries even when phrased differently**.

---

# Experimental Results

Experiments were conducted to analyze how **similarity thresholds affect cache behavior**.

## Cache Hit Rate vs Similarity Threshold

![Hit Rate](experiments/results/hit_rate_plot.png)

Observation:

The best cache hit rate occurs around a **similarity threshold of 0.80**, balancing precision and cache reuse.

---

## Query Latency vs Similarity Threshold

![Latency](experiments/results/latency_plot.png)

Lower thresholds allow more cache reuse, reducing the number of expensive vector searches and improving query latency.

---

## Semantic Cluster Visualization

![Clusters](experiments/results/cluster_map.png)

A **UMAP projection** of the embedding space reveals natural semantic groupings within the dataset, including clusters corresponding to topics such as:

- space discussions
- computer graphics
- politics
- religion

---

# API Endpoints

The system exposes a **FastAPI service**.

## Query Endpoint
POST /query

Example request:

```json
{
  "query": "techniques for realistic 3D rendering"
}
Example response:
{
  "query": "...",
  "cache_hit": true,
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
Cache Statistics
GET /cache/stats
Returns:

total entries

hit count

miss count

hit rate
Clear Cache
DELETE /cache
Clears the cache and resets statistics.

project structure

semantic-cache-system
│
├── api
├── cache
├── clustering
├── embeddings
├── preprocessing
├── vectordb
│
├── experiments
│   ├── benchmark.py
│   ├── plot_results.py
│   ├── visualize_clusters.py
│   └── results
│
├── Dockerfile
├── requirements.txt
└── README.md