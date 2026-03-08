import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import umap
import matplotlib.pyplot as plt
from api.system import SemanticSystem


print("Loading system...")

system = SemanticSystem()
system.initialize()

embeddings = system.embeddings

print("Running UMAP...")

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)

embedding_2d = reducer.fit_transform(embeddings[:2000])

print("Plotting clusters...")

plt.figure()

plt.scatter(embedding_2d[:,0], embedding_2d[:,1], s=5)

plt.title("Semantic Structure of 20 Newsgroups")

plt.savefig("experiments/results/cluster_map.png")

plt.show()