import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv("experiments/results/cache_benchmark.csv")

print(df)

# Plot hit rate vs threshold
plt.figure()

plt.plot(df["threshold"], df["hit_rate"], marker="o")

plt.title("Cache Hit Rate vs Similarity Threshold")

plt.xlabel("Similarity Threshold")

plt.ylabel("Hit Rate")

plt.grid()

plt.savefig("experiments/results/hit_rate_plot.png")

plt.show()