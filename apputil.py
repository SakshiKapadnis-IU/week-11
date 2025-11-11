# FULL K-MEANS ASSIGNMENT SCRIPT — RUN DIRECTLY OR IMPORT

# ---- install missing packages ----
import subprocess, sys

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ["numpy", "scikit-learn", "seaborn", "matplotlib"]:
    ensure(p)

# ---- imports ----
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time

# ---- load numeric diamonds ----
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

# ---- functions ----
def kmeans(X, k):
    model = KMeans(n_clusters=k, n_init="auto", random_state=0)
    model.fit(X)
    return model.cluster_centers_, model.labels_

def kmeans_diamonds(n, k):
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    times = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        times.append(time() - start)
    return np.mean(times)

# ---- RUN PLOTS WHEN EXECUTED DIRECTLY ----
if __name__ == "__main__":
    print("Running k-means timing tests... this may take a minute.")

    sns.set_theme(style="whitegrid")
    n_values = np.arange(100, 50000, 1000)
    k5_times = [kmeans_timer(n, 5, 20) for n in n_values]

    k_values = np.arange(2, 50)
    n10k_times = [kmeans_timer(10000, k, 10) for k in k_values]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.tight_layout()
    fig.suptitle("KMeans Time Complexity", y=1.08, fontsize=14)

    sns.lineplot(x=n_values, y=k5_times, ax=axes[0])
    axes[0].set_xlabel("Number of Rows (n)")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Increasing n for k=5 Clusters")

    sns.lineplot(x=k_values, y=n10k_times, ax=axes[1])
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_title("Increasing k for n=10k Samples")

    plt.show()

    print("✅ Done! Plot displayed.")