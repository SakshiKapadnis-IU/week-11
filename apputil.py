import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# load numeric columns of diamonds
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

def kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels

def kmeans_diamonds(n, k):
    X = diamonds_numeric.iloc[:n].values
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    runtimes = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        runtimes.append(time() - start)
    return np.mean(runtimes)