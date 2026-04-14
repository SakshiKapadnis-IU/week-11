import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import seaborn as sns


# Exercise 1
def kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# Exercise 2
diamonds = sns.load_dataset("diamonds")

diamonds_numeric = diamonds.select_dtypes(include=[np.number])


def kmeans_diamonds(n, k):
    data = diamonds_numeric.iloc[:n].values
    return kmeans(data, k)


# Exercise 3
def kmeans_timer(n, k, n_iter=5):
    times = []

    for _ in range(n_iter):
        start = time.time()
        kmeans_diamonds(n, k)
        end = time.time()

        times.append(end - start)

    return sum(times) / len(times)