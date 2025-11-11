import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# load numeric columns of diamonds
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include=[np.number])

def kmeans(X, k):
    """
    Perform k-means clustering on a numerical NumPy array X.

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_features)
        k (int): number of clusters

    Returns:
        tuple: (centroids, labels)
            - centroids: 2D array of shape (k, n_features)
            - labels: 1D array of shape (n_samples,) with cluster indices
    """
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels

def kmeans_diamonds(n, k):
    """
    Run k-means clustering on the first n rows of the diamonds numeric dataset.

    Parameters:
        n (int): number of rows to use
        k (int): number of clusters

    Returns:
        tuple: (centroids, labels) from kmeans
    """
    X = diamonds_numeric.iloc[:n].values
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds n_iter times and return the average runtime.

    Parameters:
        n (int): number of rows to use from diamonds dataset
        k (int): number of clusters
        n_iter (int, default=5): number of iterations to measure

    Returns:
        float: average runtime in seconds
    """
    runtimes = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        runtimes.append(time() - start)
    return np.mean(runtimes)