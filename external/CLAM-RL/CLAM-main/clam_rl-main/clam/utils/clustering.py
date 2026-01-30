from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import hdbscan

def load_from_csv(path=""):
    data = pd.read_csv(path).to_numpy()

    return data

def silhouette_best(data, max_number=20):
    best_score = 0
    best_cluster = None
    best_cluster_std = None

    for n_clusters in range(2, max_number):
        print("testing cluster number",n_clusters)
        cluster = KMeans(n_clusters=n_clusters)
        cluster_labels = cluster.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        if best_score < silhouette_avg:
            best_score = silhouette_avg
            best_cluster = cluster

            # Calculate the standard deviation for each cluster
            cluster_std = []
            for cluster_id in range(n_clusters):
                cluster_data = data[cluster_labels == cluster_id]
                cluster_std.append(np.std(cluster_data, axis=0))

            best_cluster_std = cluster_std

    return best_cluster.cluster_centers_, best_cluster_std
# data=load_from_csv("../policy_features/100_percent_feature.csv")
# chosen_cluster=silhouette_best(data)
# print(chosen_cluster.)


def hdscan_best(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    results=clusterer.fit(data)
    cluster_labels = clusterer.labels_
    core_indices = np.where(cluster_labels != -1)[0]

    # Compute centroids of core points for each cluster
    cluster_centers = []
    cluster_std=[]
    for label in np.unique(cluster_labels):
        if label != -1:
            cluster_core_points = data[cluster_labels == label]
            centroid = np.mean(cluster_core_points, axis=0)
            cluster_centers.append(centroid)
            cluster_std.append(np.std(cluster_core_points, axis=0))
    return np.array(cluster_centers), cluster_std