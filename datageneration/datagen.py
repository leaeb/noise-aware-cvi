import numpy as np
import os
import pandas as pd
from itertools import product
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

nmin = 300
K_values = [2, 3, 4]
dim_values = [2, 4, 8]
nl_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
std_values = [0.3, 0.7]

output_dir = "synthetic_datasets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_cluster_centers(K, dim, std_v, random_state):
    np.random.seed(random_state)
    centers = np.random.uniform(-8, 8, size=(K, dim))
    
    min_distance = std_v * 2
    while True:
        distances = cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        if np.all(distances >= min_distance):
            break
        else:
            centers = np.random.uniform(-8, 8, size=(K, dim))
    return centers

def create_clustered_datasets():
    for K, dim, nl, std_v in product(K_values, dim_values, nl_values, std_values):
        for dataset_num in range(5):
            centers = create_cluster_centers(K, dim, std_v, int(10 + K + dim + std_v * 10 + dataset_num))
            X, y = make_blobs(
                n_samples=nmin,
                centers=centers,
                n_features=dim,
                cluster_std=std_v,
                random_state=int(10 + K + dim + std_v * 10 + dataset_num),
            )
            data_min = np.min(X, axis=0)
            data_max = np.max(X, axis=0)
            noise_min = data_min - 0.3 * (data_max - data_min)
            noise_max = data_max + 0.3 * (data_max - data_min)

            n_noise_points = int(len(X) * nl * K)
            noise_points = np.random.uniform(
                noise_min, noise_max, size=(n_noise_points, dim)
            )

            distances = cdist(noise_points, X)
            nearest_cluster = np.argmin(distances, axis=1)
            noise_labels = np.full(n_noise_points, -1)
            threshold_distance = (
                std_v * 1.5  #distance threshold to consider noise point as part of a cluster
            )

            for i in range(n_noise_points):
                if distances[i, nearest_cluster[i]] < threshold_distance:
                    noise_labels[i] = y[nearest_cluster[i]]

            X = np.vstack([X, noise_points])
            y = np.concatenate([y, noise_labels])

            filename = f"K{K}_dim{dim}_nl{nl}_std{std_v}_set{dataset_num}.csv"
            filepath = os.path.join(output_dir, filename)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(dim)])
            df["label"] = y
            df.to_csv(filepath, index=False)

def create_uniform_datasets():
    noise_am=[0.1, 0.3, 0.5, 0.7, 0.9]
    n=1000
    for dim, na in product(dim_values, noise_am):
        for dataset_num in range(5):
            n_samples = int(n * na) 
            X = np.random.uniform(low=-9, high=9, size=(n_samples, dim))
            y = np.full(n_samples, -1)  

            filename = f"K0_dim{dim}_nl{na}_set{dataset_num}.csv"
            filepath = os.path.join(output_dir, filename)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(dim)])
            df["label"] = y
            df.to_csv(filepath, index=False)

#create_clustered_datasets()
#create_uniform_datasets()
