import numpy as np
from sklearn.datasets import make_blobs
import json
import copy

def def_dataset(n_samples,centers, cluster_std=2):
    data_blobs = make_blobs(n_samples=n_samples, centers=centers, cluster_std = cluster_std, random_state=42)[0]
    return data_blobs+np.abs(np.min(data_blobs))