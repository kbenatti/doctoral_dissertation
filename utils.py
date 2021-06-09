import numpy as np
from sklearn.datasets import make_blobs
import json
import copy

def def_dataset(n_samples,centers, cluster_std=2):
    data_blobs = make_blobs(n_samples=n_samples, centers=centers, cluster_std = cluster_std, random_state=42)[0]
    return data_blobs+np.abs(np.min(data_blobs))

def def_color(X,labels,clusters):
    centroides = [X[labels==i].mean(axis=0) for i in range(clusters)]
    mapping_color = {}
    restantes = list(range(clusters))
    if clusters == 3:
        for point, color in [(np.array((8,20)),'b'), (np.array((16,15)),'r'),
                             (np.array((4,5)),'g')]:
            ind = np.argmin(np.linalg.norm([centroides[r] for r in restantes] - point, axis=1))
            mapping_color[restantes[ind]] = color
            restantes.remove(restantes[ind])
    elif clusters == 4:
        for point, color in [(np.array((5,20)),'b'), (np.array((10,22)),'r'),
                             (np.array((5,8)),'g'),(np.array((17,15)),'c')]:
            ind = np.argmin(np.linalg.norm([centroides[r] for r in restantes] - point, axis=1))
            mapping_color[restantes[ind]] = color
            restantes.remove(restantes[ind]) 
    elif clusters == 5:
        for point, color in [(np.array((6,24)),'b'), (np.array((14,25)),'r'),
                             (np.array((9,10)),'g'),(np.array((20,16)),'c'),(np.array((18,21)),'m')]:
            ind = np.argmin(np.linalg.norm([centroides[r] for r in restantes] - point, axis=1))
            mapping_color[restantes[ind]] = color
            restantes.remove(restantes[ind]) 
    elif clusters == 6:
        for point, color in [(np.array((3,25)),'b'), (np.array((6,25)),'r'),
                             (np.array((13,25)),'m'),(np.array((18,20)),'y'),
                             (np.array((8,8)),'g'),(np.array((20,15)),'c')]:
            ind = np.argmin(np.linalg.norm([centroides[r] for r in restantes] - point, axis=1))
            mapping_color[restantes[ind]] = color
            restantes.remove(restantes[ind])
    return mapping_color

def def_marker(color):
    maker_color = {'b':'o',
                   'r':'v',
                   'm':'s',
                   'y':'^',
                   'g':'P',
                   'c':'*'}
    return maker_color[color]