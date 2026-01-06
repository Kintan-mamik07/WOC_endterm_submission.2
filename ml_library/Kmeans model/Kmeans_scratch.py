

import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def fit(self, X):
        X = np.asarray(X)
        self.centroids_ = self._init_centroids(X)

        for _ in range(self.max_iters):
            
            distances = np.array([
                [euclidean_distance(x, c) for c in self.centroids_]
                for x in X
            ])
            labels = np.argmin(distances, axis=1)

           
            new_centroids = np.array([
                X[labels == k].mean(axis=0) for k in range(self.n_clusters)
            ])
            
            for k in range(self.n_clusters):
                if np.isnan(new_centroids[k]).any():
                    new_centroids[k] = self.centroids_[k]

            
            shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = labels

            if shift < self.tol:
                break

            
        return self

    def predict(self, X):
        X = np.asarray(X)
        distances = np.array([
            [euclidean_distance(x, c) for c in self.centroids_]
            for x in X
        ])
        return np.argmin(distances, axis=1)



