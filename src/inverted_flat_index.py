import numpy as np
from scipy.cluster.vq import kmeans2, vq
import os


class InvertedFlatIndex:

    def __init__(self, d: int, k: int, probes: int,
                 clusters_path="./assets/indexes/inverted_flat_index/clusters",
                 centroids_path="./assets/indexes/inverted_flat_index/centroids",
                 train_limit=10**6, iterations=100, new_index=True) -> None:
        self.D = d                                              # initial vector dimensions
        self.K = k                                              # number of clusters
        self.K_ = probes                                        # nuber of clusters retrieved

        self.train_limit = train_limit
        self.iterations = iterations

        self.clusters_path = clusters_path
        self.centroids_path = centroids_path

        if new_index:
            if os.path.exists(self.clusters_path):
                for filename in os.listdir(self.clusters_path):
                    file_path = os.path.join(self.clusters_path, filename)
                    os.remove(file_path)
            if os.path.exists(self.centroids_path):
                os.remove(self.centroids_path)

    def generate_inverted_flat_index(self, database: np.ndarray, codebook: np.ndarray):
        training_data, predicting_data = database[:self.train_limit], database[self.train_limit:]

        self.centroids, vectors = kmeans2(training_data, self.K, minit='points', iter=self.iterations)
        np.savetxt(self.centroids_path, self.centroids)

        for i in range(self.K):
            ids, = np.where(vectors == i)
            cluster = np.column_stack((codebook[ids], ids))
            np.savetxt(self.clusters_path + f"/cluster{i}", cluster, fmt="%d")

        if predicting_data.shape[0] > 0:
            vectors, _ = vq(predicting_data, self.centroids)

            for i in range(self.K):
                ids, = np.where(vectors == i)
                cluster = np.column_stack((codebook[ids], ids))

                pre_cluster = np.loadtxt(self.clusters_path + f"/cluster{i}", dtype=int)
                post_cluster = np.vstack((pre_cluster, cluster))
                np.savetxt(self.clusters_path + f"/cluster{i}", post_cluster, fmt="%d")


    def _compute_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity