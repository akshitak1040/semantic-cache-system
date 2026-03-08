from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyClusterer:

    def __init__(self, n_clusters=25):

        self.n_clusters = n_clusters
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            random_state=42
        )


    def fit(self, embeddings):
        """
        Train the clustering model on document embeddings
        """
        self.model.fit(embeddings)


    def get_cluster_distribution(self, embedding):
        """
        Return probability distribution across clusters
        """

        probs = self.model.predict_proba([embedding])[0]

        return probs