import numpy as np
import json
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import os


class CustomerSegmentation:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.bayesian_network = None

        # Embedded data
        self.data = {
            "customers": [
                {"id": 1, "X1": 10, "X2": 100, "Z1": 25, "Z2": 2000},
                {"id": 2, "X1": 12, "X2": 120, "Z1": 27, "Z2": 2500},
                {"id": 3, "X1": 20, "X2": 300, "Z1": 35, "Z2": 4000},
                {"id": 4, "X1": 21, "X2": 310, "Z1": 36, "Z2": 4200},
                {"id": 5, "X1": 30, "X2": 500, "Z1": 45, "Z2": 6000},
                {"id": 6, "X1": 31, "X2": 510, "Z1": 46, "Z2": 6200},
                {"id": 7, "X1": 8, "X2": 90, "Z1": 22, "Z2": 1800},
                {"id": 8, "X1": 9, "X2": 110, "Z1": 23, "Z2": 1900},
                {"id": 9, "X1": 25, "X2": 350, "Z1": 40, "Z2": 5000},
                {"id": 10, "X1": 28, "X2": 400, "Z1": 42, "Z2": 5500}
            ]
        }

    def prepare_data(self):
        """Prepare data for GMM analysis"""
        X = np.array([[c['X1'], c['X2']] for c in self.data['customers']])
        return self.scaler.fit_transform(X)

    def fit_gmm(self):
        """Fit GMM model and return cluster assignments"""
        X = self.prepare_data()
        self.clusters = self.gmm.fit_predict(X)
        return self.clusters

    def setup_bayesian_network(self):
        """Setup Bayesian Network structure"""
        self.bayesian_network = BayesianNetwork([
            ('Z1', 'Cluster'),
            ('Z2', 'Cluster')
        ])

        # Create CPDs
        z1_cpd = TabularCPD(
            variable='Z1',
            variable_card=2,
            values=[[0.6], [0.4]]
        )

        z2_cpd = TabularCPD(
            variable='Z2',
            variable_card=2,
            values=[[0.5], [0.5]]
        )

        cluster_cpd = TabularCPD(
            variable='Cluster',
            variable_card=2,
            values=[
                [0.8, 0.7, 0.3, 0.2],  # Cluster 1 probabilities
                [0.2, 0.3, 0.7, 0.8]  # Cluster 2 probabilities
            ],
            evidence=['Z1', 'Z2'],
            evidence_card=[2, 2]
        )

        self.bayesian_network.add_cpds(z1_cpd, z2_cpd, cluster_cpd)

    def compute_cluster_probability(self, z1_val, z2_val):
        """Compute P(Cluster2|Z1,Z2)"""
        z1_cat = 1 if z1_val >= 35 else 0
        z2_cat = 1 if z2_val >= 4000 else 0

        inference = VariableElimination(self.bayesian_network)
        evidence = {'Z1': z1_cat, 'Z2': z2_cat}
        result = inference.query(variables=['Cluster'], evidence=evidence)
        return result.values[1]

    def visualize_clusters(self):
        """Visualize the customer segments"""
        X = self.prepare_data()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=self.clusters, cmap='viridis')
        plt.xlabel('Normalized Time on Site')
        plt.ylabel('Normalized Purchase Value')
        plt.title('Customer Segments')
        plt.colorbar(scatter, label='Cluster')
        plt.show()


def main():
    # To suppress the joblib warning, set the environment variable
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    # Initialize segmentation
    segmentation = CustomerSegmentation()

    # Fit GMM and get clusters
    clusters = segmentation.fit_gmm()
    print("\nGMM Cluster Assignments:")
    for i, cluster in enumerate(clusters, 1):
        print(f"Customer {i}: Cluster {cluster + 1}")

    # Setup and use Bayesian Network
    segmentation.setup_bayesian_network()

    # Compute P(Cluster2|Z1=35,Z2=4000)
    prob = segmentation.compute_cluster_probability(35, 4000)
    print(f"\nP(Cluster2|Z1=35,Z2=4000) = {prob:.3f}")

    # Visualize the clusters
    segmentation.visualize_clusters()


if __name__ == "__main__":
    main()