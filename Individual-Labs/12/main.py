import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Disable core detection warning

# Hardcoded input data
data = pd.DataFrame({
    'Customer': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X1': [10, 12, 20, 21, 30, 31, 8, 9, 25, 28],
    'X2': [100, 120, 300, 310, 500, 510, 90, 110, 350, 400],
    'Z1': [25, 27, 35, 36, 45, 46, 22, 23, 40, 42],
    'Z2': [2000, 2500, 4000, 4200, 6000, 6200, 1800, 1900, 5000, 5500]
})

# Prepare the features for GMM (X1 and X2)
X = data[['X1', 'X2']].values

# Fit a GMM with 2 components
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)

# Assign clusters to each customer
clusters = gmm.predict(X)
data['Cluster'] = clusters

# Determine which cluster corresponds to Cluster1 and Cluster2 based on X1 and X2
cluster_means = gmm.means_
if cluster_means[0][0] < cluster_means[1][0] and cluster_means[0][1] < cluster_means[1][1]:
    # Cluster0 is Cluster1, Cluster1 is Cluster2
    data['Cluster'] = data['Cluster'].map({0: 'Cluster1', 1: 'Cluster2'})
else:
    # Cluster0 is Cluster2, Cluster1 is Cluster1
    data['Cluster'] = data['Cluster'].map({0: 'Cluster2', 1: 'Cluster1'})

# For each cluster, fit the distribution of Z1 and Z2
# Assuming Z1 and Z2 are normally distributed within each cluster

# Cluster1
cluster1_data = data[data['Cluster'] == 'Cluster1']
mu_z1_cluster1 = cluster1_data['Z1'].mean()
sigma_z1_cluster1 = cluster1_data['Z1'].var()
mu_z2_cluster1 = cluster1_data['Z2'].mean()
sigma_z2_cluster1 = cluster1_data['Z2'].var()

# Cluster2
cluster2_data = data[data['Cluster'] == 'Cluster2']
mu_z1_cluster2 = cluster2_data['Z1'].mean()
sigma_z1_cluster2 = cluster2_data['Z1'].var()
mu_z2_cluster2 = cluster2_data['Z2'].mean()
sigma_z2_cluster2 = cluster2_data['Z2'].var()

# Compute P(Cluster2 | Z1=35, Z2=4000) using Bayes' theorem
# P(Cluster2 | Z1, Z2) = [P(Z1, Z2 | Cluster2) * P(Cluster2)] / P(Z1, Z2)

# P(Z1=35 | Cluster2) and P(Z2=4000 | Cluster2)
p_z1_given_cluster2 = norm.pdf(35, loc=mu_z1_cluster2, scale=np.sqrt(sigma_z1_cluster2))
p_z2_given_cluster2 = norm.pdf(4000, loc=mu_z2_cluster2, scale=np.sqrt(sigma_z2_cluster2))

# P(Z1=35 | Cluster1) and P(Z2=4000 | Cluster1)
p_z1_given_cluster1 = norm.pdf(35, loc=mu_z1_cluster1, scale=np.sqrt(sigma_z1_cluster1))
p_z2_given_cluster1 = norm.pdf(4000, loc=mu_z2_cluster1, scale=np.sqrt(sigma_z2_cluster1))

# Prior probabilities
p_cluster1 = len(cluster1_data) / len(data)
p_cluster2 = len(cluster2_data) / len(data)

# P(Z1, Z2)
p_z = (p_z1_given_cluster1 * p_z2_given_cluster1 * p_cluster1 +
        p_z1_given_cluster2 * p_z2_given_cluster2 * p_cluster2)

# P(Cluster2 | Z1, Z2)
p_cluster2_given_z = (p_z1_given_cluster2 * p_z2_given_cluster2 * p_cluster2) / p_z

print(f"P(Cluster2 | Z1=35, Z2=4000) = {p_cluster2_given_z:.4f}")

# Gaussian parameters for each cluster's X1 and X2
# Cluster1
cluster1_X1_mean = cluster1_data['X1'].mean()
cluster1_X1_var = cluster1_data['X1'].var()
cluster1_X2_mean = cluster1_data['X2'].mean()
cluster1_X2_var = cluster1_data['X2'].var()

# Cluster2
cluster2_X1_mean = cluster2_data['X1'].mean()
cluster2_X1_var = cluster2_data['X1'].var()
cluster2_X2_mean = cluster2_data['X2'].mean()
cluster2_X2_var = cluster2_data['X2'].var()

# Print the Gaussian parameters for each cluster
print("\nCluster1:")
print(f"X1 ~ N({cluster1_X1_mean:.2f}, {cluster1_X1_var:.2f})")
print(f"X2 ~ N({cluster1_X2_mean:.2f}, {cluster1_X2_var:.2f})")
print("\nCluster2:")
print(f"X1 ~ N({cluster2_X1_mean:.2f}, {cluster2_X1_var:.2f})")
print(f"X2 ~ N({cluster2_X2_mean:.2f}, {cluster2_X2_var:.2f})")