import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Number of clusters
CLUSTERS = 10

#%% Load data
# Load the Penguins dataset from Seaborn
penguins = sns.load_dataset("penguins")

# Drop rows with missing values
penguins.dropna(inplace=True)

# Extract features
X = penguins[['flipper_length_mm', 'body_mass_g']]
Y = penguins['species']  # Species column for coloring

#%% Modeling
# Instantiate KMeans with 3 clusters
kmeans = KMeans(n_clusters=CLUSTERS)

# Fit KMeans to the data
kmeans.fit(X)

# Get cluster labels
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

#%% Plotting
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Penguins colored by species
for species in Y.unique():
    axs[0].scatter(X[Y == species]['flipper_length_mm'], X[Y == species]['body_mass_g'], label=species)
axs[0].set_title('Penguins by Species')
axs[0].set_xlabel('Flipper Length (mm)')
axs[0].set_ylabel('Body Mass (g)')
axs[0].legend()

# Plot 2: Clusters found by KMeans
for i in range(len(np.unique(cluster_labels))):
    axs[1].scatter(
        X[cluster_labels == i]['flipper_length_mm'], 
        X[cluster_labels == i]['body_mass_g'], 
        label=f'Cluster {i+1}',
        cmap="pastel"
        )
    axs[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X')
axs[1].set_title('Clusters Found by KMeans')
axs[1].set_xlabel('Flipper Length (mm)')
axs[1].set_ylabel('Body Mass (g)')
axs[1].legend()

plt.tight_layout()
plt.show()
