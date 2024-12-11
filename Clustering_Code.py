#%% Step 1: Import Libraries
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.colors as mcolors

#%% Step 2: Load and Preprocess Data
data_path = 'gold_price_dataset_preprocessed.csv'  # Replace with the correct path
gold_data = pd.read_csv(data_path)

# Select relevant numerical features for clustering
numerical_features = ['norm_SPX', 'norm_USO', 'norm_SLV', 'norm_EUR.USD']
data = gold_data[numerical_features]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#%% Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

#%% Step 4: Elbow Method for Optimal Clusters (KMeans)
inertia = []
silhouette_scores = []
range_clusters = range(2, 11)  # Test cluster counts from 2 to 10

for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(scaled_data, labels))

#%% Plot Elbow Curve
sns.set_style("whitegrid")  
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the elbow curve
ax.plot(range_clusters, inertia, marker='o', linestyle='-', color='navy', 
        linewidth=2, markersize=8, label='Inertia')

# Title and labels
ax.set_title("Elbow Method for Optimal Clusters", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Clusters", fontsize=12)
ax.set_ylabel("Inertia", fontsize=12)

# Set x-ticks to match the cluster range for clarity
ax.set_xticks(range_clusters)

# Optional: highlight an optimal cluster count (e.g., 4)
optimal_clusters = 4
if optimal_clusters in range_clusters:
    ax.axvline(x=optimal_clusters, color='red', linestyle='--', linewidth=1.5)
    ax.text(optimal_clusters+0.1, min(inertia), f"Optimal = {optimal_clusters}", 
            color='red', fontsize=12, verticalalignment='bottom')

# Show legend and grid
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5)
fig.tight_layout()
plt.show()

#%% Step 5: KMeans Clustering and Evaluation
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Evaluation
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(scaled_data, kmeans_labels)

# Visualization
plt.figure(figsize=(10, 8))
unique_labels = np.unique(kmeans_labels)
for label in unique_labels:
    plt.scatter(pca_data[kmeans_labels == label, 0], pca_data[kmeans_labels == label, 1],
                label=f'Cluster {label}', edgecolor='k', alpha=0.7, s=50)

evaluation_text = (f"Number of Clusters: {optimal_clusters}\n"
                   f"Silhouette Score: {kmeans_silhouette:.4f}\n"
                   f"Davies-Bouldin Index: {kmeans_davies_bouldin:.4f}")
plt.text(0.02, 0.98, evaluation_text, fontsize=12, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.title(f"KMeans Clustering (Nclusters = {optimal_clusters})", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=14)
plt.ylabel("PCA Component 2", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% DBSCAN METHOD
# Step 1.1: 5-NN Distance Calculation
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)

# Sort the 5th nearest neighbor distances
distances = np.sort(distances[:, 4])

# Step 1.2: Plot the 5-NN Distance Graph
plt.figure(figsize=(12, 6))  # Larger figure size for better clarity
plt.plot(distances, color='blue', linestyle='-', linewidth=2, label='5-NN Distance')

# Optional: Add a vertical line to highlight the "elbow" point (manually determined or algorithmically found)
elbow_index = np.argmax(np.diff(distances))  # Example to find a steep change

# Add title and axis labels
plt.title("5-NN Distance Graph", fontsize=16, fontweight="bold")
plt.xlabel("Points Sorted by Distance", fontsize=12)
plt.ylabel("5-NN Distance", fontsize=12)

# Add grid and legend for clarity
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

# Improve layout
plt.tight_layout()

# Display the plot
plt.show()

# %% Step 2: Heatmap for Number of Clusters with Different eps and min_samples

# Define hyperparameter ranges
eps_values = np.linspace(0.1, 0.5, 10)  # Epsilon values
min_samples_values = range(3, 9)  # MinPts values
cluster_counts = []

# DBSCAN loop to calculate the number of clusters for each combination
for eps in eps_values:
    row = []
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        row.append(n_clusters)
    cluster_counts.append(row)

# Convert cluster counts to a NumPy array
cluster_counts = np.array(cluster_counts)

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Enhanced color map and heatmap with logarithmic scaling
norm = mcolors.LogNorm(vmin=cluster_counts.min() + 1e-6, vmax=cluster_counts.max())  # Avoid log(0)
cax = ax.imshow(cluster_counts, cmap='coolwarm', aspect='auto', interpolation='nearest', norm=norm)

# Add color bar with logarithmic scale
cbar = fig.colorbar(cax, orientation='vertical', shrink=0.8, pad=0.02)
cbar.set_label('Number of Clusters (Log Scale)', fontsize=14, labelpad=10)
cbar.ax.tick_params(labelsize=12)

# Add axis labels with enhanced styling
ax.set_title("DBSCAN Hyperparameter Tuning Heatmap (Log Scale)", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel("MinPts (min_samples)", fontsize=14, labelpad=10)
ax.set_ylabel("Epsilon (eps)", fontsize=14, labelpad=10)

# Set tick labels
ax.set_xticks(np.arange(len(min_samples_values)))
ax.set_xticklabels(min_samples_values, fontsize=12)
ax.set_yticks(np.arange(len(eps_values)))
ax.set_yticklabels([f"{eps:.2f}" for eps in eps_values], fontsize=12)

# Annotate the heatmap with cluster counts
for i in range(cluster_counts.shape[0]):
    for j in range(cluster_counts.shape[1]):
        ax.text(j, i, str(cluster_counts[i, j]), ha='center', va='center', fontsize=10, 
                color='white' if cluster_counts[i, j] > cluster_counts.max() / 2 else 'black')

# Enhanced grid and layout
ax.grid(False)  # Disable default grid to focus on heatmap
plt.tight_layout()

# Show the plot
plt.show()

# %% Step 3: Silhouette Scores for DBSCAN Hyperparameter Tuning

# Define hyperparameter ranges
eps_values = np.linspace(0.1, 0.5, 10)  # Epsilon values
min_samples_values = range(3, 9)  # MinPts values
silhouette_scores = []

# DBSCAN loop to calculate the Silhouette Score for each combination
for eps in eps_values:
    row = []
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_data)
        
        # Exclude noise points (label = -1) for Silhouette Score
        filtered_data = scaled_data[labels != -1]
        filtered_labels = labels[labels != -1]
        
        if len(set(filtered_labels)) > 1:  # Ensure at least 2 clusters
            silhouette = silhouette_score(filtered_data, filtered_labels)
        else:
            silhouette = np.nan  # Assign NaN for invalid cases
        row.append(silhouette)
    silhouette_scores.append(row)

# Convert silhouette scores to a NumPy array
silhouette_scores = np.array(silhouette_scores)

# Plotting the heatmap for Silhouette Scores
fig, ax = plt.subplots(figsize=(12, 8))

cax = ax.imshow(silhouette_scores, cmap='coolwarm', aspect='auto')

# Add color bar
cbar = fig.colorbar(cax)
cbar.set_label('Silhouette Score', fontsize=14)

# Add axis labels
ax.set_title("Silhouette Scores for DBSCAN Hyperparameter Tuning", fontsize=18, pad=20)
ax.set_xlabel("MinPts (min_samples)", fontsize=14, labelpad=10)
ax.set_ylabel("Epsilon (eps)", fontsize=14, labelpad=10)

# Set tick labels
ax.set_xticks(np.arange(len(min_samples_values)))
ax.set_xticklabels(min_samples_values, fontsize=12)
ax.set_yticks(np.arange(len(eps_values)))
ax.set_yticklabels(np.round(eps_values, 2), fontsize=12)

# Annotate the heatmap with silhouette scores
for i in range(silhouette_scores.shape[0]):
    for j in range(silhouette_scores.shape[1]):
        score = silhouette_scores[i, j]
        if not np.isnan(score):
            ax.text(j, i, f"{score:.2f}", ha='center', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.grid(False)

# %% Step 4: Find Best DBSCAN Parameters and Apply DBSCAN
best_eps = 0.32
best_min_samples = 3

# Apply DBSCAN with the chosen parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(scaled_data)

# Exclude noise points (label = -1)
filtered_data = scaled_data[labels != -1]
filtered_labels = labels[labels != -1]

# %% Step 5: Evaluate the DBSCAN Model
if len(set(filtered_labels)) > 1:
    silhouette = silhouette_score(filtered_data, filtered_labels)
    davies_bouldin = davies_bouldin_score(filtered_data, filtered_labels)
    n_clusters = len(set(filtered_labels))
else:
    silhouette = None
    davies_bouldin = None
    n_clusters = 0

# %% Step 6: Visualize DBSCAN Clustering Results
pca = PCA(n_components=2)
pca_data = pca.fit_transform(filtered_data)

plt.figure(figsize=(10, 8))
unique_labels = set(filtered_labels)

# Plot each cluster
for label in unique_labels:
    plt.scatter(pca_data[filtered_labels == label, 0], pca_data[filtered_labels == label, 1],
                label=f'Cluster {label}', edgecolor='k', alpha=0.7)

# Plot noise points separately
noise = labels == -1
if np.any(noise):
    pca_noise = pca.transform(scaled_data[noise])
    plt.scatter(pca_noise[:, 0], pca_noise[:, 1], color='black', label='Noise', marker='x')

# Add evaluation scores
evaluation_text = (f"Number of Clusters: {n_clusters}\n"
                   f"Silhouette Score: {silhouette:.4f}\n"
                   f"Davies-Bouldin Index: {davies_bouldin:.4f}")
plt.text(0.02, 0.98, evaluation_text, fontsize=12, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.title(f"DBSCAN Clustering with eps={best_eps}, min_samples={best_min_samples}", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=14)
plt.ylabel("PCA Component 2", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# %% Random Forest clustering
n_clusters = 4  # Desired number of clusters

# Generate synthetic labels for initial clustering using Agglomerative Clustering
synthetic_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(scaled_data)

# %% Step 2: Train Random Forest Classifier and Determine Feature Importances
# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(scaled_data, synthetic_labels)

# Get feature importances
importances = rf.feature_importances_
feature_names = np.array(numerical_features)

# Select the top 2 features based on importance
top_features_indices = np.argsort(importances)[-2:]
top_features = feature_names[top_features_indices]
selected_data = scaled_data[:, top_features_indices]

print(f"Selected Features for Clustering: {top_features}")

# %% Step 3: Train Random Forest with Selected Features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(selected_data, synthetic_labels)

# %% Step 4: Compute Proximity Matrix from Random Forest

# Use Random Forest proximity matrix for clustering
leaf_indices = rf_selected.apply(selected_data)
proximity_matrix = np.zeros((len(selected_data), len(selected_data)))

for tree_leafs in leaf_indices.T:
    for i, leaf_i in enumerate(tree_leafs):
        for j, leaf_j in enumerate(tree_leafs):
            if leaf_i == leaf_j:
                proximity_matrix[i, j] += 1

proximity_matrix /= rf_selected.n_estimators

# %% Step 5: Apply Hierarchical Clustering Using Proximity Matrix
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
rf_labels = hc.fit_predict(1 - proximity_matrix)  # Distance = 1 - Proximity

# %% Step 6: Evaluate the Random Forest Clustering Model
if len(set(rf_labels)) > 1:
    silhouette = silhouette_score(selected_data, rf_labels)
    davies_bouldin = davies_bouldin_score(selected_data, rf_labels)
else:
    silhouette = None
    davies_bouldin = None

# Display evaluation results
print("\nRandom Forest Clustering Evaluation:")
print(f"  Number of Clusters: {len(set(rf_labels))}")
if silhouette is not None and davies_bouldin is not None:
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
else:
    print("  Unable to compute evaluation metrics due to insufficient clusters.")

# %% Step 7: Apply PCA for Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(selected_data)

# %% Step 8: Plot the Clustering Results
plt.figure(figsize=(10, 8))
unique_labels = set(rf_labels)

# Plot each cluster
for label in unique_labels:
    plt.scatter(pca_data[rf_labels == label, 0], pca_data[rf_labels == label, 1],
                label=f'Cluster {label}', edgecolor='k', alpha=0.7)

# Add evaluation scores to the plot
evaluation_text = (f"Number of Clusters: {len(set(rf_labels))}\n"
                   f"Silhouette Score: {silhouette:.4f}\n"
                   f"Davies-Bouldin Index: {davies_bouldin:.4f}")
plt.text(0.02, 0.98, evaluation_text, fontsize=12, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Plot styling
plt.title(f"Random Forest Clustering with Top Features: {', '.join(top_features)}", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=14)
plt.ylabel("PCA Component 2", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%
