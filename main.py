import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')

print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Distribution')
plt.savefig('task2_initial_distribution.png')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('task2_elbow_method.png')
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

plt.figure(figsize=(12, 8))

colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_k):
    cluster_data = X[clusters == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6, s=100)

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            c='black', s=300, marker='*', label='Centroids', edgecolors='yellow', linewidths=2)

plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Customer Segmentation - K-Means Clustering', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('task2_kmeans_clusters.png')
plt.show()

print("\n=== Cluster Analysis ===")
cluster_analysis = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'CustomerID': 'count'
})
cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns.values]
cluster_analysis.rename(columns={'CustomerID_count': 'Customer_Count'}, inplace=True)
print(cluster_analysis)

avg_spending = df.groupby('Cluster')['Spending Score (1-100)'].mean().sort_values(ascending=False)
print("\n=== Average Spending Score per Cluster ===")
print(avg_spending)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[0])
axes[0].set_title('Annual Income by Cluster')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Annual Income (k$)')

df.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[1])
axes[1].set_title('Spending Score by Cluster')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.savefig('task2_cluster_characteristics.png')
plt.show()

print("\n=== BONUS: DBSCAN Clustering ===")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_noise = list(dbscan_clusters).count(-1)

print(f"Number of clusters found by DBSCAN: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise}")

plt.figure(figsize=(10, 6))
unique_labels = set(dbscan_clusters)
colors_dbscan = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors_dbscan):
    if k == -1:
        col = [0, 0, 0, 1]  

    class_member_mask = (dbscan_clusters == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}', alpha=0.6, s=50)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering Results')
plt.legend()
plt.savefig('task2_dbscan_clusters.png')
plt.show()

print("\nTask 2 completed successfully!")
