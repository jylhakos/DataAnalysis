"""
K-Means Clustering Example

K-Means is an unsupervised learning algorithm that groups similar data points
into k clusters by minimizing within-cluster variance.

Objective: Minimize Σ(i=1 to k) Σ(x in Ci) ||x - μi||²

Where:
- k = number of clusters
- Ci = cluster i
- μi = centroid of cluster i
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_iris_dataset
from utils.preprocessing import standardize_features


def find_optimal_clusters(X, max_clusters=10):
    """
    Use elbow method to find optimal number of clusters
    
    Args:
        X (array): Features
        max_clusters (int): Maximum number of clusters to test
    """
    print("\n" + "="*50)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)")
    print("="*50)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        print(f"k={k} -> Inertia: {kmeans.inertia_:.2f}, Silhouette: {silhouette_scores[-1]:.3f}")
    
    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inertia (Within-cluster sum of squares)
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score
    ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Recommend optimal k based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nRecommended k (based on silhouette score): {optimal_k}")
    
    return optimal_k


def train_kmeans(X, n_clusters=3):
    """
    Train K-Means clustering model
    
    Args:
        X (array): Features
        n_clusters (int): Number of clusters
        
    Returns:
        KMeans: Trained model
    """
    print(f"\nTraining K-Means with k={n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    print(f"Converged in {kmeans.n_iter_} iterations")
    
    return kmeans


def evaluate_clustering(X, labels, n_clusters):
    """
    Evaluate clustering quality
    
    Args:
        X (array): Features
        labels (array): Cluster labels
        n_clusters (int): Number of clusters
    """
    print("\n" + "="*50)
    print("CLUSTERING EVALUATION")
    print("="*50)
    
    # Silhouette score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(X, labels)
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print("  (Measures how similar points are to their own cluster vs other clusters)")
    print("  Interpretation: -1 (poor) to +1 (excellent)")
    
    # Davies-Bouldin index (lower is better)
    db_index = davies_bouldin_score(X, labels)
    print(f"\nDavies-Bouldin Index: {db_index:.4f}")
    print("  (Average similarity ratio of each cluster with its most similar cluster)")
    print("  Interpretation: Lower values indicate better clustering")
    
    # Cluster sizes
    print(f"\nCluster Sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples")


def visualize_clusters(X, labels, centroids, n_clusters):
    """
    Visualize clusters using PCA for dimensionality reduction
    
    Args:
        X (array): Features
        labels (array): Cluster labels
        centroids (array): Cluster centroids
        n_clusters (int): Number of clusters
    """
    print("\nVisualizing clusters with PCA (2D projection)...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    # Plot points colored by cluster
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i in range(n_clusters):
        cluster_points = X_pca[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i}', 
                   alpha=0.6, edgecolors='black', s=100)
    
    # Plot centroids
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='black', marker='X', s=300, 
               edgecolors='yellow', linewidths=2, label='Centroids')
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('K-Means Clustering - PCA Visualization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_with_true_labels(y_true, y_pred):
    """
    Compare clustering results with true labels (if available)
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted cluster labels
    """
    print("\n" + "="*50)
    print("COMPARISON WITH TRUE LABELS")
    print("="*50)
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    print("  (Measures similarity between true and predicted clustering)")
    print("  Range: -0.5 to 1.0 (1.0 = perfect match)")
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"\nNormalized Mutual Information: {nmi:.4f}")
    print("  (Measures mutual dependence between true and predicted labels)")
    print("  Range: 0 to 1.0 (1.0 = perfect correlation)")


def main():
    """
    Main function to demonstrate K-Means clustering
    """
    print("="*50)
    print("   K-MEANS CLUSTERING DEMONSTRATION   ")
    print("="*50)
    
    # Load dataset
    X, y = load_iris_dataset()
    
    # Standardize features
    print("\nStandardizing features...")
    X_scaled, scaler = standardize_features(X)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_scaled, max_clusters=10)
    
    # Train K-Means with optimal k (we know Iris has 3 species)
    n_clusters = 3
    print(f"\nUsing k={n_clusters} (matching the 3 Iris species)")
    kmeans = train_kmeans(X_scaled, n_clusters=n_clusters)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Evaluate clustering
    evaluate_clustering(X_scaled, labels, n_clusters)
    
    # Visualize clusters
    visualize_clusters(X_scaled, labels, centroids, n_clusters)
    
    # Compare with true labels
    compare_with_true_labels(y, labels)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Dataset: Iris (150 samples, 4 features)")
    print(f"✓ Algorithm: K-Means Clustering")
    print(f"✓ Number of Clusters: {n_clusters}")
    print(f"✓ Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    print(f"✓ K-Means successfully identified {n_clusters} natural groupings")
    print("="*50)


if __name__ == "__main__":
    main()
