"""
Principal Component Analysis (PCA) for Dimensionality Reduction

PCA finds orthogonal directions of maximum variance in the data and projects
the data onto a lower-dimensional subspace.

Objective: Maximize Var(Xw) subject to ||w|| = 1

Where:
- X = data matrix
- w = principal component direction
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_iris_dataset


def apply_pca(X, n_components=None):
    """
    Apply PCA to reduce dimensionality
    
    Args:
        X (array): Input features
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (X_transformed, pca_model)
    """
    if n_components is None:
        n_components = X.shape[1]
    
    print(f"\nApplying PCA...")
    print(f"  Original dimensions: {X.shape[1]}")
    print(f"  Target dimensions: {n_components}")
    
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    
    print(f"  Transformed shape: {X_transformed.shape}")
    
    return X_transformed, pca


def analyze_explained_variance(pca):
    """
    Analyze and visualize explained variance
    
    Args:
        pca (PCA): Fitted PCA model
    """
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("\n" + "="*50)
    print("EXPLAINED VARIANCE ANALYSIS")
    print("="*50)
    
    print("\nExplained Variance by Component:")
    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual explained variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var, 
            color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative explained variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
             'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return explained_var, cumulative_var


def visualize_pca_2d(X_pca, y, feature_names=None):
    """
    Visualize data in 2D PCA space
    
    Args:
        X_pca (array): PCA-transformed data (2D)
        y (array): Labels
        feature_names (list): Feature names
    """
    if X_pca.shape[1] < 2:
        print("Need at least 2 components for 2D visualization")
        return
    
    plt.figure(figsize=(10, 7))
    
    # Get unique classes
    classes = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot each class
    for i, cls in enumerate(classes):
        mask = (y == cls)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i], label=cls, alpha=0.6,
                   edgecolors='black', s=100)
    
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title('PCA: 2D Visualization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_pca_3d(X_pca, y):
    """
    Visualize data in 3D PCA space
    
    Args:
        X_pca (array): PCA-transformed data (3D)
        y (array): Labels
    """
    if X_pca.shape[1] < 3:
        print("Need at least 3 components for 3D visualization")
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique classes
    classes = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot each class
    for i, cls in enumerate(classes):
        mask = (y == cls)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  c=colors[i], label=cls, alpha=0.6,
                  edgecolors='black', s=100)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title('PCA: 3D Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def analyze_component_loadings(pca, feature_names):
    """
    Analyze feature contributions to principal components
    
    Args:
        pca (PCA): Fitted PCA model
        feature_names (list): Names of original features
    """
    print("\n" + "="*50)
    print("COMPONENT LOADINGS")
    print("="*50)
    print("\nFeature contributions to each principal component:")
    
    loadings = pca.components_
    
    for i, loading in enumerate(loadings):
        print(f"\nPrincipal Component {i+1}:")
        for j, (feature, weight) in enumerate(zip(feature_names, loading)):
            print(f"  {feature:20s}: {weight:+.4f}")
    
    # Plot loadings heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Loading Value')
    plt.yticks(range(len(loadings)), [f'PC{i+1}' for i in range(len(loadings))])
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.title('PCA Component Loadings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_dimensionality_reduction(X, y, feature_names):
    """
    Compare different numbers of components
    
    Args:
        X (array): Features
        y (array): Labels
        feature_names (list): Feature names
    """
    print("\n" + "="*50)
    print("COMPARING DIFFERENT NUMBER OF COMPONENTS")
    print("="*50)
    
    # Test different numbers of components
    n_features = X.shape[1]
    
    for n_comp in range(1, n_features + 1):
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(X)
        total_var = np.sum(pca.explained_variance_ratio_)
        
        print(f"\n{n_comp} component(s): {total_var:.4f} ({total_var*100:.2f}% variance retained)")
        print(f"  Dimensionality reduction: {n_features} → {n_comp}")
        print(f"  Data compression ratio: {(1 - n_comp/n_features)*100:.1f}%")


def main():
    """
    Main function to demonstrate PCA
    """
    print("="*50)
    print(" PCA DIMENSIONALITY REDUCTION ")
    print("="*50)
    
    # Load dataset
    X, y = load_iris_dataset()
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Standardize features (important for PCA)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA with all components
    X_pca_full, pca_full = apply_pca(X_scaled)
    
    # Analyze explained variance
    explained_var, cumulative_var = analyze_explained_variance(pca_full)
    
    # Find number of components needed for 95% variance
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")
    
    # Analyze component loadings
    analyze_component_loadings(pca_full, feature_names)
   
    # Visualize in 2D
    X_pca_2d, pca_2d = apply_pca(X_scaled, n_components=2)
    visualize_pca_2d(X_pca_2d, y)
    
    # Visualize in 3D
    X_pca_3d, pca_3d = apply_pca(X_scaled, n_components=3)
    visualize_pca_3d(X_pca_3d, y)
    
    # Compare different dimensionality reductions
    compare_dimensionality_reduction(X_scaled, y, feature_names)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Algorithm: Principal Component Analysis (PCA)")
    print(f"✓ Original Dimensions: {X.shape[1]}")
    print(f"✓ Components for 95% variance: {n_components_95}")
    print(f"✓ First 2 PCs explain: {cumulative_var[1]*100:.2f}% of variance")
    print(f"✓ PCA successfully reduced dimensionality while preserving information")
    print("="*50)


if __name__ == "__main__":
    main()
