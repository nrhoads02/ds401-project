import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from fastdtw import fastdtw
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union


def compute_multivariate_dtw(series1: np.ndarray, series2: np.ndarray, radius: int = 10) -> float:
    """
    Compute DTW distance between two multivariate time series.
    
    Parameters:
        series1 (np.ndarray): First multivariate time series (n_samples, n_features)
        series2 (np.ndarray): Second multivariate time series (n_samples, n_features)
        radius (int): Constraint for warping path - how far it can deviate from diagonal
        
    Returns:
        float: DTW distance between the series
    """
    n_features = series1.shape[1]
    
    # Normalize each feature to have similar scale influence
    combined = np.vstack([series1, series2])
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    series1_norm = (series1 - mean) / std
    series2_norm = (series2 - mean) / std
    
    # For multivariate series, we compute DTW for each dimension then sum
    total_distance = 0
    for i in range(n_features):
        distance, _ = fastdtw(series1_norm[:, i], series2_norm[:, i], radius=radius)
        total_distance += distance
    
    # Return average distance across dimensions
    return total_distance / n_features


def multi_feature_temporal_clustering(
        df: pl.DataFrame, 
        features: List[str] = None,
        eps: float = 0.5, 
        min_samples: int = 5,
        radius: int = 10,
        return_distance_matrix: bool = False
    ) -> Dict[str, int]:
    """
    Cluster stocks based on their temporal volatility patterns using multiple features.
    
    Parameters:
        df (pl.DataFrame): DataFrame with volatility features for multiple stocks
        features (List[str]): List of column names to use for clustering. If None, defaults
                             to ['log_returns', 'YZVol_30', 'YZVol_30_future', 'ParkinsonVol_30', 'STD_30']
        eps (float): DBSCAN epsilon parameter - max distance between samples to be considered neighbors
        min_samples (int): DBSCAN min_samples parameter - number of samples in a neighborhood for a point to be a core point
        radius (int): Constraint parameter for DTW algorithm
        return_distance_matrix (bool): Whether to return the distance matrix in addition to cluster assignments
        
    Returns:
        Dict[str, int]: Dictionary mapping ticker symbols to cluster assignments
        np.ndarray (optional): Distance matrix if return_distance_matrix is True
    """
    # Default features if none provided
    if features is None:
        features = ['log_returns', 'YZVol_30', 'YZVol_30_future', 'ParkinsonVol_30', 'STD_30']
    
    # Check if all requested features exist in the dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features {missing_features} not found in DataFrame. Available features: {df.columns}")
    
    print(f"Clustering using features: {features}")
    
    # Group by ticker
    ticker_groups = df.group_by("act_symbol")
    
    # Create feature matrices for each ticker
    ticker_features = {}
    
    for name, group in ticker_groups:
        # Sort by date to ensure temporal order
        ticker_data = group.sort("date")
        
        # Check for missing values
        # In Polars, checking for nulls is a bit tricky - we'll use a simpler approach
        has_nulls = False
        for feature in features:
            if ticker_data[feature].null_count() > 0:
                has_nulls = True
                break
                
        if has_nulls:
            print(f"Warning: {name} has missing values in selected features. Filling with 0.")
            ticker_data = ticker_data.fill_null(0)  # Simpler approach to fill nulls
        
        # Extract key temporal features into a multi-dimensional time series
        # Each row is a time point, each column is a different feature
        feature_array = ticker_data.select(features).to_numpy()
        
        # Only include tickers with sufficient data
        if len(feature_array) > 30:  # Minimum length threshold
            ticker_features[name] = feature_array
    
    tickers = list(ticker_features.keys())
    n_tickers = len(tickers)
    
    print(f"Computing DTW distances between {n_tickers} tickers...")
    
    # Compute distances between tickers using multi-dimensional DTW
    distance_matrix = np.zeros((n_tickers, n_tickers))
    
    for i in range(n_tickers):
        if i % 20 == 0:
            print(f"Processing ticker {i+1}/{n_tickers}")
            
        for j in range(i+1, n_tickers):
            # Compute multi-dimensional DTW distance
            distance = compute_multivariate_dtw(
                ticker_features[tickers[i]], 
                ticker_features[tickers[j]],
                radius=radius
            )
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    
    print("Applying DBSCAN to cluster tickers...")
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)
    
    # Map tickers to their cluster assignments
    ticker_clusters = {ticker: int(cluster) for ticker, cluster in zip(tickers, clusters)}
    
    # Count number of stocks per cluster
    cluster_counts = {}
    for cluster in clusters:
        if cluster not in cluster_counts:
            cluster_counts[cluster] = 0
        cluster_counts[cluster] += 1
    
    print("Clustering complete!")
    print(f"Found {len(set(clusters))} clusters")
    for cluster, count in sorted(cluster_counts.items()):
        if cluster == -1:
            print(f"  Noise points: {count} tickers")
        else:
            print(f"  Cluster {cluster}: {count} tickers")
    
    if return_distance_matrix:
        return ticker_clusters, distance_matrix, tickers
    
    return ticker_clusters


def find_optimal_eps(
        distance_matrix: np.ndarray, 
        k: int = 4,
        plot_path: str = 'k_distance_plot.png'
    ) -> np.ndarray:
    """
    Find optimal eps parameter for DBSCAN by plotting k-distance graph.
    
    Parameters:
        distance_matrix (np.ndarray): Distance matrix between samples
        k (int): Number of nearest neighbors to consider
        plot_path (str): Path to save the k-distance plot
        
    Returns:
        np.ndarray: Sorted k-distances (inspect for the "knee" point)
    """
    # Calculate distances to kth nearest neighbor
    distances = np.sort(distance_matrix, axis=1)[:, 1:k+1]
    k_distances = np.mean(distances, axis=1)
    
    # Sort distances for knee detection
    sorted_distances = np.sort(k_distances)
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.xlabel('Points (sorted)')
    plt.ylabel(f'Mean distance to {k} nearest neighbors')
    plt.title('K-distance Graph for DBSCAN eps Parameter Estimation')
    plt.grid(True)
    
    # Draw guidelines to help find the knee point
    # Find the steepest increase
    diffs = np.diff(sorted_distances)
    max_diff_idx = np.argmax(diffs)
    knee_idx = max_diff_idx + 1
    
    plt.axvline(x=knee_idx, color='r', linestyle='--', label=f'Suggested knee at {knee_idx}')
    plt.axhline(y=sorted_distances[knee_idx], color='g', linestyle='--', 
                label=f'Suggested eps ≈ {sorted_distances[knee_idx]:.2f}')
    
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"K-distance plot saved to {plot_path}")
    print(f"Suggested eps value based on knee point: {sorted_distances[knee_idx]:.2f}")
    
    return sorted_distances


def visualize_clusters(
        df: pl.DataFrame, 
        ticker_clusters: Dict[str, int],
        feature: str = 'YZVol_30',
        n_examples: int = 3,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
    """
    Visualize representative examples from each cluster.
    
    Parameters:
        df (pl.DataFrame): DataFrame with volatility features
        ticker_clusters (Dict[str, int]): Mapping of tickers to cluster assignments
        feature (str): Feature to plot for visualization
        n_examples (int): Number of examples to show per cluster
        figsize (Tuple[int, int]): Figure size
    """
    # Group tickers by cluster
    clusters = {}
    for ticker, cluster in ticker_clusters.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(ticker)
    
    # Create figure
    n_clusters = len(clusters)
    fig, axes = plt.subplots(n_clusters, n_examples, figsize=figsize, squeeze=False)
    
    for i, (cluster, tickers) in enumerate(sorted(clusters.items())):
        # Select n_examples random tickers from this cluster (or all if fewer)
        if len(tickers) <= n_examples:
            sample_tickers = tickers
        else:
            np.random.seed(42)  # For reproducibility
            sample_tickers = np.random.choice(tickers, n_examples, replace=False)
        
        for j, ticker in enumerate(sample_tickers):
            if j >= n_examples:
                break
                
            # Get data for this ticker
            ticker_data = df.filter(pl.col("act_symbol") == ticker).sort("date")
            
            # Plot the feature
            dates = ticker_data["date"].to_numpy()
            values = ticker_data[feature].to_numpy()
            
            ax = axes[i, j]
            ax.plot(dates, values)
            ax.set_title(f"{ticker} (Cluster {cluster})")
            
            # Only show year in x-axis for cleaner visualization
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # If this is the first column, add y-axis label
            if j == 0:
                if cluster == -1:
                    ax.set_ylabel("Noise", fontsize=12)
                else:
                    ax.set_ylabel(f"Cluster {cluster}", fontsize=12)
    
    # Adjust layout and add overall title
    plt.tight_layout()
    plt.suptitle(f"Cluster Examples: {feature} Over Time", fontsize=16, y=1.02)
    plt.savefig(f'cluster_examples_{feature}.png', bbox_inches='tight')
    plt.close()
    
    print(f"Cluster examples visualization saved to cluster_examples_{feature}.png")


def visualize_distance_matrix(
        distance_matrix: np.ndarray, 
        tickers: List[str],
        clusters: List[int],
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
    """
    Visualize the distance matrix with tickers ordered by cluster.
    
    Parameters:
        distance_matrix (np.ndarray): Distance matrix between tickers
        tickers (List[str]): List of ticker symbols
        clusters (List[int]): Cluster assignment for each ticker
        figsize (Tuple[int, int]): Figure size
    """
    # Create a DataFrame for easier manipulation
    df_distance = pd.DataFrame(distance_matrix, index=tickers, columns=tickers)
    
    # Add cluster information
    df_distance['cluster'] = clusters
    
    # Sort by cluster
    df_distance = df_distance.sort_values('cluster')
    
    # Get the sorted tickers and remove the cluster column
    sorted_tickers = df_distance.index.tolist()
    df_distance = df_distance.drop('cluster', axis=1)
    
    # Reorder columns to match row ordering
    df_distance = df_distance[sorted_tickers]
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_distance, 
        cmap='viridis_r',
        xticklabels=False,
        yticklabels=False
    )
    
    # Add cluster boundaries if there are multiple clusters
    unique_clusters = sorted(set(clusters))
    if len(unique_clusters) > 1:
        # Find indices where cluster changes
        boundaries = []
        current_cluster = clusters[0]
        for i, cluster in enumerate(clusters[1:], 1):
            if cluster != current_cluster:
                boundaries.append(i)
                current_cluster = cluster
        
        # Draw boundaries
        for b in boundaries:
            plt.axhline(y=b, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=b, color='red', linestyle='-', linewidth=1)
    
    plt.title('Distance Matrix (ordered by cluster)')
    plt.savefig('distance_matrix_by_cluster.png', bbox_inches='tight')
    plt.close()
    
    print("Distance matrix visualization saved to distance_matrix_by_cluster.png")


def cluster_analysis(df: pl.DataFrame, ticker_clusters: Dict[str, int], feature_sets: Dict[str, List[str]]) -> None:
    """
    Analyze characteristics of each cluster across different feature sets.
    
    Parameters:
        df (pl.DataFrame): DataFrame with volatility features
        ticker_clusters (Dict[str, int]): Mapping of tickers to cluster assignments
        feature_sets (Dict[str, List[str]]): Different groups of features to analyze
    """
    # Create a DataFrame with cluster assignments
    cluster_df = pl.DataFrame({
        "act_symbol": list(ticker_clusters.keys()),
        "cluster": list(ticker_clusters.values())
    })
    
    # Join with the original data
    df_with_clusters = df.join(cluster_df, on="act_symbol", how="inner")
    
    # Analyze each feature set
    for set_name, features in feature_sets.items():
        print(f"\n=== Analysis of {set_name} ===")
        
        # For each feature in this set
        for feature in features:
            if feature not in df.columns:
                print(f"Warning: Feature {feature} not found in DataFrame, skipping.")
                continue
                
            # Calculate statistics by cluster
            stats = df_with_clusters.group_by("cluster").agg(
                pl.mean(feature).alias(f"{feature}_mean"),
                pl.std(feature).alias(f"{feature}_std"),
                pl.min(feature).alias(f"{feature}_min"),
                pl.max(feature).alias(f"{feature}_max"),
                pl.count().alias("count")
            ).sort("cluster")
            
            print(f"\nStatistics for {feature}:")
            print(stats)
            
            # Create boxplot
            plt.figure(figsize=(12, 6))
            
            # Convert to pandas for seaborn
            pdf = df_with_clusters.select(["cluster", feature]).to_pandas()
            
            # Replace -1 with "Noise" for better visualization
            pdf["cluster"] = pdf["cluster"].replace(-1, "Noise")
            
            # Create boxplot
            sns.boxplot(x="cluster", y=feature, data=pdf)
            
            plt.title(f"Distribution of {feature} by Cluster")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'cluster_boxplot_{feature}.png')
            plt.close()


def evaluate_clustering(distance_matrix: np.ndarray, clusters: List[int]) -> float:
    """
    Evaluate clustering quality using silhouette score.
    
    Parameters:
        distance_matrix (np.ndarray): Distance matrix between samples
        clusters (List[int]): Cluster assignments
        
    Returns:
        float: Silhouette score (-1 to 1, higher is better)
    """
    # Filter out noise points (-1) for silhouette calculation
    valid_indices = [i for i, c in enumerate(clusters) if c != -1]
    
    if len(valid_indices) < 2 or len(set([clusters[i] for i in valid_indices])) < 2:
        print("Not enough valid clusters for silhouette evaluation")
        return 0
    
    valid_distance_matrix = distance_matrix[np.ix_(valid_indices, valid_indices)]
    valid_clusters = [clusters[i] for i in valid_indices]
    
    # Calculate silhouette score
    score = silhouette_score(valid_distance_matrix, valid_clusters, metric='precomputed')
    
    print(f"Silhouette Score: {score:.3f} (range -1 to 1, higher is better)")
    return score


def parameter_sweep(
        df: pl.DataFrame,
        features: List[str],
        eps_values: List[float],
        min_samples_values: List[int]
    ) -> Dict[Tuple[float, int], Dict[str, Any]]:
    """
    Perform parameter sweep to find optimal DBSCAN parameters.
    
    Parameters:
        df (pl.DataFrame): DataFrame with volatility features
        features (List[str]): Features to use for clustering
        eps_values (List[float]): List of eps values to try
        min_samples_values (List[int]): List of min_samples values to try
        
    Returns:
        Dict: Dictionary mapping parameter combinations to evaluation results
    """
    results = {}
    
    # First, compute the distance matrix once (expensive operation)
    print("Computing distance matrix for parameter sweep...")
    _, distance_matrix, tickers = multi_feature_temporal_clustering(
        df, features=features, return_distance_matrix=True
    )
    
    print(f"Running parameter sweep with {len(eps_values)} eps values and {len(min_samples_values)} min_samples values")
    
    # Try different parameter combinations
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nTesting eps={eps}, min_samples={min_samples}")
            
            # Apply DBSCAN with these parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            clusters = dbscan.fit_predict(distance_matrix)
            
            # Count clusters and noise points
            n_clusters = len(set([c for c in clusters if c != -1]))
            n_noise = list(clusters).count(-1)
            
            print(f"Found {n_clusters} clusters and {n_noise} noise points")
            
            # Calculate silhouette score if possible
            try:
                if n_clusters >= 2:
                    score = evaluate_clustering(distance_matrix, clusters)
                else:
                    score = float('nan')
                    print("Not enough clusters for silhouette score")
            except Exception as e:
                score = float('nan')
                print(f"Error calculating silhouette score: {e}")
            
            # Store results
            results[(eps, min_samples)] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': n_noise / len(clusters) * 100,
                'silhouette': score,
                'clusters': clusters,
            }
    
    # Display results summary
    print("\nParameter Sweep Results:")
    print(f"{'eps':>8} {'min_samples':>12} {'n_clusters':>12} {'noise_pct':>12} {'silhouette':>12}")
    print("-" * 60)
    
    for (eps, min_samples), res in sorted(results.items()):
        print(f"{eps:8.2f} {min_samples:12d} {res['n_clusters']:12d} {res['noise_pct']:12.1f}% {res['silhouette']:12.3f}")
    
    # Find best configuration based on silhouette score (if available)
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['silhouette']) and v['n_clusters'] >= 2}
    
    if valid_results:
        best_params = max(valid_results.items(), key=lambda x: x[1]['silhouette'])[0]
        print(f"\nBest parameters based on silhouette score: eps={best_params[0]}, min_samples={best_params[1]}")
        print(f"Silhouette score: {valid_results[best_params]['silhouette']:.3f}")
    else:
        print("\nNo valid configurations found with multiple clusters")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("This module provides functions for temporal volatility clustering.")
    print("Import the module and use its functions in your analysis workflow.")