import polars as pl
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_optimal_eps_and_clusters(
    df, 
    start_date="2012-01-01", 
    end_date="2024-08-01",
    min_data_pct=70,
    n_components=3,
    eps_range=None,
    min_samples_range=None,
    max_stocks=None,
    max_noise_pct=20,  # Maximum acceptable noise percentage
    min_clusters=3     # Minimum number of clusters desired
):
    """
    Find optimal DBSCAN parameters prioritizing low noise and balanced clusters.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with time-series data
    start_date, end_date : str
        Date range for analysis
    min_data_pct : float
        Minimum percentage of trading days required
    n_components : int
        Number of PCA components
    eps_range : list or None
        Range of eps values to test
    min_samples_range : list or None
        Range of min_samples values to test
    max_stocks : int or None
        Maximum number of stocks to use
    max_noise_pct : float
        Maximum acceptable percentage of noise points (default: 20%)
    min_clusters : int
        Minimum number of clusters desired (default: 3)
        
    Returns:
    --------
    tuple: (best_eps, best_min_samples, best_n_clusters, best_noise_pct)
    """
    print("Finding optimal DBSCAN parameters prioritizing low noise and balanced clusters...")
    
    # Use the core function to prepare data but don't run clustering yet
    stock_clusters, pca_df, _ = cluster_stocks_by_timeseries(
        df, 
        start_date=start_date,
        end_date=end_date,
        min_data_pct=min_data_pct,
        algorithm='kmeans',  # Dummy choice, we won't use it
        n_components=n_components,
        max_stocks=max_stocks
    )
    
    # Extract PCA components for clustering tests
    X_pca = stock_clusters.select([f'PC{i+1}' for i in range(n_components)]).to_numpy()
    
    # Default parameter ranges if not provided
    if eps_range is None:
        # Calculate distances to nearest neighbors to establish range
        nbrs = NearestNeighbors(n_neighbors=20).fit(X_pca)
        distances, _ = nbrs.kneighbors(X_pca)
        
        # Use percentiles to establish broader range for eps
        eps_min = max(0.001, np.percentile(distances[:, 5], 10))  # Lower percentile but ensure > 0
        eps_max = np.percentile(distances[:, 15], 90)  # Higher percentile for a wider range
        
        # Create a range with more points to explore more options
        # Ensure we don't use invalid small values
        eps_range = np.linspace(eps_min, eps_max, 15)
    else:
        # Ensure all values in the provided range are valid (> 0)
        eps_range = np.array([max(0.001, eps) for eps in eps_range])
    
    if min_samples_range is None:
        # More options for min_samples to find more balanced clusters
        min_samples_range = [3, 5, 8, 10, 15, 20, 25]
    
    # Track best parameters
    best_params = {
        'eps': None,
        'min_samples': None,
        'score': -float('inf'),  # Will use a custom score
        'n_clusters': 0,
        'noise_pct': 100,
        'cluster_balance': 0
    }
    
    # Grid search
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            print(f"Testing eps={eps:.3f}, min_samples={min_samples}")
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_pca)
            
            # Count clusters and noise
            unique_labels = np.unique(labels)
            cluster_labels = [l for l in unique_labels if l != -1]
            n_clusters = len(cluster_labels)
            n_noise = list(labels).count(-1) if -1 in labels else 0
            noise_pct = n_noise / len(labels) * 100
            
            # Calculate silhouette score if we have valid clusters
            silhouette = -1
            if n_clusters > 1 and n_noise < len(labels):
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > n_clusters:  # Ensure we have enough points
                    try:
                        silhouette = silhouette_score(
                            X_pca[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                    except:
                        pass
            
            # Calculate cluster balance metric
            # (standard deviation of cluster sizes relative to perfect balance)
            cluster_balance = 0
            if n_clusters > 1:
                # Count points in each cluster
                cluster_sizes = [np.sum(labels == label) for label in cluster_labels]
                # Ideal size would be equal distribution of non-noise points
                ideal_size = (len(labels) - n_noise) / n_clusters
                # Calculate coefficient of variation (lower is better)
                if ideal_size > 0:
                    cluster_std = np.std(cluster_sizes)
                    cluster_balance = 1 - min(1, cluster_std / ideal_size)  # 0 to 1, higher is better
            
            # Custom scoring that prioritizes:
            # 1. Low noise percentage (heavily weighted)
            # 2. Balanced clusters
            # 3. Having at least min_clusters
            # 4. Good separation (silhouette score)
            
            # Start with a base score of -infinity
            score = -float('inf')
            
            # Only consider parameter sets with acceptable noise percentage
            if noise_pct <= max_noise_pct and n_clusters >= min_clusters:
                # Calculate score with heavy penalty for noise and reward for balance
                score = (
                    -noise_pct * 0.1 +               # Penalize noise (higher weight)
                    cluster_balance * 5 +            # Reward balanced clusters (high weight)
                    min(3, n_clusters - min_clusters) * 0.5 +  # Modest reward for extra clusters
                    max(0, silhouette) * 2           # Reward good separation
                )
            
            # Store result
            result = {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_pct': noise_pct,
                'silhouette': silhouette,
                'cluster_balance': cluster_balance,
                'score': score
            }
            results.append(result)
            
            print(f"  Found {n_clusters} clusters, {noise_pct:.1f}% noise, balance={cluster_balance:.2f}, score={score:.1f}")
            
            # Check if this is better
            if score > best_params['score']:
                best_params = result
    
    # Convert results to polars DataFrame for analysis
    results_df = pl.DataFrame(results)
    
    # Create visualization of the parameter search
    plt.figure(figsize=(15, 15))
    
    # Create a matrix of results for heatmap
    if results_df.height > 0:
        # Get unique values for both axes
        unique_eps = sorted(results_df['eps'].unique().to_list())
        unique_min_samples = sorted(results_df['min_samples'].unique().to_list())
        
        # Create matrices for the heatmaps
        n_clusters_matrix = np.zeros((len(unique_min_samples), len(unique_eps)))
        noise_pct_matrix = np.zeros((len(unique_min_samples), len(unique_eps)))
        balance_matrix = np.zeros((len(unique_min_samples), len(unique_eps)))
        score_matrix = np.zeros((len(unique_min_samples), len(unique_eps)))
        score_matrix.fill(-10)  # Fill with a low value to distinguish from actual zeros
        
        # Fill matrices with values
        for i, min_samp in enumerate(unique_min_samples):
            for j, eps_val in enumerate(unique_eps):
                # Get rows matching these parameters
                matches = results_df.filter(
                    (pl.col('min_samples') == min_samp) & 
                    (pl.col('eps') == eps_val)
                )
                
                if matches.height > 0:
                    n_clusters_matrix[i, j] = matches['n_clusters'][0]
                    noise_pct_matrix[i, j] = matches['noise_pct'][0]
                    balance_matrix[i, j] = matches['cluster_balance'][0]
                    score_matrix[i, j] = matches['score'][0]
        
        # Plot heatmaps
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(n_clusters_matrix, annot=True, fmt='g', cmap='viridis', ax=ax1,
                   xticklabels=[f"{e:.3f}" for e in unique_eps],
                   yticklabels=unique_min_samples)
        ax1.set_title('Number of Clusters')
        ax1.set_xlabel('eps')
        ax1.set_ylabel('min_samples')
        
        ax2 = plt.subplot(2, 2, 2)
        sns.heatmap(noise_pct_matrix, annot=True, fmt='.1f', cmap='rocket_r', ax=ax2,
                   xticklabels=[f"{e:.3f}" for e in unique_eps],
                   yticklabels=unique_min_samples)
        ax2.set_title('Noise Percentage')
        ax2.set_xlabel('eps')
        ax2.set_ylabel('min_samples')
        
        ax3 = plt.subplot(2, 2, 3)
        # Create a masked array for balance where we only show values for valid parameter sets
        masked_balance = np.ma.masked_where(score_matrix <= -10, balance_matrix)
        sns.heatmap(masked_balance, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax3,
                   xticklabels=[f"{e:.3f}" for e in unique_eps],
                   yticklabels=unique_min_samples)
        ax3.set_title('Cluster Balance (higher is better)')
        ax3.set_xlabel('eps')
        ax3.set_ylabel('min_samples')
        
        ax4 = plt.subplot(2, 2, 4)
        # Create a masked array for score where we only show values for valid parameter sets
        masked_score = np.ma.masked_where(score_matrix <= -10, score_matrix)
        sns.heatmap(masked_score, annot=True, fmt='.1f', cmap='plasma', ax=ax4,
                   xticklabels=[f"{e:.3f}" for e in unique_eps],
                   yticklabels=unique_min_samples)
        ax4.set_title('Overall Score (higher is better)')
        ax4.set_xlabel('eps')
        ax4.set_ylabel('min_samples')
    
    plt.tight_layout()
    plt.show()
    
    # Return the best parameters
    if best_params['eps'] is not None and best_params['score'] > -float('inf'):
        print("\n" + "="*50)
        print(f"Best parameters: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}")
        print(f"  {best_params['n_clusters']} clusters, {best_params['noise_pct']:.1f}% noise")
        print(f"  Cluster balance: {best_params['cluster_balance']:.2f}, Score: {best_params['score']:.1f}")
        print("="*50)
        return best_params['eps'], best_params['min_samples'], best_params['n_clusters'], best_params['noise_pct']
    else:
        print("\nNo good clustering parameters found that meet the criteria.")
        print(f"Try adjusting max_noise_pct (currently {max_noise_pct}%) or min_clusters (currently {min_clusters})")
        return None, None, 0, 100

def cluster_stocks_by_timeseries(
    df, 
    start_date="2012-01-01", 
    end_date="2024-08-01",
    min_data_pct=70,
    algorithm="dbscan",  # Choose 'dbscan' or 'kmeans'
    n_clusters=8,  # For kmeans
    eps=None,  # Auto-determined if None
    min_samples=10,  # Increased to require denser clusters
    n_components=3,
    max_stocks=None
):
    """
    Cluster stocks by their volatility patterns with balanced cluster sizes.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with time-series data for multiple stocks
    start_date, end_date : str
        Date range for analysis
    min_data_pct : float
        Minimum percentage of trading days required
    algorithm : str
        'dbscan' or 'kmeans'
    n_clusters : int
        Number of clusters for kmeans
    eps : float or None
        DBSCAN epsilon (auto-calculated if None)
    min_samples : int
        DBSCAN min_samples
    n_components : int
        Number of PCA components
    max_stocks : int or None
        Maximum number of stocks to include
    """
    print("Preparing data for stock clustering...")
    
    # Parse dates and filter data
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    period_df = df.filter((pl.col('date') >= start) & (pl.col('date') <= end))
    
    # Count total available trading days in the period
    total_days = period_df['date'].n_unique()
    print(f"Period contains {total_days} trading days")
    
    # Define key volatility indicators (focusing on the most relevant)
    exclude_patterns = [
        'act_symbol', 'date', 'open', 'high', 'low', 'close', 'volume',
        'gvz', 'ovx', 'vix', 'vvix', 'vxapl', 'vxazn', 'vxeem'
    ]
    
    feature_cols = [col for col in df.columns 
                  if not any(pattern in col for pattern in exclude_patterns)]
    print(f"Using {len(feature_cols)} volatility indicators for clustering")
    
    # Find stocks with sufficient data
    stock_counts = period_df.group_by('act_symbol').agg(pl.count('date').alias('days'))
    min_days = int(total_days * min_data_pct / 100)
    valid_stocks = stock_counts.filter(pl.col('days') >= min_days)['act_symbol'].to_list()
    print(f"Found {len(valid_stocks)} stocks with at least {min_data_pct}% of trading days")
    
    # Limit number of stocks if requested
    if max_stocks and len(valid_stocks) > max_stocks:
        print(f"Sampling {max_stocks} stocks to manage memory usage")
        np.random.seed(42)
        valid_stocks = np.random.choice(valid_stocks, max_stocks, replace=False).tolist()
    
    # Process stocks and compute statistics
    print("Computing indicator statistics for each stock...")
    
    # Process stocks in batches to reduce memory usage
    batch_size = 500
    stock_stats = []
    
    for i in range(0, len(valid_stocks), batch_size):
        batch_stocks = valid_stocks[i:i+batch_size]
        print(f"Processing stocks {i+1} to {min(i+batch_size, len(valid_stocks))}...")
        
        # Filter data for this batch
        batch_df = period_df.filter(pl.col('act_symbol').is_in(batch_stocks))
        
        # Compute statistics more selectively to focus on time series characteristics
        stats_df = (batch_df
                   .group_by('act_symbol')
                   .agg([
                       # Mean (average level)
                       *[pl.col(col).mean().alias(f"{col}_mean") for col in feature_cols],
                       # Standard deviation (variability)
                       *[pl.col(col).std().alias(f"{col}_std") for col in feature_cols],
                       # Skewness (asymmetry)
                       *[pl.col(col).skew().alias(f"{col}_skew") for col in feature_cols],
                       # Median (robust central tendency)
                       *[pl.col(col).median().alias(f"{col}_median") for col in feature_cols],
                       # Quantiles (distribution shape)
                       *[pl.col(col).quantile(0.75).alias(f"{col}_q75") for col in feature_cols],
                       *[pl.col(col).quantile(0.25).alias(f"{col}_q25") for col in feature_cols]
                   ]))
        
        stock_stats.append(stats_df)
    
    # Combine all batches
    all_stats_df = pl.concat(stock_stats)
    print(f"Computed statistics for {all_stats_df.height} stocks")
    
    # Fill null values with zeros
    all_stats_df = all_stats_df.fill_null(0)
    
    # Extract feature matrix
    feature_names = [col for col in all_stats_df.columns if col != 'act_symbol']
    X = all_stats_df.select(feature_names).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale the data
    print("Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance: {np.sum(explained_variance):.3f}")
    
    # Create PCA DataFrame
    pca_df = pl.DataFrame({
        'act_symbol': all_stats_df['act_symbol'],
        **{f'PC{i+1}': X_pca[:, i] for i in range(n_components)}
    })
    
    # Apply clustering algorithm
    if algorithm.lower() == 'kmeans':
        print(f"Clustering with K-means (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        
    else:  # DBSCAN
        # If eps is not provided, use nearest neighbors to estimate it
        if eps is None:
            print("Estimating optimal eps value...")
            
            # Calculate distances to nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(50, X_pca.shape[0]-1)).fit(X_pca)
            distances, _ = nbrs.kneighbors(X_pca)
            
            # Sort distances to the kth neighbor
            k = min_samples
            kth_distances = np.sort(distances[:, k])
            
            # Find the "elbow point" in the distance curve
            from scipy.signal import find_peaks
            # Use the negative of distances to find peaks instead of valleys
            neg_dists = -np.diff(kth_distances)
            # Find peaks with prominence at least 10% of the range
            peaks, _ = find_peaks(neg_dists, prominence=0.1*(np.max(neg_dists)-np.min(neg_dists)))
            
            if len(peaks) > 0:
                # Use the first significant peak as the elbow point
                elbow_idx = peaks[0]
                eps_value = kth_distances[elbow_idx]
            else:
                # If no clear peak, use a percentile-based approach
                eps_value = np.percentile(kth_distances, 25)  # 25th percentile
            
            print(f"Estimated optimal eps={eps_value:.3f}")
            
            # Plot the k-distance graph
            plt.figure(figsize=(10, 6))
            plt.plot(kth_distances)
            plt.axhline(y=eps_value, color='r', linestyle='--')
            plt.xlabel(f'Points (sorted by distance to {k}th neighbor)')
            plt.ylabel(f'Distance to {k}th neighbor')
            plt.title(f'K-distance Graph (k={k})')
            plt.grid(True, alpha=0.3)
            plt.text(len(kth_distances)*0.6, eps_value*1.1, f'eps={eps_value:.3f}', 
                    color='red', fontsize=12)
            plt.show()
        else:
            eps_value = eps
        
        print(f"Clustering with DBSCAN (eps={eps_value}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca)
    
    # Count clusters
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) if -1 in labels else 0
    
    print(f"Found {n_clusters_found} clusters and {n_noise} noise points")
    
    # Calculate cluster sizes and check balance
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise points: {count} ({count/len(labels)*100:.1f}%)")
        else:
            print(f"Cluster {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Create final DataFrame with clusters
    stock_clusters = pca_df.with_columns(pl.Series(name='cluster', values=labels))
    
    return stock_clusters, pca_df, pca

def plot_stock_clusters(stock_clusters, pca=None):
    """
    Visualize stock clusters in PCA space.
    
    Parameters:
    -----------
    stock_clusters : pl.DataFrame
        DataFrame with act_symbol, PCA components, and cluster assignments
    pca : PCA model, optional
        Fitted PCA model for plotting explained variance
    """
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create 2D scatter plot
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Extract data
    pc1 = stock_clusters['PC1'].to_numpy()
    pc2 = stock_clusters['PC2'].to_numpy()
    clusters = stock_clusters['cluster'].to_numpy()
    
    # Plot clusters
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_clusters))))
    
    for i, cluster in enumerate(unique_clusters):
        mask = clusters == cluster
        if cluster == -1:
            ax1.scatter(pc1[mask], pc2[mask], c='gray', label='Noise', alpha=0.3, s=20)
        else:
            ax1.scatter(pc1[mask], pc2[mask], label=f'Cluster {cluster}', 
                       alpha=0.8, s=40, color=colors[i % len(colors)])
    
    ax1.set_title('Stock Clusters (PC1 vs PC2)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create 3D scatter plot if we have enough components
    if 'PC3' in stock_clusters.columns:
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        pc3 = stock_clusters['PC3'].to_numpy()
        
        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            if cluster == -1:
                ax2.scatter(pc1[mask], pc2[mask], pc3[mask], c='gray', label='Noise', 
                           alpha=0.3, s=20)
            else:
                ax2.scatter(pc1[mask], pc2[mask], pc3[mask], label=f'Cluster {cluster}', 
                           alpha=0.8, s=40, color=colors[i % len(colors)])
        
        ax2.set_title('Stock Clusters (3D)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
    
    # Plot explained variance
    if pca is not None:
        ax3 = fig.add_subplot(2, 2, 3)
        
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        components = range(1, len(variance_ratio) + 1)
        
        ax3.bar(components, variance_ratio, alpha=0.7, label='Individual')
        ax3.step(components, cumulative_variance, where='mid', color='red', label='Cumulative')
        ax3.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('PCA Explained Variance')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot cluster sizes
    ax4 = fig.add_subplot(2, 2, 4)
    
    cluster_counts = stock_clusters.group_by('cluster').agg(pl.count().alias('count'))
    cluster_counts = cluster_counts.sort('cluster')
    
    cluster_sizes = {}
    for i, row in enumerate(cluster_counts.iter_rows(named=True)):
        cluster_sizes[row['cluster']] = row['count']
    
    # Plot regular clusters
    regular_clusters = [c for c in unique_clusters if c != -1]
    if regular_clusters:
        regular_counts = [cluster_sizes.get(c, 0) for c in regular_clusters]
        ax4.bar(regular_clusters, regular_counts, color=colors[:len(regular_clusters)])
    
    # Plot noise separately if present
    if -1 in unique_clusters:
        noise_count = cluster_sizes.get(-1, 0)
        ax4.bar(-1, noise_count, color='gray')
    
    ax4.set_title('Cluster Sizes')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Stocks')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some stats about clusters
    print("\nCluster statistics:")
    for cluster in np.unique(clusters):
        count = np.sum(clusters == cluster)
        if cluster == -1:
            print(f"Noise: {count} stocks ({count/len(clusters)*100:.1f}%)")
        else:
            print(f"Cluster {cluster}: {count} stocks ({count/len(clusters)*100:.1f}%)")
    
    # Show a few stocks from each cluster
    print("\nSample stocks by cluster:")
    for cluster in np.sort(np.unique(clusters)):
        cluster_stocks = stock_clusters.filter(pl.col('cluster') == cluster)['act_symbol'].to_list()
        if len(cluster_stocks) > 0:
            if cluster == -1:
                print(f"Noise examples: {', '.join(cluster_stocks[:15])}")
            else:
                print(f"Cluster {cluster} examples: {', '.join(cluster_stocks[:15])}")

# Example usage:
# Step 1: Use K-means for balanced clusters
# cluster_df, pca_df, pca = cluster_stocks_by_timeseries(
#     ohlcv, 
#     n_components=3,
#     algorithm='kmeans',
#     n_clusters=8  # Try different values for more or fewer clusters
# )
# plot_stock_clusters(cluster_df, pca)

# Step 2: Or try DBSCAN with auto-estimated epsilon
# cluster_df, pca_df, pca = cluster_stocks_by_timeseries(
#     ohlcv, 
#     n_components=3,
#     algorithm='dbscan',
#     eps=None,  # Auto-estimate
#     min_samples=10
# )
# plot_stock_clusters(cluster_df, pca)