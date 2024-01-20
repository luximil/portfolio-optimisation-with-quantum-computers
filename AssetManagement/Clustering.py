# ----------------------------------------------------------------------------
# Written by Francisco Jose Manjon Cabeza Garcia for his Master Thesis at    |
# Technische Universität Berlin.                                             |
#                                                                            |
# This file contains a collection of methods to cluster assets into groups   |
# exibiting similar characteristics.                                         |
# ----------------------------------------------------------------------------

from k_means_constrained import KMeansConstrained
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def kmeansAssetClustering(asset_features: pd.DataFrame,
                          max_cluster_size: int,
                          min_cluster_size: int,
                          max_cluster_count: int,
                          min_cluster_count: int) -> list[list[str]]:
    """Cluster asset ticker into groups according to the passed asset features
    and using the constrained k-means algorithm.

    Parameters
    ----------
    asset_features : pandas.DataFrame
        Dataframe with the asset tickers or names as row index and the
        features as columns, i.e., for each asset there is one row containing
        all the features used for the clustering procedure. The row index
        items are returned in clusters.
    max_cluster_size : int
        Maximum number of elements in each cluster.
    min_cluster_size : int
        Minimum number of elements in each cluster.
    max_cluster_count : int
        Maximum number of clusters.
    min_cluster_count : int
        Minimum number of clusters.

    Returns
    -------
    clustered_assets : list[list[str]]
        List of clusters with their elements. Each element of the list is also
        a list with the string objects in the asset_features row index
        contained in the cluster.
    """

    assert max_cluster_size >= min_cluster_size, \
        ("The maximum number of elements in each cluster cannot be strictly"
         " less than the minimum number of elements in each cluster.")
    assert max_cluster_count >= min_cluster_count, \
        ("The maximum number of clusters cannot be strictly less than the"
         " minimum number of clusters.")

    # Initialise the best clustering result and its inertia.
    best_clustering = None
    best_clustering_inertia = np.inf

    # Iterate over all number of clusters within the interval
    # [min_cluster_count, max_cluster_count].
    for n_clusters in range(min_cluster_count, max_cluster_count+1):
        # Optimise the constrained k-means clustering for n_clusters cluster
        # count.
        constrained_kmeans_cluster = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=min_cluster_size,
            size_max=max_cluster_size,
            random_state=0
        )
        clustering = constrained_kmeans_cluster.fit(asset_features)

        # If the constrained k-means clustering for n_clusters cluster count
        # is better than the best one found until now, update the best one
        # with the current one values.
        if clustering.inertia_ < best_clustering_inertia:
            best_clustering = clustering
            best_clustering_inertia = best_clustering_inertia

    # Create a list containing a list for each computed cluster which in turn
    # contains the names of the assets from the asset_features row index.
    clustered_assets = [
        asset_features.index[best_clustering.labels_ == i].to_list()
        for i in range(best_clustering.labels_.max())
        ]

    return clustered_assets


def CategoricalAssetClustering(asset_features: pd.DataFrame) -> pd.Series:
    """Cluster assets into groups according to the features in the passed
    DataFrame columns.

    Parameters
    ----------
    asset_features : pandas.DataFrame
        pandas DataFrame row-indexed by asset name/ticker and with the
        features as columns.

    Returns
    -------
    clustered_assets : pandas.Series
        pandas DataFrame row-indexed by unique combination of feature tuples
        and containing a list of asset names/tickers contained in the cluster
        defined by that unique combination of features' values.
    """

    # Extract the names of the features in the input data.
    features = asset_features.columns.to_list()
    # Reset the index of the input DataFrame to use the index as column to
    # aggregate by the features.
    clustered_assets = asset_features.reset_index(names="index")
    # Group by the features' columns and get the list of asset tickers with
    # each feature values combinations.
    clustered_assets = clustered_assets.groupby(features)["index"].apply(list)

    return clustered_assets


def AssetCorrelationDistance(asset_1: pd.Series | np.ndarray,
                             asset_2: pd.Series | np.ndarray) -> float:
    """Compute the distance between two asset timeseries using their
    correlation coefficient (corr) as in (0.5 * (1-corr))^(0.5).

    Parameters
    ----------
    asset_1 : pandas.Series or numpy.ndarray
        1-dimensional array containing the timeseries with the features of
        asset 1.
    asset_2 : pandas.Series or numpy.ndarray
        1-dimensional array containing the timeseries with the features of
        asset 2.

    Returns
    -------
    corr_dist_measure : float
        Distance between the two asset timeseries passed using their
        correlation coefficient (corr) as in (0.5 * (1-corr))^(0.5).
    """

    correlation = np.corrcoef(asset_1, asset_2)[0, 1]
    corr_dist_measure = np.sqrt(0.5 * (1 - correlation))

    return corr_dist_measure


def ComputeClusterDistance(cluster_1: pd.Series | pd.DataFrame,
                           cluster_2: pd.Series | pd.DataFrame,
                           distance_measure,
                           aggregation_function=np.nanmax) -> float:
    """Compute the distance between two clusters by computing the distances
    between each pair of elements from the two clusters and aggregating them.

    Parameters
    ----------
    cluster_1 : pandas.Series or pandas.DataFrame
        Cluster 1 elements with their features as rows, i.e., each element in
        the cluster has a column in `cluster_1` containing all the features
        used to compute the distance to `cluster_2` elements.
    cluster_2 : pandas.Series or pandas.DataFrame
        Cluster 2 elements with their features as rows, i.e., each element in
        the cluster has a column in `cluster_2` containing all the features
        used to compute the distance to `cluster_1` elements.
    distance_measure
        Function that takes two pandas.Series objects as input and returns a
        distance measure between the two. Each pair of `cluster_1` element and
        `cluster_2` element is passed onto this method to compute the distance
        between the two elements.
    aggregation_function, default numpy.nanmax
        Function that takes a `numpy.ndarray` object as input and returns a
        `float`. The table with the pairwise distances between `cluster_1` and
        `cluster_2` elements computed using the `distance_measure` method is
        passed onto this function to aggregate the distances accordingly. By
        default this function is the `numpy.nanmax` method, which returns the
        maximum non-NaN element in the array/table of pairwise distances.

    Returns
    -------
    cluster_distance : float
        Aggregated distance measure returned by the `aggregation_function`
        method.
    """

    # Check that the data for the elements in cluster 1 and cluster 2 have the
    # same features, i.e., the timeseries are equal in length and timestamps.
    assert cluster_1.index.equals(cluster_2.index), \
        ("The features of the elements in cluster_1 and cluster_2 must match.")

    # If the passed objects are pandas.Series objects, parse them into
    # pandas.DataFrame objects such that we can later transpose them and get a
    # row vector instead of a column vector (pandas.Series are column vectors).
    if isinstance(cluster_1, pd.Series):
        cluster_1 = pd.DataFrame(cluster_1)

    if isinstance(cluster_2, pd.Series):
        cluster_2 = pd.DataFrame(cluster_2)

    # Transpose the pandas.DataFrame objects to have the features, i.e.,
    # returns' timestamps as columns and the elements, i.e., asset tickers as
    # rows.
    cluster_1 = cluster_1.T
    cluster_2 = cluster_2.T

    # Compute the distances between all pairs of elements from cluster 1 and
    # cluster 2.
    all_pairs_distances = cdist(cluster_1, cluster_2, metric=distance_measure)
    # Compute the distance between the two clusters by aggregating all element
    # distances between clusters according to the aggregation function passed
    # as parameter.
    cluster_distance = aggregation_function(all_pairs_distances)

    return cluster_distance


def QuasiHierarchicalAssetClustering(asset_features: pd.DataFrame,
                                     max_cluster_size: int,
                                     distance_measure,
                                     distance_measures_aggregation_function,
                                     verbose: bool = False
                                     ) -> list[pd.DataFrame]:
    """Cluster assets following the hierarchical clustering method defined by
    Marcos Lopez de Prado in his book Advances in Financial Machine Learning.
    However, the clusters are not merged together after they reach the maximum
    number of assets given as parameter. Moreover, no tree or forest structure
    is returned, but a list where each element is pandas.DataFrame with the
    features of all assets in a cluster. Hence the name quasi-hierarchical
    asset clustering.

    Parameters
    ----------
    asset_features : pandas.DataFrame
        DataFrame with the asset features (e.g. returns' timeseries) as rows,
        i.e., each row corresponds to the features of a single asset.
    max_cluster_size : int
        Maximum number of assets allowed in each cluster. When two clusters
        are identified to be merged, they are only merged if the resulting
        cluster has size less than or equal to this parameter.
    distance_measure
        Function that takes two pandas.Series objects as input and returns a
        distance measure between the two. Each pair of elements from two
        different clusters is passed onto this method to compute the distance
        between the two elements.
    distance_measures_aggregation_function
        Function that takes a `numpy.ndarray` object as input and returns a
        `float`. The table with the pairwise distances between elements from 2
        clusters computed using the `distance_measure` method is passed onto
        this function to aggregate the distances accordingly. For example, the
        `numpy.nanmax` method, which returns the maximum non-NaN element in
        the array/table of pairwise distances.
    verbose : bool, default False
        If `True`, the two clusters to merge, the resulting merged cluster,
        the number of clusters and the cluster sizes at each iteration are
        printed.

    Returns
    -------
    clusters : list[pandas.DataFrame]
        List of pandas.DataFrame objects representing each asset cluster and
        containing the input asset features as columns.
    """

    assert max_cluster_size > 1, \
        ("The maximum cluster size must be strictly greater than 0.")

    # Initialise clusters by assigning each asset to a separate cluster.
    clusters = [asset_ts for _, asset_ts in asset_features.iterrows()]
    # Initialise the cluster sizes by setting them all to 1, because we have
    # initially assigned each asset to a separate cluster.
    cluster_sizes = np.ones(len(clusters))
    # Initialise the cluster distances.
    cluster_distances = np.full(
        shape=(asset_features.shape[0], asset_features.shape[0]),
        fill_value=np.nan
        )
    for i, cluster_1 in enumerate(clusters):
        for j, cluster_2 in enumerate(clusters):
            # Only fill the entries above the upper diagonal. No need to
            # compute the distance between an asset and itself (diagonal
            # elements) or compute the distances twice. Distance measures must
            # by definition be symmetric.
            if i < j:
                cluster_distances[i, j] = distance_measure(cluster_1,
                                                           cluster_2)

    while not np.isnan(cluster_distances).all():
        # There are clusters which can be merged respecting the maximum
        # cluster size of the resulting merged cluster.
        # Get the index of the two clusters which are the nearest.
        clusters_to_merge = np.unravel_index(
            np.nanargmin(cluster_distances, axis=None),
            shape=cluster_distances.shape
            )

        if verbose:
            print(f"Clusters to merge {clusters_to_merge}")

        if cluster_sizes[list(clusters_to_merge)].sum() > max_cluster_size:
            # The clusters which are intended to be merged would result in a
            # new cluster with size strictly greater than the maximum allowed.
            # Set the distance between these two clusters to NaN to avoid
            # selecting this pair again.
            cluster_distances[*clusters_to_merge] = np.nan

            if verbose:
                print(f"Clusters {clusters_to_merge} will not be merged"
                      " because the resulting cluster would contain"
                      f" {cluster_sizes[list(clusters_to_merge)].sum()}"
                      " elements, which are more than allowed.")

            # Skip this merge.
            continue

        # Merge the two clusters into one.
        cluster_1 = clusters.pop(clusters_to_merge[0])
        # Pop -1 because cluster_1 was already removed from the clusters list,
        # i.e., there is one element less, and cluster_2 index > cluster_1
        # index.
        cluster_2 = clusters.pop(clusters_to_merge[1]-1)
        new_cluster = pd.concat([cluster_1, cluster_2], axis=1)
        clusters.append(new_cluster)

        if verbose:
            print(f"New cluster\n{new_cluster.columns}")
            print(f"New number of clusters {len(clusters)}")

        # Update distance measures.
        # Remove the distance measures of the two clusters which were merged.
        cluster_distances = np.delete(cluster_distances,
                                      list(clusters_to_merge),
                                      axis=0
                                      )
        cluster_distances = np.delete(cluster_distances,
                                      list(clusters_to_merge),
                                      axis=1
                                      )
        # Compute the distances between the remaining clusters and the new one.
        distances_to_new_cluster = [
            ComputeClusterDistance(
                cluster_1=cluster,
                cluster_2=new_cluster,
                distance_measure=distance_measure,
                aggregation_function=distance_measures_aggregation_function)
            for cluster in clusters[:-1]]
        # Set the distance of the new cluster to itself to NaN.
        distances_to_new_cluster.append(np.nan)
        # Update the cluster distances matrix by adding a column with the
        # distances to the new cluster computed.
        distances_to_new_cluster = np.array(distances_to_new_cluster)
        cluster_distances = np.r_[
            cluster_distances,
            np.full((1, cluster_distances.shape[1]), np.nan)
            ]
        cluster_distances = np.c_[cluster_distances, distances_to_new_cluster]

        # Update cluster sizes.
        cluster_sizes = np.delete(cluster_sizes, list(clusters_to_merge))
        cluster_sizes = np.append(cluster_sizes, new_cluster.shape[1])

        if verbose:
            print(f"Current cluster sizes {cluster_sizes}")

    return clusters
