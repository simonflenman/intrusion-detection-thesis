import hdbscan

def build_hdbscan(min_cluster_size: int = 50,
                  min_samples:     int = None,
                  metric:          str = 'euclidean') -> hdbscan.HDBSCAN:
    """
    Returns an HDBSCAN instance with your chosen hyperparameters.
    - min_cluster_size: smallest cluster you care about
    - min_samples: if None, defaults to min_cluster_size
    - metric: distance metric
    """
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=(min_samples if min_samples is not None else min_cluster_size),
        metric=metric,
        cluster_selection_method='eom'
    )
