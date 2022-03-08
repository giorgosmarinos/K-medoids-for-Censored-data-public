from itertools import combinations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import distance_metrics
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.externals.joblib import Parallel, delayed
from lifelines import CoxPHFitter
from scipy.spatial.distance import jensenshannon, euclidean
from sklearn.preprocessing import normalize
from lifelines.datasets import load_rossi
import time
from sklearn.metrics import silhouette_score


# Modified fitness methods are currently under development
# So far, we have mostly completed the Silhouette method which needed to be 
# Modified in order to be able to return results for the proposed algorithm (Modified K medoids)


def run_Cox_PH_model(data, time_column, death_column):
    '''Run a Cox PH model to compute probabilities'''
    
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) 
    cph.fit(data, time_column, death_column)
    survival_probabilities_Cox_PH = cph.predict_survival_function(data)

    return survival_probabilities_Cox_PH


def silhouette_score_slow(X, labels, data, time_column, death_column, metric='euclidean', sample_size=None,
                          random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples.
    This method is computationally expensive compared to the reference one.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    To clarrify, b is the distance between a sample and the nearest cluster
    that b is not a part of.
    This function returns the mean Silhoeutte Coefficient over all samples.
    To obtain the values for each sample, use silhouette_samples
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    sample_size : int or None
        The size of the sample to use when computing the Silhouette
        Coefficient. If sample_size is None, no sampling is used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
    http://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    probs_ = run_Cox_PH_model(data, time_column, death_column)
    probs = probs_.T.values

    if sample_size is not None:
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            raise ValueError('Distance matrix cannot be precomputed')
        else:
            X, labels, data = X[indices], labels[indices], probs[indices]
    return np.mean(silhouette_samples_slow(X, labels, data, time_column, death_column, metric=metric, **kwds))


def silhouette_samples_slow(X, labels, data, time_column, death_column, metric='euclidean', **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhoeutte Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    This function returns the Silhoeutte Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by metrics.pairwise.pairwise_distances. If X is the distance
        array itself, use "precomputed" as the metric.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array, shape = [n_samples]
        Silhouette Coefficient for each samples.
    References
    ----------
    Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
    http://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    metric = distance_metrics()[metric]
    n = labels.shape[0]
    A = np.array([_intra_cluster_distance_slow(X, labels, metric, i, data, time_column, death_column)
                  for i in range(n)])
    B = np.array([_nearest_cluster_distance_slow(X, labels, metric, i, data, time_column, death_column)
                  for i in range(n)])
    sil_samples = (B - A) / np.maximum(A, B)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def _intra_cluster_distance_slow(X, labels, metric, i, data, time_column, death_column):
    """Calculate the mean intra-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    metric: function
        Pairwise metric function
    i : int
        Sample index being calculated. It is excluded from calculation and
        used to determine the current label
    Returns
    -------
    a : float
        Mean intra-cluster distance for sample i
    """
    indices = np.where(labels == labels[i])[0]
    if len(indices) == 0:
        return 0.
    first_distance = np.array([jensenshannon(data.values[i], data.values[j]) for j in indices if not i == j])
    second_distance = np.array([normalize(euclidean(X[i], X[j]).reshape(1,-1)) for j in indices if not i == j])
    joint_dist = np.mean([np.add(first_distance, second_distance)])
    return joint_dist


def _nearest_cluster_distance_slow(X, labels, metric, i, data, time_column, death_column):
    """Calculate the mean nearest-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    metric: function
        Pairwise metric function
    i : int
        Sample index being calculated. It is used to determine the current
        label.
    Returns
    -------
    b : float
        Mean nearest-cluster distance for sample i
    """
    label = labels[i]   
    b = np.min(
            [np.mean(
                [np.add(normalize(euclidean(X[i], X[j]).reshape(1,-1)),jensenshannon(data.values[i], data.values[j])) for j in np.where(labels == cur_label)[0]]
            ) for cur_label in set(labels) if not cur_label == label])

    
    return b





if __name__ == '__main__':
    
    lista_me_results = []
    rossi = load_rossi()

    # K-Medoids needs to run here 
    kmeans = KMeans(n_clusters=2, random_state=0).fit(rossi.values)
    labels = kmeans.labels_

    #Scikit Learn Silhouette
    t0 = time.time()
    s = silhouette_score(rossi.values, labels)
    t = time.time() - t0
    print('Scikit silhouette (%fs): %f' % (t, s))

    #Custom Silhouette score 
    t0 = time.time()
    s = silhouette_score_slow(X = rossi.values, labels = labels, data = rossi, time_column= 'week', death_column = 'arrest', metric='euclidean', sample_size=None,
                            random_state=None)
    t = time.time() - t0
    print('Slow silhouette (%fs): %f' % (t, s))
    