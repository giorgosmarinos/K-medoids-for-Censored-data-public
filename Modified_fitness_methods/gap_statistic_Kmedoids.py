# We'll wrap the `MeanShift` algorithm from sklearn

from sklearn.cluster import MeanShift
import Modified_K_Medoids_Joint_Distance_
from Modified_K_Medoids_Joint_Distance_ import KMedoids_version_6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
try:
    from sklearn.datasets.samples_generator import make_blobs
except ImportError:
    from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter 
from distances import Jensen_Shanon_distance, example_distance_func


# Cox Proportional Hazard Model 
def run_Cox_PH_model(data, time_column, death_column):
    '''Run a Cox PH model'''

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) 
    cph.fit(data, time_column, death_column)

    cph.print_summary()

    survival_probabilities_Cox_PH = cph.predict_survival_function(data)

    return survival_probabilities_Cox_PH


# Use this for custom clustering model 
def special_clustering_func(X, k):
    """
    Special clustering function that is used for the definition 
    alternative clustering method
    
    These user defined functions *must* take the X and a k 
    and can take an arbitrary number of other kwargs, which can
    be pass with `clusterer_kwargs` when initializing OptimalK
    """
    
    # Here you can do whatever clustering algorithm you heart desires,
    # but we'll do a simple wrap of the MeanShift model in sklearn.
    
    surv_probs = run_Cox_PH_model(data, 'week', 'arrest')

    m = KMedoids_version_6(X,n_clusters=k, diarkeia='week', gegonota='arrest', dist_func=Jensen_Shanon_distance, dist_func_2=example_distance_func, min_n_obs=180)
    m.fit(X,surv_probs.values.T, diarkeia='week', gegonota='arrest', plotit=True, verbose=True)
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return m.n_clusters #m.cluster_centers_, m.predict(X)


# Make some data
#X, y = make_blobs(n_samples=50, n_features=2, centers=3, random_state=25)

data = load_rossi()
special_clustering_func(data, k=2)

#print(n_c)
#print(pred)

# Define the OptimalK instance, but pass in our own clustering function
optimalk = OptimalK(clusterer=special_clustering_func)

# Use the callable instance as normal.
n_clusters = optimalk(data, n_refs=3, cluster_array=range(1, 4))

#print(n_clusters)