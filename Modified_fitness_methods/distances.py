import numpy as np 
import pandas as pd 
from lifelines.statistics import logrank_test 
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

#Example distance
def example_distance_func(data1, data2):
    '''example distance function'''
    return np.sqrt(np.sum((data1 - data2)**2))


#log rank distance (Do not use)
def log_rank_dist_func(data1, data2, time_col, event_col):
    '''distance function based on the log rank test statistic'''
    result = logrank_test(data1[:time_col], data2[:time_col], event_observed_A=data1[:event_col], event_observed_B=data2[:event_col])
    return result.p_value


def euclidean_distance(data1, data2):
    return euclidean(data1, data2)


#Jensen Shanon distance
def Jensen_Shanon_distance(data1, data2):
    dist = jensenshannon(data1, data2)
    return dist


#wasserstein distance
def wasserstein_distance_(data1, data2):
    dist = wasserstein_distance(data1, data2)
    return dist
