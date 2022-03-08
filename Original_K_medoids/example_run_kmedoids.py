from Kmedoids import KMedoids
import numpy as np
import matplotlib.pyplot as plt
from lifelines.datasets import load_rossi

def example_distance_func(data1, data2):
    '''example distance function'''
    return np.sqrt(np.sum((data1 - data2)**2))

if __name__ == '__main__':
    #X = np.random.normal(0,3,(500,2))
    data = load_rossi()
    X = data.values[:, 2:]
    model = KMedoids(n_clusters=2, dist_func=example_distance_func)
    model.fit(X, plotit=True, verbose=True)
    plt.show()