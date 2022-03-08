from Modified_K_Medoids_Joint_Distance_ import KMedoids_version_6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
import lifelines
from lifelines import datasets
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from scipy.spatial.distance import jensenshannon
from lifelines.datasets import load_rossi
from lifelines.datasets import load_panel_test
from scipy.stats import wasserstein_distance
from lifelines.datasets import load_kidney_transplant
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


#####
from fitness_methods.distances import Jensen_Shanon_distance,  example_distance_func, wasserstein_distance_, log_rank_dist_func, euclidean_distance
from datasets import load_rossi_data, load_regression_dataset_lifelines, load_transplant_data, load_FLchain_data, load_support2_data, load_friendster_data
from plots import TSNE_plot, PCA_plot
from CoxPH_model import run_Cox_PH_model



'''

# In order to obtain the results you should run this script  
# Before execute the script make sure that you have choose the dataset you want to test the algorithm with 
# Also make sure that you have tunr appropriately the KMedoids_version_6() and fit() functions

'''


if __name__ == '__main__':

    #Random Data
    #X = np.random.normal(0,3,(500,2))

    print('####################################################################')

    #load a censored dataset from lifelines
    #survival_probabilities_Cox_PH, data = load_regression_dataset_lifelines()

    print('####################################################################')

    #LOAD ROSSI DATASET
    survival_probabilities_Cox_PH, data = load_rossi_data()

    print('####################################################################')

    #LOAD Transplant Dataset
    #survival_probabilities_Cox_PH, data = load_transplant_data()

    print('####################################################################')

    #FLCHAIN DATA
    #survival_probabilities_Cox_PH, data = load_FLchain_data()

    print('####################################################################')

    #Load support data
    #data = load_support2_data()

    #Min Max scaling the data
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(data.values)

    #Create new Dataframe from scaled data
    #columns = data.columns
    #x_scaled_df = pd.DataFrame(x_scaled, columns=columns)

    #Run parametric CoxPH regression
    #survival_probabilities_Cox_PH = run_Cox_PH_model(data, 'futime', 'death')
 
    #use KMedoids model (the traditional one)
    # model = KMedoids(n_clusters=2, dist_func=example_distance_func)

    #apply KMedoids algorithm in its first (traditional) version
    # model.fit(data.values, plotit=True, verbose=True)

    #use KMedoids model (the new modified versions)
    #model = KMedoids_new(data,n_clusters=2, diarkeia='week', gegonota='arrest', dist_func=Jensen_Shanon_distance, dist_func_2=example_distance_func)

    #apply KMedoids algorithm - using the Modified_K_Medoids.py or Modified_K_Medoids_new.py
    #model.fit(data,survival_probabilities_Cox_PH.values.T, diarkeia='week', gegonota='arrest', plotit=True, verbose=True)

    #For the Modified K medoids. py script
    #model = KMedoids(data,n_clusters=2, dist_func=Jensen_Shanon_distance)

    # apply KMedoids algorithm - using the Modified_K_Medoids.py or Modified_K_Medoids_new.py
    #model.fit(data,survival_probabilities_Cox_PH.values.T, plotit=True, verbose=True)

    #K Medoids 3rd version for k = 2
    #model = KMedoids_version_3(data,n_clusters=2, diarkeia='week', gegonota='arrest', dist_func=Jensen_Shanon_distance, dist_func_2=example_distance_func)

    # apply KMedoids algorithm from Modified version 3
    #model.fit(data,survival_probabilities_Cox_PH.values.T, diarkeia='week', gegonota='arrest', plotit=True, verbose=True)
    
    #K Medoids 4rth version for k = 3
    #model = KMedoids_version_4(data,n_clusters=3, diarkeia='futime', gegonota='death', dist_func=Jensen_Shanon_distance, dist_func_2=example_distance_func)

    # apply KMedoids algorithm from Modified version 4
    #model.fit(data,survival_probabilities_Cox_PH.values.T, diarkeia='futime', gegonota='death', plotit=True, verbose=True)
    
    #K Medoids 5th version for k = 3 BUT with JOINT distance function 
    model = KMedoids_version_6(data,n_clusters=2, diarkeia='week', gegonota='arrest', dist_func=Jensen_Shanon_distance, dist_func_2=example_distance_func, min_n_obs=180)

    # apply KMedoids algorithm from Modified version 5
    model.fit(data,survival_probabilities_Cox_PH.values.T, diarkeia='week', gegonota='arrest', plotit=True, verbose=True)
    
    
    
    