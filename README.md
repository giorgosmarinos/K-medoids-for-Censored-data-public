# K Medoids algorithm for censored data 

#### This code corresponds to the pre-print version of paper **K-Medoids algorithm modification for censored data**. 
#### In this paper we present a modified version of K-medoids algorithm which can discover meaningful clusters both in terms of features but also in terms of survivability.
#### For this purpose we have introduced a custom distance metric which considers lifetime distribution differences and spatial differences like in the traditional K-medoids algorithm (e.g., euclidean distance)

#### To reproduce the experiments that are described in the paper you need to:

* Specify the parameters of the KMedoids_version_6 and fit functions inside the run_k_medoids.py and then execute the script:
  * data: *your data
  * n_clusters: *number of desired clusters    
  * diarkeia: *Define the time column, which would be different for various datasets (and certainly needs to be specified)
  * gegonota: *Define the binary event column
  * dist_func: *Distance function which is used to calculate the differencies between probability distributions. So far, for that parameter there is only one option which is Jensen Shanon distance 
  * dist_func_2: *Distance function which is used to calculate the differencies between features in the dataset. Currently standardized euclidean distance it is used. 
  * min_n_obs: *Define the minimun number of clusters you would like to obtain for each cluster. This parameter is useful to avoid return cluster that has no observations.


#### However, the repository is currently under development.
#### Stay tuned and the repository will be updated soon. Cheers!!!
