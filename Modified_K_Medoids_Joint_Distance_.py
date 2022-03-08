import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython import embed
import time
import pandas as pd 
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from scipy.spatial.distance import jensenshannon
from lifelines.statistics import logrank_test
from lifelines.datasets import load_kidney_transplant
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from lifelines.statistics import multivariate_logrank_test
from sklearn.preprocessing import normalize, scale, MinMaxScaler



def _get_init_centers(n_clusters, n_samples):
    '''return random points as initial centers'''
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0,n_samples)
        if not _ in init_ids:
            init_ids.append(_)
    return init_ids


#Jensen Shanon distance
def _get_distance(data1, data2):
    dist = jensenshannon(data1, data2)
    return dist


def example_distance_func(data1, data2):
    '''example distance function'''
    return np.sqrt(np.sum((data1 - data2)**2))


def euclidean_distance(data1, data2):
    return euclidean(data1, data2)


def normalization(distance_matrix):
    
    if type(distance_matrix) != 'numpy.ndarray':
        distance_matrix = np.array(distance_matrix)
    
    scaler = MinMaxScaler()
    scaler.fit(distance_matrix.reshape(-1,1))
    scaler.transform(distance_matrix.reshape(-1,1))
    
    return distance_matrix
    
    

#compute the cost for example distance 
def _get_cost(n_clusters, data, X, centers_id, dist_func, dist_func_2, diarkeia, gegonota, min_n_obs):
    '''return total cost and cost of each cluster'''
    #st = time.time()
    dist_mat = np.zeros((len(X),len(centers_id)))
    dist_mat_2 = np.zeros((len(X),len(centers_id)))
    # compute distance matrix
    for j in range(len(centers_id)):
        center = X[centers_id[j],:]
        center_2 = data.values[centers_id[j],:]
        for i in range(len(X)):
            if i == centers_id[j]:
                dist_mat[i,j] = 0.
                dist_mat_2[i,j] = 0.
            else:
                dist_mat[i,j] = dist_func(X[i,:], center)
                dist_mat_2[i,j] = dist_func_2(data.values[i,:], center_2)
    #print( 'time: ', -st+time.time())
    dist_mat_2 = normalize(dist_mat_2)    
    dist_mat_sumed = np.add(dist_mat, dist_mat_2)    
    mask = np.argmin(dist_mat_sumed,axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))
    for i in range(len(centers_id)):
        mem_id = np.where(mask==i)        
        members[mem_id] = i
        costs[i] = np.nansum(dist_mat_sumed[mem_id,i])   
    if len(np.unique(members)) == n_clusters:    
        if any(j < min_n_obs for j in Counter(members).values()):                            
            log_rank_sc = 2            
        else:            
            data['labels'] = members            
            if n_clusters > 2:                
                results = multivariate_logrank_test(data[diarkeia], data['labels'], data[gegonota])                
                log_rank_sc = results.p_value                                
            elif n_clusters == 2:                
                group_1 = data[data['labels'] == 1.0]
                group_2 = data[data['labels'] == 0.0]
            
                kmf1 = KaplanMeierFitter() 
                kmf1.fit(group_1[diarkeia], group_1[gegonota])
                
                kmf2 = KaplanMeierFitter() 
                kmf2.fit(group_2[diarkeia], group_2[gegonota])
                
                results = logrank_test(group_1[diarkeia], group_2[diarkeia], event_observed_A=group_1[gegonota], event_observed_B=group_2[gegonota])
                
                log_rank_sc = results.p_value
            
        
    elif len(np.unique(members)) < n_clusters:
        
        log_rank_sc = 2
    
                
    return members, costs, np.sum(costs), dist_mat_sumed, log_rank_sc



def run_Cox_PH_model(data,train_samples):
    '''Run a Cox PH model''' 
    
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0) # sparse solutions,
    cph.fit(data[:train_samples], 'week', 'arrest')
    
    return cph
    

def _kmedoids_run(data, X, n_clusters, dist_func, dist_func_2, diarkeia, gegonota, min_n_obs, max_iter=10, tol=0.0001, verbose=True):
    '''run algorithm return centers, members, and etc.'''
    
    st = time.time()
    
    # Get initial centers
    n_samples, n_features = X.shape
    init_ids = _get_init_centers(n_clusters,n_samples)
    if verbose:
        print ('Initial centers are ', init_ids)
    centers = init_ids
    members, costs, tot_cost, dist_mat, log_rank_score = _get_cost(n_clusters, data, X, init_ids,dist_func, dist_func_2, diarkeia, gegonota, min_n_obs)
    print('1_costs:',costs)
    print('1_tot_cost:',tot_cost)
    print('log_rank_score:', log_rank_score)
    print(Counter(members).values())
    #break_out_flag = False
        
    cc,SWAPED = 0, True
    while True:
        SWAPED = False
        for i in range(n_samples):
            if not i in centers:
                for j in range(len(centers)):
                    centers_ = deepcopy(centers)
                    centers_[j] = i          
                    members_, costs_, tot_cost_, dist_mat_, log_rank_score_= _get_cost(n_clusters, data, X, centers_,dist_func, dist_func_2, diarkeia, gegonota, min_n_obs)
                
                    if tot_cost_ < tot_cost and log_rank_score_ != 2 and log_rank_score_ < 0.01:#  and tot_cost_ < tot_cost:
                        
                        print('log_rank_score_:', log_rank_score_)
                        print('this is where new variables come in')
                        members, costs, tot_cost, dist_mat,log_rank_score= \
                        members_, costs_, tot_cost_, dist_mat_,log_rank_score_
                        centers = centers_
                        SWAPED = True
                        if verbose: 
                            print ('Change centers to ', centers)
                               
                        
        print('cc:',cc)
        if cc > max_iter:
            if verbose:
                print ('End Searching by reaching maximum iteration', max_iter)
            break
        if not SWAPED:
            
            if any(j < min_n_obs for j in Counter(members).values()):
                
                # Get initial centers AGAIN
                n_samples, n_features = X.shape
                init_ids = _get_init_centers(n_clusters,n_samples)
                
                if verbose:
                    print ('Initial centers are ', init_ids)
                    
                centers = init_ids
                members, costs, tot_cost, dist_mat, log_rank_score = _get_cost(n_clusters, data, X, init_ids,dist_func, dist_func_2, diarkeia, gegonota, min_n_obs)
                    
                continue
            
            elif all(j >= min_n_obs for j in Counter(members).values()):
                
                if verbose:
                    print ('End Searching by no swaps')
                    break
        cc += 1
        print(cc)
        
    print( 'time: ', -st+time.time())
    
    return centers,members, costs, tot_cost, dist_mat



class KMedoids_version_6(object):
    '''
    Main API of KMedoids Clustering

    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance

    Attributes
    --------
        labels_    :  cluster labels for each data item
        centers_   :  cluster centers id
        costs_     :  array of costs for each cluster
        n_iter_    :  number of iterations for the best trail

    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)

        predict(X): predict cluster id given a test dataset.
    '''
    def __init__(self,data, n_clusters, diarkeia, gegonota, dist_func, dist_func_2 ,min_n_obs, max_iter=10, tol=0.0001):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.dist_func_2 = dist_func_2
        self.max_iter = max_iter
        self.tol = tol
        self.data = data
        self.diarkeia = diarkeia
        self.gegonota = gegonota
        self.min_n_obs = min_n_obs

    def fit(self, data,X,diarkeia, gegonota, plotit=True, verbose=True):
        centers,members, costs,tot_cost, dist_mat = _kmedoids_run(data,
                X,self.n_clusters, self.dist_func, self.dist_func_2, self.diarkeia, self.gegonota, min_n_obs = self.min_n_obs, max_iter=self.max_iter, tol=self.tol,verbose=verbose)
        
        print(centers)
        print(members)
        
        if plotit:
            data['labels'] = members
            fig, ax = plt.subplots(1,1)
            colors = ['b','g','r','c','m','y','k']
            if self.n_clusters > len(colors):
                raise ValueError('we need more colors')
            
            for i in range(len(centers)):
                X_c = data.values[members==i,:]
                plt.figure()
                ax.scatter(X_c[:,0],X_c[:,1],c=colors[i],alpha=0.5,s=30)
                ax.scatter(X[centers[i],0],X[centers[i],1],c=colors[i],alpha=1., s=250,marker='*')
                plt.show()
                    
        if self.n_clusters == 3:
            
            data['labels'] = members
            
            group_1 = data[data['labels'] == 0.0]
            print(group_1)
            group_2 = data[data['labels'] == 1.0]
            print(group_2)
            group_3 = data[data['labels'] == 2.0]
            print(group_3)
    
            kmf1 = KaplanMeierFitter() 
            kmf1.fit(group_1[self.diarkeia], group_1[self.gegonota])
            
            kmf2 = KaplanMeierFitter() 
            kmf2.fit(group_2[self.diarkeia], group_2[self.gegonota])
            
            kmf3 = KaplanMeierFitter() 
            kmf3.fit(group_3[self.diarkeia], group_3[self.gegonota])
            
            ax = kmf1.plot()
            ax = kmf2.plot(ax = ax)
            ax = kmf3.plot(ax = ax)
            
            #ax.set_ylim([0.0, 1.0])
    
            
            print(group_1.shape)
            print(group_2.shape)
            print(group_3.shape)
            
            
            #results = multivariate_logrank_test(group_1[self.diarkeia], group_2[self.diarkeia], event_observed_A=group_1[self.gegonota], event_observed_B=group_2[self.gegonota])
            
            results = multivariate_logrank_test(data[diarkeia], data['labels'], data[gegonota])
            
            results.print_summary()
            print(results.p_value)        
            print(results.test_statistic) 
                        
            #print(data)
            
            #TSNE
            TSNE_plot(data)
            
            #PCA
            PCA_plot(data)
            
        elif self.n_clusters == 2:
            
            data['labels'] = members
            
            group_1 = data[data['labels'] == 0.0]
            print(group_1)
            group_2 = data[data['labels'] == 1.0]
            print(group_2)

    
            kmf1 = KaplanMeierFitter() 
            kmf1.fit(group_1[self.diarkeia], group_1[self.gegonota])
            
            kmf2 = KaplanMeierFitter() 
            kmf2.fit(group_2[self.diarkeia], group_2[self.gegonota])
            
                        
            ax = kmf1.plot()
            ax = kmf2.plot(ax = ax)
            
            #ax.set_ylim([0.0, 1.0])
    
            
            print(group_1.shape)
            print(group_2.shape)
            
            
            results = logrank_test(group_1[self.diarkeia], group_2[self.diarkeia], event_observed_A=group_1[self.gegonota], event_observed_B=group_2[self.gegonota])
            
            
            results.print_summary()
            print(results.p_value)        
            print(results.test_statistic) 
                        
            #print(data)
            
            #TSNE
            TSNE_plot(data)
            
            #PCA
            PCA_plot(data)
            
        return   

    def predict(self,X):
        raise NotImplementedError()
