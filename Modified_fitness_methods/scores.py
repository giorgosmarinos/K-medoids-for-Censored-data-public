from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette method 
def silhouette_(X, labels):
    score = silhouette_score(X, labels)
    return score

# Calinski score - to be used 
def calinski_score(X, labels):
    score = calinski_harabasz_score(X, labels)
    return score

# Bouldin score - to be usef 
def bouldin_score(X, labels):
    score = davies_bouldin_score(X, labels)
    return score
