from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def silhouette_(X, labels):
    score = silhouette_score(X, labels)
    return score


def calinski_score(X, labels):
    score = calinski_harabasz_score(X, labels)
    return score


def bouldin_score(X, labels):
    score = davies_bouldin_score(X, labels)
    return score
