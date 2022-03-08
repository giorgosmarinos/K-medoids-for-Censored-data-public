import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

def TSNE_plot(data):

    #TSNE
    model = TSNE(n_components=2, random_state=0)
    data_array = data.drop(columns=['labels'])
    X_embedded = model.fit_transform(data_array.values)

    print(X_embedded)

    # Plot
    plt.figure()
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = data['labels'])
    plt.title('TSNE plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def PCA_plot(data):

    #PCA
    pca = PCA(n_components = 2) # Choose number of components
    X_embedded = pca.fit_transform(data.values)

    # Plot
    plt.figure()
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = data['labels'])
    plt.title('PCA plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()