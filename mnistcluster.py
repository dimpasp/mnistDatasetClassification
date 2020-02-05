import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import pylab as pl
from sklearn.cluster import KMeans, SpectralClustering, FeatureAgglomeration
from sklearn.metrics.cluster import completeness_score
from sklearn import (manifold, decomposition)
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
import time
import datetime as dt


start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
mnist = fetch_openml('mnist_784')
print(mnist.data.shape)
X, y = np.float32(mnist.data[:70000])/ 255., np.float32(mnist.target[:70000])
X, y = shuffle(X,y)
X_train, y_train = np.float32(X[:5000])/255., np.float32(y[:5000])
X_test, y_test = np.float32(X[60000:])/ 255., np.float32(y[60000:])
X_reduced = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, n_neighbors=5).fit_transform(X_train)


spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
X = spectral.fit(X_reduced)
y_pred = spectral.fit_predict(X_reduced)
fig = plt.figure()

for i in range(0, X_reduced.shape[0]):
    if spectral.labels_[i] == 0:
        c1 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='red')
    elif spectral.labels_[i] == 1:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='blue')
    elif spectral.labels_[i] == 2:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='yellow')
    elif spectral.labels_[i] == 3:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='green')
    elif spectral.labels_[i] == 4:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='gray')
    elif spectral.labels_[i] == 5:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='orange')
    elif spectral.labels_[i] == 6:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='black')  
    elif spectral.labels_[i] == 7:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='azure')
    elif spectral.labels_[i] == 8:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='beige')
    elif spectral.labels_[i] == 9:
        c2 = pl.scatter(X_reduced[i,0], X_reduced[i,1], c='yellow')   
        
print("pososto plirotitas :")
print (completeness_score(y_train, y_pred))
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))





