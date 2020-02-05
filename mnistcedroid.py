import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
mnist = fetch_openml('mnist_784')
mnist.keys()
print("data field is 70k x 784 array, each row represents pixels from 28x28=784 image")
X1 = mnist.data
y1 = mnist.target
print("shape of data: ", X1.shape)

#if test size=0.25then i have 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.4)
print("εγινε το train")
pca_model = PCA(n_components=2)
pca_model.fit(X_train)
X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)
X_train[:5]
kVals = range(1, 4, 1)
accuracies = []
for xx in range(1, 4, 1):
    
    clf = NearestCentroid(shrink_threshold=xx)
    clf.fit(X1, y1)
    y1_pred = clf.predict(X1)
    print(xx, np.mean(y1 == y1_pred))
    


print(classification_report(y1, y1_pred))    
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))    