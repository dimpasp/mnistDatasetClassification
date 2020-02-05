import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
mnist = fetch_openml('mnist_784')
mnist.keys()
print("data field is 70k x 784 array, each row represents pixels from 28x28=784 image")
images = mnist.data
targets = mnist.target
X_data = images/255.0
Y = targets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.4  )
pca_model = PCA(n_components=2)
pca_model.fit(X_train)
X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)
X_train[:5]
param_C = 1000
param_gamma = 0.01 
classifier = svm.SVC(C=param_C,gamma=param_gamma)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
#υπολογισμός χρόνου υλοποίησης
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
