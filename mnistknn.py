import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, svm, metrics
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
mnist = fetch_openml('mnist_784')
mnist.keys()
print("data field is 70k x 784 array, each row represents pixels from 28x28=784 image")
X1 = mnist.data
y1 = mnist.target
print("shape of data: ", X1.shape)
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.4)
pca_model = PCA(n_components=2)
pca_model.fit(trainData)
trainData = pca_model.transform(trainData)
testData = pca_model.transform(testData)
# take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)
kVals = range(1, 100, 1)
accuracies = []
for k in range(1, 100, 1):
    # train the classifier with the current value of `k`
    modelfork = KNeighborsClassifier(n_neighbors=k)
    modelfork.fit(trainData, trainLabels)
    score = modelfork.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
modelafterk = KNeighborsClassifier(n_neighbors=kVals[i])
modelafterk.fit(trainData, trainLabels)
predictions = modelafterk.predict(valData)
print(classification_report(valLabels, predictions))  

