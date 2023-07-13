import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import time

train_data = pd.read_csv("mnist_train.csv", header=None)
test_data = pd.read_csv("mnist_test.csv", header=None)

train_labels = train_data.iloc[:, 0].to_numpy()
train_data = train_data.iloc[:, 1:]  # Spalte 0 entfernen

test_labels = test_data.iloc[:, 0].to_numpy()
test_data = test_data.iloc[:, 1:]  # Spalte 0 entfernen

# Z-Transform for train and test datasets
scaler = StandardScaler()
train_data_z = scaler.fit_transform(train_data)
test_data_z = scaler.transform(test_data)

# PCA for train and test datasets
pca = PCA(n_components=0.95)
train_data_pca = pca.fit_transform(train_data_z)
test_data_pca = pca.transform(test_data_z)

def knn_nearest_neighbors(train_data_pca, test_data_pca, k, train_labels, batch_size):
    result = []

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_data_pca)

    num_test_samples = len(test_data_pca)
    num_batches = num_test_samples // batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size

        distances, indices = nn.kneighbors(test_data_pca[batch_start:batch_end])
        neighbour_labels = train_labels[indices]

        # Perform majority voting using KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(train_data_pca, train_labels)

        start_time = time.time()
        batch_result = knn_classifier.predict(test_data_pca[batch_start:batch_end])
        end_time = time.time()
        result.append(batch_result)

        print("Runtime for batch", i+1, "is", end_time - start_time, "seconds")

    result = np.concatenate(result)

    return result

results = []
accuracies = []
k = 4
batch_size = 200

result = knn_nearest_neighbors(train_data_pca, test_data_pca, k, train_labels, batch_size)

