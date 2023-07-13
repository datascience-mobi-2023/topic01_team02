import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

train_data = pd.read_csv("../mnist_train.csv", header=None)
test_data = pd.read_csv("../mnist_test.csv", header=None)

train_labels = train_data.iloc[:, 0].to_numpy()
train_data = train_data.iloc[:, 1:]  # Spalte 0 entfernen

test_labels = test_data.iloc[:, 0].to_numpy()
test_data = test_data.iloc[:, 1:]  # Spalte 0 entfernen


# Z-Transform for train and test datasets
train_data_z = StandardScaler().fit_transform(train_data)
test_data_z = StandardScaler().fit_transform(test_data)

# PCA for train and test datasets
pca = PCA(n_components=.95)
pca.fit(train_data_z)
pca.fit(test_data_z)
train_data_pca = pca.transform(train_data_z)
test_data_pca = pca.transform(test_data_z)

def knn_kdtree(train_data_pca, test_data_pca, k, train_labels, testsize):

    result = np.array([])
    kd_tree = spatial.KDTree(train_data_pca,leafsize=10)
    
    for i in range(0, len(test_data_pca), testsize):
        
        dist, neighbour_index = kd_tree.query(test_data_pca[i:i+testsize, None],p=2,k=k, workers = -1)
        neighbour_label = train_labels[neighbour_index]
        group_result = [mode(neighbour_label, axis=2, keepdims=True)[0]]
        group_result = np.squeeze(np.array(group_result)).astype(int)  # Konvertierung zu einem Integer-Array
        result = result.astype(int)
        result = np.concatenate((result, group_result), axis=0)  # Verkettung der Ergebnisse

    return result

results = []
k=4

# Calculate knn with kdtree
result = knn_kdtree(train_data_pca, test_data_pca, k, train_labels, 200)

# Confusion matrix for the best k (k=4)
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_labels, result)

# Ground truth and predicted lists
v_true = []
v_pred = []
cm = disp.confusion_matrix

# For each cell in confusion matrix add truths and predictions to the lists
for gr in range(len(cm)):
    for prediction in range(len(cm)):
        v_true += [gr] * cm[gr][prediction]
        v_pred += [prediction] * cm[gr][prediction]

print("Classification report from confusion matrix:\n"
    f"{metrics.classification_report(v_true, v_pred)}\n")