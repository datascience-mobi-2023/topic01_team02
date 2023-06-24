import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
from scipy.stats import mode

train_data = pd.read_csv("mnist_train.csv", header=None)
test_data = pd.read_csv("mnist_test.csv", header=None)

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
k_values = range(2, 21)

for k in k_values:
    result = knn_kdtree(train_data_pca, test_data_pca, k, train_labels, 200)
    results.append(result)

results_df = pd.DataFrame(results).transpose()
results_df.insert(0, "Real Label", test_labels)  # Hinzufügen der Spalte für echte Testlabels
results_df.columns = ["Real Label"] + list(k_values)  # Umbenennen der Spalten
results_df.to_csv("knn_results.csv", index=False)


accuracies = []
# Test k values from 1 to 10
for k in range(2, 11):
    # Perform KNN classification
    predictions = knn_kdtree(train_data_pca, test_data_pca, k, train_labels, 200)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels[:len(predictions)]) * 100
    accuracies.append(accuracy)
    print("Accuracy for k =", k, "is", accuracy)

# Find k with the highest accuracy
best_k = np.argmax(accuracies) + 1
best_accuracy = accuracies[best_k - 1]
print("\nBest k:", best_k)
print("Best accuracy:", best_accuracy)




