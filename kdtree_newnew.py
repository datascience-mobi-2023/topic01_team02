import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


train_data = pd.read_csv("mnist_train.csv", header=None)
test_data = pd.read_csv("mnist_test.csv", header=None)

train_labels = train_data.iloc[:, 0].to_numpy()
train_data = train_data.iloc[:, 1:]  # Remove column 0

test_labels = test_data.iloc[:, 0].to_numpy()
test_data = test_data.iloc[:, 1:]  # Remove column 0


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
    num_test_samples = len(test_data_pca)
    num_batches = num_test_samples // batch_size

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size

        # Perform majority voting using KNeighborsClassifier

        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(train_data_pca, train_labels)
        batch_result = knn_classifier.predict(test_data_pca[batch_start:batch_end])
        

        #selected_algorithm = knn_classifier._fit_method
        #print("Ausgewählter Algorithmus:", selected_algorithm)

        result.append(batch_result)

    result = np.concatenate(result)

    return result

results = []
accuracies = []

# Test k values from 2 to 50
k_values = range(1, 5)
batch_size = 200

for k in k_values:
    # Calculate knn with nearest neighbors

    result = knn_nearest_neighbors(train_data_pca, test_data_pca, k, train_labels, batch_size)
    results.append(result)

    # Calculate accuracy

    accuracy = np.mean(result == test_labels[:len(result)]) * 100
    accuracies.append(accuracy)
    print("Accuracy for k =", k, "is", accuracy)


# Create csv file with real labels and predicted labels with different k-values

results_df = pd.DataFrame(results).transpose()
results_df.insert(0, "Real Label", test_labels)  # Add column for real label
results_df.columns = ["Real Label"] + list(k_values)  # Rename columns
results_df.to_csv("knn_results_optimized.csv", index=False)

# Find k with the highest accuracy

best_k = k_values[np.argmax(accuracies)]
best_accuracy = accuracies[np.argmax(accuracies)]
print("\nBest k:", best_k)
print("Best accuracy:", best_accuracy)


# Create csv file with k-values and accuracies

accuracy_df = pd.DataFrame({"k": k_values, "Accuracy": accuracies})
accuracy_df.to_csv("accuracy_results_optimized.csv", index=False)
