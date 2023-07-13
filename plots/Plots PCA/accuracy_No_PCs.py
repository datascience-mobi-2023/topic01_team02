import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


train_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_train.csv", header=None)
test_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_test.csv", header=None)

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


# Define the numbers of principal components to evaluate
component_values = list(range(5, 101, 5)) + list(range(100, 784, 25))

def knn_nearest_neighbors(train_data_pca, test_data_pca, k, train_labels, batch_size):
    result = [] #empty list 
    nn = NearestNeighbors(n_neighbors=k) 
    nn.fit(train_data_pca)
    num_test_samples = len(test_data_pca) #number of test images 
    num_batches = num_test_samples // batch_size #defines no. of batches (1 batch = test data images processed at same time) 
 
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        distances, indices = nn.kneighbors(test_data_pca[batch_start:batch_end])
        neighbour_labels = train_labels[indices]

        # Perform majority voting using KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(train_data_pca, train_labels)
        batch_result = knn_classifier.predict(test_data_pca[batch_start:batch_end])
        result.append(batch_result)
    result = np.concatenate(result)
    return result

results = []
accuracies = []

max_accuracy = 0.0
min_accuracy = 100.0
max_component = 0
min_component = 0

for n_components in component_values:
    # Reduce dimensionality using PCA
    train_data_pca_subset = train_data_pca[:, :n_components]
    test_data_pca_subset = test_data_pca[:, :n_components]

    # Calculate k-NN using KNN
    result = knn_nearest_neighbors(train_data_pca_subset, test_data_pca_subset, 4, train_labels, 200)

    # Calculate accuracy
    accuracy = np.mean(result == test_labels[:len(result)]) * 100
    accuracies.append(accuracy)
    print("Accuracy for", n_components, "components is", accuracy)

    # Update max and min accuracy values
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_component = n_components
    if n_components >= 75 and accuracy < min_accuracy:
        min_accuracy = accuracy
        min_component = n_components

# Plot accuracy vs. number of components
plt.plot(component_values, accuracies)
plt.xlabel("Number of Components")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Number of Components")

# Mark highest accuracy
plt.scatter(max_component, max_accuracy, color='red', label='Max Accuracy')
plt.annotate(f'{max_accuracy:.2f}% ({max_component})', (max_component, max_accuracy), xytext=(10, -10),
             textcoords='offset points', color='red')

# Mark lowest accuracy after 75 components
if min_component >= 75:
    plt.scatter(min_component, min_accuracy, color='green', label='Min Accuracy after 75 PCs')
    plt.annotate(f'{min_accuracy:.2f}% ({min_component})', (min_component, min_accuracy), xytext=(10, 10),
                 textcoords='offset points', color='green')

plt.legend()
plt.show()

