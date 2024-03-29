import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import time

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")
train_data_Lab = test_data

train_labels = train_data.iloc[:, 0]
train_data = train_data.drop(['0'], axis=1)

test_labels = test_data.iloc[:, 0]
test_data = test_data.drop(['0'], axis=1)

for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    im = ax.imshow(train_data.iloc[i].values.reshape(28, 28))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(train_labels.iloc[i])
plt.show()

# Z-Transform for train and test datasets
train_data_z = StandardScaler().fit_transform(train_data)
test_data_z = StandardScaler().fit_transform(test_data)

# PCA for train and test datasets
pca = PCA(n_components=0.95)
pca.fit(train_data_z)
pca.fit(test_data_z)
train_data_pca = pca.transform(train_data_z)
test_data_pca = pca.transform(test_data_z)

# Calculation of Euclidean Distance
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

def labels_of_nearest_neighbours(distances, labels, k):
    # Reshape distances and labels from a row to a column
    distances = distances.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    # Join the distances and labels together into an array and convert into a dataframe
    distances_and_labels = np.concatenate((distances, labels), axis=1)
    sorted_labels_df = pd.DataFrame(distances_and_labels, columns=['distance', 'label'])

    # Sort euclidean distances and labels in the dataframe in ascending order and return the first k labels
    sorted_labels_df = sorted_labels_df.sort_values('distance')
    return sorted_labels_df['label'].head(k).values

def most_common_label(labels_array):
    # Identify the most common label in k nearest neighbours
    most_common = Counter(labels_array).most_common(1)
    return most_common[0][0]

def main_KNN(test_point, train_data_points, train_data_points_labels, k):
    start_time = time.time()
    # Create an empty list of the euclidean distances
    list_of_distances = []

    # calculate the euclidean distances between the train data points and the new test data point
    for i in range(len(train_data_points)):
        distance = euclidean_distance(test_point, train_data_points[i])
        # add the euclidean distance to the list of euclidean distances
        list_of_distances.append(distance)

    # Identify the most common label in the k nearest neighbours and return the predicted label
    k_nearest_labels = labels_of_nearest_neighbours(np.array(list_of_distances),
                                                    np.array(train_data_points_labels), k)
    predicted_label = most_common_label(k_nearest_labels)
    elapsed_time = time.time() - start_time
    return predicted_label, elapsed_time

for i in range(len(test_data_pca)):
    final_result, runtime = main_KNN(test_data_pca[i], train_data_pca, train_labels, k=4)
    print("Real digit: " + str(test_labels[i]) + "; Predicted digit: " + str(final_result))
    print("Runtime: " + str(runtime) + " seconds")
    print("---------------------------")

