import pandas as pd
import numpy as np
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

train_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_train.csv", header=None)
test_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_test.csv", header=None)

train_labels = train_data.iloc[:, 0]
train_data = train_data.iloc[:, 1:]

test_labels = test_data.iloc[:, 0]
test_data = test_data.iloc[:, 1:]

# Z-Transform for train and test datasets
train_data_z = StandardScaler().fit_transform(train_data)
test_data_z = StandardScaler().fit_transform(test_data)

n_components_range = [0.95]
average_runtimes = []

for n_components in n_components_range:
    # PCA for train and test datasets
    pca = PCA(n_components=n_components)
    train_data_pca = pca.fit_transform(train_data_z)
    test_data_pca = pca.transform(test_data_z)
    
    # Calculation of Euclidean Distance
    def euclidean_distance(row1, row2):
        distance = 0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i])**2
        return np.sqrt(distance)

    def labels_of_nearest_neighbours(distances, labels, k):
        distances = distances.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        distances_and_labels = np.concatenate((distances, labels), axis=1)
        sorted_labels_df = pd.DataFrame(
            distances_and_labels, columns=['distance', 'label'])
        sorted_labels_df = sorted_labels_df.sort_values('distance')
        return sorted_labels_df['label'].head(k).values

    def most_common_label(labels_array):
        most_common = Counter(labels_array).most_common(1)
        return most_common[0][0]


    def main_KNN(test_point, train_data_points, train_data_points_labels, k):
        list_of_distances = []
        for i in range(len(train_data_points)):
            distance = euclidean_distance(test_point, train_data_points[i])
            list_of_distances.append(distance)
        k_nearest_labels = labels_of_nearest_neighbours(
            np.array(list_of_distances), np.array(train_data_points_labels), k)
        predicted_label = most_common_label(k_nearest_labels)
        return int(predicted_label)

    num_images = 50  # Number of test images for which to measure the runtime
    execution_times = []

    for i in range(num_images):
        test_point = test_data_pca[i]
        runtime = timeit.timeit(lambda: main_KNN(test_point, train_data_pca, train_labels, k=4), number=1)
        execution_times.append(runtime)
    average_runtime = np.mean(execution_times)
    average_runtimes.append(average_runtime) 

total_test_data_images = len(test_data_pca)
average_runtime_overall = [(average_runtime * total_test_data_images) / 3600 for average_runtime in average_runtimes]
print(average_runtime_overall)

# Plotting the execution time against the number of principal components
plt.plot(n_components_range, average_runtime_overall)
plt.xlabel('Variance explained by number of Principal Components')
plt.ylabel('Execution Time (hours)')
plt.title('Execution Time vs. Principal Components')
plt.show()


