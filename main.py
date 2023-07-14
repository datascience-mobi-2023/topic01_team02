import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

train_data = pd.read_csv("mnist_train.csv", header=None)
test_data = pd.read_csv("mnist_test.csv", header=None)

train_labels = train_data.iloc[:, 0]
train_data = train_data.iloc[:, 1:] # Verwende alle Spalten außer der ersten und setze den Index zurück

test_labels = test_data.iloc[:,0]
test_data = test_data.iloc[:, 1:]

for i in range(10):
    ax= plt.subplot(1,10 ,i+1)
    im=ax.imshow(train_data.iloc[i].values.reshape(28,28))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(train_labels.iloc[i])
plt.show()

# Z-Transform for train and test datasets
train_data_z = StandardScaler().fit_transform(train_data)
test_data_z = StandardScaler().fit_transform(test_data)

# PCA for train and test datasets
pca = PCA(n_components=.95)
pca.fit(train_data_z)
pca.fit(test_data_z)
train_data_pca = pca.transform(train_data_z)
test_data_pca = pca.transform(test_data_z)

# Calculation of Euclidean Distance
def euclidean_distance(row1, row2):
 distance = 0
 for i in range(len(row1)-1):
    distance += (row1[i] - row2[i])**2
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
    
    # Create an empty list of the euclidean distances
    list_of_distances = []
    
    # calculate the euclidean distances between the train data points and the new test data point
    for i in range(len(train_data_points)):
        distance = euclidean_distance(test_point, train_data_points[i])
        # add the euclidean distance to the list of euclidean distances
        list_of_distances.append(distance)
    
    # Identify the most common label in the k nearest neighbours and return the predicted label
    k_nearest_labels = labels_of_nearest_neighbours(np.array(list_of_distances), np.array(train_data_points_labels), k)
    predicted_label = most_common_label(k_nearest_labels)
    return int(predicted_label)

predicted_labels = []
for i in range(len(test_data_pca)):
    test_point = test_data_pca[i]
    predicted_label = main_KNN(test_point, train_data_pca, train_labels, k=3)
    predicted_labels.append(predicted_label)
    print(predicted_label)

# Create a dataframe to store the true and predicted labels
results_df = pd.DataFrame({'True Label': test_labels, 'Predicted Label': predicted_labels})

# Calculate accuracy for k=3
accuracy = sum(results_df['True Label'] == results_df['Predicted Label']) / len(results_df) * 100
print("Accuracy for k=3:", accuracy)

# Save results to a CSV file
results_df.to_csv('knn_results.csv', index=False)

# Save accuracy to a separate CSV file
accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})
accuracy_df.to_csv('knn_accuracy.csv', index=False)