import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_data = pd.read_csv("../mnist_train.csv")
test_data = pd.read_csv("../mnist_test.csv")

train_labels = train_data.iloc[:,0]
train_data = train_data.drop(['0'], axis=1)

test_labels = test_data.iloc[:,0]
test_data = test_data.drop(['0'], axis=1)

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

print(train_data_pca)

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

