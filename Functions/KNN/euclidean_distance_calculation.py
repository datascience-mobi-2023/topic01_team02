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

# Calculate Euclidean Distance for each test sample against all train samples

ed_list = []
for i in range(len(train_data_pca)):
    ed = euclidean_distance(test_data_pca[0], train_data_pca[i])
    ed_list.append(ed)
   
# Create a DataFrame from the distances list
df_distances = pd.DataFrame(ed_list)
df_distances = df_distances.sort_values(by=0)

# Save DataFrame to a CSV file
df_distances.to_csv("euclidean_distances.csv", index=False)
