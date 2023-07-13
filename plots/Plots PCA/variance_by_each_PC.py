import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_train.csv")
test_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_test.csv")

train_labels = train_data.iloc[:, 0]
train_data = train_data.drop(['0'], axis=1)

test_labels = test_data.iloc[:, 0]
test_data = test_data.drop(['0'], axis=1)

# Z-Transform for train and test datasets
train_data_z = StandardScaler().fit_transform(train_data)
test_data_z = StandardScaler().fit_transform(test_data)

# PCA for train dataset
pca = PCA(n_components=train_data.shape[1])
train_data_pca = pca.fit_transform(train_data_z)
variance_explained = pca.explained_variance_ratio_

components_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
variance_explained_to_plot = [variance_explained[i - 1] for i in components_to_plot]

plt.bar(range(1, len(components_to_plot) + 1), variance_explained_to_plot)
plt.xlabel("Component")
plt.ylabel("Variance Explained")
plt.title("Variance Explained by Specific Components")

# Annotate variance values above each bar
for i, variance in enumerate(variance_explained_to_plot):
    plt.annotate(f'{variance:.2f}', (i + 1, variance), xytext=(0, 5),
                 textcoords='offset points', ha='center')

plt.xticks(range(1, len(components_to_plot) + 1), components_to_plot)
plt.show()
