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

# PCA for train and test datasets
pca = PCA(n_components=train_data.shape[1])
train_data_pca = pca.fit_transform(train_data_z)
variance_explained = pca.explained_variance_ratio_

# Plot the number of principal components and variance explained
fig, ax = plt.subplots()
plt.plot(np.cumsum(variance_explained))
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Principal Components', pad=20)  # Set the title pad to increase the distance from the plot

# Set the target variances and their respective principal components to mark
target_variances = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
target_components = []

# Find the number of principal components for each target variance
for target_variance in target_variances:
    index = np.argmax(np.cumsum(variance_explained) >= target_variance)
    target_components.append(index)

# Add vertical and horizontal dotted lines for each target variance
for target_variance, target_component in zip(target_variances, target_components):
    # Add vertical dotted line
    plt.axvline(x=target_component, color='gray', linestyle='--')
    plt.text(target_component, 1.01, str(target_component), transform=plt.gca().get_xaxis_transform(), ha='center')

    # Add horizontal dotted line
    plt.axhline(y=target_variance, color='gray', linestyle='--')
    plt.text(len(variance_explained) - 0.5, target_variance, str(target_variance), ha='right', va='center')

plt.show()
