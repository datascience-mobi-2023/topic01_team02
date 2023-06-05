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
