import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the train data
train_data = pd.read_csv("/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_train.csv")

# Separate labels from the train data
train_labels = train_data.iloc[:, 0]
train_data = train_data.drop(['0'], axis=1)

# Z-Transform for train dataset
train_data_z = StandardScaler().fit_transform(train_data)

# Perform PCA on the train data
pca = PCA(n_components=2)
components = pca.fit_transform(train_data_z)

# Create a DataFrame with components and labels
df = pd.DataFrame(components, columns=['Component 1', 'Component 2'])
df['Label'] = train_labels.astype(str)

# Define colors for different labels
label_colors = {
    '0': 'red',
    '1': 'blue',
    '2': 'green',
    '3': 'orange',
    '4': 'purple',
    '5': 'brown',
    '6': 'pink',
    '7': 'gray',
    '8': 'olive',
    '9': 'cyan'
}

# Create a scatter plot
plt.figure(figsize=(8, 8))
for label, color in label_colors.items():
    indices = df['Label'] == label
    plt.scatter(df.loc[indices, 'Component 1'], df.loc[indices, 'Component 2'], c=color, label=label, alpha=0.7)

# Set labels and title
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Scatter Plot of MNIST Digits")

# Add legend
plt.legend()

# Show the plot
plt.show()
