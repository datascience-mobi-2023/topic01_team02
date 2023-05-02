import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv()
test_data = pd.read_csv()

train_labels = train_data.iloc[:,0]
train_data = train_data.drop(['0'], axis=1)

for i in range(10):
    ax= plt.subplot(1,10 ,i+1)
    im=ax.imshow(train_data.iloc[i].values.reshape(28,28))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(train_labels.iloc[i])
plt.show()
