####check if data is balanced or inbalanced

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

train_labels = train_data.iloc[:,0]
test_labels = test_data.iloc[:,0]

c_train = Counter(train_labels)
c_test = Counter(test_labels)

print(c_train)
print(c_test)

total_train_samples = len(train_labels)
total_test_samples = len(test_labels)

percentage_share_train = {label: (count / total_train_samples) * 100 for label, count in c_train.items()}
percentage_share_test = {label: (count / total_test_samples) * 100 for label, count in c_test.items()}

sorted_percentage_share_train = {label: percentage_share_train[label] for label in range(10)}
sorted_percentage_share_test = {label: percentage_share_test[label] for label in range(10)}

print("Training set label percentage share:")
for label, percentage in sorted_percentage_share_train.items():
    print(f"Label {label}: {percentage:.1f}%")

print("\nTest set label percentage share:")
for label, percentage in sorted_percentage_share_test.items():
    print(f"Label {label}: {percentage:.1f}%")

sorted_percentage_share_train_values = [round(percentage, 1) for percentage in sorted_percentage_share_train.values()]
sorted_percentage_share_test_values = [round(percentage, 1) for percentage in sorted_percentage_share_test.values()]

print(sorted_percentage_share_train_values)
print(sorted_percentage_share_test_values)

labels = np.arange(10)

bar_width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(labels, sorted_percentage_share_train_values, bar_width, label='Training set')
rects2 = ax.bar(labels + bar_width, sorted_percentage_share_test_values, bar_width, label='Test set')

ax.set_xlabel('Labels')
ax.set_ylabel('Percentage share')
ax.set_title('Percentage share of Labels in training and test dataset')
ax.set_xticks(labels + bar_width / 2)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

