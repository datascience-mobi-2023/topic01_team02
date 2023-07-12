import pandas as pd
import matplotlib.pyplot as plt

own_acc = pd.read_csv("concat_accuracies.csv", names=['k', 'Accuracy'], skiprows=1, nrows=9)
kdt_acc = pd.read_csv("accuracy_kd_scipy.csv", names=['k', 'Accuracy'], skiprows=1, nrows=9)
cla_acc = pd.read_csv("accuracy_classifier.csv", names=['k', 'Accuracy'], skiprows=2, nrows=9)

# Set up the figure and axes
fig, ax = plt.subplots()

# Plot the data for each set of k values and accuracies
ax.plot(own_acc['k'], own_acc['Accuracy'], label='Own Accuracy')
ax.plot(kdt_acc['k'], kdt_acc['Accuracy'], label='KDT Accuracy')
ax.plot(cla_acc['k'], cla_acc['Accuracy'], label='Classifier Accuracy')

# Set the labels and title
ax.set_xlabel('k values')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison')

# Add a legend
ax.legend()

# Show the plot
plt.show()
