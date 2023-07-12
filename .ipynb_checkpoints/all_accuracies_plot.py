import pandas as pd

Accuracy_k2= pd.read_csv("knn_accuracy_2.csv")
Accuracy_k3= pd.read_csv("knn_accuracy_3.csv")
Accuracy_k4= pd.read_csv("knn_accuracy_4.csv")
Accuracy_k5= pd.read_csv("knn_accuracy_5.csv")
Accuracy_k6= pd.read_csv("knn_accuracy_6.csv")
Accuracy_k7= pd.read_csv("knn_accuracy_7.csv")


knn_accuracies = ["knn_accuracy_2.csv", "knn_accuracy_3.csv", "knn_accuracy_4.csv",
             "knn_accuracy_5.csv", "knn_accuracy_6.csv", "knn_accuracy_7.csv"]


Accuracy_knn = pd.DataFrame()

for i, filename in enumerate(knn_accuracies, start=2):
    data = pd.read_csv(filename)
    data.insert(0, "k", i)  
    Accuracy_knn = pd.concat([Accuracy_knn, data], ignore_index=True)

print(Accuracy_knn)

Accuracy_KDtrees_scipy= pd.read_csv("accuracy_kd_scipy.csv")


Accuracy_classifier= pd.read_csv("accuracy_classifier.csv")


#create plot for all accuracies from k=2 to k=7

import matplotlib.pyplot as plt

# Data
k_values = [2, 3, 4, 5, 6, 7]
knn_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

# Plotting
plt.figure(figsize=(8, 6))

# Plotting knn accuracies
for i, filename in enumerate(knn_accuracies, start=2):
    accuracies = pd.read_csv(filename)['accuracy']
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color=knn_colors[i-2], label=f'Accuracy_knn ({i})')

# Plotting KDtrees accuracy
plt.plot(k_values, Accuracy_KDtrees_scipy['accuracy'], marker='o', linestyle='-', color='cyan', label='Accuracy_KDtrees_scipy')

# Plotting classifier accuracy
plt.plot(k_values, Accuracy_classifier['accuracy'], marker='o', linestyle='-', color='magenta', label='Accuracy_classifier')

# Set axis labels and title
plt.xlabel('k-values')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

# Add legend
plt.legend()

# Show the plot
plt.show()