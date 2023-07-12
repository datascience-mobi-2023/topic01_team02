import pandas as pd

k2 = pd.read_csv("knn_accuracy_2.csv")
k3 = pd.read_csv("knn_accuracy_3.csv")
k4 = pd.read_csv("knn_accuracy_4.csv")
k5 = pd.read_csv("knn_accuracy_5.csv")
k6 = pd.read_csv("knn_accuracy_6.csv")
k7 = pd.read_csv("knn_accuracy_7.csv")
k8 = pd.read_csv("knn_accuracy_8.csv")
k9 = pd.read_csv("knn_accuracy_9.csv")
k10 = pd.read_csv("knn_accuracy_10.csv")

# Add 'k' column to each DataFrame
k2['k'] = 2
k3['k'] = 3
k4['k'] = 4
k5['k'] = 5
k6['k'] = 6
k7['k'] = 7
k8['k'] = 8
k9['k'] = 9
k10['k'] = 10


# Concatenate the DataFrames vertically
concatenated_df = pd.concat([k2, k3, k4, k5, k6, k7, k8, k9, k10])

# Reset the index
concatenated_df.reset_index(drop=True, inplace=True)

# Reorder the columns
concatenated_df = pd.DataFrame(concatenated_df, columns=['k', 'Accuracy'])

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv("concat_accuracies.csv", index=False)

