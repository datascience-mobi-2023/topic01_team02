import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the train and test datasets from CSV files
train_data = pd.read_csv('/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_train.csv')
test_data = pd.read_csv('/Users/alexk/Documents/MoBi/4. FS/BioInfo/topic01_team02/mnist_test.csv')

# Separate features (pixels) and labels in train and test datasets
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape the training data
x_test = x_test.reshape(-1, 28, 28, 1)  # Reshape the testing data
x_train = x_train / 255.0  # Normalize the training data
x_test = x_test / 255.0  # Normalize the testing data
y_train = to_categorical(y_train)  # Convert the training labels to one-hot encoding
y_test = to_categorical(y_test)  # Convert the testing labels to one-hot encoding

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy:', accuracy)
