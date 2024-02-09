import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data('./seven_number.jpg')

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build a simple neural network
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images to a 1D array
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, verbose=False)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_accuracy}')

# Visualize predictions on a sample image
sample_image = test_images[0]
sample_label = test_labels[0]

# Reshape the image to (1, 28, 28) to match the input shape of the model
sample_image = sample_image.reshape((1, 28, 28))

# Make predictions
predictions = model.predict(sample_image)

# Display the results
predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
true_label = tf.argmax(sample_label).numpy()

plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {predicted_label}, True: {true_label}')
plt.show()
