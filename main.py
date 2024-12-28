import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import cv2 as cv

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize the images to [0, 1]
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names in the CIFAR-10 dataset
class_names = ['Planes', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

"""
# Plotting the first 16 images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])  # Hide x ticks
    plt.yticks([])  # Hide y ticks
    plt.imshow(training_images[i])  # No need for `cmap=plt.cm.binary` as images are RGB
    plt.xlabel(class_names[training_labels[i][0]])  # Correct index for the label

plt.show()
"""

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

"""
# Build the model using Sequential API
model = models.Sequential() 

# First Conv2D layer - 32 filters, 3x3 kernel, ReLU activation function
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  

# MaxPooling layer - 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Second Conv2D layer - 64 filters, 3x3 kernel, ReLU activation function
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# MaxPooling layer - 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Third Conv2D layer - 64 filters, 3x3 kernel, ReLU activation function
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten layer to convert 2D matrices into 1D array for the dense layers
model.add(layers.Flatten())  

# Fully connected Dense layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Output Dense layer with 10 units (one for each class) and softmax activation for multi-class classification
model.add(layers.Dense(10, activation='softmax'))  # Output layer has 10 units for 10 classes

# Compile the model
model.compile(
    optimizer='adam',  # Using Adam optimizer for efficient training
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
    metrics=['accuracy']  # Track accuracy during training and evaluation
)

# Train the model
model.fit(
    training_images,  # Input training images (features)
    training_labels,  # True labels for the training images
    epochs=10,  # Number of times to iterate over the entire training dataset
    validation_data=(testing_images, testing_labels)  # Data used for validation (to evaluate the model's performance)
)

# Evaluate the model on the testing data (data it has not seen before)
loss, accuracy = model.evaluate(testing_images, testing_labels)

# Print the results
print(f'Loss: {loss}')  # Print the loss (error) value
print(f'Accuracy: {accuracy}')  # Print the accuracy (how many correct predictions the model made)

# Save the trained model to a file for future use
model.save('image_classifier_model.keras')  # Save the model in the Keras format
"""


# Load the trained model
model = load_model('image_classifier_model.keras')

# Load and preprocess the image
img = cv.imread('deer.jpg')  # Load the image
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Resize the image to match the input shape of the model (32x32)
img = cv.resize(img, (32, 32))

# Normalize the image (scale pixel values to range [0, 1])
img = img / 255.0

# Expand dimensions to match the input shape of the model (1, 32, 32, 3)
img = np.expand_dims(img, axis=0)

# Make a prediction
prediction = model.predict(img)

# Get the predicted class index
index = np.argmax(prediction)

# Print the prediction
print(f'Prediction is {class_names[index]}')


