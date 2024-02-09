import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np


def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def classify_image(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    _, label, confidence = decoded_predictions[0]
    return label, confidence

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Replace 'path_to_your_image.jpg' with the actual path to your image file
image_path = '/home/jordan/Desktop/Python_examples/TensorFlow/image_classificator/number_seven.jpg'

# Classify the image
label, confidence = classify_image(model, image_path)

# Print the result
print(f'The image is classified as: {label} with confidence: {confidence}')

