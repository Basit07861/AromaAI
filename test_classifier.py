import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained model
model = tf.keras.models.load_model("flower_model.h5")

# Define image size & classes (replace with your actual classes)
IMG_SIZE = 224
class_names = ['daisy', 'rose', 'tulip', 'sunflower', 'dandelion']  # update if needed

# Path to the image you want to test
test_image_path = "test_images/my_flower.jpg"  # put your test image here

# Load & preprocess image
img = Image.open(test_image_path).resize((IMG_SIZE, IMG_SIZE))
img_array = np.array(img) / 255.0   # normalize to 0-1
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_idx]

print(f"✅ Predicted class: {predicted_class}")
print("🔢 Prediction probabilities:", predictions[0])
