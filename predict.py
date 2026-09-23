import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "gesture_model.keras"
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "fist",
    "open_palm",
    "swipe_left",
    "swipe_right",
    "thumbs_up"
]


def predict_gesture(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]

    gesture = class_names[predicted_index]

    print("Gesture:", gesture)
    print("Confidence:", round(float(confidence) * 100, 2), "%")


image_path = input("Enter image path: ")

predict_gesture(image_path)
