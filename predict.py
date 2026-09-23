import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------------
# Configuration
# -----------------------------------

MODEL_PATH = "gesture_model.keras"
CLASS_PATH = "class_names.json"
IMG_SIZE = 128

# -----------------------------------
# Load Model
# -----------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# -----------------------------------
# Load Class Names
# -----------------------------------

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_gesture(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    confidence = prediction[0][predicted_index]

    gesture = class_names[predicted_index]

    print("\n-------------------------")
    print("Predicted Gesture:", gesture)
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )
    print("-------------------------\n")


# -----------------------------------
# User Input
# -----------------------------------

image_path = input(
    "Enter image path: "
)

predict_gesture(image_path)
