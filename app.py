import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Gesture-Based Computer Control",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Gesture-Based Computer Control")
st.write("Upload a hand gesture image to predict the gesture.")

# -----------------------------
# Load Trained CNN Model
# -----------------------------

MODEL_PATH = "gesture_model.keras"
IMG_SIZE = 128

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# IMPORTANT:
# Keep this order the same as the
# class order used during training.

class_names = [
    "fist",
    "open_palm",
    "swipe_left",
    "swipe_right",
    "thumbs_up"
]

# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Gesture",
        width=300
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------

    processed_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(processed_image) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index]

    gesture = class_names[predicted_index]

    # -----------------------------
    # Display Result
    # -----------------------------

    st.success(
        f"Detected Gesture: {gesture.replace('_', ' ').title()}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )
