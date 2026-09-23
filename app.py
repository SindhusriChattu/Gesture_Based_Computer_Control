import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Gesture-Based Computer Control",
    page_icon="✋",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("✋ Gesture-Based Computer Control")

st.write(
    "Upload a hand gesture image "
    "to predict the gesture using CNN."
)

# -----------------------------------
# Configuration
# -----------------------------------

MODEL_PATH = "gesture_model.keras"
CLASS_PATH = "class_names.json"
IMG_SIZE = 128

# -----------------------------------
# Load Model
# -----------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# -----------------------------------
# Load Class Names
# -----------------------------------

@st.cache_data
def load_class_names():

    with open(
        CLASS_PATH,
        "r"
    ) as file:

        return json.load(file)


# -----------------------------------
# Initialize
# -----------------------------------

try:

    model = load_model()

    class_names = load_class_names()

except Exception as error:

    st.error(
        "Unable to load the trained CNN model."
    )

    st.info(
        "Make sure gesture_model.keras and "
        "class_names.json are present in "
        "the project repository."
    )

    st.stop()


# -----------------------------------
# Upload Image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Gesture",
        width=300
    )

    # -------------------------------
    # Preprocessing
    # -------------------------------

    processed_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        processed_image
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -------------------------------
    # CNN Prediction
    # -------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    confidence = float(
        prediction[0][predicted_index]
    )

    gesture = class_names[
        predicted_index
    ]

    # -------------------------------
    # Display Result
    # -------------------------------

    display_gesture = (
        gesture
        .replace("_", " ")
        .title()
    )

    st.success(
        f"Detected Gesture: {display_gesture}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # -------------------------------
    # Action Mapping
    # -------------------------------

    actions = {

        "thumbs_up":
            "Play / Start",

        "open_palm":
            "Pause",

        "swipe_right":
            "Next",

        "swipe_left":
            "Previous",

        "fist":
            "Stop"
    }

    action = actions.get(
        gesture,
        "No action assigned"
    )

    st.write(
        f"**Mapped Action:** {action}"
    )
