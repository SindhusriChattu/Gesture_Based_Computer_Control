import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# -----------------------------------
# Configuration
# -----------------------------------

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = "dataset"

# -----------------------------------
# Data Preprocessing
# -----------------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -----------------------------------
# Display Class Information
# -----------------------------------

print("\nClass Mapping:")
print(train_data.class_indices)

# -----------------------------------
# CNN Model
# -----------------------------------

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# -----------------------------------
# Compile Model
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------
# Model Summary
# -----------------------------------

model.summary()

# -----------------------------------
# Train Model
# -----------------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# -----------------------------------
# Save Model
# -----------------------------------

model.save("gesture_model.keras")

# -----------------------------------
# Save Class Names
# -----------------------------------

# Keras assigns class indexes alphabetically.
# We save the exact order so prediction
# uses the same mapping.

class_names = [
    name
    for name, index in sorted(
        train_data.class_indices.items(),
        key=lambda item: item[1]
    )
]

with open("class_names.json", "w") as file:
    json.dump(class_names, file)

# -----------------------------------
# Final Output
# -----------------------------------

print("\nTraining completed successfully.")

print("\nClass names:")
print(class_names)

print("\nSaved files:")
print("gesture_model.keras")
print("class_names.json")
