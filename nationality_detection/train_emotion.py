import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "fer2013.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_model.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 15

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading FER2013 dataset...")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")
print("Dataset loaded successfully.")


# ============================================================
# CONVERT PIXELS
# ============================================================

print("\nPreparing images...")

pixels = df["pixels"].values

X = np.array([
    np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
    for pixel_string in pixels
])

X = X.reshape(
    -1,
    IMG_SIZE,
    IMG_SIZE,
    1
)

X = X.astype("float32") / 255.0


# ============================================================
# LABELS
# ============================================================

y = df["emotion"].values

y = to_categorical(
    y,
    NUM_CLASSES
)


# ============================================================
# DATA SPLIT
# ============================================================

print("\nSplitting dataset...")

train_mask = df["Usage"] == "Training"
validation_mask = df["Usage"] == "PublicTest"
test_mask = df["Usage"] == "PrivateTest"

X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print(f"Training images:   {len(X_train)}")
print(f"Validation images: {len(X_validation)}")
print(f"Test images:       {len(X_test)}")


# ============================================================
# CREATE MODEL
# ============================================================

print("\nBuilding CNN model...")

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# TEST
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved successfully:\n"
    f"{MODEL_PATH}"
)

print("\nTraining complete.")