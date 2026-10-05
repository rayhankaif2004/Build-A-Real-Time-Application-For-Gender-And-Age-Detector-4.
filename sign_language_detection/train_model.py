import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =========================
# SETTINGS
# =========================

CLASSES = ["A", "B", "C", "D", "E", "nothing"]

IMG_SIZE = 64

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model.keras"
)

# =========================
# LOAD DATASET
# =========================

X = []
y = []

print("\nLoading dataset...\n")

for label, class_name in enumerate(CLASSES):

    class_folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_folder):
        print(f"ERROR: Folder not found: {class_folder}")
        exit()

    images = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name}: {len(images)} images")

    for image_name in images:

        image_path = os.path.join(
            class_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        X.append(image)
        y.append(label)

# =========================
# PREPARE DATA
# =========================

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print("\nTotal images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train = to_categorical(
    y_train,
    num_classes=len(CLASSES)
)

y_test = to_categorical(
    y_test,
    num_classes=len(CLASSES)
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# =========================
# CREATE MODEL
# =========================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        len(CLASSES),
        activation="softmax"
    )
])

# =========================
# COMPILE
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# =========================
# EVALUATE
# =========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL TRAINING COMPLETE")
print("==============================")

print(f"Test accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)