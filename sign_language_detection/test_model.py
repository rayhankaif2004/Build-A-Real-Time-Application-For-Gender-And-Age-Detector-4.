import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# SETTINGS
# =========================

MODEL_PATH = "model/sign_language_model.keras"

CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "nothing"
]

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)

IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]

# =========================
# START CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\nCamera started.")
print("Show your hand to the camera.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Create input image
    image = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    predicted_class = CLASSES[predicted_index]

    # Display result
    text = f"{predicted_class} ({confidence:.1f}%)"

    cv2.putText(
        frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Language Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()