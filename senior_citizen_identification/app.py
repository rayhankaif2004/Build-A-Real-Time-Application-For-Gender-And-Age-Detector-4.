import os
import csv
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "AGE_GENDER_DETECTION-main",
    "Age_Sex_Detection.h5"
)

LOG_FILE = os.path.join(BASE_DIR, "senior_citizen_log.csv")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Senior Citizen Identification",
    page_icon="👴",
    layout="wide"
)

st.title("Senior Citizen Identification")

st.write(
    "Upload a video containing multiple people. "
    "The application detects faces, predicts age and gender, "
    "identifies senior citizens, and records the results."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_age_gender_model():
    return load_model(MODEL_PATH, compile=False)


try:
    model = load_age_gender_model()
except Exception as e:
    st.error("Unable to load the Age & Gender model.")
    st.code(str(e))
    st.stop()


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# CSV LOGGING
# ============================================================

def save_log(age, gender, is_senior):

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow(
                [
                    "Date",
                    "Time",
                    "Age",
                    "Gender",
                    "Senior Citizen"
                ]
            )

        now = datetime.now()

        writer.writerow(
            [
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                age,
                gender,
                "Yes" if is_senior else "No"
            ]
        )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_age_gender(face):

    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)

    # Original project model:
    # output 0 = gender
    # output 1 = age

    if isinstance(prediction, list) and len(prediction) >= 2:

        gender_output = prediction[0]
        age_output = prediction[1]

        gender_value = float(
            np.array(gender_output).reshape(-1)[0]
        )

        age_value = float(
            np.array(age_output).reshape(-1)[0]
        )

    else:

        values = np.array(prediction).reshape(-1)

        if len(values) >= 2:

            gender_value = float(values[0])
            age_value = float(values[1])

        else:

            return None, None

    gender = "Male" if gender_value < 0.5 else "Female"

    age = int(round(age_value))

    age = max(1, min(age, 100))

    return age, gender


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(input_path, output_path):

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return False, 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 20

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    detected_people = 0

    # Process every few frames for better speed
    frame_skip = 5

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Only perform detection on selected frames
        if frame_count % frame_skip == 0:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            detected_people = len(faces)

            for (x, y, w, h) in faces:

                face = frame[
                    y:y + h,
                    x:x + w
                ]

                if face.size == 0:
                    continue

                try:
                    age, gender = predict_age_gender(face)

                except Exception as e:
                    print("Prediction error:", e)
                    continue

                if age is None:
                    continue

                is_senior = age > 60

                # Senior citizens = red box
                if is_senior:

                    box_color = (0, 0, 255)

                    label = (
                        f"Senior Citizen | "
                        f"{gender} | {age}"
                    )

                else:

                    box_color = (0, 255, 0)

                    label = f"{gender} | {age}"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x, y - 30),
                    (x + w, y),
                    box_color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # Save result
                save_log(
                    age,
                    gender,
                    is_senior
                )

        writer.write(frame)

    cap.release()
    writer.release()

    return True, detected_people


# ============================================================
# GUI
# ============================================================

st.markdown("---")

st.subheader("Upload Video")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "m4v"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button(
        "Analyze Video",
        type="primary"
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_input:

            temp_input.write(
                uploaded_video.read()
            )

            input_path = temp_input.name

        output_path = os.path.join(
            tempfile.gettempdir(),
            "senior_citizen_output.mp4"
        )

        with st.spinner(
            "Processing video. Please wait..."
        ):

            success, people_count = process_video(
                input_path,
                output_path
            )

        if success:

            st.success(
                "Video processed successfully."
            )

            st.metric(
                "People detected in processed frame",
                people_count
            )

            st.subheader(
                "Processed Video"
            )

            st.video(output_path)

            if os.path.exists(LOG_FILE):

                with open(
                    LOG_FILE,
                    "rb"
                ) as file:

                    st.download_button(
                        "Download CSV Report",
                        data=file,
                        file_name="senior_citizen_log.csv",
                        mime="text/csv"
                    )

        else:

            st.error(
                "Could not process the video."
            )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.subheader("Detection Rules")

st.write(
    "• Age above 60 → Senior Citizen"
)

st.write(
    "• Age 60 or below → Regular detection"
)

st.write(
    "• Senior citizens are highlighted with a red rectangle."
)

st.write(
    "• Age, gender and detection time are stored in CSV."
)

st.caption(
    "Predictions are estimates from the trained Age & Gender model "
    "and may not be accurate for every recording."
)