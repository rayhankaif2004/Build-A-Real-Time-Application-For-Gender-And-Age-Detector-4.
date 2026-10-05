import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

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
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="centered"
)

st.title("Sign Language Detection")
st.write("Show your hand to the camera and the model will predict the sign.")

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_sign_model():
    return load_model(MODEL_PATH)

model = load_sign_model()

# =========================
# VIDEO PROCESSOR
# =========================

class SignLanguageProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        # Resize for model
        image = cv2.resize(img, (64, 64))

        # Normalize
        image = image.astype("float32") / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Prediction
        prediction = model.predict(image, verbose=0)

        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index] * 100

        predicted_class = CLASSES[predicted_index]

        # Display prediction
        text = f"{predicted_class} ({confidence:.1f}%)"

        cv2.putText(
            img,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================
# START CAMERA
# =========================

webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

st.markdown("---")

st.subheader("Supported Signs")

st.write("A  |  B  |  C  |  D  |  E  |  Nothing")