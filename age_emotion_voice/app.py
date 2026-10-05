import os
import tempfile
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SAMPLE_RATE = 16000


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Age & Emotion Detection Through Voice",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    gender_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "gender_model.pkl"
        )
    )

    gender_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "gender_encoder.pkl"
        )
    )

    age_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "age_model.pkl"
        )
    )

    emotion_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "emotion_model.pkl"
        )
    )

    emotion_encoder = joblib.load(
        os.path.join(
            MODEL_DIR,
            "emotion_encoder.pkl"
        )
    )

    feature_columns = joblib.load(
        os.path.join(
            MODEL_DIR,
            "feature_columns.pkl"
        )
    )

    # Optional senior model.
    # It was trained as an additional model, but the
    # application follows the project rule:
    # predicted age > 60 = senior citizen.
    senior_model_path = os.path.join(
        MODEL_DIR,
        "senior_model.pkl"
    )

    if os.path.exists(senior_model_path):
        senior_model = joblib.load(
            senior_model_path
        )
    else:
        senior_model = None

    training_info_path = os.path.join(
        MODEL_DIR,
        "training_info.pkl"
    )

    if os.path.exists(training_info_path):
        training_info = joblib.load(
            training_info_path
        )
    else:
        training_info = {}

    return (
        gender_model,
        gender_encoder,
        age_model,
        emotion_model,
        emotion_encoder,
        feature_columns,
        senior_model,
        training_info
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "The uploaded audio contains no usable audio."
        )

    features = {}

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    for i in range(20):

        features[
            f"mfcc_{i}_mean"
        ] = float(
            np.mean(mfcc[i])
        )

        features[
            f"mfcc_{i}_std"
        ] = float(
            np.std(mfcc[i])
        )

    # --------------------------------------------------------
    # MEL SPECTROGRAM
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=20
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    for i in range(20):

        features[
            f"mel_{i}_mean"
        ] = float(
            np.mean(mel_db[i])
        )

        features[
            f"mel_{i}_std"
        ] = float(
            np.std(mel_db[i])
        )

    # --------------------------------------------------------
    # ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = float(
        np.mean(zcr)
    )

    features["zcr_std"] = float(
        np.std(zcr)
    )

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=y)

    features["rms_mean"] = float(
        np.mean(rms)
    )

    features["rms_std"] = float(
        np.std(rms)
    )

    # --------------------------------------------------------
    # SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["centroid_mean"] = float(
        np.mean(centroid)
    )

    features["centroid_std"] = float(
        np.std(centroid)
    )

    # --------------------------------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["bandwidth_mean"] = float(
        np.mean(bandwidth)
    )

    features["bandwidth_std"] = float(
        np.std(bandwidth)
    )

    return features


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        gender_model,
        gender_encoder,
        age_model,
        emotion_model,
        emotion_encoder,
        feature_columns,
        senior_model,
        training_info

    ) = load_models()

except Exception as error:

    st.error(
        "Could not load the trained ML models."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "Age & Emotion Detection Through Voice"
)

st.write(
    "Upload a male voice recording to predict age. "
    "For predicted ages above 60, the application "
    "also predicts emotion and identifies the speaker "
    "as a senior citizen."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        "The application uses machine-learning models "
        "trained on the CREMA-D speech dataset."
    )

    if training_info:

        if "gender_accuracy" in training_info:

            st.write(
                f"Voice sex classifier accuracy: "
                f"{training_info['gender_accuracy'] * 100:.2f}%"
            )

        if "age_mae_years" in training_info:

            st.write(
                f"Male voice age MAE: "
                f"{training_info['age_mae_years']:.2f} years"
            )

        if "senior_accuracy" in training_info:

            st.write(
                f"Senior citizen classifier accuracy: "
                f"{training_info['senior_accuracy'] * 100:.2f}%"
            )

        if "emotion_accuracy" in training_info:

            st.write(
                f"Male voice emotion accuracy: "
                f"{training_info['emotion_accuracy'] * 100:.2f}%"
            )


# ============================================================
# SEPARATOR
# ============================================================

st.markdown("---")


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "Upload Voice Recording"
)

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


# ============================================================
# SHOW AUDIO
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    st.success(
        "Voice recording uploaded successfully."
    )


# ============================================================
# ANALYZE
# ============================================================

st.markdown("---")


if st.button(
    "Analyze Voice",
    type="primary"
):

    if uploaded_file is None:

        st.warning(
            "Please upload a voice recording first."
        )

        st.stop()

    temp_path = None

    try:

        # ----------------------------------------------------
        # SAVE UPLOADED AUDIO TEMPORARILY
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        features = extract_features(
            temp_path
        )

        feature_df = pd.DataFrame(
            [features]
        )

        # Make absolutely sure the order matches
        # the order used during model training.

        feature_df = feature_df.reindex(
            columns=feature_columns
        )

        # ----------------------------------------------------
        # VOICE SEX PREDICTION
        # ----------------------------------------------------

        gender_prediction = gender_model.predict(
            feature_df
        )

        gender_label = gender_encoder.inverse_transform(
            gender_prediction
        )[0]

        gender_label = str(
            gender_label
        ).lower()

        # ----------------------------------------------------
        # FEMALE VOICE
        # ----------------------------------------------------

        if gender_label != "male":

            st.error(
                "Upload male voice."
            )

            st.stop()

        # ----------------------------------------------------
        # MALE VOICE
        # ----------------------------------------------------

        st.success(
            "Male voice detected."
        )

        # ----------------------------------------------------
        # AGE PREDICTION
        # ----------------------------------------------------

        predicted_age = age_model.predict(
            feature_df
        )[0]

        predicted_age = int(
            round(predicted_age)
        )

        # Keep age within a sensible CREMA-D range.

        predicted_age = max(
            1,
            min(
                predicted_age,
                100
            )
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Prediction Results"
        )

        st.metric(
            "Predicted Age",
            f"{predicted_age} years"
        )

        # ====================================================
        # SENIOR CITIZEN
        # ====================================================

        if predicted_age > 60:

            st.success(
                "Senior Citizen"
            )

            # ------------------------------------------------
            # EMOTION PREDICTION
            # ------------------------------------------------

            emotion_prediction = emotion_model.predict(
                feature_df
            )

            emotion_label = emotion_encoder.inverse_transform(
                emotion_prediction
            )[0]

            st.metric(
                "Predicted Emotion",
                str(emotion_label)
            )

            st.info(
                "Predicted age is above 60, so the "
                "speaker is classified as a senior citizen "
                "and emotion prediction is applied."
            )

        else:

            st.info(
                "Age is 60 or below. "
                "Emotion prediction is not applied."
            )

    except Exception as error:

        st.error(
            "An error occurred while analyzing the voice."
        )

        st.code(
            str(error)
        )

    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
        # ----------------------------------------------------

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "This application uses machine-learning models "
    "trained on the CREMA-D speech dataset. "
    "Predictions are estimates and may not be accurate "
    "for every recording."
)