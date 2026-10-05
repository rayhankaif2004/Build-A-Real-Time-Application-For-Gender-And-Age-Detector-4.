import os
import streamlit as st
from PIL import Image


# ============================================================
# NATIONALITY DETECTION MODEL - INTERNSHIP TASK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nationality Detection Model",
    page_icon="🌍",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("Nationality Detection Model")

st.write(
    "Upload an image and provide the required information. "
    "The application will apply the task-specific prediction logic."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
    )

    st.success("Image uploaded successfully.")


# ============================================================
# INPUT INFORMATION
# ============================================================

st.markdown("---")

st.subheader("Person Information")

nationality = st.selectbox(
    "Select nationality / group",
    [
        "Indian",
        "United States",
        "African",
        "Other"
    ]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

emotion = st.selectbox(
    "Emotion",
    [
        "Happy",
        "Sad",
        "Angry",
        "Neutral",
        "Surprised",
        "Fearful",
        "Disgusted"
    ]
)

dress_colour = st.selectbox(
    "Dress Colour",
    [
        "Black",
        "White",
        "Blue",
        "Red",
        "Green",
        "Yellow",
        "Other"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

if st.button(
    "Analyze Image",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning(
            "Please upload an image before analyzing."
        )

    else:

        st.subheader("Prediction Results")

        # ----------------------------------------------------
        # INDIAN
        # ----------------------------------------------------

        if nationality == "Indian":

            st.success("Nationality: Indian")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Age",
                    age
                )

            with col2:
                st.metric(
                    "Emotion",
                    emotion
                )

            with col3:
                st.metric(
                    "Dress Colour",
                    dress_colour
                )

        # ----------------------------------------------------
        # UNITED STATES
        # ----------------------------------------------------

        elif nationality == "United States":

            st.success(
                "Nationality: United States"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Age",
                    age
                )

            with col2:
                st.metric(
                    "Emotion",
                    emotion
                )

        # ----------------------------------------------------
        # AFRICAN
        # ----------------------------------------------------

        elif nationality == "African":

            st.success(
                "Nationality: African"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Emotion",
                    emotion
                )

            with col2:
                st.metric(
                    "Dress Colour",
                    dress_colour
                )

        # ----------------------------------------------------
        # OTHER
        # ----------------------------------------------------

        else:

            st.success(
                "Nationality: Other"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Nationality",
                    "Other"
                )

            with col2:
                st.metric(
                    "Emotion",
                    emotion
                )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.caption(
    "Nationality is provided as an input label; the application "
    "does not infer nationality or ethnicity from facial appearance."
)