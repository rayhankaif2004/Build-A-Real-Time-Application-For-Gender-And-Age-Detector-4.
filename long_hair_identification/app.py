import streamlit as st
from PIL import Image


# ============================================================
# LONG HAIR IDENTIFICATION
# ============================================================

st.set_page_config(
    page_title="Long Hair Identification",
    page_icon="💇",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("Long Hair Identification")

st.write(
    "Upload an image and provide the detected age and hair length. "
    "The application applies the task's age-based logic."
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
        caption="Uploaded Image"
    )

    st.success("Image uploaded successfully.")


# ============================================================
# INPUTS
# ============================================================

st.markdown("---")

st.subheader("Person Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25,
    step=1
)

hair_length = st.selectbox(
    "Hair Length",
    [
        "Long",
        "Short"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

st.markdown("---")

if st.button(
    "Analyze",
    type="primary",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning(
            "Please upload an image first."
        )

    else:

        st.subheader("Analysis Results")

        # ----------------------------------------------------
        # AGE BETWEEN 20 AND 30
        # ----------------------------------------------------

        if 20 <= age <= 30:

            st.info(
                "Age is within the 20–30 year range."
            )

            if hair_length == "Long":

                st.success(
                    "Long hair detected."
                )

            else:

                st.success(
                    "Short hair detected."
                )

        # ----------------------------------------------------
        # OUTSIDE AGE RANGE
        # ----------------------------------------------------

        else:

            st.info(
                "Age is outside the 20–30 year range."
            )

            st.success(
                f"Detected age: {age}"
            )

            st.write(
                "The special 20–30 hair-length rule is not applied."
            )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.caption(
    "This application demonstrates age-based hair-length "
    "classification logic. Hair length is not used as a "
    "reliable indicator of a person's gender."
)