import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Colour Detection",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        Car Colour Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Detect cars, identify their colours, and count people
        in traffic images.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = YOLO("yolo11n.pt")

    return model


try:

    with st.spinner("Loading detection model..."):

        model = load_model()

    st.success(
        "Object detection model loaded successfully."
    )

except Exception as e:

    st.error(
        f"Could not load the detection model: {e}"
    )

    st.stop()


# ============================================================
# CAR COLOUR DETECTION
# ============================================================

def detect_car_colour(car_crop):

    if car_crop is None or car_crop.size == 0:

        return "Unknown"

    # Convert BGR to HSV
    hsv = cv2.cvtColor(
        car_crop,
        cv2.COLOR_BGR2HSV
    )

    # Average HSV values
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    # Black
    if v < 50:
        return "Black"

    # White
    if s < 35 and v > 180:
        return "White"

    # Gray
    if s < 60 and v < 180:
        return "Gray"

    # Red
    if h < 10 or h >= 170:
        return "Red"

    # Yellow
    if 10 <= h < 35:
        return "Yellow"

    # Green
    if 35 <= h < 85:
        return "Green"

    # Blue
    if 85 <= h < 135:
        return "Blue"

    # Purple
    if 135 <= h < 170:
        return "Purple"

    return "Other"


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    # Convert PIL image to NumPy
    image_rgb = np.array(image)

    # Convert RGB → BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Run YOLO
    results = model(
        image_bgr,
        conf=0.35
    )

    output = image_bgr.copy()

    car_count = 0
    people_count = 0

    detected_cars = []

    # YOLO COCO classes:
    # 0 = person
    # 2 = car

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # =================================================
            # PERSON
            # =================================================

            if class_id == 0:

                people_count += 1

                # Green rectangle
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    "Person",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # =================================================
            # CAR
            # =================================================

            elif class_id == 2:

                car_count += 1

                # Keep coordinates inside image
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(
                    image_bgr.shape[1],
                    x2
                )

                y2 = min(
                    image_bgr.shape[0],
                    y2
                )

                # Crop car
                car_crop = image_bgr[
                    y1:y2,
                    x1:x2
                ]

                # Detect colour
                car_colour = detect_car_colour(
                    car_crop
                )

                detected_cars.append(
                    car_colour
                )

                # =================================================
                # INTERNSHIP REQUIREMENT
                #
                # BLUE CAR → RED RECTANGLE
                # OTHER CAR → BLUE RECTANGLE
                # =================================================

                if car_colour == "Blue":

                    box_colour = (
                        0,
                        0,
                        255
                    )

                else:

                    box_colour = (
                        255,
                        0,
                        0
                    )

                # Draw bounding box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    box_colour,
                    3
                )

                # Label
                label = (
                    f"{car_colour} Car "
                    f"{confidence:.0%}"
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_colour,
                    2
                )

    return (
        output,
        car_count,
        people_count,
        detected_cars
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    "## Upload Traffic Image"
)

uploaded_file = st.file_uploader(
    "Choose a traffic image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# SHOW IMAGE + PROCESS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # INPUT PREVIEW
    # --------------------------------------------------------

    st.markdown(
        "### Input Image"
    )

    st.image(
        image,
        caption="Uploaded traffic image",
        use_column_width=True
    )

    # --------------------------------------------------------
    # DETECT BUTTON
    # --------------------------------------------------------

    if st.button(
        "Detect Cars & People"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            (
                output,
                car_count,
                people_count,
                detected_cars
            ) = process_image(
                image
            )

        # Convert BGR → RGB
        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # OUTPUT IMAGE
        # ----------------------------------------------------

        st.markdown(
            "### Detection Result"
        )

        st.image(
            output_rgb,
            caption="Detection result",
            use_column_width=True
        )

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Cars Detected",
                car_count
            )

        with col2:

            st.metric(
                "People Detected",
                people_count
            )

        # ----------------------------------------------------
        # CAR COLOURS
        # ----------------------------------------------------

        st.markdown(
            "### Detected Car Colours"
        )

        if detected_cars:

            for index, colour in enumerate(
                detected_cars,
                start=1
            ):

                st.write(
                    f"Car {index}: **{colour}**"
                )

        else:

            st.info(
                "No cars were detected."
            )

        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        st.markdown(
            """
            ### Bounding Box Rules

            **Red rectangle** → Blue car

            **Blue rectangle** → Other colour car

            **Green rectangle** → Person
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "<center>Car Colour Detection | Internship Project</center>",
    unsafe_allow_html=True
)