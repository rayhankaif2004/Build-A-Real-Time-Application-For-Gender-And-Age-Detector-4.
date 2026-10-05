# Nationality Detection Model

## Overview

This project is a Streamlit-based application developed as part of the internship tasks.

The application allows a user to upload an image and provide the required person information. Based on the selected nationality/group, the application displays the corresponding information required by the task specification.

## Features

- Image upload through a graphical user interface
- Image preview
- Nationality/group selection
- Age input
- Emotion selection
- Dress-colour selection
- Task-specific prediction result display
- Streamlit-based GUI

## Task Logic

The application provides different outputs depending on the selected group:

### Indian
Displays:
- Nationality
- Age
- Emotion
- Dress colour

### United States
Displays:
- Nationality
- Age
- Emotion

### African
Displays:
- Nationality/group
- Emotion
- Dress colour

### Other
Displays:
- Nationality/group
- Emotion

## Project Structure

```text
nationality_detection/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── model/
│
├── app.py
├── README.md
└── requirements.txt
```

## Installation

Create or activate a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Open the terminal in the `nationality_detection` directory and run:

```bash
streamlit run app.py
```

The Streamlit application will provide a local URL in the terminal. Open that URL in a web browser.

## Usage

1. Upload an image.
2. Select the applicable nationality/group.
3. Enter/select the required information.
4. Click **Analyze Image**.
5. Review the displayed results.

## Limitations

The current implementation uses nationality/group as a user-provided input rather than attempting to infer nationality or ethnicity from facial appearance.

The current version is a functional GUI implementation. The `dataset` and `model` directories are reserved for the machine-learning training and model files.

## Technologies Used

- Python
- Streamlit
- Pillow

## Project Status

GUI implementation completed and tested. Machine-learning model integration is a separate step.