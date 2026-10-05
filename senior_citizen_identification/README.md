# Senior Citizen Identification

## Overview

A computer vision application that processes video containing multiple people and identifies senior citizens using age and gender predictions from a trained machine-learning model.

## Features

- Upload video
- Detect multiple faces
- Predict age
- Predict gender
- Identify people above 60 as senior citizens
- Highlight senior citizens with a red rectangle
- Display age and gender
- Save age, gender and detection time to CSV
- Streamlit GUI

## Senior Citizen Rule

A person is classified as a senior citizen when the predicted age is greater than 60.

## Technologies

- Python
- TensorFlow
- OpenCV
- NumPy
- Streamlit

## Model

The application reuses the trained Age & Gender model from the main project:

`AGE_GENDER_DETECTION-main/Age_Sex_Detection.h5`

## Installation

```bash
pip install -r requirements.txt