# Long Hair Identification

## Overview

The Long Hair Identification application is a Streamlit-based computer vision demonstration that allows users to upload an image and provide information about the person's age and hair length.

The application applies age-based logic to classify the provided hair length.

## Features

- Upload JPG, JPEG, and PNG images
- Display the uploaded image
- Enter the person's age
- Select hair length as Long or Short
- Apply the 20–30 age-based classification rule
- Display analysis results through a Streamlit interface

## Project Structure

```text
long_hair_identification/
│
├── dataset/
├── model/
├── app.py
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Streamlit
- Pillow

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Run the Application

From inside the `long_hair_identification` folder, run:

```bash
streamlit run app.py
```

The application will open in the browser.

## How to Use

1. Upload an image.
2. Enter the person's age.
3. Select the hair length.
4. Click **Analyze**.
5. View the analysis results.

## Classification Logic

For people between 20 and 30 years old, the application applies the hair-length rule and reports whether long or short hair was selected.

For ages outside the 20–30 range, the application displays the detected age and indicates that the special age-based hair-length rule is not applied.

## Disclaimer

This application is a demonstration project. Hair length is not a reliable indicator of a person's gender or other personal attributes.