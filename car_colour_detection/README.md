# Car Colour Detection

## Project Overview

This project detects cars and people in traffic images using YOLO object detection.

For each detected car, the application estimates the car colour and displays a bounding box around the vehicle.

The application also counts:
- Total number of cars
- Total number of people

## Features

- Upload a traffic image
- Detect cars
- Detect people
- Estimate car colour
- Count detected cars
- Count detected people
- Display bounding boxes
- Display detection confidence
- User-friendly Streamlit GUI

## Car Colour Detection

The application estimates the following colours:

- Black
- White
- Gray
- Red
- Yellow
- Green
- Blue
- Purple
- Other

## Bounding Box Rules

- Red rectangle = Blue car
- Blue rectangle = Other colour car
- Green rectangle = Person

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Ultralytics YOLO

## Model

The application uses the YOLO11 Nano pretrained object detection model for detecting cars and people.

The car-colour classification is performed using image colour analysis in the HSV colour space.

## How to Run

Install the required dependencies:

```bash
python -m pip install -r requirements.txt