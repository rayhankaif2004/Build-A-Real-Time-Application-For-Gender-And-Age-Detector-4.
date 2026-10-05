# Sign Language Detection

A real-time American Sign Language (ASL) alphabet detection application built using Python, OpenCV, TensorFlow/Keras, and Streamlit.

## Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize six classes of hand signs:

- A
- B
- C
- D
- E
- Nothing

The application uses a webcam to capture the user's hand and predicts the corresponding sign in real time.

## Features

- Real-time webcam detection
- ASL alphabet recognition
- Six supported classes
- Confidence percentage displayed with predictions
- Custom dataset collected using OpenCV
- CNN-based image classification model
- Streamlit web interface

## Dataset

A custom dataset was collected using the project's data collection script.

The dataset contains:

- 300 images of A
- 300 images of B
- 300 images of C
- 300 images of D
- 300 images of E
- 300 images of Nothing

Total dataset size: **1,800 images**

The images were resized to 64 × 64 pixels before training.

## Model

The project uses a CNN with:

- Convolutional layers
- Max pooling layers
- Dropout
- Dense layers
- Softmax output layer

The trained