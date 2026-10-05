# Age & Emotion Detection Through Voice

## Overview

This project implements a machine-learning based voice analysis application using the CREMA-D speech dataset.

The application accepts a male voice recording and predicts the speaker's estimated age. If the predicted age is above 60 years, the application additionally identifies the speaker as a senior citizen and predicts the emotion expressed in the recording.

Female voice recordings are rejected according to the internship task requirement.

## Features

- Upload WAV voice recordings through a Streamlit GUI
- Machine-learning based male/female voice classification
- Male voice age prediction
- Senior citizen identification for predicted age above 60
- Emotion prediction for predicted senior citizens
- Female voice rejection with the message:
  
  `Upload male voice.`

- Model information displayed in the application
- Simple and user-friendly Streamlit interface

## Machine Learning Models

The project uses separate Random Forest models for:

1. Voice sex classification
2. Male voice age prediction
3. Senior citizen identification
4. Male voice emotion classification

The trained models are stored in the `model/` directory.

## Dataset

The models were trained using the CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) speech dataset.

The dataset contains speech recordings with emotion and speaker demographic information.

Only the information required for this project, such as age, sex and emotion labels, is used.

## Audio Features

Audio features are extracted using Librosa, including:

- MFCC
- Mel spectrogram features
- Zero Crossing Rate
- RMS energy
- Spectral Centroid
- Spectral Bandwidth

These features are used as input to the machine-learning models.

## Project Structure

```text
age_emotion_voice/
│
├── dataset/
│   └── metadata/
│       └── speaker_metadata.csv
│
├── model/
│   ├── age_model.pkl
│   ├── emotion_encoder.pkl
│   ├── emotion_model.pkl
│   ├── feature_columns.pkl
│   ├── gender_encoder.pkl
│   ├── gender_model.pkl
│   ├── senior_model.pkl
│   └── training_info.pkl
│
├── app.py
├── train_model.py
├── README.md
└── requirements.txt