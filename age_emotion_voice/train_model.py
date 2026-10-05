import os
import re
import warnings

import audb
import joblib
import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METADATA_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "metadata",
    "speaker_metadata.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 16000


# ============================================================
# CREMA-D EMOTION LABELS
# ============================================================

EMOTION_MAP = {
    "ANG": "Angry",
    "DIS": "Disgusted",
    "FEA": "Fearful",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):
    """
    Extract numerical audio features from one WAV file.
    """

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Empty audio file")

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
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

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
        features[f"mel_{i}_mean"] = float(np.mean(mel_db[i]))
        features[f"mel_{i}_std"] = float(np.std(mel_db[i]))

    # --------------------------------------------------------
    # ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=y)

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # --------------------------------------------------------
    # SPECTRAL CENTROID
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    # --------------------------------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))

    return features


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("VOICE AGE & EMOTION MODEL TRAINING")
print("==============================================")
print()


# ============================================================
# CHECK METADATA
# ============================================================

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"\nMetadata file not found:\n{METADATA_FILE}\n\n"
        "Make sure speaker_metadata.csv exists inside:\n"
        "age_emotion_voice/dataset/metadata/"
    )


# ============================================================
# LOAD CREMA-D
# ============================================================

print("Loading CREMA-D...")

db = audb.load(
    "crema-d",
    version="1.3.0",
    format="wav",
    sampling_rate=SAMPLE_RATE
)

print("CREMA-D loaded.")
print("Total audio files:", len(db.files))


# ============================================================
# LOAD SPEAKER METADATA
# ============================================================

print()
print("Loading speaker metadata...")

speaker_df = pd.read_csv(METADATA_FILE)

required_columns = [
    "speaker_id",
    "age",
    "sex"
]

for column in required_columns:
    if column not in speaker_df.columns:
        raise ValueError(
            f"Missing required column in speaker_metadata.csv: {column}"
        )

speaker_df["speaker_id"] = speaker_df["speaker_id"].astype(int)
speaker_df["age"] = pd.to_numeric(
    speaker_df["age"],
    errors="coerce"
)

speaker_df["sex"] = (
    speaker_df["sex"]
    .astype(str)
    .str.lower()
    .str.strip()
)

speaker_df = speaker_df.dropna(
    subset=["speaker_id", "age", "sex"]
)

speaker_lookup = speaker_df.set_index(
    "speaker_id"
).to_dict("index")

print("Total speakers:", len(speaker_df))


# ============================================================
# BUILD DATASET
# ============================================================

print()
print("Extracting audio features...")
print("This can take some time because there are 7,441 audio files.")
print()

rows = []

total_files = len(db.files)

for index, audio_path in enumerate(db.files, start=1):

    filename = os.path.basename(audio_path)

    # Expected:
    # 1001_DFA_ANG_XX.wav

    match = re.match(
        r"(\d+)_.*?_([A-Z]{3})_[A-Z]{2}\.wav$",
        filename
    )

    if match is None:
        continue

    speaker_id = int(match.group(1))
    emotion_code = match.group(2)

    if speaker_id not in speaker_lookup:
        continue

    if emotion_code not in EMOTION_MAP:
        continue

    speaker_info = speaker_lookup[speaker_id]

    try:

        features = extract_features(audio_path)

        row = {
            "speaker_id": speaker_id,
            "age": float(speaker_info["age"]),
            "sex": str(speaker_info["sex"]).lower(),
            "emotion": EMOTION_MAP[emotion_code]
        }

        row.update(features)

        rows.append(row)

    except Exception as error:

        print(
            f"Error processing {filename}: {error}"
        )

    if index % 250 == 0 or index == total_files:

        print(
            f"Processed {index}/{total_files} files "
            f"({index / total_files * 100:.1f}%)"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = pd.DataFrame(rows)

if data.empty:
    raise RuntimeError(
        "No audio features were extracted."
    )


print()
print("==============================================")
print("FEATURE EXTRACTION COMPLETE")
print("==============================================")

print("Samples:", len(data))
print("Features:", len(data.columns))

print()
print("Sex distribution:")
print(data["sex"].value_counts())

print()
print("Emotion distribution:")
print(data["emotion"].value_counts())


# ============================================================
# FEATURE COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "speaker_id",
    "age",
    "sex",
    "emotion"
]

FEATURE_COLUMNS = [
    column
    for column in data.columns
    if column not in TARGET_COLUMNS
]

X = data[FEATURE_COLUMNS]


# ============================================================
# 1. VOICE SEX CLASSIFIER
# ============================================================

print()
print("==============================================")
print("TRAINING VOICE SEX CLASSIFIER")
print("==============================================")

gender_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gender_splitter.split(
        X,
        data["sex"],
        groups=data["speaker_id"]
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = data["sex"].iloc[train_idx]
y_test = data["sex"].iloc[test_idx]

gender_encoder = LabelEncoder()

y_train_encoded = gender_encoder.fit_transform(
    y_train
)

y_test_encoded = gender_encoder.transform(
    y_test
)

gender_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

gender_model.fit(
    X_train,
    y_train_encoded
)

gender_predictions = gender_model.predict(
    X_test
)

gender_accuracy = accuracy_score(
    y_test_encoded,
    gender_predictions
)

print(
    f"Voice sex classifier accuracy: "
    f"{gender_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE SEX MODEL
# ============================================================

joblib.dump(
    gender_model,
    os.path.join(
        MODEL_DIR,
        "gender_model.pkl"
    )
)

joblib.dump(
    gender_encoder,
    os.path.join(
        MODEL_DIR,
        "gender_encoder.pkl"
    )
)


# ============================================================
# MALE DATA ONLY
# ============================================================

male_data = data[
    data["sex"] == "male"
].copy()

if male_data.empty:
    raise RuntimeError(
        "No male voice samples were found."
    )

print()
print("Male voice samples:", len(male_data))
print(
    "Male speakers:",
    male_data["speaker_id"].nunique()
)


# ============================================================
# 2. MALE VOICE AGE MODEL
# ============================================================

print()
print("==============================================")
print("TRAINING MALE VOICE AGE MODEL")
print("==============================================")

male_X = male_data[FEATURE_COLUMNS]
male_y = male_data["age"]

age_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

age_train_idx, age_test_idx = next(
    age_splitter.split(
        male_X,
        male_y,
        groups=male_data["speaker_id"]
    )
)

age_X_train = male_X.iloc[age_train_idx]
age_X_test = male_X.iloc[age_test_idx]

age_y_train = male_y.iloc[age_train_idx]
age_y_test = male_y.iloc[age_test_idx]


# ------------------------------------------------------------
# AGE SAMPLE WEIGHTS
# Give additional importance to older speakers.
# ------------------------------------------------------------

age_weights = np.ones(
    len(age_y_train),
    dtype=float
)

age_weights[
    (age_y_train >= 50) &
    (age_y_train <= 60)
] = 1.5

age_weights[
    age_y_train > 60
] = 3.0


age_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2
)

age_model.fit(
    age_X_train,
    age_y_train,
    sample_weight=age_weights
)

age_predictions = age_model.predict(
    age_X_test
)

age_mae = mean_absolute_error(
    age_y_test,
    age_predictions
)

print(
    f"Male voice age MAE: "
    f"{age_mae:.2f} years"
)


# ============================================================
# SAVE AGE MODEL
# ============================================================

joblib.dump(
    age_model,
    os.path.join(
        MODEL_DIR,
        "age_model.pkl"
    )
)


# ============================================================
# 3. SENIOR CITIZEN CLASSIFIER
# ============================================================

print()
print("==============================================")
print("TRAINING SENIOR CITIZEN CLASSIFIER")
print("==============================================")

# Senior citizen according to project requirement:
# age > 60

male_data["senior"] = (
    male_data["age"] > 60
).astype(int)

senior_X = male_data[FEATURE_COLUMNS]
senior_y = male_data["senior"]

senior_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

senior_train_idx, senior_test_idx = next(
    senior_splitter.split(
        senior_X,
        senior_y,
        groups=male_data["speaker_id"]
    )
)

senior_X_train = senior_X.iloc[
    senior_train_idx
]

senior_X_test = senior_X.iloc[
    senior_test_idx
]

senior_y_train = senior_y.iloc[
    senior_train_idx
]

senior_y_test = senior_y.iloc[
    senior_test_idx
]

senior_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

senior_model.fit(
    senior_X_train,
    senior_y_train
)

senior_predictions = senior_model.predict(
    senior_X_test
)

senior_accuracy = accuracy_score(
    senior_y_test,
    senior_predictions
)

print(
    f"Senior citizen classifier accuracy: "
    f"{senior_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE SENIOR MODEL
# ============================================================

joblib.dump(
    senior_model,
    os.path.join(
        MODEL_DIR,
        "senior_model.pkl"
    )
)


# ============================================================
# 4. MALE VOICE EMOTION MODEL
# ============================================================

print()
print("==============================================")
print("TRAINING MALE VOICE EMOTION MODEL")
print("==============================================")

emotion_X = male_data[FEATURE_COLUMNS]
emotion_y = male_data["emotion"]

emotion_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

emotion_train_idx, emotion_test_idx = next(
    emotion_splitter.split(
        emotion_X,
        emotion_y,
        groups=male_data["speaker_id"]
    )
)

emotion_X_train = emotion_X.iloc[
    emotion_train_idx
]

emotion_X_test = emotion_X.iloc[
    emotion_test_idx
]

emotion_y_train = emotion_y.iloc[
    emotion_train_idx
]

emotion_y_test = emotion_y.iloc[
    emotion_test_idx
]

emotion_encoder = LabelEncoder()

emotion_train_encoded = emotion_encoder.fit_transform(
    emotion_y_train
)

emotion_test_encoded = emotion_encoder.transform(
    emotion_y_test
)

emotion_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

emotion_model.fit(
    emotion_X_train,
    emotion_train_encoded
)

emotion_predictions = emotion_model.predict(
    emotion_X_test
)

emotion_accuracy = accuracy_score(
    emotion_test_encoded,
    emotion_predictions
)

print(
    f"Male voice emotion accuracy: "
    f"{emotion_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE EMOTION MODEL
# ============================================================

joblib.dump(
    emotion_model,
    os.path.join(
        MODEL_DIR,
        "emotion_model.pkl"
    )
)

joblib.dump(
    emotion_encoder,
    os.path.join(
        MODEL_DIR,
        "emotion_encoder.pkl"
    )
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

joblib.dump(
    FEATURE_COLUMNS,
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

training_info = {
    "dataset": "CREMA-D",
    "dataset_version": "1.3.0",
    "sampling_rate": SAMPLE_RATE,

    "total_audio_files": total_files,
    "processed_samples": len(data),
    "speakers": int(
        data["speaker_id"].nunique()
    ),

    "features": len(FEATURE_COLUMNS),

    "gender_accuracy": float(
        gender_accuracy
    ),

    "age_mae_years": float(
        age_mae
    ),

    "senior_accuracy": float(
        senior_accuracy
    ),

    "emotion_accuracy": float(
        emotion_accuracy
    ),

    "emotion_classes": list(
        emotion_encoder.classes_
    )
}

joblib.dump(
    training_info,
    os.path.join(
        MODEL_DIR,
        "training_info.pkl"
    )
)


# ============================================================
# FINISHED
# ============================================================

print()
print("==============================================")
print("TRAINING COMPLETE")
print("==============================================")

print()
print("Models saved in:")
print(MODEL_DIR)

print()
print("Created files:")

for filename in sorted(
    os.listdir(MODEL_DIR)
):
    print(" -", filename)

print()
print("Training summary:")
print(
    f"Voice sex accuracy: "
    f"{gender_accuracy * 100:.2f}%"
)

print(
    f"Male age MAE: "
    f"{age_mae:.2f} years"
)

print(
    f"Senior citizen accuracy: "
    f"{senior_accuracy * 100:.2f}%"
)

print(
    f"Male emotion accuracy: "
    f"{emotion_accuracy * 100:.2f}%"
)

print()
print("The voice ML models are ready.")