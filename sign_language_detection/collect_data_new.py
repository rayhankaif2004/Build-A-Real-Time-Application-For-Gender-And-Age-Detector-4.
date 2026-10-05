import cv2
import os
import time

# =========================
# SETTINGS
# =========================

CLASSES = ["A", "B", "C", "D", "E", "nothing"]

IMAGES_PER_CLASS = 300

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)

CAMERA_INDEX = 0

# =========================
# CREATE FOLDERS
# =========================

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)
    os.makedirs(folder, exist_ok=True)

print("\nDataset folders are ready.")

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# =========================
# COLLECT DATA
# =========================

for class_name in CLASSES:

    class_folder = os.path.join(DATASET_DIR, class_name)

    existing_images = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(existing_images)

    print("\n====================================")
    print(f"GET READY: {class_name}")
    print(f"Existing images: {count}")
    print(f"Target images: {IMAGES_PER_CLASS}")
    print("Position your hand inside the white box.")
    print("Press SPACE to start collecting.")
    print("Press ESC to stop.")
    print("====================================")

    # Wait for SPACE
    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # White collection box
        box_size = 300
        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Class: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to start",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Language Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("\nCollection stopped.")
            exit()

        if key == 32:
            break

    # =========================
    # COLLECT IMAGES
    # =========================

    print(f"\nCollecting {class_name}...")

    while count < IMAGES_PER_CLASS:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        box_size = 300
        x1 = (width - box_size) // 2
        y1 = (height - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Progress
        cv2.putText(
            frame,
            f"Class: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/{IMAGES_PER_CLASS}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "ESC = stop",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Language Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("\nCollection stopped.")
            exit()

        # Crop the box
        roi = frame[y1:y2, x1:x2]

        # Save image
        filename = os.path.join(
            class_folder,
            f"{class_name}_{count + 1:04d}.jpg"
        )

        cv2.imwrite(filename, roi)

        count += 1

        # Small delay
        time.sleep(0.08)

    print(f"Finished {class_name}: {count} images.")

# =========================
# FINISHED
# =========================

cap.release()
cv2.destroyAllWindows()

print("\n====================================")
print("DATASET COLLECTION COMPLETE")
print("====================================")

for class_name in CLASSES:

    folder = os.path.join(DATASET_DIR, class_name)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name}: {len(images)} images")

print("\nDataset is ready for model training.")