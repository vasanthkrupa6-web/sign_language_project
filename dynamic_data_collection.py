import cv2
import mediapipe as mp
import numpy as np
import os
import time


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dynamic_dataset"

SIGN_NAME = "phone"

NUM_SAMPLES = 100

SEQUENCE_LENGTH = 100


# ============================================================
# CREATE FOLDER
# ============================================================

save_path = os.path.join(
    DATASET_PATH,
    SIGN_NAME
)

os.makedirs(save_path, exist_ok=True)


print("=" * 60)
print("       SIGNECHO DYNAMIC DATA COLLECTION")
print("=" * 60)
print(f"Sign              : {SIGN_NAME}")
print(f"Samples            : {NUM_SAMPLES}")
print(f"Frames/sample      : {SEQUENCE_LENGTH}")
print(f"Save location      : {save_path}")
print("=" * 60)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def extract_landmarks(hand_landmarks):

    landmarks = []

    # Wrist becomes origin
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z

    for landmark in hand_landmarks.landmark:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        landmarks.extend([
            x,
            y,
            z
        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    # Scale normalization
    max_value = np.max(
        np.abs(landmarks)
    )

    if max_value > 0:
        landmarks = landmarks / max_value

    return landmarks


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ ERROR: Camera could not be opened.")
    exit()


print("\nInstructions:")
print("1. Press SPACE to start recording.")
print("2. Wait for 3-second countdown.")
print("3. Perform the PHONE gesture naturally.")
print("4. The system records 30 valid frames.")
print("5. Press Q to quit.")
print()


# ============================================================
# COLLECTION
# ============================================================

sample_count = 0

while sample_count < NUM_SAMPLES:

    # --------------------------------------------------------
    # WAITING SCREEN
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"DYNAMIC SIGN: {SIGN_NAME.upper()}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sample: {sample_count}/{NUM_SAMPLES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press SPACE to record",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SignEcho Dynamic Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key != 32:
        continue


    # ========================================================
    # COUNTDOWN
    # ========================================================

    print(
        f"\nRecording sample "
        f"{sample_count + 1}/{NUM_SAMPLES}..."
    )

    for countdown in [3, 2, 1]:

        ret, countdown_frame = cap.read()

        if not ret:
            break

        countdown_frame = cv2.flip(
            countdown_frame,
            1
        )

        cv2.putText(
            countdown_frame,
            str(countdown),
            (280, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 255, 255),
            5
        )

        cv2.putText(
            countdown_frame,
            f"GET READY: {SIGN_NAME.upper()}",
            (80, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SignEcho Dynamic Collector",
            countdown_frame
        )

        cv2.waitKey(1000)


    # ========================================================
    # RECORD SEQUENCE
    # ========================================================

    sequence = []

    while len(sequence) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)


        # ----------------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            features = extract_landmarks(
                hand_landmarks
            )

            if features.shape == (63,):

                sequence.append(features)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"RECORDING: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "PERFORM PHONE GESTURE",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "SignEcho Dynamic Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()


    # ========================================================
    # SAVE SEQUENCE
    # ========================================================

    if len(sequence) == SEQUENCE_LENGTH:

        sequence_array = np.array(
            sequence,
            dtype=np.float32
        )

        filename = os.path.join(
            save_path,
            f"{SIGN_NAME}_{sample_count + 1:04d}.npy"
        )

        np.save(
            filename,
            sequence_array
        )

        sample_count += 1

        print(
            f"✅ Saved sample "
            f"{sample_count}/{NUM_SAMPLES}"
            f" | Shape: {sequence_array.shape}"
        )

    else:

        print(
            "⚠️ Sample discarded."
            " Not enough valid hand frames."
        )

    time.sleep(0.5)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


print("\n" + "=" * 60)
print("       DYNAMIC COLLECTION COMPLETED")
print("=" * 60)

print(f"Sign: {SIGN_NAME}")
print(f"Samples collected: {sample_count}")
print(f"Location: {save_path}")

print("=" * 60)