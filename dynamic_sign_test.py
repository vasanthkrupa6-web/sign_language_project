import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import pyttsx3
import time
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/dynamic_sign_model.keras"
LABEL_PATH = "model/dynamic_labels.pkl"

SEQUENCE_LENGTH = 100
FEATURES = 63

CONFIDENCE_THRESHOLD = 0.70

# Number of consecutive predictions that must agree
STABLE_PREDICTIONS = 5

# Time the prediction must remain stable
STABLE_TIME = 0.8

# After speaking, wait before accepting another prediction
COOLDOWN_TIME = 2.0


# ============================================================
# LOAD MODEL
# ============================================================

print("🔄 Loading dynamic SignEcho model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

print("✅ Dynamic model loaded!")
print(f"Signs: {labels}")


# ============================================================
# PHRASES
# ============================================================

phrases = {
    "sorry": "I am sorry.",
    "toilets": "I need to use toilets.",
    "emergency": "I am in Trouble, Please help.",
    "thankyou": "Thank you so much.",
    "hello": "Hello, How are you.",
    "hungry": "I am feeling Hungry.",
    "bye": "Bye, See you again.",
    "yes": "Yes.",
    "no": "No.",
    "where": "Where is this Place?",
    "drink": "I am Thirsty, i need to drink water.",
    "phone": "May i use your Phone?"
}


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak(text):

    print(f"\n🔊 {text}")

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            150
        )

        engine.setProperty(
            "volume",
            1.0
        )

        voices = engine.getProperty(
            "voices"
        )

        if len(voices) > 0:

            engine.setProperty(
                "voice",
                voices[0].id
            )

        engine.say(text)

        engine.runAndWait()

        engine.stop()

        del engine

    except Exception as e:

        print(
            f"❌ Voice error: {e}"
        )


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = (
    mp.solutions.drawing_utils
)

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.7,

    min_tracking_confidence=0.7
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(
    hand_landmarks
):

    landmarks = []

    # --------------------------------------------------------
    # WRIST
    # --------------------------------------------------------

    wrist_x = (
        hand_landmarks.landmark[0].x
    )

    wrist_y = (
        hand_landmarks.landmark[0].y
    )

    wrist_z = (
        hand_landmarks.landmark[0].z
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    for landmark in hand_landmarks.landmark:

        x = (
            landmark.x
            - wrist_x
        )

        y = (
            landmark.y
            - wrist_y
        )

        z = (
            landmark.z
            - wrist_z
        )

        landmarks.extend([
            x,
            y,
            z
        ])


    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(landmarks)
    )

    if max_value > 0:

        landmarks = (
            landmarks / max_value
        )


    return landmarks


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "❌ Could not open camera."
    )

    exit()


print("\n")
print("=" * 60)
print("        SIGNECHO DYNAMIC REAL-TIME TEST")
print("=" * 60)

print(
    f"Signs: {labels}"
)

print()
print(
    "Perform ONE gesture at a time."
)

print()
print(
    "Hold the gesture naturally."
)

print()
print(
    "The system needs several consistent predictions."
)

print()
print(
    "Remove your hand between different gestures."
)

print()
print(
    "Press Q to quit."
)

print("=" * 60)
print()


# ============================================================
# VARIABLES
# ============================================================

sequence = []

# Stores recent predictions
prediction_history = deque(
    maxlen=STABLE_PREDICTIONS
)

last_prediction = None

prediction_start_time = 0

last_spoken = None

last_speech_time = 0

gesture_active = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "❌ Camera frame error."
        )

        break


    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        features = extract_landmarks(
            hand_landmarks
        )


        if features.shape == (FEATURES,):

            sequence.append(
                features
            )


            # Keep only latest 30 frames

            if len(sequence) > SEQUENCE_LENGTH:

                sequence.pop(0)


        # ----------------------------------------------------
        # DRAW HAND
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(

            frame,

            hand_landmarks,

            mp_hands.HAND_CONNECTIONS
        )


    else:

        # ====================================================
        # HAND REMOVED
        # ====================================================

        sequence = []

        prediction_history.clear()

        last_prediction = None

        prediction_start_time = 0

        last_spoken = None

        gesture_active = False


        cv2.putText(

            frame,

            "Show your hand",

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


    # ========================================================
    # PREDICT AFTER 30 FRAMES
    # ========================================================

    if len(sequence) == SEQUENCE_LENGTH:

        input_sequence = np.array(
            sequence,
            dtype=np.float32
        )


        input_sequence = np.expand_dims(
            input_sequence,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probabilities = model.predict(

            input_sequence,

            verbose=0

        )[0]


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]


        prediction_index = (
            top_indices[0]
        )


        prediction = labels[
            prediction_index
        ]


        confidence = probabilities[
            prediction_index
        ]


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            # ------------------------------------------------
            # ADD TO HISTORY
            # ------------------------------------------------

            prediction_history.append(
                prediction
            )


            # ------------------------------------------------
            # DISPLAY MAIN PREDICTION
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Sign: {prediction.upper()}",

                (20, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2
            )


            cv2.putText(

                frame,

                f"Confidence: {confidence * 100:.1f}%",

                (20, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )


            # ------------------------------------------------
            # TOP 3 DISPLAY
            # ------------------------------------------------

            y_position = 170

            for rank, index in enumerate(
                top_indices
            ):

                name = labels[index]

                score = (
                    probabilities[index]
                    * 100
                )

                text = (
                    f"{rank + 1}. "
                    f"{name.upper()} "
                    f"{score:.1f}%"
                )

                cv2.putText(

                    frame,

                    text,

                    (20, y_position),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2
                )

                y_position += 30


            # =================================================
            # CHECK IF PREDICTION CHANGED
            # =================================================

            if prediction != last_prediction:

                last_prediction = prediction

                prediction_start_time = (
                    time.time()
                )


            else:

                elapsed = (

                    time.time()
                    - prediction_start_time
                )


            # =================================================
            # MAJORITY VOTE
            # =================================================

            stable_prediction = None

            if len(
                prediction_history
            ) == STABLE_PREDICTIONS:

                counts = {}

                for p in prediction_history:

                    counts[p] = (
                        counts.get(p, 0)
                        + 1
                    )


                stable_prediction = max(
                    counts,
                    key=counts.get
                )


                stable_count = counts[
                    stable_prediction
                ]


                # ---------------------------------------------
                # DISPLAY STABILITY
                # ---------------------------------------------

                cv2.putText(

                    frame,

                    f"Stable: "
                    f"{stable_prediction.upper()} "
                    f"({stable_count}/"
                    f"{STABLE_PREDICTIONS})",

                    (20, 280),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 0),

                    2
                )


                # =================================================
                # SPEAK ONLY AFTER STABILITY
                # =================================================

                if (

                    stable_prediction == prediction

                    and

                    stable_count
                    >= STABLE_PREDICTIONS

                    and

                    elapsed >= STABLE_TIME

                    and

                    time.time()
                    - last_speech_time
                    >= COOLDOWN_TIME

                    and

                    not gesture_active

                ):

                    phrase = phrases.get(

                        stable_prediction,

                        stable_prediction
                    )


                    speak(
                        phrase
                    )


                    last_spoken = (
                        stable_prediction
                    )

                    last_speech_time = (
                        time.time()
                    )

                    gesture_active = True


        else:

            # =================================================
            # LOW CONFIDENCE
            # =================================================

            prediction_history.clear()

            last_prediction = None

            prediction_start_time = 0


            cv2.putText(

                frame,

                "Low confidence",

                (20, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 0, 255),

                2
            )


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(

        frame,

        f"Frames: "
        f"{len(sequence)}/"
        f"{SEQUENCE_LENGTH}",

        (20, 130),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ========================================================
    # GESTURE STATUS
    # ========================================================

    if gesture_active:

        cv2.putText(

            frame,

            "Gesture accepted - remove hand",

            (20, 320),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "SignEcho Dynamic Recognition",

        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print(
    "\n✅ Dynamic SignEcho stopped."
)