import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time

print("🔄 Loading model...")
with open("model/sign_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
labels = data["labels"]
print(f"✅ Model loaded! Signs: {labels}\n")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

last_sign = ""
last_spoken = 0
COOLDOWN = 2.5        # seconds before repeating same sign
CONFIDENCE = 0.85     # minimum confidence to show prediction

print("🎥 Camera started. Show a sign! Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = ""
    confidence = 0.0

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            for lm in hand_lm.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            proba = model.predict_proba([np.array(landmarks)])[0]
            confidence = max(proba)
            if confidence >= CONFIDENCE:
                prediction = model.predict([np.array(landmarks)])[0]

    # Display prediction on screen
    if prediction:
        cv2.putText(frame, f"Sign: {prediction.upper()}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No sign detected", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Sign Language Detector", frame)

    # Speak the sign aloud
    now = time.time()
    if prediction and (prediction != last_sign or (now - last_spoken) > COOLDOWN):
        print(f"🗣️  Detected: {prediction} ({confidence*100:.1f}%)")
        engine.say(prediction)
        engine.runAndWait()
        last_sign = prediction
        last_spoken = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Closed. Goodbye!")