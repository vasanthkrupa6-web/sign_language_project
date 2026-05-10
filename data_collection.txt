import cv2
import mediapipe as mp
import numpy as np
import os

# =============================================
# CHANGE THIS for each new sign you collect
SIGN_NAME = "hello"
SAMPLES = 200
# =============================================

SAVE_DIR = f"dataset/{SIGN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
count = 0
collect = False

print(f"\n👋 Ready to collect '{SIGN_NAME}'")
print("Press 's' to START collecting, 'q' to QUIT\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            for lm in hand_lm.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Status display
    if collect:
        status = f"Collecting '{SIGN_NAME}': {count}/{SAMPLES}"
        color = (0, 255, 0)
    else:
        status = f"Press 'S' to collect '{SIGN_NAME}'"
        color = (0, 165, 255)

    cv2.putText(frame, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Progress bar
    if collect and SAMPLES > 0:
        bar_width = int((count / SAMPLES) * 400)
        cv2.rectangle(frame, (10, 60), (410, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 255, 0), -1)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        collect = True
        print("✅ Collection started!")

    elif key == ord('q') or key == ord('Q'):
        print("❌ Quit by user.")
        break

    # Save landmark data
    if collect and len(landmarks) == 63:
        np.save(f"{SAVE_DIR}/{count}.npy", np.array(landmarks))
        count += 1
        if count >= SAMPLES:
            print(f"\n🎉 Done! {SAMPLES} samples saved for '{SIGN_NAME}'.")
            print(f"📂 Saved in: {SAVE_DIR}/\n")
            break

cap.release()
cv2.destroyAllWindows()