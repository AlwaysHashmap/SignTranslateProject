import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Local helper modules
from packages import *
from functions import (mediapipe_detection, keypoint_value_extraction, extract_words_from_output_textfile)

MODEL_PATH = "SL_Best_Model.keras"
GLOBAL_MIN_PATH = "global_min.npy"
GLOBAL_MAX_PATH = "global_max.npy"
ACTIONS_TXT = "Sign_Language_Words.txt"

FRAMES_PER_SEQUENCE = 30
FEATURE_DIM = 1662  # Must match the new dimension with face

MIN_DETECTION_CONF = 0.75
MIN_TRACKING_CONF = 0.75

# Frame-to-frame movement threshold
MOVEMENT_THRESHOLD = 0.10  # if you see random triggers, try 0.02 or 0.03
# Total-sequence movement threshold
SEQUENCE_MOVEMENT_THRESHOLD = 0.70  # if your movement is subtle, reduce this

def main():
    # Load model
    model = load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Load global_min, global_max
    if not (os.path.exists(GLOBAL_MIN_PATH) and os.path.exists(GLOBAL_MAX_PATH)):
        print("[ERROR] Missing global_min.npy or global_max.npy.")
        return
    global_min = np.load(GLOBAL_MIN_PATH)
    global_max = np.load(GLOBAL_MAX_PATH)
    print("Loaded global_min and global_max.")

    # Load actions (class labels)
    actions = extract_words_from_output_textfile(ACTIONS_TXT)
    print(f"Actions ({len(actions)}): {actions}")

    # Initialize Mediapipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF
    )

    # Start webcam
    cap = cv2.VideoCapture(0)
    frame_buffer = []
    prev_keypoints = None  # to compare with current frame for gating

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty camera frame. Exiting...")
            break

        # Mediapipe detection
        results = mediapipe_detection(frame, holistic)
        keypoints = keypoint_value_extraction(results)

        # If shape is (1, 258), squeeze
        if keypoints.shape == (1, FEATURE_DIM):
            keypoints = np.squeeze(keypoints, axis=0)

        if keypoints.shape != (FEATURE_DIM,):
            print(f"[WARNING] Keypoints shape {keypoints.shape} != ({FEATURE_DIM},). Skipping.")
            prev_keypoints = None
            continue

        if prev_keypoints is not None:
            dist = np.linalg.norm(keypoints - prev_keypoints)
            if dist > MOVEMENT_THRESHOLD:
                frame_buffer.append(keypoints)
            else:
                # Skip if No significant movement
                pass
        else:
            pass

        prev_keypoints = keypoints

        if len(frame_buffer) == FRAMES_PER_SEQUENCE:
            seq_array = np.array(frame_buffer)  # shape (30, 258)

            total_movement = 0.0
            for i in range(1, len(seq_array)):
                total_movement += np.linalg.norm(seq_array[i] - seq_array[i-1])

            if total_movement < SEQUENCE_MOVEMENT_THRESHOLD:
                # Very little overall movement => skip classification
                print(f"No movement, skipping classification (total movement: {total_movement:.2f})")
                frame_buffer = []
                continue

            gm = global_min.reshape(1, FEATURE_DIM)
            gM = global_max.reshape(1, FEATURE_DIM)
            seq_array = (seq_array - gm) / (gM - gm + 1e-8)

            seq_array = seq_array[np.newaxis, ...]  # (1, 30, 258)
            predictions = model.predict(seq_array)
            pred_idx = np.argmax(predictions[0])
            pred_action = actions[pred_idx]
            confidence = predictions[0][pred_idx]

            print(f"Predicted: {pred_action} (conf: {confidence:.2f})")

            frame_buffer = []

        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Exiting Main.")

if __name__ == "__main__":
    main()