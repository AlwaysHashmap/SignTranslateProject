import cv2 # openCV
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import mediapipe as mediapipe

mediapipe_holistic = mediapipe.solutions.holistic       # MediaPipe Holistic Model (Detecting)
mediapipe_drawing = mediapipe.solutions.drawing_utils   # MediaPipe Drawing Utilities (Drawing)

def mediapipe_detection(image, model):                  # Image = Frame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert BGR -> RGB
    image.flags.writeable = False
    results = model.process(image)                      # MediaPipe Model Prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert back to RGB -> BGR
    return image, results

def mediapipe_detection_draw_landmarks(image, results):
    mediapipe_drawing.draw_landmarks(image, results.face_landmarks, mediapipe_holistic.FACE_CONNECTIONS)
    mediapipe_drawing.draw_landmarks(image, results.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS)
    mediapipe_drawing.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)
    mediapipe_drawing.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)                                           # Accessing Webcam
    with mediapipe_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:     # Access MediaPipe Model
        while cap.isOpened():
            ret, frame = cap.read()                                     # Read Feed

            image, results = mediapipe_detection(frame, holistic)       # Start Detection

            mediapipe_detection_draw_landmarks(image, results)          # Draw Landmarks

            cv2.imshow('Input Sign Language', image)                    # Show Feed

            if cv2.waitKey(10) & 0xFF == ord('q'):                      # End Feed
                break
        cap.release()
        cv2.destroyAllWindows()