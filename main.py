from packages import *
from functions import *

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