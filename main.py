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


# Problem 1
# There is not problem recognizing words, however there is a problem with sentences.
# We need to figure out a way to detected multiple words in sentence however this is
# impossible in the current program as it only detects single words.

# One of the solutions is to record the video instead and use Dynamic Time warping.
# However, I feel like this will greatly increase the calculation time and be slow.

# The Second option is to record the video and use Loop Frame Windows.
# For instance, we can analyze every 30 seconds before moving forward and detect
# any signs. If the sign is detected and moved forward until the next sign is detected
# we can split the videos into different frames and kinda move on to the next word.
# This will increase the calculation time by alot so we need to find a viable function
# to keep the time complexity low.

# Another solution would be using Motion thresholding with keypoints.
# We could set up a motion threshold where if a movement gradually slows down or stops
# we can detect and conclude that a word in Sing language has completed and kinda
# isolate the sentence. This approach will be easier for LSTM to distinguish between
# individual signs.