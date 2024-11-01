from packages import *

def mediapipe_detection(image, model):                  # Image = Frame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert BGR -> RGB
    image.flags.writeable = False
    results = model.process(image)                      # MediaPipe Model Prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert back to RGB -> BGR
    return image, results

def mediapipe_detection_draw_landmarks(image, results):     # For Debugging
    mediapipe_drawing.draw_landmarks(image, results.face_landmarks, mediapipe_holistic.FACEMESH_TESSELATION,
                                     mediapipe_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),  # Dot Color/Size
                                     mediapipe_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))  # Line Color/Size

    mediapipe_drawing.draw_landmarks(image, results.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS,
                                     mediapipe_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3), # Dot Color/Size
                                     mediapipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)) # Line Color/Size

    mediapipe_drawing.draw_landmarks(image, results.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                     mediapipe_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3), # Dot Color/Size
                                     mediapipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)) # Line Color/Size

    mediapipe_drawing.draw_landmarks(image, results.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS,
                                     mediapipe_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3), # Dot Color/Size
                                     mediapipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)) # Line Color/Size

def keypoint_value_extraction(results): # If landmark is detected, get the xyz points and flatten into one array. If no landmark is detected, fill the array with 0's for error handling
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()     # Flatten into 1-D array
    else:
        face = np.zeros(468 * 3)  # Fill with zeros if no face detected

    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()     # Flatten into 1-D array
    else:
        pose = np.zeros(33 * 4)  # Fill with zeros if no pose detected

    if results.left_hand_landmarks:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()     # Flatten into 1-D array
    else:
        left_hand = np.zeros(21 * 3)  # Fill with zeros if no left hand detected

    if results.right_hand_landmarks:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()   # Flatten into 1-D array
    else:
        right_hand = np.zeros(21 * 3)  # Fill with zeros if no right hand detected

    keypoints = np.concatenate([face, pose, left_hand, right_hand]) # Concatenate into one array

    return keypoints