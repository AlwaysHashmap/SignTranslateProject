import numpy as np
import cv2
import os

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

keypoints_folder = 'text2avatarData/열'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                             # Thumb connections
    (0, 5), (5, 6), (6, 7), (7, 8),                             # Index finger connections
    (0, 9), (9, 10), (10, 11), (11, 12),                        # Middle finger connections
    (0, 13), (13, 14), (14, 15), (15, 16),                      # Ring finger connections
    (0, 17), (17, 18), (18, 19), (19, 20)                       # Pinky finger connections
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),                             # Left face connections
    (0, 4), (4, 5), (5, 6), (6, 8),                             # Right face connections
    (9, 10),                                                    # Mouth connections
    (11, 12), (11, 23), (12, 24), (23, 24),                     # Torso connections
    (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19), # Left arm connections
    (12, 14), (14, 16), (16, 22), (16, 20), (16, 18), (18, 20), # Right arm connections
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),           # Left leg connections
    (24, 26), (26, 28), (28, 32), (30, 32), (28, 30)            # Right leg connections
]

for sequence in os.listdir(keypoints_folder):
    sequence_path = os.path.join(keypoints_folder, sequence)
    if not os.path.isdir(sequence_path):
        continue

    # Iterate through each frame file in the sequence
    for frame_file in sorted(os.listdir(sequence_path), key=lambda x: int(x.split('.')[0])):
        # Load keypoints from the .npy file
        frame_path = os.path.join(sequence_path, frame_file)
        keypoints = np.load(frame_path)

        # Create an empty black canvas
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)

        # Extract pose, left hand, and right hand keypoints from the keypoints array
        pose_keypoints = keypoints[:NUM_POSE_LANDMARKS * 4].reshape((NUM_POSE_LANDMARKS, 4))
        left_hand_keypoints = keypoints[NUM_POSE_LANDMARKS * 4: NUM_POSE_LANDMARKS * 4 + NUM_HAND_LANDMARKS * 3].reshape((NUM_HAND_LANDMARKS, 3))
        right_hand_keypoints = keypoints[NUM_POSE_LANDMARKS * 4 + NUM_HAND_LANDMARKS * 3:].reshape((NUM_HAND_LANDMARKS, 3))

        # Draw pose keypoints
        for idx, keypoint in enumerate(pose_keypoints):
            x, y, z, visibility = keypoint
            if visibility > 0.5:  # Only draw visible landmarks
                x = int(x * 500)  # Scale to canvas size
                y = int(y * 500)
                cv2.circle(canvas, (x, y), 2, (255, 255, 0), -1)  # Draw pose keypoints in light blue

        # Draw left hand keypoints
        for keypoint in left_hand_keypoints:
            x, y, z = keypoint
            x = int(x * 500)  # Scale to canvas size
            y = int(y * 500)
            cv2.circle(canvas, (x, y), 2, (255, 0, 0), -1)  # Draw left hand keypoints in blue

        # Draw right hand keypoints
        for keypoint in right_hand_keypoints:
            x, y, z = keypoint
            x = int(x * 500)  # Scale to canvas size
            y = int(y * 500)
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)  # Draw right hand keypoints in green

        # Draw connections for pose skeleton
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if pose_keypoints[start_idx][3] > 0.5 and pose_keypoints[end_idx][3] > 0.5:  # Ensure valid and visible points
                x1, y1, _ = pose_keypoints[start_idx][:3]
                x2, y2, _ = pose_keypoints[end_idx][:3]
                x1, y1 = int(x1 * 500), int(y1 * 500)
                x2, y2 = int(x2 * 500), int(y2 * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Draw thinner lines for pose connections

        # Draw connections for left and right hand skeletons
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if left_hand_keypoints[start_idx][2] != 0 and left_hand_keypoints[end_idx][2] != 0:  # Ensure valid points
                x1, y1, _ = left_hand_keypoints[start_idx]
                x2, y2, _ = left_hand_keypoints[end_idx]
                x1, y1 = int(x1 * 500), int(y1 * 500)
                x2, y2 = int(x2 * 500), int(y2 * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw thinner lines for left hand

            if right_hand_keypoints[start_idx][2] != 0 and right_hand_keypoints[end_idx][2] != 0:  # Ensure valid points
                x1, y1, _ = right_hand_keypoints[start_idx]
                x2, y2, _ = right_hand_keypoints[end_idx]
                x1, y1 = int(x1 * 500), int(y1 * 500)
                x2, y2 = int(x2 * 500), int(y2 * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw thinner lines for right hand

        # Show the skeleton visualization for the current frame
        cv2.imshow("Skeleton Visualization", canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit visualization
            break

# Release all windows
cv2.destroyAllWindows()
