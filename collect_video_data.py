from packages import *
from functions import *

PATH = os.path.join('text2avatarData')
#actions = np.array(['고민', '뻔뻔', '수어'])
#actions = np.array(['고민', '뻔뻔'])
actions = np.array(['Idle'])

sequences = 10
video_folder = 'edited_video'

if __name__ == '__main__':
    # Create 'text2avatarData' directory structure
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence)))
        except FileExistsError:
            pass

    for action in actions:
        video_path = os.path.join(video_folder, f"{action}.mp4")  # Video file path for each action

        # Check if video file exists
        if not os.path.isfile(video_path):
            print(f"Video file for action '{action}' not found.")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_index = 0

        with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            while True:
                ret, image = cap.read()
                if not ret:
                    print(f"End of video for action '{action}' reached.")
                    break

                # Process the image and extract landmarks
                results = mediapipe_detection(image, holistic)
                mediapipe_detection_draw_landmarks(image, results)

                # Display the frame with landmarks
                cv2.imshow('Camera', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                # Save keypoints to the appropriate action/sequence directory
                sequence_index = frame_index // sequences
                frame_number = frame_index % sequences
                frame_path = os.path.join(PATH, action, str(sequence_index), f"{frame_number}.npy")
                keypoints = keypoint_value_extraction(results)
                np.save(frame_path, keypoints)

                frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    print("Processing completed for all videos.")

    # Deleting Empty Sequences
    for root, dirs, files in os.walk(PATH, topdown=False):  # topdown=False allows us to delete subfolders first
        # Check if a directory is empty (no files and no subdirectories)
        if not dirs and not files:
            os.rmdir(root)  # Remove the empty directory
