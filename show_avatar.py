from packages import *
from functions import *

# Example list of Korean words
prompt = (
    "Given the sentence '의사가 공원에 대해서 알려드릴겁니다', create a natural Korean word array "
    "by extracting only meaningful words without including unnecessary particles, suffixes, or endings. "
    "Focus only on key concepts or actions. Ensure the result is ['의사', '공원', '알려주다']. "
    "Provide only the array as output, without any additional explanations or formatting."
)
#words = converted_list

# For Test
words = "간호사가 입원에 대해서 알려드릴겁니다"

# Change words -> converted_list for actual change
prompt = prompt + " " + words

client = Groq(
    api_key=''
)
completion = client.chat.completions.create(
    #model="llama3-8b-8192",
    #model="gemma2-9b-it",
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)
print("변환된 단어: ")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

word_list = ['간호사','입원', '알려주다']

# Paths for the words
base_path = 'collected_data/text2avatarData'
idle_path = os.path.join(base_path, 'Idle')

# Load 'Idle' Frames
idle_frames = load_word_data(idle_path)

# Debug: Print frame counts
# print(f"Number of frames for Idle: {len(idle_frames)}")

# Initialize combined_frames with Idle at the start (Start with Idle)
combined_frames = idle_frames.copy()
previous_frames = idle_frames

# Define transitions
num_transition_frames = 4  # Number of frames in the transition
for word in word_list:
    word_path = os.path.join(base_path, word)
    word_frames = load_word_data(word_path)

    # Debug: Print frame counts
    print(f"\nNumber of frames for {word}: {len(word_frames)}")

    # Create transition from previous word to the current word
    transition_frames = interpolate_keypoints(previous_frames[-1], word_frames[0], num_transition_frames)
    combined_frames += transition_frames + word_frames

    # Update previous_frames to the current word's frames
    previous_frames = word_frames

# Add Idle at the end (End with Idle)
final_transition = interpolate_keypoints(previous_frames[-1], idle_frames[0], num_transition_frames)
combined_frames += final_transition + idle_frames

# Debug: Print total frame count
print(f"Total combined frames: {len(combined_frames)}")

if __name__ == '__main__':
    # Visualize the combined sequence
    for frame in combined_frames:
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)

        # Extract keypoints
        pose_keypoints = frame[:33 * 4].reshape((33, 4))  # 33 pose landmarks
        left_hand_keypoints = frame[33 * 4:33 * 4 + 21 * 3].reshape((21, 3))  # Left hand landmarks
        right_hand_keypoints = frame[33 * 4 + 21 * 3:].reshape((21, 3))  # Right hand landmarks

        # Draw pose connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            if pose_keypoints[start_idx][3] > 0.5 and pose_keypoints[end_idx][3] > 0.5:  # Visibility > 0.5
                x1, y1 = int(pose_keypoints[start_idx][0] * 500), int(pose_keypoints[start_idx][1] * 500)
                x2, y2 = int(pose_keypoints[end_idx][0] * 500), int(pose_keypoints[end_idx][1] * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Draw thinner lines for pose connections

        # Draw pose keypoints
        for keypoint in pose_keypoints:
            x, y, z, visibility = keypoint
            if visibility > 0.5:  # Only draw visible keypoints
                x = int(x * 500)
                y = int(y * 500)
                cv2.circle(canvas, (x, y), 2, (255, 255, 0), -1)  # Draw pose keypoints in light blue

        # Draw left hand connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if left_hand_keypoints[start_idx][2] != 0 and left_hand_keypoints[end_idx][2] != 0:
                x1, y1 = int(left_hand_keypoints[start_idx][0] * 500), int(left_hand_keypoints[start_idx][1] * 500)
                x2, y2 = int(left_hand_keypoints[end_idx][0] * 500), int(left_hand_keypoints[end_idx][1] * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw thinner lines for left hand

        # Draw left hand keypoints
        for keypoint in left_hand_keypoints:
            x, y, z = keypoint
            x = int(x * 500)
            y = int(y * 500)
            cv2.circle(canvas, (x, y), 2, (255, 0, 0), -1)  # Draw left hand keypoints in blue

        # Draw right hand connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if right_hand_keypoints[start_idx][2] != 0 and right_hand_keypoints[end_idx][2] != 0:
                x1, y1 = int(right_hand_keypoints[start_idx][0] * 500), int(right_hand_keypoints[start_idx][1] * 500)
                x2, y2 = int(right_hand_keypoints[end_idx][0] * 500), int(right_hand_keypoints[end_idx][1] * 500)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw thinner lines for right hand

        # Draw right hand keypoints
        for keypoint in right_hand_keypoints:
            x, y, z = keypoint
            x = int(x * 500)
            y = int(y * 500)
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)  # Draw right hand keypoints in green

        # Display the skeleton
        cv2.imshow("Skeleton Visualization", canvas)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

