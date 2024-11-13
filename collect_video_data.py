from packages import *
from functions import *

PATH = os.path.join('text2avatarData')
#actions = np.array(['화장실', '고민', '어디', '있다'])
# ['감기', '감사합니다', '고열', '구내염', '귀', '근육통', '눈', '다리', '두드러기', '두통', '등', '따끔거리다', '멍들다', '목', '몸',
# '몸살', '무릎', '물다', '발목', '부러지다', '붕대', '뼈', '사마귀', '설사', '소화불량', '손',
# '아프다', '안녕하세요', '어깨', '어지럽다', '얼굴', '열', '의사', '자주', '찰과상', '코로나',
# '토하다', '피', '피부', '허리', '호흡곤란', '화상']
#actions = np.array(['머리'])
# actions = np.array(['아프다'])
actions = np.array(['열'])

# 머리 아프다. 열 있다.
#
sequences = 50

if __name__ == '__main__':
    # Create 'text2avatarData' directory structure
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence)))
        except FileExistsError:
            pass

    cap = cv2.VideoCapture('video1.mp4')

    frame_index = 0
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while True:
            ret, image = cap.read()

            # Break the loop if the video ends
            if not ret:
                print("End of video reached.")
                break

            results = mediapipe_detection(image, holistic)
            mediapipe_detection_draw_landmarks(image, results)

            cv2.imshow('Camera', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Extract landmarks and save to .npy files
            keypoints = keypoint_value_extraction(results)
            frame_path = os.path.join(PATH, actions[0], str(frame_index // sequences), str(frame_index % sequences))
            np.save(frame_path, keypoints)

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
