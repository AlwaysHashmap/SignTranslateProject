from packages import *
from functions import *

PATH = os.path.join('collected_data/data')
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
frames = 30

if __name__ == '__main__':
    # Create 'data' Directory. For each word, create sequence amount with frame amount
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence)))
        except:
            pass

    cap = cv2.VideoCapture(0)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        for action, sequence, frame in product(actions, range(sequences), range(frames)):
            # Sequence의 First Frame 이면 대기. 'Space'로 진행
            if frame == 0:
                while True:
                    if keyboard.is_pressed(' '):
                        break

                    ret, image = cap.read()
                    results = mediapipe_detection(image, holistic)
                    mediapipe_detection_draw_landmarks(image, results)

                    image = myPutText(image, 'Recording data "{}" & Sequence number = {}.'.format(action, sequence), (10, 10), 25, (255, 0, 0))
                    image = myPutText(image, 'Press "Space" when you are ready.', (10, 40), 25, (255, 0, 0))

                    cv2.imshow('Camera', image)
                    cv2.waitKey(1)

                    # Check if 'Camera' is closed
                    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                        break
            else:
                ret, image = cap.read()

                results = mediapipe_detection(image, holistic)
                mediapipe_detection_draw_landmarks(image, results)

                image = myPutText(image, 'Recording data "{}" & Sequence number = {}.'.format(action, sequence), (10, 10), 25, (255, 0, 0))
                cv2.imshow('Camera', image)
                cv2.waitKey(1)

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Extract landmarks and save them to arrays (.npy files)
            keypoints = keypoint_value_extraction(results)
            frame_path = os.path.join(PATH, action, str(sequence), str(frame))
            np.save(frame_path, keypoints)

        cap.release()
        cv2.destroyAllWindows()
