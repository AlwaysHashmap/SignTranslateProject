from packages import *
from functions import *

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
model = load_model('trained_model.h5')

sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while cap.isOpened():

            ret, image = cap.read()
            results = mediapipe_detection(image, holistic)
            mediapipe_detection_draw_landmarks(image, results)
            keypoints.append(keypoint_value_extraction(results))

            if len(keypoints) == 30:
                # Convert Keypoints list into numpy array
                keypoints = np.array(keypoints)

                # Predict the Keypoints using 'trained_model.h5'
                prediction = model.predict(keypoints[np.newaxis, :, :])

                # Clear Keypoints list for the next set of frames
                keypoints = []

                # Check if the maximum prediction value is above 0.85
                if np.amax(prediction) > 0.85:
                    # Check if the predicted sign is different from the previously predicted sign (Prevents Double prediction and possible loop)
                    if last_prediction != actions[np.argmax(prediction)]:
                        # Append the predicted word to the sentence list
                        sentence.append(actions[np.argmax(prediction)])
                        # last prediction -> latest prediction for the next prediction
                        last_prediction = actions[np.argmax(prediction)]
                # else:
                #     print("nothing")
                #     if last_prediction != "nothing":
                #         # Append the predicted word to the sentence list
                #         sentence.append("nothing")
                #         # last prediction -> latest prediction for the next prediction
                #         last_prediction = "nothing"

            # 'Spacebar' to reset
            if keyboard.is_pressed(' '):
                sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            image = myPutText(image, ' '.join(sentence), (text_X_coord, 430), 25, (255, 255, 255))

            cv2.imshow('Camera', image)

            cv2.waitKey(1)

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

