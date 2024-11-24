from packages import *
from functions import *

PATH = os.path.join('collected_data/data')
actions = np.array(os.listdir(PATH))
model = load_model('trained_model.h5')

sentence, keypoints, last_prediction = [], [], []

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

                print(np.amax(prediction))
                print(actions[np.argmax(prediction)])
                # Check if the maximum prediction value is above 0.85
                #if np.amax(prediction) > 0.15 and np.amax(prediction) < 0.35:
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

        converted_list = [str(element) for element in sentence]
        #print(converted_list)
        print("Detected Words: ['머리', '아프다', '열', '있다']")
        #print("변환된 문장: 머리가 아파요. 열이 있어요.")

        # Example list of Korean words
        prompt = "Given the words ['머리', '아프다', '열', '있다'], create a natural, spoken Korean sentence that flows as if someone were casually describing their symptoms. Avoid added phrases like 'I apologize for the mistake' and prioritize a conversational tone. Keep it short, fluent, and clear, like '머리가 아프고 열이 있어요'. This is just an example, don't use those words. I will give you new words. Only uses these words and actually put theses words in the sentence. Please be formal. Just give me the answer I don't need an explaination"
        #words = converted_list

        # For Test
        words = "['머리', '아프다', '열', '있다']"
        #words = "['눈', '아프다', '기침', '있다']"
        #words = "['가슴', '하고', '귀', '아프다']"

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
        print("변환된 문장: ")
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
