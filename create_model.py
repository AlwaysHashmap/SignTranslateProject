from packages import *
from functions import *

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
sequences = 30
frames = 30

# {'action[0]' : 0, 'action[1]' : 1}
label_map = {label:num for num, label in enumerate(actions)}
landmarks, labels = [], []

if __name__ == '__main__':
    # Load landmarks and correspond labels by each action
    for action, sequence in product(actions, range(sequences)):
        temp = []
        for frame in range(frames):
            npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
            temp.append(npy)
        landmarks.append(temp)
        labels.append(label_map[action])

    # Convert landmarks and labels to numpy arrays
    X, Y = np.array(landmarks), to_categorical(labels).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

    # Build Model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(30,1662))) # 30 frames, 126 Keypoint Values
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # Train the model
    model.fit(X_train, Y_train, epochs=500)
    # Save the trained model
    model.save('trained_model.h5')


    # Make predictions on the test set
    predictions = np.argmax(model.predict(X_test), axis=1)
    # Get the true labels from the test set
    test_labels = np.argmax(Y_test, axis=1)
    # Calculate the accuracy of the predictions
    accuracy = metrics.accuracy_score(test_labels, predictions)