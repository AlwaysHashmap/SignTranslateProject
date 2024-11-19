from keras.src.layers import BatchNormalization, Dropout

from packages import *
from functions import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = os.path.join('collected_data/data')
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

    # Model setup with increased complexity and lower dropout
    model = Sequential()
    model.add(Input(shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train with validation split to monitor progress
    history = model.fit(X_train, Y_train, epochs=1820)

    # Predict test labels and compare to true labels
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    Y_true = np.argmax(Y_test, axis=1)

    # Print classification report for detailed metrics
    print(classification_report(Y_true, Y_pred, target_names=actions))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=actions, yticklabels=actions, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    model.save('trained_model.h5')


    # Make predictions on the test set
    predictions = np.argmax(model.predict(X_test), axis=1)
    # Get the true labels from the test set
    test_labels = np.argmax(Y_test, axis=1)
    # Calculate the accuracy of the predictions
    accuracy = metrics.accuracy_score(test_labels, predictions)