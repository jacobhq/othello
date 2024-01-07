import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(8, 8)))  # Othello board is 8x8
    model.add(Dense(64, activation='relu'))  # Hidden layer
    model.add(Dense(64, activation='softmax'))  # Output layer, one neuron for each possible move

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_training_data(num_games):
    X = []
    Y = []

    for _ in range(num_games):
        board = np.random.randint(2, size=(8, 8))  # Random board
        move = np.random.randint(64)  # Random move
        X.append(board)
        Y.append(tf.keras.utils.to_categorical(move, num_classes=64))  # One-hot encode the move

    return np.array(X), np.array(Y)

model = create_model()
X, Y = generate_training_data(10000)
model.fit(X, Y, epochs=10)