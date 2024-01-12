import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.game import BLACK, human_strategy, play, random_strategy, convert_board_numeric, legal_moves
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf


def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(8, 8)))  # Othello board is 8x8
    model.add(Dense(64, activation='relu'))  # Hidden layer
    # Output layer, one neuron for each possible move
    model.add(Dense(64, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def generate_training_data(num_games):
    X = []
    Y = []

    for i in range(num_games):
        board = play(random_strategy, random_strategy, trainer=True)
        moves = legal_moves(BLACK, board)
        for move in moves:
            X.append(convert_board_numeric(board))
            Y.append(tf.keras.utils.to_categorical(
                move, num_classes=64))  # One-hot encode the move

        print(f"After game {i+1}, X size: {len(X)}, Y size: {len(Y)}")

    return np.array(X), np.array(Y)


def model_strategy(model, player, board):
    board_numeric = np.array([convert_board_numeric(board)])
    predictions = model.predict(board_numeric)[0]
    legal_moves_player = legal_moves(player, board)
    return legal_moves_player[np.argmax(predictions[legal_moves_player])]


def play_with_model(model):
    play(model_strategy(model), human_strategy)


model = create_model()
X, Y = generate_training_data(10000)
model.fit(X, Y, epochs=10)
play_with_model(model)
