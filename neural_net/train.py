# neural_network.py

import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from game.game import *

def convert_board_numeric(board):
    return [0 if char == '?' else 1 if char == '@' else -1 for char in board]

def create_neural_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(squares()), activation='softmax')
    ])
    return model

def generate_training_data(num_samples, black_strategy, white_strategy):
    X_train = []  # List to store numeric board states
    y_train = []  # List to store corresponding next moves

    for _ in range(num_samples):
        board = initial_board()  # Start with the initial board
        player = BLACK  # Start with the Black player

        while player is not None:
            move = get_move(strategy(player, board), player, board)

            # Save the current board state and the corresponding next move
            X_train.append(convert_board_numeric(board))
            y_train.append(move)

            make_move(move, player, board)  # Make the move on the board
            player = next_player(board, player)  # Switch to the next player

    return X_train, y_train

def train_neural_network(X_train, y_train):
    model = create_neural_network(len(X_train[0]))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Data splitting
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Training
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Evaluation (optional)
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

    return model

# ... (Other utility functions related to the neural network, if needed)

# Train the neural network when this module is executed directly
if __name__ == "__main__":
    num_samples = 1000
    X_train, y_train = generate_training_data(num_samples, random_strategy, random_strategy)
    trained_model = train_neural_network(X_train, y_train)
    # You can save the trained model if needed: trained_model.save("othello_neural_network_model")
