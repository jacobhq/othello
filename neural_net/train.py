import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from game import initial_board, play, maximizer, weighted_score
from game.game import OUTER

# Define the neural network model
def create_model(input_size):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Assuming regression for Q-value estimation
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Convert Othello board to a flat representation
def board_to_input(board):
    flattened_board = np.array([sq for sq in board if sq != OUTER]).flatten()
    print("Flattened Board:", flattened_board)
    return flattened_board

# Generate training data using the game simulation
def generate_training_data(num_games=100):
    X_train, y_train = [], []

    for _ in range(num_games):
        board = initial_board()
        _, final_score = play(maximizer(weighted_score), maximizer(weighted_score), False)

        # Flatten the board and add it to the input data
        X_train.append(board_to_input(board))

        # Use the final score as the target Q-value
        y_train.append(final_score)

    return np.array(X_train), np.array(y_train)

def train_and_save_model():
    # Define input size based on the flattened board representation
    input_size = 100

    # Create the neural network model
    model = create_model(input_size)

    # Generate training data using game simulation
    X_train, y_train = generate_training_data(num_games=100)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('othello_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    train_and_save_model()
