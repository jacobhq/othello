from game import play, maximizer, weighted_score, random_strategy, print_board
from game import convert_board_numeric

# Define strategies for both players using the random_strategy function (comp vs comp)
black_strategy = maximizer(weighted_score)
white_strategy = random_strategy

# Play the game using the defined strategies (comp vs comp)
final_board, final_score = play(black_strategy, white_strategy)

# Print the final board and score
print("Final board:")
print(print_board(final_board))
print(f"Final Score: {final_score}")
print(convert_board_numeric(final_board))

# Determine the winner and print the result
if final_score > 0:
    print("Black (@) wins!")
elif final_score < 0:
    print("White (o) wins!")
else:
    print("It's a tie!")
