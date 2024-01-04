from game import play, human_strategy, maximizer, weighted_score

# Define the strategies for the human and computer players
human_player_strategy = human_strategy
computer_player_strategy = maximizer(weighted_score)

# Play the game with a human player and the computer
final_board, final_score = play(human_player_strategy,
                                computer_player_strategy)

# Print the final board and score
print("Final board:")
print(print_board(final_board))
print(f"Final Score: {final_score}")

# Determine the winner and print the result
if final_score > 0:
  print("You win!")
elif final_score < 0:
  print("Computer wins!")
else:
  print("It's a tie!")
