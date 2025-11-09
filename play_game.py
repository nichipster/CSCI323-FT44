from models.alphago_baseline import AlphaGoBaseline

# Create two AlphaGo instances
alphago1 = AlphaGoBaseline.load('experiments/checkpoints', num_simulations=50)
alphago2 = AlphaGoBaseline(num_simulations=50)

# Play game
game_data, winner = alphago1.play_game(opponent=alphago2, show_board=True)

print(f"Game completed in {len(game_data)} moves")
print(f"Winner: {'Black' if winner > 0 else 'White' if winner < 0 else 'Draw'}")