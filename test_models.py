import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_interface import BaselineModel, PureMCTSModel, PolicyOnlyModel, RandomMCTSModel
from utils.go_game import GoGame

models = [
    BaselineModel(num_simulations=10),
    PureMCTSModel(num_simulations=10),
    PolicyOnlyModel(),
    RandomMCTSModel(num_simulations=10)
]

for model in models:
    print(f"Testing {model.name}...")
    game = GoGame()
    move = model.select_move(game)
    print(f"  Selected move: {move} ✓")

print("\n✓ ALL MODELS WORKING!")