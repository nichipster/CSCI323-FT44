from models.mcts import MCTS

# 1️⃣ Random MCTS (default rollout)
if __name__ == "__main__":
    random_mcts = MCTS(policy_net=None, value_net=None, num_simulations=3)