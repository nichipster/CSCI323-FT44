from models.mcts import MCTS
import numpy as np

# pattern-based rollout function 
def pattern_based_rollout(game_state):
    """
    Rollout that is a bit smarter than random:
    - prefer moves next to own stones
    - slightly prefer moves next to enemy stones (fighting)
    - big bonus if the move captures
    """
    sim_game = game_state.copy()
    original_player = game_state.current_player

    while not sim_game.is_game_over():
        legal_moves = sim_game.get_legal_moves()

        # if only pass is available
        if len(legal_moves) == 1:
            move = legal_moves[0]
            sim_game.make_move(*move)
            continue

        scored_moves = []
        for (r, c) in legal_moves:
            if (r, c) == (-1, -1):
                scored_moves.append(((r, c), 0))
                continue

            score = 0

            # look around
            for dr, dc in [(0,1),(1,0),(-1,0),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < sim_game.size and 0 <= nc < sim_game.size:
                    if sim_game.board[nr, nc] == sim_game.current_player:
                        score += 2        # friendly stones nearby
                    elif sim_game.board[nr, nc] == -sim_game.current_player:
                        score += 1        # enemy stones nearby

            # capturing is great
            if sim_game._captures_opponent(r, c):
                score += 5

            scored_moves.append(((r, c), score))

        # pick best scored move
        best_move = max(scored_moves, key=lambda x: x[1])[0]
        sim_game.make_move(*best_move)

    # final result from original player's view
    winner = sim_game.get_winner()
    return 1 if winner == original_player else -1 


# 2️⃣ Pure MCTS (with pattern rollout)
if __name__ == "__main__":
    pure_mcts = MCTS(policy_net=None, value_net=None, num_simulations=3)
    pure_mcts._rollout = lambda g: pattern_based_rollout(g)  # override rollout function