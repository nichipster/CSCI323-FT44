import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Monte Carlo Tree Search (MCTS)
==============================
Implements MCTS algorithm with neural network guidance:
1. Selection: Choose promising nodes using UCT formula
2. Expansion: Add new child node
3. Simulation: Rollout to game end
4. Backpropagation: Update node statistics

UCT Formula: Q(node)/N(node) + c*sqrt(ln(N(parent))/N(node))
- Q(node): Total reward from this node
- N(node): Number of visits
- c: Exploration constant (typically √2 or 1.4)
"""

class MCTSNode:
    """
    Node in the Monte Carlo Search Tree

    Attributes:
        game_state (GoGame): Board state at this node
        parent (MCTSNode): Parent node
        move (tuple): Move that led to this node
        children (list): Child nodes
        visits (int): Number of times this node was visited
        value_sum (float): Sum of simulation results
        prior (float): Prior probability from policy network
        untried_moves (list): Legal moves not yet expanded
    """

    def __init__(self, game_state, parent=None, move=None, prior=0):
        """
        Initialize MCTS node

        Args:
            game_state (GoGame): Current board state
            parent (MCTSNode): Parent node in tree
            move (tuple): Move that led to this state
            prior (float): Policy network probability for this move
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.prior = prior
        self.untried_moves = game_state.get_legal_moves()

    def is_fully_expanded(self):
        """Check if all legal moves have been tried"""
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        """
        Select best child using Upper Confidence Bound for Trees (UCT)

        UCT balances exploration and exploitation:
        - High Q/N (exploitation): Choose moves that won frequently
        - High sqrt term (exploration): Choose less-visited moves

        Args:
            c_param (float): Exploration constant (higher = more exploration)

        Returns:
            MCTSNode: Child with highest UCT value
        """
        choices_weights = []
        for child in self.children:
            # Exploitation term: average value
            exploit = child.value_sum / (child.visits + 1e-8)

            # Exploration term: encourage visiting less-explored nodes
            explore = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))

            # UCT value = exploitation + exploration
            uct_value = exploit + explore
            choices_weights.append(uct_value)

        return self.children[np.argmax(choices_weights)]

    def expand(self, move, prior=0):
        """
        Create and add a new child node

        Args:
            move (tuple): Move to expand (row, col)
            prior (float): Policy network probability

        Returns:
            MCTSNode: Newly created child node
        """
        # Create new game state with move applied
        new_game = self.game_state.copy()
        new_game.make_move(move[0], move[1])

        # Create child node
        child = MCTSNode(new_game, parent=self, move=move, prior=prior)
        self.children.append(child)
        self.untried_moves.remove(move)

        return child

    def update(self, result):
        """
        Update node statistics after simulation

        Args:
            result (float): Simulation result (1 for win, -1 for loss)
        """
        self.visits += 1
        self.value_sum += result

    def get_average_value(self):
        """Get average value (win rate) for this node"""
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance

    Combines tree search with neural network evaluation:
    - Policy network guides move selection
    - Value network evaluates leaf nodes
    - Mixing parameter λ balances network vs rollout

    Parameters:
        num_simulations: Number of MCTS iterations per move
        c_param: Exploration constant in UCT
        lambda_param: Weight for value network (0=rollout only, 1=network only)
    """

    def __init__(self, policy_net=None, value_net=None, num_simulations=800,
                 c_param=1.4, lambda_param=0.5):
        """
        Initialize MCTS

        Args:
            policy_net (PolicyNetwork): Policy network for move selection
            value_net (ValueNetwork): Value network for position evaluation
            num_simulations (int): Number of simulations per move
            c_param (float): Exploration constant
            lambda_param (float): Mixing weight (0-1) between value net and rollout
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.lambda_param = lambda_param

    def search(self, game_state):
        """
        Perform MCTS to find best move

        Algorithm:
        1. Run num_simulations iterations
        2. Each iteration: Select -> Expand -> Simulate -> Backpropagate
        3. Return move with highest visit count

        Args:
            game_state (GoGame): Current game state

        Returns:
            tuple: Best move (row, col) or (-1, -1) for pass
        """
        root = MCTSNode(game_state)

        for simulation in range(self.num_simulations):
            node = root
            search_game = game_state.copy()

            # === SELECTION ===
            # Traverse tree using UCT until reaching unexpanded node
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                search_game.make_move(node.move[0], node.move[1])

            # === EXPANSION ===
            # Add new child node if game not over
            if not search_game.is_game_over() and node.untried_moves:
                # Use policy network to select most promising untried move
                if self.policy_net:
                    features = search_game.get_board_features()
                    probs = np.squeeze(self.policy_net.predict(np.expand_dims(features, axis=0), verbose=0))

                    # Find best untried move according to policy
                    best_move = None
                    best_prob = -1
                    for move in node.untried_moves:
                        if move == (-1, -1):
                            move_idx = game_state.size * game_state.size
                        else:
                            move_idx = move[0] * game_state.size + move[1]
                        if probs[move_idx] > best_prob:
                            best_prob = probs[move_idx]
                            best_move = move

                    node = node.expand(best_move, prior=best_prob)
                else:
                    # Random expansion if no policy network
                    move = node.untried_moves[np.random.randint(len(node.untried_moves))]
                    node = node.expand(move)

                search_game.make_move(node.move[0], node.move[1])

            # === SIMULATION/EVALUATION ===
            # Evaluate position using rollout and/or value network
            if search_game.is_game_over():
                result = search_game.get_winner()
                if result != game_state.current_player:
                    result = -1
                else:
                    result = 1
            else:
                # Rollout simulation
                rollout_result = self._rollout(search_game)

                # Mix rollout with value network if available
                if self.value_net:
                    features = node.game_state.get_board_features()
                    value_estimate = float(np.squeeze(self.value_net.predict(np.expand_dims(features, axis=0), verbose=0)))
                    # Weighted combination: λ*rollout + (1-λ)*value_net
                    result = self.lambda_param * rollout_result + \
                            (1 - self.lambda_param) * value_estimate
                else:
                    result = rollout_result

            # === BACKPROPAGATION ===
            # Update statistics for all nodes in path
            while node is not None:
                node.update(result)
                result = -result  # Flip sign for opponent
                node = node.parent

        # Return move with highest visit count (most reliable)
        if not root.children:
            return (-1, -1)  # Pass if no moves available

        visit_counts = [(child.move, child.visits) for child in root.children]
        best_move = max(visit_counts, key=lambda x: x[1])[0]

        return best_move

    def _rollout(self, game_state):
      sim_game = game_state.copy()
      original_player = game_state.current_player

      while not sim_game.is_game_over():
          legal_moves = sim_game.get_legal_moves()
          move = legal_moves[np.random.randint(len(legal_moves))]  # random rollout
          sim_game.make_move(move[0], move[1])

      winner = sim_game.get_winner()
      return 1 if winner == original_player else -1


if __name__ == "__main__":
  # Test MCTS
  print("Testing MCTS...")
  test_game = GoGame()
  mcts = MCTS(policy_net=None, value_net=None, num_simulations=5)  # Fewer sims for testing
  move = mcts.search(test_game)
  print(f"✓ MCTS selected move: {move}")
  print("✓ MCTS working correctly!")   