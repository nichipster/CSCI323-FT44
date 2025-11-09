"""
Unified Model Interface for Go AI Tournament

Implements 4 model variants with clear characteristics:
1. Baseline: Policy Net + MCTS + Neural Rollouts
2. Pure MCTS: No Neural Net + MCTS + Pattern-Based Rollouts
3. Policy-Only: Policy Net + No Search
4. Random MCTS: No Neural Net + MCTS + Random Rollouts
"""

import torch
import numpy as np
from copy import deepcopy
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import PolicyNetwork
from utils.go_game import GoGame


class BaseModel:
    """Base class for all Go AI models"""
    
    def __init__(self, name="Base"):
        self.name = name
        self.move_times = []
        
    def select_move(self, game_state):
        """
        Select a move for the given game state
        
        Args:
            game_state: GoGame instance
            
        Returns:
            move_idx: Integer 0-80 for board, 81 for pass
        """
        raise NotImplementedError
        
    def get_model_info(self):
        """Return dictionary with model characteristics"""
        return {"name": self.name}


class PolicyOnlyModel(BaseModel):
    """
    Model 3: Policy-Only
    - Uses Policy Network
    - No MCTS search
    - Direct move selection from policy
    """
    
    def __init__(self, policy_net=None, device='cpu'):
        super().__init__(name="Policy-Only")
        self.device = device
        
        if policy_net is None:
            self.policy_net = PolicyNetwork().to(device)
        else:
            self.policy_net = policy_net.to(device)
            
        self.policy_net.eval()
    
    def select_move(self, game_state):
        """Select move directly from policy network"""
        state = game_state.get_state().unsqueeze(0).to(self.device)
        legal_mask = game_state.get_legal_moves_mask().to(self.device)
        
        with torch.no_grad():
            move_probs = self.policy_net(state).squeeze(0)
            move_probs = move_probs * legal_mask
            
            if move_probs.sum() == 0:
                return 81  # Pass if no legal moves
                
            move_probs = move_probs / move_probs.sum()
            
            # Select highest probability move
            move_idx = torch.argmax(move_probs).item()
        
        return move_idx
    
    def get_model_info(self):
        return {
            "name": self.name,
            "uses_policy_network": True,
            "uses_mcts": False,
            "rollout_type": "none"
        }


class RandomMCTSModel(BaseModel):
    """
    Model 4: Random MCTS (Baseline comparison)
    - No neural networks
    - Pure MCTS with random rollouts
    - Simplest baseline
    """
    
    def __init__(self, num_simulations=100, c_param=1.4):
        super().__init__(name="Random-MCTS")
        self.num_simulations = num_simulations
        self.c_param = c_param
    
    def select_move(self, game_state):
        """Run MCTS with random rollouts"""
        return self._mcts_search(game_state)
    
    def _mcts_search(self, game_state):
        """Monte Carlo Tree Search with random rollouts"""
        root = MCTSNode(game_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_game = game_state.copy()
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                search_game.make_move(node.move)
            
            # Expansion
            if not search_game.is_game_over() and node.untried_moves:
                move = np.random.choice(node.untried_moves)
                node = node.expand(move)
                search_game.make_move(move)
            
            # Simulation (Random rollout)
            result = self._random_rollout(search_game, game_state.current_player)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                result = -result
                node = node.parent
        
        # Return most visited move
        if not root.children:
            return 81  # Pass
        
        return max(root.children, key=lambda c: c.visits).move
    
    def _random_rollout(self, game_state, original_player):
        """Play random moves until game ends"""
        sim_game = game_state.copy()
        
        while not sim_game.is_game_over():
            legal_moves = sim_game.get_legal_moves()
            move = np.random.choice(legal_moves)
            sim_game.make_move(move)
        
        # Return result from original player's perspective
        score = sim_game._calculate_score()
        if original_player == GoGame.BLACK:
            return 1 if score > 0 else -1
        else:
            return 1 if score < 0 else -1
    
    def get_model_info(self):
        return {
            "name": self.name,
            "uses_policy_network": False,
            "uses_mcts": True,
            "rollout_type": "random",
            "num_simulations": self.num_simulations
        }


class PureMCTSModel(BaseModel):
    """
    Model 2: Pure MCTS
    - No neural networks
    - MCTS with pattern-based rollouts
    - Smarter than random but no learning
    """
    
    def __init__(self, num_simulations=100, c_param=1.4):
        super().__init__(name="Pure-MCTS")
        self.num_simulations = num_simulations
        self.c_param = c_param
    
    def select_move(self, game_state):
        """Run MCTS with pattern-based rollouts"""
        return self._mcts_search(game_state)
    
    def _mcts_search(self, game_state):
        """Monte Carlo Tree Search with pattern rollouts"""
        root = MCTSNode(game_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_game = game_state.copy()
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                search_game.make_move(node.move)
            
            # Expansion
            if not search_game.is_game_over() and node.untried_moves:
                move = np.random.choice(node.untried_moves)
                node = node.expand(move)
                search_game.make_move(move)
            
            # Simulation (Pattern-based rollout)
            result = self._pattern_rollout(search_game, game_state.current_player)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                result = -result
                node = node.parent
        
        # Return most visited move
        if not root.children:
            return 81
        
        return max(root.children, key=lambda c: c.visits).move
    
    def _pattern_rollout(self, game_state, original_player):
        """Rollout with pattern-based heuristics"""
        sim_game = game_state.copy()
        
        while not sim_game.is_game_over():
            legal_moves = sim_game.get_legal_moves()
            
            if len(legal_moves) == 1:
                move = legal_moves[0]
                sim_game.make_move(move)
                continue
            
            # Score moves based on patterns
            scored_moves = []
            for move_idx in legal_moves:
                if move_idx == 81:  # Pass
                    scored_moves.append((move_idx, 0))
                    continue
                
                row = move_idx // GoGame.BOARD_SIZE
                col = move_idx % GoGame.BOARD_SIZE
                score = self._score_move(sim_game, row, col)
                scored_moves.append((move_idx, score))
            
            # Select best scored move
            best_move = max(scored_moves, key=lambda x: x[1])[0]
            sim_game.make_move(best_move)
        
        # Return result
        final_score = sim_game._calculate_score()
        if original_player == GoGame.BLACK:
            return 1 if final_score > 0 else -1
        else:
            return 1 if final_score < 0 else -1
    
    def _score_move(self, game_state, row, col):
        """Score a move based on patterns"""
        score = 0
        
        # Check neighbors
        for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < GoGame.BOARD_SIZE and 0 <= nc < GoGame.BOARD_SIZE:
                if game_state.board[nr, nc] == game_state.current_player:
                    score += 2  # Friendly stones nearby
                elif game_state.board[nr, nc] != GoGame.EMPTY:
                    score += 1  # Enemy stones nearby
        
        # Check for captures (simplified)
        opponent = GoGame.WHITE if game_state.current_player == GoGame.BLACK else GoGame.BLACK
        for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < GoGame.BOARD_SIZE and 0 <= nc < GoGame.BOARD_SIZE:
                if game_state.board[nr, nc] == opponent:
                    score += 5  # Potential capture
        
        return score
    
    def get_model_info(self):
        return {
            "name": self.name,
            "uses_policy_network": False,
            "uses_mcts": True,
            "rollout_type": "pattern-based",
            "num_simulations": self.num_simulations
        }


class BaselineModel(BaseModel):
    """
    Model 1: AlphaGo Baseline
    - Policy Network for move guidance
    - MCTS for search
    - Neural rollouts for evaluation
    """
    
    def __init__(self, policy_net=None, num_simulations=100, c_param=1.4, device='cpu'):
        super().__init__(name="Baseline")
        self.device = device
        self.num_simulations = num_simulations
        self.c_param = c_param
        
        if policy_net is None:
            self.policy_net = PolicyNetwork().to(device)
        else:
            self.policy_net = policy_net.to(device)
            
        self.policy_net.eval()
    
    def select_move(self, game_state):
        """Run MCTS with policy network guidance and neural rollouts"""
        return self._mcts_search(game_state)
    
    def _mcts_search(self, game_state):
        """MCTS with neural network guidance"""
        root = MCTSNode(game_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_game = game_state.copy()
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                search_game.make_move(node.move)
            
            # Expansion (guided by policy network)
            if not search_game.is_game_over() and node.untried_moves:
                move = self._select_move_by_policy(search_game, node.untried_moves)
                node = node.expand(move)
                search_game.make_move(move)
            
            # Simulation (Neural rollout)
            result = self._neural_rollout(search_game, game_state.current_player)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                result = -result
                node = node.parent
        
        # Return most visited move
        if not root.children:
            return 81
        
        return max(root.children, key=lambda c: c.visits).move
    
    def _select_move_by_policy(self, game_state, legal_moves):
        """Use policy network to select from legal moves"""
        state = game_state.get_state().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            move_probs = self.policy_net(state).squeeze(0)
            
            # Find best legal move
            best_move = None
            best_prob = -1
            for move in legal_moves:
                if move_probs[move] > best_prob:
                    best_prob = move_probs[move]
                    best_move = move
        
        return best_move if best_move is not None else np.random.choice(legal_moves)
    
    def _neural_rollout(self, game_state, original_player):
        """Rollout using policy network"""
        sim_game = game_state.copy()
        
        while not sim_game.is_game_over():
            state = sim_game.get_state().unsqueeze(0).to(self.device)
            legal_mask = sim_game.get_legal_moves_mask().to(self.device)
            
            with torch.no_grad():
                move_probs = self.policy_net(state).squeeze(0)
                move_probs = move_probs * legal_mask
                
                if move_probs.sum() == 0:
                    move = 81
                else:
                    move_probs = move_probs / move_probs.sum()
                    # Sample from distribution
                    move = torch.multinomial(move_probs, 1).item()
            
            sim_game.make_move(move)
        
        # Return result
        final_score = sim_game._calculate_score()
        if original_player == GoGame.BLACK:
            return 1 if final_score > 0 else -1
        else:
            return 1 if final_score < 0 else -1
    
    def get_model_info(self):
        return {
            "name": self.name,
            "uses_policy_network": True,
            "uses_mcts": True,
            "rollout_type": "neural",
            "num_simulations": self.num_simulations
        }


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.untried_moves = game_state.get_legal_moves()
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=1.4):
        """Select child with highest UCT value"""
        choices_weights = []
        for child in self.children:
            exploit = child.value_sum / (child.visits + 1e-8)
            explore = c_param * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-8))
            uct_value = exploit + explore
            choices_weights.append(uct_value)
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self, move):
        """Create new child node"""
        new_game = self.game_state.copy()
        new_game.make_move(move)
        
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        
        return child
    
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.value_sum += result
    
    def copy(self):
        """Create a deep copy of the game state"""
        return deepcopy(self)


# Add copy method to GoGame for simulations
def _copy_game(self):
    """Create a deep copy of the game"""
    new_game = GoGame()
    new_game.board = self.board.copy()
    new_game.current_player = self.current_player
    new_game.ko_point = self.ko_point
    new_game.move_history = self.move_history.copy()
    new_game.pass_count = self.pass_count
    new_game.captured_stones = self.captured_stones.copy()
    return new_game

GoGame.copy = _copy_game


# Add is_game_over method to GoGame
def _is_game_over(self):
    """Check if game is over"""
    return self.pass_count >= 2 or len(self.move_history) >= 200

GoGame.is_game_over = _is_game_over


if __name__ == "__main__":
    print("Testing model interface...")
    
    # Test each model
    models = [
        BaselineModel(num_simulations=10),
        PureMCTSModel(num_simulations=10),
        PolicyOnlyModel(),
        RandomMCTSModel(num_simulations=10)
    ]
    
    game = GoGame()
    
    for model in models:
        print(f"\n{model.name}:")
        print(f"  Info: {model.get_model_info()}")
        move = model.select_move(game)
        print(f"  Selected move: {move}")
    
    print("\n✓ All models working!")
