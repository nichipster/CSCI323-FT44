"""
Monte Carlo Tree Search (MCTS) for Go

Based on AlphaGo paper (Silver et al., 2016)
Combines policy network with neural rollouts for position evaluation
"""

import numpy as np
import torch
import math
from copy import deepcopy


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, game_state, parent=None, move=None, prior=0.0):
        """
        Args:
            game_state: Current game state
            parent: Parent node
            move: Move that led to this node
            prior: Prior probability from policy network
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def uct_score(self, c_puct=1.0):
        """
        Calculate UCT (Upper Confidence bound for Trees) score
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            UCT score balancing exploitation and exploration
        """
        # Exploitation term
        q_value = self.value()
        
        # Exploration term
        if self.parent is None:
            return q_value
        
        u_value = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct=1.0):
        """Select child with highest UCT score"""
        return max(self.children.values(), key=lambda child: child.uct_score(c_puct))
    
    def expand(self, policy_probs):
        """
        Expand node by adding children for legal moves
        
        Args:
            policy_probs: Move probabilities from policy network
        """
        legal_moves = self.game_state.get_legal_moves()
        
        for move in legal_moves:
            if move not in self.children:
                # Create child node
                child_game = deepcopy(self.game_state)
                child_game.make_move(move)
                
                prior = policy_probs[move].item() if isinstance(policy_probs, torch.Tensor) else policy_probs[move]
                self.children[move] = MCTSNode(child_game, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def update(self, value):
        """
        Update node statistics
        
        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance
    
    Uses policy network for move selection and neural rollouts for evaluation
    """
    
    def __init__(self, policy_net, num_simulations=800, c_puct=1.0, device='cpu'):
        """
        Args:
            policy_net: Policy network for move probabilities
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            device: Device for neural networks
        """
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        
        # Move network to device
        self.policy_net.to(device)
        
        # Set to evaluation mode
        self.policy_net.eval()
    
    def search(self, game_state):
        """
        Perform MCTS search from current game state
        
        Args:
            game_state: Current game state
            
        Returns:
            move_probs: Probability distribution over moves
        """
        root = MCTSNode(deepcopy(game_state))
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf
            while node.is_expanded and len(node.children) > 0:
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Get current game state
            current_game = node.game_state
            
            # Check if game is over
            done = len(current_game.get_legal_moves()) <= 1  # Only pass available
            
            if not done:
                # Expansion: expand leaf node
                if not node.is_expanded:
                    # Get policy network predictions
                    state = current_game.get_state().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        policy_probs = self.policy_net(state).squeeze(0).cpu()
                    
                    # Mask illegal moves
                    legal_mask = current_game.get_legal_moves_mask()
                    policy_probs = policy_probs * legal_mask
                    policy_probs = policy_probs / policy_probs.sum()
                    
                    # Expand node
                    node.expand(policy_probs)
                
                # Evaluation: evaluate leaf using neural rollout
                value = self._rollout(current_game)
            else:
                # Game is over
                _, score = current_game.make_move(81)  # Pass
                value = np.tanh(score / 20.0)  # Normalize score
            
            # Backpropagation: update all nodes in search path
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Flip value for opponent
        
        # Return move probabilities based on visit counts
        return self._get_move_probs(root)
    
    def _rollout(self, game_state):
        """
        Fast rollout simulation using policy network
        
        Args:
            game_state: Game state to rollout from
            
        Returns:
            value: Estimated value of position
        """
        rollout_game = deepcopy(game_state)
        current_player = rollout_game.current_player
        
        max_moves = 100  # Prevent infinite games
        for _ in range(max_moves):
            legal_moves = rollout_game.get_legal_moves()
            
            if len(legal_moves) <= 1:  # Only pass available
                break
            
            # Use policy network to guide rollout
            state = rollout_game.get_state().unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_probs = self.policy_net(state).squeeze(0).cpu()
            
            # Mask illegal moves and sample
            legal_mask = rollout_game.get_legal_moves_mask()
            policy_probs = policy_probs * legal_mask
            
            if policy_probs.sum() > 0:
                policy_probs = policy_probs / policy_probs.sum()
                move = torch.multinomial(policy_probs, 1).item()
            else:
                move = np.random.choice(legal_moves)
            
            done, reward = rollout_game.make_move(move)
            if done:
                break
        
        # Get final score
        _, score = rollout_game.make_move(81)  # Pass to end game
        
        # Normalize score to [-1, 1]
        value = np.tanh(score / 20.0)
        
        # Return value from original player's perspective
        if rollout_game.current_player != current_player:
            value = -value
        
        return value
    
    def _get_move_probs(self, root, temperature=1.0):
        """
        Get move probability distribution based on visit counts
        
        Args:
            root: Root node of search tree
            temperature: Temperature for move selection (1.0 = proportional, 0 = argmax)
            
        Returns:
            move_probs: Tensor of move probabilities
        """
        visits = np.zeros(82)
        for move, child in root.children.items():
            visits[move] = child.visit_count
        
        if temperature == 0:
            # Argmax selection
            best_move = np.argmax(visits)
            probs = np.zeros(82)
            probs[best_move] = 1.0
        else:
            # Proportional to visit count
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
        
        return torch.FloatTensor(probs)
    
    def get_best_move(self, game_state, temperature=0.0):
        """
        Get best move using MCTS
        
        Args:
            game_state: Current game state
            temperature: Temperature for move selection
            
        Returns:
            best_move: Index of best move
            move_probs: Probability distribution over moves
        """
        move_probs = self.search(game_state)
        
        if temperature == 0:
            best_move = torch.argmax(move_probs).item()
        else:
            best_move = torch.multinomial(move_probs, 1).item()
        
        return best_move, move_probs


if __name__ == "__main__":
    # Test MCTS
    import sys
    sys.path.append('..')
    from models.policy_net import PolicyNetwork
    from utils.go_game import GoGame
    
    print("Testing MCTS...")
    
    # Create network
    policy_net = PolicyNetwork()
    
    # Create MCTS
    mcts = MCTS(policy_net, num_simulations=100)
    
    # Create game
    game = GoGame()
    
    # Get best move
    print("Running MCTS search...")
    best_move, move_probs = mcts.get_best_move(game)
    
    print(f"Best move: {best_move}")
    print(f"Move probability: {move_probs[best_move]:.4f}")
    print(f"Top 5 moves:")
    top_moves = torch.topk(move_probs, 5)
    for prob, move in zip(top_moves.values, top_moves.indices):
        print(f"  Move {move.item()}: {prob.item():.4f}")
