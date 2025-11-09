"""
AlphaGo Baseline System

Combines Policy Network + MCTS + Neural Rollouts
Based on Silver et al. (2016) "Mastering the game of Go with deep neural networks and tree search"

Note: This is a simplified baseline that uses neural rollouts for position evaluation
rather than a separate value network, making it more suitable for educational purposes
and limited computational resources.
"""

import torch
import numpy as np
from copy import deepcopy
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import PolicyNetwork
from utils.mcts_neural_rollouts import MCTS
from utils.go_game import GoGame


class AlphaGoBaseline:
    """
    Complete AlphaGo baseline system
    
    Components:
    1. Policy Network (SL): Predicts expert human moves
    2. MCTS: Tree search guided by policy network
    3. Neural Rollouts: Fast simulations using policy network
    """
    
    def __init__(self, 
                 policy_net=None,
                 num_simulations=800,
                 c_puct=1.0,
                 device='cpu'):
        """
        Args:
            policy_net: Trained policy network (or None for random initialization)
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for MCTS
            device: Device for neural networks ('cpu' or 'cuda')
        """
        self.device = device
        
        # Initialize policy network
        if policy_net is None:
            self.policy_net = PolicyNetwork().to(device)
        else:
            self.policy_net = policy_net.to(device)
        
        # Initialize MCTS (using rollouts for evaluation)
        self.mcts = MCTS(
            policy_net=self.policy_net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device
        )
        
        self.num_simulations = num_simulations
        
    def select_move(self, game_state, temperature=0.0):
        """
        Select move using full AlphaGo pipeline
        
        Args:
            game_state: Current game state
            temperature: Temperature for move selection
                        0 = deterministic (argmax)
                        >0 = stochastic (proportional to visit count)
        
        Returns:
            move: Selected move index
            move_probs: MCTS move probability distribution
        """
        move, move_probs = self.mcts.get_best_move(game_state, temperature)
        return move, move_probs
    
    def play_game(self, opponent=None, show_board=False):
        """
        Play a complete game
        
        Args:
            opponent: Another AlphaGoBaseline instance (or None for self-play)
            show_board: Whether to print board after each move
            
        Returns:
            game_data: List of (state, move_probs, reward) tuples
            winner: 1 for black, -1 for white, 0 for draw
        """
        game = GoGame()
        game_data = []
        
        if opponent is None:
            opponent = self  # Self-play
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while move_count < max_moves:
            # Get current state
            state = game.get_state()
            
            # Select move (AlphaGo for black, opponent for white)
            if game.current_player == GoGame.BLACK:
                move, move_probs = self.select_move(game, temperature=1.0 if move_count < 30 else 0)
            else:
                move, move_probs = opponent.select_move(game, temperature=1.0 if move_count < 30 else 0)
            
            # Store game data
            game_data.append((state, move_probs, 0))  # Reward updated at end
            
            # Make move
            done, reward = game.make_move(move)
            
            if show_board:
                print(f"\nMove {move_count + 1}: Player {game.current_player} plays {move}")
                game.render()
            
            if done:
                # Game over - assign rewards
                final_score = reward
                winner = 1 if final_score > 0 else (-1 if final_score < 0 else 0)
                
                # Update rewards in game data
                for i in range(len(game_data)):
                    state, probs, _ = game_data[i]
                    # Alternate rewards (black gets +1 for win, white gets -1)
                    player = GoGame.BLACK if i % 2 == 0 else GoGame.WHITE
                    game_reward = winner if player == GoGame.BLACK else -winner
                    game_data[i] = (state, probs, game_reward)
                
                if show_board:
                    print(f"\nGame Over! Final score: {final_score:.1f}")
                    print(f"Winner: {'Black' if winner > 0 else 'White' if winner < 0 else 'Draw'}")
                
                return game_data, winner
            
            move_count += 1
        
        # Max moves reached
        print("Game reached maximum moves!")
        return game_data, 0
    
    def predict_move(self, game_state):
        """
        Predict move probabilities using policy network only (no MCTS)
        
        Args:
            game_state: Game state
            
        Returns:
            move_probs: Probability distribution over moves
        """
        state = game_state.get_state().unsqueeze(0).to(self.device)
        legal_mask = game_state.get_legal_moves_mask().to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            move_probs = self.policy_net(state).squeeze(0)
            move_probs = move_probs * legal_mask
            move_probs = move_probs / move_probs.sum()
        
        return move_probs.cpu()
    
    def save(self, checkpoint_dir):
        """
        Save model checkpoints
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        policy_path = os.path.join(checkpoint_dir, 'policy_net.pth')
        self.policy_net.save(policy_path)
        
        print(f"Model saved to {checkpoint_dir}")
    
    @classmethod
    def load(cls, checkpoint_dir, num_simulations=800, device='cpu'):
        """
        Load model from checkpoints
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            num_simulations: Number of MCTS simulations
            device: Device for models
            
        Returns:
            AlphaGoBaseline instance with loaded model
        """
        policy_path = os.path.join(checkpoint_dir, 'policy_net.pth')
        policy_net = PolicyNetwork.load(policy_path, device)
        
        print(f"Model loaded from {checkpoint_dir}")
        
        return cls(
            policy_net=policy_net,
            num_simulations=num_simulations,
            device=device
        )
    
    def get_info(self):
        """Get system information"""
        policy_params = sum(p.numel() for p in self.policy_net.parameters())
        
        info = {
            'policy_network_params': policy_params,
            'total_params': policy_params,
            'num_simulations': self.num_simulations,
            'device': self.device,
            'evaluation_method': 'Neural Rollouts'
        }
        
        return info


def demo_baseline():
    """Demonstrate the baseline system"""
    print("=" * 60)
    print("AlphaGo Baseline System Demo")
    print("=" * 60)
    
    # Create baseline system
    print("\nInitializing AlphaGo Baseline...")
    alphago = AlphaGoBaseline(num_simulations=100, device='cpu')
    
    # Show system info
    info = alphago.get_info()
    print("\nSystem Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create game
    game = GoGame()
    print("\nInitial board:")
    game.render()
    
    # Select a move
    print("\nRunning MCTS search (100 simulations)...")
    move, move_probs = alphago.select_move(game, temperature=1.0)
    
    print(f"\nSelected move: {move}")
    print(f"Move probability: {move_probs[move]:.4f}")
    
    # Show top 5 moves
    print("\nTop 5 moves:")
    top_moves = torch.topk(move_probs, min(5, len(move_probs)))
    for prob, move_idx in zip(top_moves.values, top_moves.indices):
        print(f"  Move {move_idx.item()}: {prob.item():.4f}")
    
    # Play a quick game
    print("\n" + "=" * 60)
    print("Playing a demonstration game (self-play)...")
    print("=" * 60)
    
    alphago_fast = AlphaGoBaseline(num_simulations=50, device='cpu')
    game_data, winner = alphago_fast.play_game(show_board=True)
    
    print(f"\nGame completed in {len(game_data)} moves")
    print(f"Winner: {'Black' if winner > 0 else 'White' if winner < 0 else 'Draw'}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_baseline()
