"""
Training Pipeline for AlphaGo Baseline

Implements supervised learning for policy network
Based on AlphaGo paper methodology (simplified for 9x9 board)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import PolicyNetwork
from models.alphago_baseline import AlphaGoBaseline
from utils.go_game import GoGame


class GoDataset(Dataset):
    """Dataset for Go game positions"""
    
    def __init__(self, states, moves):
        """
        Args:
            states: List of board states (4, 9, 9)
            moves: List of expert moves (integers 0-81)
        """
        self.states = torch.stack([s for s in states])
        self.moves = torch.LongTensor(moves)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx]


class TrainingPipeline:
    """
    Training pipeline for AlphaGo baseline
    
    Training stage:
    - Supervised Learning (SL) for Policy Network
    """
    
    def __init__(self, 
                 policy_net=None,
                 device='cpu',
                 checkpoint_dir='experiments/checkpoints'):
        """
        Args:
            policy_net: Policy network (or None for new)
            device: Training device
            checkpoint_dir: Directory for saving checkpoints
        """
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize network
        if policy_net is None:
            self.policy_net = PolicyNetwork().to(device)
        else:
            self.policy_net = policy_net.to(device)
        
        # Training history
        self.history = {
            'policy_loss': [],
            'policy_accuracy': []
        }
    
    def train_policy_supervised(self, 
                                train_data, 
                                val_data=None,
                                epochs=10,
                                batch_size=32,
                                learning_rate=0.001,
                                save_best=True):
        """
        Train policy network with supervised learning
        
        Args:
            train_data: Tuple of (states, moves)
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_best: Whether to save best model
            
        Returns:
            history: Training history dictionary
        """
        print("\n" + "="*60)
        print("Training Policy Network (Supervised Learning)")
        print("="*60)
        
        # Create datasets
        states, moves = train_data
        train_dataset = GoDataset(states, moves)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_states, val_moves = val_data
            val_dataset = GoDataset(val_states, val_moves)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.policy_net.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_states, batch_moves in pbar:
                batch_states = batch_states.to(self.device)
                batch_moves = batch_moves.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.policy_net(batch_states)
                loss = criterion(outputs, batch_moves)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_moves.size(0)
                train_correct += predicted.eq(batch_moves).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # Validation phase
            if val_data is not None:
                self.policy_net.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_states, batch_moves in val_loader:
                        batch_states = batch_states.to(self.device)
                        batch_moves = batch_moves.to(self.device)
                        
                        outputs = self.policy_net(batch_states)
                        loss = criterion(outputs, batch_moves)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_moves.size(0)
                        val_correct += predicted.eq(batch_moves).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100. * val_correct / val_total
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
                print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
                
                # Save best model
                if save_best and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint('policy_net_best.pth', self.policy_net, epoch, avg_val_loss)
                    print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")
                
                # Update history
                self.history['policy_loss'].append({
                    'epoch': epoch + 1,
                    'train': avg_train_loss,
                    'val': avg_val_loss
                })
                self.history['policy_accuracy'].append({
                    'epoch': epoch + 1,
                    'train': train_accuracy,
                    'val': val_accuracy
                })
            else:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
                
                self.history['policy_loss'].append({
                    'epoch': epoch + 1,
                    'train': avg_train_loss
                })
                self.history['policy_accuracy'].append({
                    'epoch': epoch + 1,
                    'train': train_accuracy
                })
        
        # Save final model
        self._save_checkpoint('policy_net_final.pth', self.policy_net, epochs, avg_train_loss)
        print(f"\nTraining completed! Final model saved.")
        
        return self.history
    
    def generate_selfplay_data(self, num_games=100, num_simulations=100):
        """
        Generate training data through self-play
        
        Args:
            num_games: Number of self-play games to generate
            num_simulations: MCTS simulations per move
            
        Returns:
            states: List of board states
            moves: List of moves played
            outcomes: List of game outcomes
        """
        print("\n" + "="*60)
        print(f"Generating Self-Play Data ({num_games} games)")
        print("="*60)
        
        # Create AlphaGo system
        alphago = AlphaGoBaseline(
            policy_net=self.policy_net,
            num_simulations=num_simulations,
            device=self.device
        )
        
        all_states = []
        all_moves = []
        all_outcomes = []
        
        for game_idx in tqdm(range(num_games), desc="Playing games"):
            # Play game
            game_data, winner = alphago.play_game(show_board=False)
            
            # Extract states, moves, and outcomes
            for state, move_probs, reward in game_data:
                all_states.append(state)
                # Get the move that was actually played (highest probability)
                move = torch.argmax(move_probs).item()
                all_moves.append(move)
                all_outcomes.append(reward)
        
        print(f"\nGenerated {len(all_states)} training positions")
        print(f"  Black wins: {sum(1 for v in all_outcomes if v > 0)}")
        print(f"  White wins: {sum(1 for v in all_outcomes if v < 0)}")
        print(f"  Draws: {sum(1 for v in all_outcomes if v == 0)}")
        
        return all_states, all_moves, all_outcomes
    
    def _save_checkpoint(self, filename, model, epoch, loss):
        """Save model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        model.save(filepath)
    
    def save_history(self, filename='training_history.json'):
        """Save training history to JSON"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")


def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic training data for demonstration
    
    Returns:
        states: List of random board states
        moves: List of random moves
    """
    print("Generating synthetic training data...")
    
    states = []
    moves = []
    
    for _ in range(num_samples):
        # Random board state
        state = torch.randn(4, 9, 9)
        states.append(state)
        
        # Random move
        move = np.random.randint(0, 82)
        moves.append(move)
    
    return states, moves


def demo_training_pipeline():
    """Demonstrate the training pipeline"""
    print("\n" + "="*60)
    print("AlphaGo Training Pipeline Demo")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create pipeline
    pipeline = TrainingPipeline(device=device)
    
    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    train_states, train_moves = generate_synthetic_data(num_samples=500)
    val_states, val_moves = generate_synthetic_data(num_samples=100)
    
    # Train policy network
    print("\n" + "="*60)
    print("Policy Network Training")
    print("="*60)
    
    policy_history = pipeline.train_policy_supervised(
        train_data=(train_states, train_moves),
        val_data=(val_states, val_moves),
        epochs=3,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Save training history
    pipeline.save_history()
    
    print("\n" + "="*60)
    print("Training pipeline demo completed!")
    print("="*60)


if __name__ == "__main__":
    demo_training_pipeline()
