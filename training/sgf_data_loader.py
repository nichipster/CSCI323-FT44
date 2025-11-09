"""
SGF Data Loader for AlphaGo Training

Loads real game records from SGF (Smart Game Format) files
and converts them to training data for the policy network.
"""

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from sgfmill import sgf, boards
import json
from datetime import datetime


class SGFDataLoader:
    """Load and preprocess SGF game files for training"""
    
    BOARD_SIZE = 9
    
    def __init__(self, sgf_dir='data/sgf_games'):
        self.sgf_dir = sgf_dir
        self.board_size = self.BOARD_SIZE
        
    def load_sgf_file(self, filepath):
        """Load a single SGF file and extract positions"""
        try:
            with open(filepath, 'rb') as f:
                game = sgf.Sgf_game.from_bytes(f.read())
            
            board_size = game.get_size()
            if board_size != self.BOARD_SIZE:
                return None
            
            main_sequence = game.get_main_sequence()
            board = boards.Board(self.BOARD_SIZE)
            positions = []
            
            for node in main_sequence:
                move_info = node.get_move()
                if move_info is None:
                    continue
                
                color, move = move_info
                if move is None:
                    continue
                
                state = self._board_to_state(board, color)
                row, col = move
                move_idx = row * self.BOARD_SIZE + col
                positions.append((state, move_idx))
                
                try:
                    board.play(row, col, color)
                except:
                    pass
            
            return positions
        except:
            return None
    
    def _board_to_state(self, board, current_color):
        """Convert sgfmill board to tensor representation"""
        state = np.zeros((4, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        
        player_color = 'b' if current_color == 'b' else 'w'
        opponent_color = 'w' if current_color == 'b' else 'b'
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                stone = board.get(row, col)
                if stone == player_color:
                    state[0, row, col] = 1.0
                elif stone == opponent_color:
                    state[1, row, col] = 1.0
                else:
                    state[2, row, col] = 1.0
        
        return torch.FloatTensor(state)
    
    def load_all_games(self, max_games=None, min_moves=20, max_moves=200):
        """Load all SGF files from directory"""
        print("\n" + "="*70)
        print("Loading SGF Game Data")
        print("="*70)
        
        sgf_pattern = os.path.join(self.sgf_dir, '**', '*.sgf')
        sgf_files = glob.glob(sgf_pattern, recursive=True)
        
        if len(sgf_files) == 0:
            print(f"\n⚠ No SGF files found in {self.sgf_dir}")
            print("Please download game data first!")
            return [], []
        
        print(f"\nFound {len(sgf_files)} SGF files")
        
        if max_games is not None:
            sgf_files = sgf_files[:max_games]
            print(f"Loading first {max_games} games...")
        
        all_states = []
        all_moves = []
        games_processed = 0
        games_skipped = 0
        
        for filepath in tqdm(sgf_files, desc="Processing SGF files"):
            positions = self.load_sgf_file(filepath)
            
            if positions is None or len(positions) == 0:
                games_skipped += 1
                continue
            
            if len(positions) < min_moves or len(positions) > max_moves:
                games_skipped += 1
                continue
            
            for state, move in positions:
                all_states.append(state)
                all_moves.append(move)
            
            games_processed += 1
        
        print(f"\n✓ Successfully processed {games_processed} games")
        print(f"  Skipped: {games_skipped} games")
        print(f"  Total positions: {len(all_states)}")
        
        return all_states, all_moves
    
    def save_processed_data(self, states, moves, output_dir='data/processed'):
        """Save processed data to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        states_tensor = torch.stack(states)
        moves_tensor = torch.LongTensor(moves)
        
        train_path = os.path.join(output_dir, 'train_data.pt')
        torch.save({
            'states': states_tensor,
            'moves': moves_tensor,
            'metadata': {
                'num_positions': len(states),
                'board_size': self.BOARD_SIZE,
                'created': datetime.now().isoformat()
            }
        }, train_path)
        
        print(f"\n✓ Saved {len(states)} positions to {train_path}")
        
        stats = {
            'total_positions': len(states),
            'board_size': self.BOARD_SIZE,
            'created': datetime.now().isoformat()
        }
        
        stats_path = os.path.join(output_dir, 'data_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved statistics to {stats_path}")
    
    @staticmethod
    def load_processed_data(data_path='data/processed/train_data.pt'):
        """Load previously processed data"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found: {data_path}")
        
        data = torch.load(data_path)
        
        print(f"\n✓ Loaded {len(data['states'])} positions from {data_path}")
        if 'metadata' in data:
            print(f"  Created: {data['metadata'].get('created', 'Unknown')}")
        
        return data['states'], data['moves']
    
    def create_train_val_split(self, states, moves, val_split=0.1, shuffle=True):
        """Split data into training and validation sets"""
        total_size = len(states)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        if shuffle:
            indices = torch.randperm(total_size)
            states = [states[i] for i in indices]
            moves = [moves[i] for i in indices]
        
        train_states = states[:train_size]
        train_moves = moves[:train_size]
        val_states = states[train_size:]
        val_moves = moves[train_size:]
        
        print(f"\nData Split:")
        print(f"  Training: {len(train_states)} positions")
        print(f"  Validation: {len(val_states)} positions")
        
        return train_states, train_moves, val_states, val_moves


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SGF Data Loader Demo")
    print("="*70)
    
    loader = SGFDataLoader(sgf_dir='data/sgf_games')
    states, moves = loader.load_all_games(max_games=100)
    
    if len(states) == 0:
        print("\n⚠ No game data found!")
        print("\n📥 Download Instructions:")
        print("   1. Visit: https://u-go.net/gamerecords/")
        print("   2. Filter for 9×9 games")
        print("   3. Save SGF files to: data/sgf_games/")
    else:
        print(f"\nDataset Statistics:")
        print(f"  Total positions: {len(states)}")
        print(f"  State shape: {states[0].shape}")
        
        train_states, train_moves, val_states, val_moves = \
            loader.create_train_val_split(states, moves, val_split=0.1)
        
        loader.save_processed_data(states, moves)
        print("\n✓ Demo completed!")
