"""
Simplified 9x9 Go Game Environment

Implements basic Go rules for a 9x9 board
"""

import numpy as np
import torch
from copy import deepcopy


class GoGame:
    """
    9x9 Go game environment
    
    Board representation:
    - 0: Empty
    - 1: Black stone
    - 2: White stone
    """
    
    BOARD_SIZE = 9
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    
    def __init__(self):
        """Initialize empty board"""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = self.BLACK
        self.ko_point = None  # For ko rule
        self.move_history = []
        self.pass_count = 0
        self.captured_stones = {self.BLACK: 0, self.WHITE: 0}
        
    def reset(self):
        """Reset game to initial state"""
        self.__init__()
        return self.get_state()
    
    def get_state(self):
        """
        Get current board state as tensor
        
        Returns:
            Tensor of shape (4, 9, 9) with:
            - Channel 0: Current player stones
            - Channel 1: Opponent stones
            - Channel 2: Empty positions
            - Channel 3: Move history (last move)
        """
        state = np.zeros((4, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK
        
        # Channel 0: Current player stones
        state[0] = (self.board == self.current_player).astype(np.float32)
        
        # Channel 1: Opponent stones
        state[1] = (self.board == opponent).astype(np.float32)
        
        # Channel 2: Empty positions
        state[2] = (self.board == self.EMPTY).astype(np.float32)
        
        # Channel 3: Last move indicator
        if len(self.move_history) > 0:
            last_move = self.move_history[-1]
            if last_move != 81:  # Not a pass
                row, col = last_move // self.BOARD_SIZE, last_move % self.BOARD_SIZE
                state[3, row, col] = 1.0
        
        return torch.FloatTensor(state)
    
    def get_legal_moves(self):
        """
        Get list of legal moves
        
        Returns:
            List of move indices (0-80 for board positions, 81 for pass)
        """
        legal_moves = []
        
        # Check each board position
        for i in range(self.BOARD_SIZE * self.BOARD_SIZE):
            row, col = i // self.BOARD_SIZE, i % self.BOARD_SIZE
            if self.is_legal_move(row, col):
                legal_moves.append(i)
        
        # Pass is always legal
        legal_moves.append(81)
        
        return legal_moves
    
    def get_legal_moves_mask(self):
        """
        Get binary mask of legal moves
        
        Returns:
            Tensor of shape (82,) with 1 for legal moves, 0 for illegal
        """
        mask = torch.zeros(82)
        legal_moves = self.get_legal_moves()
        mask[legal_moves] = 1.0
        return mask
    
    def is_legal_move(self, row, col):
        """Check if a move is legal"""
        # Must be empty
        if self.board[row, col] != self.EMPTY:
            return False
        
        # Check ko rule
        if self.ko_point is not None and (row, col) == self.ko_point:
            return False
        
        # Simulate move to check for suicide
        temp_board = deepcopy(self.board)
        temp_board[row, col] = self.current_player
        
        # Check if move captures opponent stones
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                if temp_board[nr, nc] == opponent:
                    if self._count_liberties(temp_board, nr, nc) == 0:
                        return True  # Captures opponent stones
        
        # Check if own group has liberties (not suicide)
        if self._count_liberties(temp_board, row, col) > 0:
            return True
        
        return False
    
    def make_move(self, move_idx):
        """
        Make a move
        
        Args:
            move_idx: Integer 0-80 for board position, 81 for pass
            
        Returns:
            done: Boolean indicating if game is over
            reward: Reward for the move (0 during game, final score at end)
        """
        if move_idx == 81:  # Pass
            self.pass_count += 1
            self.move_history.append(81)
            self.ko_point = None
            
            # Game ends after two consecutive passes
            if self.pass_count >= 2:
                return True, self._calculate_score()
            
            self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
            return False, 0
        
        self.pass_count = 0
        row, col = move_idx // self.BOARD_SIZE, move_idx % self.BOARD_SIZE
        
        if not self.is_legal_move(row, col):
            # Illegal move - penalize heavily
            return True, -100
        
        # Place stone
        self.board[row, col] = self.current_player
        self.move_history.append(move_idx)
        
        # Remove captured opponent stones
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK
        captured_groups = []
        single_capture = None
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                if self.board[nr, nc] == opponent:
                    if self._count_liberties(self.board, nr, nc) == 0:
                        captured = self._remove_group(nr, nc)
                        captured_groups.extend(captured)
                        if len(captured) == 1:
                            single_capture = captured[0]
        
        # Ko rule: if exactly one stone was captured, mark it
        if len(captured_groups) == 1:
            self.ko_point = single_capture
        else:
            self.ko_point = None
        
        self.captured_stones[self.current_player] += len(captured_groups)
        
        # Switch player
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
        
        return False, 0
    
    def _count_liberties(self, board, row, col):
        """Count liberties of a group"""
        if board[row, col] == self.EMPTY:
            return 0
        
        color = board[row, col]
        visited = set()
        liberties = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                    if board[nr, nc] == self.EMPTY:
                        liberties.add((nr, nc))
                    elif board[nr, nc] == color and (nr, nc) not in visited:
                        stack.append((nr, nc))
        
        return len(liberties)
    
    def _remove_group(self, row, col):
        """Remove a group of stones and return their positions"""
        color = self.board[row, col]
        captured = []
        stack = [(row, col)]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            if self.board[r, c] == color:
                self.board[r, c] = self.EMPTY
                captured.append((r, c))
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                        if self.board[nr, nc] == color:
                            stack.append((nr, nc))
        
        return captured
    
    def _calculate_score(self):
        """
        Calculate final score (simplified Chinese rules)
        
        Returns:
            Score from Black's perspective (positive = Black winning)
        """
        # Count stones and territory
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)
        
        # Add captured stones
        black_score += self.captured_stones[self.BLACK]
        white_score += self.captured_stones[self.WHITE]
        
        # Count territory (simplified - just empty points adjacent to stones)
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i, j] == self.EMPTY:
                    black_neighbors = 0
                    white_neighbors = 0
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.BOARD_SIZE and 0 <= nj < self.BOARD_SIZE:
                            if self.board[ni, nj] == self.BLACK:
                                black_neighbors += 1
                            elif self.board[ni, nj] == self.WHITE:
                                white_neighbors += 1
                    
                    if black_neighbors > 0 and white_neighbors == 0:
                        black_score += 1
                    elif white_neighbors > 0 and black_neighbors == 0:
                        white_score += 1
        
        # Komi (compensation for white)
        komi = 7.5
        white_score += komi
        
        return black_score - white_score
    
    def render(self):
        """Print board to console"""
        print("\n  ", end="")
        for i in range(self.BOARD_SIZE):
            print(f" {i}", end="")
        print()
        
        for i in range(self.BOARD_SIZE):
            print(f"{i} ", end="")
            for j in range(self.BOARD_SIZE):
                if self.board[i, j] == self.EMPTY:
                    print(" .", end="")
                elif self.board[i, j] == self.BLACK:
                    print(" X", end="")
                else:
                    print(" O", end="")
            print()
        
        print(f"\nCurrent player: {'Black' if self.current_player == self.BLACK else 'White'}")
        print(f"Captured - Black: {self.captured_stones[self.BLACK]}, White: {self.captured_stones[self.WHITE]}")


if __name__ == "__main__":
    # Test the game
    game = GoGame()
    print("Initial board:")
    game.render()
    
    # Make some test moves
    game.make_move(40)  # Center
    game.render()
    
    game.make_move(30)
    game.render()
    
    # Check legal moves
    legal_moves = game.get_legal_moves()
    print(f"\nNumber of legal moves: {len(legal_moves)}")
    
    # Get state
    state = game.get_state()
    print(f"State shape: {state.shape}")
