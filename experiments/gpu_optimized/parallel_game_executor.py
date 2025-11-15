"""
Parallel Game Executor for GPU-Optimized Tournament
===================================================

Executes multiple games in parallel to maximize GPU batch utilization.

KEY INNOVATION: Instead of playing games sequentially, we play multiple games
simultaneously. When each game needs a neural network inference, we batch all
those requests together for maximum GPU efficiency.

Expected GPU utilization: 85-95% (vs 10-20% in sequential version)
Expected speedup: 8-12x faster overall tournament time
"""

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread, Event
import time
from queue import Queue, Empty
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.go_game import GoGame
from experiments.gpu_optimized.batch_inference_engine import create_inference_engine


class ParallelGameExecutor:
    """
    Executes multiple Go games in parallel with batched GPU inference.
    
    Architecture:
    1. Multiple game threads run simultaneously
    2. When a game needs inference, it submits a request to a shared queue
    3. A dedicated batch processing thread collects requests and processes them in batches
    4. Results are returned to waiting games
    
    This maximizes GPU utilization by ensuring the GPU always has work to do.
    """
    
    def __init__(self, inference_engine, max_parallel_games=32, device='cuda'):
        """
        Args:
            inference_engine: Batched inference engine for GPU acceleration
            max_parallel_games: Maximum number of games to run simultaneously
            device: torch device ('cuda' or 'cpu')
        """
        self.inference_engine = inference_engine
        self.max_parallel_games = max_parallel_games
        self.device = device
        
        # Inference request queue
        self.inference_queue = Queue(maxsize=1000)
        self.inference_results = {}
        self.result_lock = Lock()
        self.next_request_id = 0
        self.request_id_lock = Lock()
        
        # Batch processing thread
        self.batch_thread = None
        self.stop_event = Event()
        self.active = False
        
        # Statistics
        self.games_completed = 0
        self.total_moves = 0
        self.total_inference_time = 0.0
        self.stats_lock = Lock()
    
    def start(self):
        """Start the batch processing thread"""
        if self.active:
            return
        
        self.active = True
        self.stop_event.clear()
        self.batch_thread = Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        
        print(f"✓ Parallel executor started with {self.max_parallel_games} parallel games")
    
    def stop(self):
        """Stop the batch processing thread"""
        if not self.active:
            return
        
        self.stop_event.set()
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        self.active = False
    
    def _batch_processor(self):
        """
        Background thread that collects inference requests and processes them in batches.
        
        This is the KEY to GPU efficiency - it ensures the GPU is always processing
        large batches instead of individual requests.
        """
        print(f"  🔄 Batch processor thread started")
        batch_wait_time = 0.005  # 5ms - increased from 2ms for better collection
        
        while not self.stop_event.is_set():
            # Collect requests for a short time
            start_time = time.time()
            requests = []
            states = []
            
            # Quick collection phase
            while time.time() - start_time < batch_wait_time:
                try:
                    request = self.inference_queue.get(timeout=0.001)
                    requests.append(request)
                    states.append(request['state'])
                    
                    # Don't let batch get too large
                    if len(requests) >= self.inference_engine.batch_size:
                        break
                except Empty:
                    pass
            
            # Process batch if we have requests
            if requests:
                batch_start = time.time()
                
                try:
                    # Run inference on batch
                    results = self.inference_engine.predict_batch(states)
                    
                    batch_time = time.time() - batch_start
                    
                    # Store results
                    with self.result_lock:
                        for request, result in zip(requests, results):
                            request_id = request['id']
                            self.inference_results[request_id] = result
                            request['event'].set()
                    
                    # Update stats
                    with self.stats_lock:
                        self.total_inference_time += batch_time
                except Exception as e:
                    print(f"  ⚠️ Batch processing error: {e}")
                    # Set events anyway so games don't hang
                    with self.result_lock:
                        for request in requests:
                            request['event'].set()
            else:
                # Small sleep if no requests
                time.sleep(0.001)
    
    def request_inference(self, state):
        """
        Request inference for a game state.
        Returns action probabilities.
        
        This method is called by game threads when they need the policy network.
        """
        # Generate unique request ID
        with self.request_id_lock:
            request_id = self.next_request_id
            self.next_request_id += 1
        
        # Create request
        request_event = Event()
        request = {
            'id': request_id,
            'state': state,
            'event': request_event
        }
        
        # Submit request
        self.inference_queue.put(request)
        
        # Wait for result (with timeout)
        if not request_event.wait(timeout=30.0):  # Increased timeout for heavy MCTS load
            raise TimeoutError(f"Inference request {request_id} timed out")
        
        # Get result
        with self.result_lock:
            result = self.inference_results.pop(request_id)
        
        return result
    
    def execute_games_parallel(self, game_specs, max_moves=200):
        """
        Execute multiple games in parallel.
        
        Args:
            game_specs: List of game specifications, each containing:
                {
                    'black_model': model instance,
                    'white_model': model instance,
                    'game_id': unique identifier
                }
            max_moves: Maximum moves per game
            
        Returns:
            List of game results
        """
        if not self.active:
            self.start()
        
        results = []
        results_lock = Lock()
        
        def play_single_game(spec):
            """Play a single game (runs in thread pool)"""
            result = self._play_game_with_batched_inference(
                spec['black_model'],
                spec['white_model'],
                spec.get('game_id', 0),
                max_moves
            )
            
            with results_lock:
                results.append(result)
            
            with self.stats_lock:
                self.games_completed += 1
            
            return result
        
        # Execute games in thread pool
        with ThreadPoolExecutor(max_workers=self.max_parallel_games) as executor:
            futures = [executor.submit(play_single_game, spec) for spec in game_specs]
            
            # Wait for completion with progress
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                completed += 1
                try:
                    future.result()
                    if completed % max(1, total // 20) == 0:  # Update every 5%
                        print(f"  Progress: {completed}/{total} games ({completed/total*100:.1f}%)")
                except Exception as e:
                    print(f"  ⚠️ Game error: {e}")
        
        return results
    
    def _play_game_with_batched_inference(self, black_model, white_model, game_id, max_moves):
        """
        Play a single game using batched inference.
        
        When this game needs inference, it submits a request to the shared queue
        which will be batched with requests from other games.
        """
        game = GoGame()
        game_data = {
            'game_id': game_id,
            'black_model': black_model.name if hasattr(black_model, 'name') else 'Unknown',
            'white_model': white_model.name if hasattr(white_model, 'name') else 'Unknown',
            'moves': [],
            'move_times': {'black': [], 'white': []},
        }
        
        start_time = time.time()
        move_count = 0
        
        while move_count < max_moves:
            current_model = black_model if game.current_player == GoGame.BLACK else white_model
            player_name = 'black' if game.current_player == GoGame.BLACK else 'white'
            
            move_start = time.time()
            
            # Get move using batched inference
            move = self._get_model_move_batched(current_model, game)
            
            move_time = time.time() - move_start
            game_data['move_times'][player_name].append(move_time)
            
            # Make move
            done, reward = game.make_move(move)
            move_count += 1
            
            with self.stats_lock:
                self.total_moves += 1
            
            if done:
                break
        
        # Determine winner
        score = game._calculate_score()
        if score > 0:
            winner = game_data['black_model']
        elif score < 0:
            winner = game_data['white_model']
        else:
            winner = 'Draw'
        
        game_data.update({
            'winner': winner,
            'final_score': score,
            'num_moves': move_count,
            'game_duration': time.time() - start_time,
            'avg_move_time_black': np.mean(game_data['move_times']['black']) if game_data['move_times']['black'] else 0,
            'avg_move_time_white': np.mean(game_data['move_times']['white']) if game_data['move_times']['white'] else 0,
            'completed': move_count < max_moves
        })
        
        return game_data
    
    def _get_model_move_batched(self, model, game):
        """
        Get move from model using batched inference.
        
        For models with neural networks, this uses the shared batch inference.
        For models without neural networks (pure MCTS, random), calls directly.
        """
        # Check if model needs neural network inference
        if not hasattr(model, 'uses_neural_net') or not model.uses_neural_net:
            # Model doesn't use neural net - call directly
            return model.select_move(game)
        
        # Model needs neural network - use batched inference
        # We'll modify the model's inference calls to use our batch engine
        if hasattr(model, 'set_inference_engine'):
            model.set_inference_engine(self)
        
        return model.select_move(game)
    
    def get_stats(self):
        """Return executor statistics"""
        with self.stats_lock:
            stats = {
                'games_completed': self.games_completed,
                'total_moves': self.total_moves,
                'total_inference_time': self.total_inference_time,
                'avg_moves_per_game': self.total_moves / max(1, self.games_completed),
                'avg_inference_time_per_move': self.total_inference_time / max(1, self.total_moves)
            }
        
        # Add inference engine stats
        stats.update(self.inference_engine.get_stats())
        
        return stats


def estimate_optimal_parallel_games(device):
    """
    Estimate optimal number of parallel games based on GPU memory.
    
    Returns recommended number of parallel games to maximize throughput
    without running out of memory.
    """
    if device.type != 'cuda':
        return 8  # CPU: moderate parallelism
    
    # Get GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_memory / 1e9
    
    # Rough estimate: each game needs ~50MB for MCTS tree + states
    # Policy network needs ~200MB
    # Leave 1GB buffer for PyTorch overhead
    
    available_memory = total_gb - 1.2  # Leave buffer
    memory_per_game = 0.05  # 50MB per game
    
    max_games = int(available_memory / memory_per_game)
    
    # Cap at reasonable limits
    recommended = min(max_games, 16)  # Don't exceed 16 parallel games for MCTS stability
    recommended = max(recommended, 8)  # At least 8 games
    
    return recommended
