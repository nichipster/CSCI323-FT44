"""
Configuration file for training hyperparameters
"""

# Training Configuration
TRAINING_CONFIG = {
    # Model architecture
    'input_channels': 4,
    'num_filters': 48,
    
    # Training parameters
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    
    # Learning rate schedule
    'lr_schedule': 'step',  # 'step', 'exponential', or 'cosine'
    'lr_step_size': 5,      # Decay every N epochs
    'lr_gamma': 0.5,        # Multiply LR by this factor
    
    # Data
    'train_split': 0.9,
    'validation_split': 0.1,
    'num_workers': 4,
    
    # Early stopping
    'early_stopping': True,
    'patience': 5,
    'min_delta': 0.001,
    
    # Checkpointing
    'save_best_only': True,
    'checkpoint_dir': 'experiments/checkpoints/',
    
    # Logging
    'log_interval': 100,  # Log every N batches
    'tensorboard_dir': 'experiments/logs/',
}

# MCTS Configuration
MCTS_CONFIG = {
    'simulations': 1000,
    'exploration_constant': 1.4,  # UCB constant
    'cpuct': 5.0,                # AlphaGo-style exploration
    'temperature': 1.0,           # Softmax temperature
    'rollout_depth': 81,         # Maximum rollout depth (full 9x9 board)
    'time_limit': 10.0,          # seconds per move (optional)
    'virtual_loss': 3,           # For parallel MCTS
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'games_per_matchup': 50,
    'komi': 7.5,                 # Standard Go komi
    'scoring': 'area',           # 'area' or 'territory'
    'time_control': None,        # No time limit
    'resignation_threshold': -20.0,  # Points behind to resign
    'alternating_colors': True,  # Ensure fair comparison
    'record_games': True,
    'save_sgf': True,
}

# Data paths
DATA_CONFIG = {
    'sgf_dir': 'data/sgf_games/',
    'processed_dir': 'data/processed/',
    'train_file': 'data/processed/train_data.pt',
    'test_file': 'data/processed/test_data.pt',
}

# Device configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'device': 'cuda' if True else 'cpu',  # Will be set automatically
    'num_gpus': 1,
}
