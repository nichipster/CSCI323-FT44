# AlphaGo Methodology for 9×9 Go

**CSCI323 Group Project - Team FT44**

A comparative evaluation of Monte Carlo Tree Search and neural network approaches for playing 9×9 Go, inspired by DeepMind's AlphaGo.

## Project Overview

This project implements and compares four AI models:

1. **Baseline**: Policy Network + MCTS with neural rollouts (AlphaGo-inspired)
2. **Pure MCTS**: Traditional MCTS with pattern-based heuristics  
3. **Policy-Only**: Trained policy network without search
4. **Random MCTS**: MCTS with random rollouts (control baseline)

## Repository Structure

```
CSCI323 FT44/
├── data/
│   ├── sgf_games/           # Expert Go games (470 games from various sources)
│   └── processed/           # Preprocessed training data
├── models/
│   ├── policy_net.py        # CNN policy network architecture
│   └── model_interface.py   # Four model implementations
├── training/
│   ├── sgf_data_loader.py   # SGF parsing and data preparation
│   └── train_pipeline.py    # Training loop
├── utils/
│   ├── go_game.py          # Go game logic and rules
│   └── mcts_neural_rollouts.py
├── experiments/
│   ├── checkpoints/         # Trained model weights
│   ├── gpu_optimized/       # GPU tournament implementation
│   └── tournament_results/  # Game results and statistics
├── train.py                 # Train policy network
└── run_tournament.py        # Run model comparisons
```

## Quick Start

### Setup

```bash
# Install dependencies
pip install torch numpy sgfmill matplotlib pandas seaborn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Training the Policy Network

The policy network has already been trained on 470 expert games and achieves ~24.4% validation accuracy.

```bash
# Train from scratch (optional)
python train.py
```

**Training configuration:**
- Dataset: 470 SGF files (9×9 Go)  
- Architecture: 7-layer CNN (64→128→256 filters)
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Training time: ~15 minutes on RTX 4060 Ti

### Running Tournaments

Compare all four models in a round-robin tournament:

```bash
python run_tournament.py
```

**Tournament settings:**
- Games per matchup: 20
- GPU-optimized: Batched neural network inference
- Parallel game execution

## Model Specifications

| Model | Neural Network | MCTS | Rollout Strategy | Simulations |
|-------|---------------|------|------------------|-------------|
| Baseline | ✓ (Policy) | ✓ | Neural network | 100 |
| Pure MCTS | ✗ | ✓ | Pattern heuristics | 100 |
| Policy-Only | ✓ (Policy) | ✗ | Direct policy | N/A |
| Random MCTS | ✗ | ✓ | Random | 100 |

## Key Results

Based on our tournament evaluation:

- **Policy-Only** achieved the highest win rate (85% vs Baseline)
- **Baseline** underperformed despite theoretical expectations
- Limited training data (470 games) and weak policy accuracy (24.4%) hindered MCTS performance
- Pattern-based MCTS outperformed random MCTS significantly

**Key Finding**: Weak policy networks can hurt MCTS performance rather than help it, demonstrating the importance of component quality in hybrid AI systems.

## Using the Models

### Play a Single Game

```python
from models.model_interface import BaselineModel, PolicyOnlyModel
from utils.go_game import GoGame

# Initialize models
model1 = BaselineModel(num_simulations=100)
model2 = PolicyOnlyModel()

# Create game
game = GoGame()

# Play until game over
while not game.is_game_over():
    if game.current_player == GoGame.BLACK:
        move = model1.select_move(game)
    else:
        move = model2.select_move(game)
    game.make_move(move)

# Get result
print(f"Final score: {game._calculate_score()}")
```

### GPU Optimization

The tournament system supports GPU batching for efficient neural network inference:

```python
from experiments.gpu_optimized.gpu_tournament import GPUTournament

# Run GPU-optimized tournament
tournament = GPUTournament(
    device='cuda',
    games_per_matchup=20,
    batch_size=32
)
results = tournament.run()
```

### MCTS Implementation
- **UCT formula**: exploit + c_param × √(ln(parent.visits) / child.visits)
- **Exploration constant**: c_param = 1.4
- **Simulation limit**: 100 iterations per move
- **Rollout depth**: Until game termination

## Files

- **`train.py`**: Train policy network from SGF data
- **`run_tournament.py`**: Execute round-robin tournament
- **`play_game.py`**: Play interactive games
- **`visualize_training.py`**: Plot training curves

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
sgfmill>=1.1.1
matplotlib>=3.7.0
pandas>=2.0.0
seaborn>=0.12.0
```
