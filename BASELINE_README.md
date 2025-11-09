# AlphaGo Baseline System

Complete implementation of the AlphaGo baseline for 9×9 Go, following the methodology from [Silver et al., 2016](https://www.nature.com/articles/nature16961).

## System Components

The baseline system consists of four main components:

### 1. **Policy Network** (`models/policy_net.py`)
- Convolutional neural network for move prediction
- Architecture: 5 convolutional layers + 1 fully connected layer
- Input: 9×9×4 board state tensor
- Output: 82-dimensional probability distribution (81 positions + pass)
- Training: Supervised learning on expert games

### 2. **MCTS** (`utils/mcts.py`)
- Monte Carlo Tree Search algorithm
- Guided by policy and value networks
- Combines tree search with neural network evaluations
- Uses neural rollouts for fast simulations

### 3. **Complete System** (`models/alphago_baseline.py`)
- Integrates all components
- Provides unified interface for game playing
- Implements the full AlphaGo pipeline

## Architecture Details

### Policy Network
```
Input: (batch, 4, 9, 9)
  ↓
Conv1: 4→48 filters (3×3) + BatchNorm + ReLU
  ↓
Conv2: 48→96 filters (3×3) + BatchNorm + ReLU
  ↓
Conv3: 96→96 filters (3×3) + BatchNorm + ReLU
  ↓
Conv4: 96→96 filters (3×3) + BatchNorm + ReLU
  ↓
Conv5: 96→1 filter (1×1) + ReLU
  ↓
Flatten: (batch, 81)
  ↓
FC: 81→82
  ↓
Softmax
  ↓
Output: (batch, 82) [move probabilities]
```

### Value Network
```
Input: (batch, 4, 9, 9)
  ↓
Conv1-5: Same as Policy Network
  ↓
Flatten: (batch, 81)
  ↓
FC1: 81→256 + ReLU
  ↓
FC2: 256→1
  ↓
Tanh
  ↓
Output: (batch, 1) [position value in [-1,1]]
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy tqdm

# Optional: For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Training the Baseline

#### Option 1: Full Training Pipeline (Recommended)

```bash
# Train both policy and value networks with synthetic data
python train.py --mode full --epochs 10 --num_samples 2000
```

#### Option 2: Train Policy Network Only

```bash
python train.py --mode policy --epochs 10 --num_samples 2000
```

#### Option 3: Train with Self-Play

```bash
# Generate self-play data and train value network
python train.py --mode value --selfplay_games 100 --mcts_sims 100
```

#### Training Options

```bash
python train.py --help

Options:
  --mode {policy,value,full}  Training mode
  --epochs EPOCHS             Number of training epochs (default: 10)
  --batch_size BATCH_SIZE     Batch size (default: 32)
  --learning_rate LR          Learning rate (default: 0.001)
  --num_samples N             Training samples for synthetic data (default: 1000)
  --selfplay_games N          Self-play games for value network (default: 50)
  --mcts_sims N               MCTS simulations per move (default: 100)
  --device {cpu,cuda,auto}    Device (default: auto)
  --checkpoint_dir DIR        Checkpoint directory (default: experiments/checkpoints)
```

### Using the Trained System

```python
from models.alphago_baseline import AlphaGoBaseline
from utils.go_game import GoGame

# Load trained system
alphago = AlphaGoBaseline.load('experiments/checkpoints', num_simulations=400)

# Create a game
game = GoGame()

# Get best move
move, move_probs = alphago.select_move(game, temperature=0)
print(f"Best move: {move}")

# Make move
game.make_move(move)
game.render()

# Evaluate position
value = alphago.evaluate_position(game)
print(f"Position value: {value:.3f}")
```

### Playing a Complete Game

```python
from models.alphago_baseline import AlphaGoBaseline

# Create two AlphaGo instances
alphago1 = AlphaGoBaseline(num_simulations=400)
alphago2 = AlphaGoBaseline(num_simulations=400)

# Play game
game_data, winner = alphago1.play_game(opponent=alphago2, show_board=True)

print(f"Game completed in {len(game_data)} moves")
print(f"Winner: {'Black' if winner > 0 else 'White' if winner < 0 else 'Draw'}")
```

## Demo and Testing

### Test Individual Components

```bash
# Test policy network
python models/policy_net.py

# Test value network
python models/value_net.py

# Test MCTS
python utils/mcts.py

# Test Go game environment
python utils/go_game.py
```

### Run Full Demo

```bash
# Demonstrate complete baseline system
python models/alphago_baseline.py
```

## Training Pipeline

The training follows the AlphaGo paper methodology:

### Stage 1: Supervised Learning (Policy Network)
1. **Data**: Expert game positions and moves
2. **Loss**: Cross-entropy loss
3. **Objective**: Maximize likelihood of expert moves
4. **Duration**: ~10 epochs on synthetic data

### Stage 2: Self-Play Data Generation
1. **Method**: Play games using policy network + MCTS
2. **Purpose**: Generate training data for value network
3. **Games**: 50-100 self-play games
4. **MCTS**: 100 simulations per move

### Stage 3: Reinforcement Learning (Value Network)
1. **Data**: Self-play positions with game outcomes
2. **Loss**: Mean squared error
3. **Objective**: Predict game winner
4. **Duration**: ~10 epochs

## File Structure

```
CSCI323 FT44/
├── models/
│   ├── policy_net.py          # Policy network
│   ├── value_net.py           # Value network
│   └── alphago_baseline.py    # Complete system
├── utils/
│   ├── go_game.py             # Go game environment
│   └── mcts.py                # MCTS implementation
├── training/
│   └── train_pipeline.py      # Training pipeline
├── experiments/
│   ├── checkpoints/           # Saved models
│   └── logs/                  # Training logs
└── train.py                   # Quick start script
```

## Key Features

### 1. Complete AlphaGo Pipeline
- ✅ Policy network for move prediction
- ✅ Value network for position evaluation
- ✅ MCTS with neural guidance
- ✅ Neural rollouts for fast simulation

### 2. Training Infrastructure
- ✅ Supervised learning pipeline
- ✅ Self-play data generation
- ✅ Reinforcement learning for value network
- ✅ Checkpoint saving and loading
- ✅ Training history tracking

### 3. Go Game Implementation
- ✅ Legal move validation
- ✅ Capture detection
- ✅ Ko rule enforcement
- ✅ Scoring (simplified Chinese rules)
- ✅ Board visualization

### 4. MCTS Features
- ✅ UCT selection
- ✅ Neural network expansion
- ✅ Value network evaluation
- ✅ Policy network rollouts
- ✅ Backpropagation

## Performance Characteristics

### Network Sizes
- **Policy Network**: ~58,000 parameters
- **Value Network**: ~79,000 parameters
- **Total**: ~137,000 parameters

### Computational Requirements
- **Training**: CPU or GPU (GPU recommended)
- **Inference**: CPU sufficient for 9×9 board
- **MCTS**: ~100-800 simulations per move
- **Time per move**: 1-10 seconds (depending on simulations)

## Differences from Original AlphaGo

This implementation is adapted for learning purposes:

| Feature | Original AlphaGo | This Baseline |
|---------|------------------|---------------|
| Board size | 19×19 | 9×9 |
| Policy network layers | 13 | 5 |
| Network filters | 192 | 48-96 |
| Input features | 48 | 4 |
| Training data | 30M positions | Synthetic/Self-play |
| MCTS simulations | 1,600 | 100-800 |
| Value network rollouts | Yes | Yes |

## Common Issues and Solutions

### Issue 1: Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 16

# Reduce MCTS simulations
python train.py --mcts_sims 50
```

### Issue 2: Slow Training
```bash
# Use GPU if available
python train.py --device cuda

# Reduce training samples
python train.py --num_samples 500
```

### Issue 3: Poor Performance
- Train for more epochs (--epochs 20)
- Increase training data (--num_samples 5000)
- Increase MCTS simulations (--mcts_sims 400)

## References

1. Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." *Nature*, 529(7587), 484-489.

2. Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." *Nature*, 550(7676), 354-359.

## License

This implementation is for educational purposes as part of CSCI323 course project.

## Contributors

[Your Team Members]

---

For questions or issues, please contact the course instructors or refer to the project documentation.
