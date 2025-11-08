# CSCI323 AlphaGo Project - 9×9 Go AI Implementation

**Team:** FT44  
**Project Type:** Comparative Evaluation (Breadth)  
**Board Size:** 9×9 Go  

## Project Overview

This project implements and evaluates multiple Go AI approaches on 9×9 boards, comparing traditional Monte Carlo Tree Search (MCTS) with neural network-enhanced methods inspired by DeepMind's AlphaGo.

## Models Implemented

1. **Baseline Model:** Policy Network + MCTS (Supervised Learning)
2. **Pure MCTS:** Traditional Go AI using pattern-based rollouts
3. **Policy Network Only:** Neural network without search
4. **Random MCTS:** MCTS with random rollout policy

## Repository Structure

```
CSCI323 FT44/
├── data/                   # Training and test data
│   ├── sgf_games/         # Raw SGF game files
│   ├── processed/         # Preprocessed training data
│   └── raw/              # Raw downloaded data
├── models/                # Neural network architectures
│   ├── __init__.py
│   ├── policy_net.py     # Policy network definition
│   ├── mcts.py           # MCTS implementation
│   └── baseline_mcts.py  # Pure MCTS baseline
├── training/              # Training scripts
│   ├── __init__.py
│   ├── train_policy.py   # Policy network training
│   ├── configs.py        # Hyperparameters
│   └── data_loader.py    # Dataset preparation
├── evaluation/            # Evaluation framework
│   ├── __init__.py
│   ├── tournament.py     # Model vs model games
│   ├── metrics.py        # Performance calculations
│   └── visualize.py      # Results visualization
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── go_board.py       # Go game logic wrapper
│   └── sgf_parser.py     # SGF file parsing
├── experiments/           # Experimental results
│   ├── logs/             # Training logs
│   ├── checkpoints/      # Model checkpoints
│   └── results/          # Game results and statistics
├── report/                # Project documentation
│   ├── figures/          # Plots and diagrams
│   └── drafts/           # Report drafts
├── presentation/          # Presentation materials
├── docs/                  # Additional documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, RTX 4060 Ti or better)
- 50GB free disk space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "CSCI323 FT44"

# Create virtual environment
conda create -n go_ai python=3.9
conda activate go_ai

# Install dependencies
pip install -r requirements.txt
```

### Download Training Data

```bash
# Download 9×9 KGS games (instructions in data/README.md)
python utils/download_data.py --board_size 9 --output data/sgf_games/
```

## Quick Start

### 1. Train Policy Network

```bash
python training/train_policy.py --epochs 20 --batch_size 64 --gpu 0
```

### 2. Run Evaluation Tournament

```bash
python evaluation/tournament.py --games 50 --models all
```

### 3. Generate Results

```bash
python evaluation/visualize.py --results experiments/results/tournament_results.csv
```

## Usage Examples

### Training the Policy Network

```python
from training.train_policy import train_policy_network
from training.configs import TRAINING_CONFIG

# Train with default configuration
model, history = train_policy_network(
    data_dir='data/processed/',
    config=TRAINING_CONFIG,
    device='cuda'
)
```

### Running a Single Game

```python
from models.policy_net import PolicyNetwork
from models.mcts import MCTS
from utils.go_board import GoBoard

# Load trained model
policy_net = PolicyNetwork.load('experiments/checkpoints/best_policy.pth')

# Initialize MCTS with policy network
mcts = MCTS(policy_network=policy_net, simulations=1000)

# Play a game
board = GoBoard(size=9)
move = mcts.select_move(board)
```

## Evaluation Metrics

- **Win Rate:** Percentage of games won between model pairs
- **ELO Rating:** Relative strength calculated from tournament results
- **Move Prediction Accuracy:** % of expert moves correctly predicted
- **Computational Efficiency:** Average time per move
- **Playing Style:** Qualitative analysis of strategic patterns

## Expected Results

| Model | Expected ELO | Move Accuracy | Strength |
|-------|-------------|---------------|----------|
| Baseline (Policy+MCTS) | ~2000 | 30-35% | Strong Amateur |
| Pure MCTS | ~1500 | 12% | Medium |
| Policy-Only | ~1200 | 30-35% | Medium |
| Random MCTS | ~800 | 1% | Weak |

## Project Timeline

- **Nov 7-9:** Setup, data preparation, baseline implementation
- **Nov 9:** Presentation submission
- **Nov 10-13:** Full evaluation, analysis
- **Nov 14-16:** Report writing and final submission

## Team Contributions

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Member 1 | Training Lead | GPU training, infrastructure setup |
| Member 2 | Literature Review | Background theory, related work |
| Member 3 | MCTS Implementation | Search algorithms, baselines |
| Member 4 | Evaluation | Tournament framework, metrics |
| Member 5 | Neural Networks | Architecture, training support |
| Member 6 | Report Lead | Documentation, integration |

## References

1. Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature, 529(7587), 484-489.
2. Browne, C., et al. (2012). "A survey of Monte Carlo tree search methods." IEEE Transactions on Computational Intelligence and AI in games, 4(1), 1-43.
3. Clark, C., & Storkey, A. (2015). "Training deep convolutional neural networks to play go." International Conference on Machine Learning, 1766-1774.

## License

This project is for educational purposes as part of CSCI323 coursework at University of Wollongong.

## Contact

For questions or issues, contact team members via course communication channels.

---

*Last Updated: November 7, 2025*
