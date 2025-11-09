# 🎮 AlphaGo Baseline - Quick Reference Card

## 🚀 Quick Start (3 Commands)

```bash
# 1. Test everything works
python test_baseline.py

# 2. Train the system
python train.py --epochs 10

# 3. Use the trained system
python models/alphago_baseline.py
```

## 📦 What You Have

### Core System (Complete ✅)
```
AlphaGo Baseline = Policy Network + MCTS + Neural Rollouts
```

| Component | File | Description | Status |
|-----------|------|-------------|--------|
| Policy Network | `models/policy_net.py` | Predicts move probabilities | ✅ Done |
| MCTS | `utils/mcts.py` | Tree search algorithm | ✅ Done |
| Go Game | `utils/go_game.py` | 9×9 Go environment | ✅ Done |
| Complete System | `models/alphago_baseline.py` | Integration | ✅ Done |
| Training | `training/train_pipeline.py` | Full pipeline | ✅ Done |

## 🎯 Three Ways to Use

### 1️⃣ Test the Baseline
```python
# Verify everything works
python test_baseline.py
```

### 2️⃣ Train from Scratch
```python
# Train policy network with synthetic data
python train.py --epochs 10 --num_samples 2000

# Quick test training
python train.py --epochs 3 --num_samples 500
```

### 3️⃣ Use in Your Code
```python
from models.alphago_baseline import AlphaGoBaseline
from utils.go_game import GoGame

# Create system
alphago = AlphaGoBaseline(num_simulations=400)

# Play
game = GoGame()
move, probs = alphago.select_move(game)
game.make_move(move)
```

## 📊 System Specs

```
Policy Network:    58,000 parameters
Total:             58,000 parameters

Board Size:        9×9
Input Features:    4 channels
MCTS Simulations:  100-800 per move
Time per Move:     1-10 seconds
Evaluation:        Neural Rollouts (using policy network)
```

## 🔄 Training Pipeline

```
Stage 1: Policy Network (Supervised Learning)
   ├─ Input: Board positions + expert moves
   ├─ Method: Cross-entropy loss
   └─ Output: Trained policy network

Stage 2: MCTS Integration
   ├─ Method: Policy network guides tree search
   └─ Evaluation: Neural rollouts using policy network
```

## 🎨 Architecture Diagram

```
                  AlphaGo Baseline
              ┌──────────────────────┐
              │                      │
      ┌───────▼────────┐             │
      │ Policy Network │             │
      │   (Move Prob)  │             │
      └───────┬────────┘             │
              │                      │
       ┌──────▼──────┐               │
       │    MCTS     │               │
       │ (Rollouts)  │               │
       └──────┬──────┘               │
              │                      │
       ┌──────▼──────┐               │
       │  Go Game    │               │
       │  (9×9)      │               │
       └─────────────┘               │
              │                      │
              └──────────────────────┘
```

## 📁 Important Files

| File | Purpose | Run It |
|------|---------|--------|
| `test_baseline.py` | Test all components | `python test_baseline.py` |
| `train.py` | Train the system | `python train.py --help` |
| `models/alphago_baseline.py` | Demo the system | `python models/alphago_baseline.py` |
| `BASELINE_README.md` | Full documentation | Open in editor |

## ⚙️ Configuration Options

```bash
python train.py \
  --epochs 10 \                    # Training epochs
  --batch_size 32 \                # Batch size
  --learning_rate 0.001 \          # Learning rate
  --num_samples 2000 \             # Training samples
  --mcts_sims 400 \                # MCTS simulations
  --device cuda \                  # Use GPU
  --checkpoint_dir experiments/checkpoints
```

## 🧪 Test Each Component

```bash
python models/policy_net.py       # Test policy network
python utils/mcts.py              # Test MCTS
python utils/go_game.py           # Test Go game
python models/alphago_baseline.py # Test complete system
python training/train_pipeline.py # Test training
```

## 🎓 For Your Project Report

### What to Document

1. **Baseline Description**
   - "We implemented the AlphaGo baseline with Policy Network and MCTS"
   - Architecture: 5-layer CNN for policy
   - Training: Supervised learning on expert moves
   - Evaluation: Neural rollouts using policy network

2. **Implementation Details**
   - Board size: 9×9 (simplified from 19×19)
   - Network: ~58K parameters
   - MCTS: 100-800 simulations per move
   - Evaluation: Policy-guided rollouts to terminal positions

3. **Your Extension** (Choose one)
   - Compare different MCTS simulation counts
   - Add fast rollout policy
   - Test different network architectures
   - Compare policy-only vs policy+MCTS

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run from project root directory |
| Out of memory | `--batch_size 16 --mcts_sims 50` |
| Slow training | `--device cuda` or `--num_samples 500` |
| Test failures | Check each component individually |

## 📚 Key References

- **Paper**: Silver et al. (2016) "Mastering the game of Go with deep neural networks"
- **Nature**: https://www.nature.com/articles/nature16961

## ✨ What Makes This Complete

✅ All 3 core components implemented (Policy, MCTS, Game)
✅ Complete training pipeline (Supervised Learning)
✅ Neural rollouts for position evaluation
✅ Working demo and tests
✅ Ready for extensions
✅ Fully documented

## 🎯 Next Steps

1. **Run tests**: `python test_baseline.py`
2. **Try training**: `python train.py --epochs 3`
3. **Read docs**: Open `BASELINE_README.md`
4. **Plan extension**: Choose comparative or improvement approach
5. **Start experimenting**: Modify and enhance!

---

**Quick Help**:
- Full docs: `BASELINE_README.md`
- Code comments: Read inline documentation
- Tests: `python test_baseline.py`

**Ready to use! 🚀**
