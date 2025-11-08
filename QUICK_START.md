# QUICK START GUIDE

## Repository Setup Complete! ✅

Your CSCI323 AlphaGo project repository has been successfully created.

---

## 📁 Repository Structure

```
CSCI323 FT44/
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup_check.py              # Environment verification script
├── 📄 .gitignore                  # Git ignore rules
│
├── 📁 data/                       # Training data
│   ├── sgf_games/                # Raw SGF files (empty - needs download)
│   ├── processed/                # Preprocessed tensors
│   └── raw/                      # Temporary downloads
│
├── 📁 models/                     # Neural network code
│   ├── policy_net.py            # ✅ Policy network (READY)
│   ├── mcts.py                  # TODO: MCTS implementation
│   └── baseline_mcts.py         # TODO: Pure MCTS
│
├── 📁 training/                   # Training scripts
│   ├── configs.py               # ✅ Hyperparameters (READY)
│   ├── train_policy.py          # TODO: Training loop
│   └── data_loader.py           # TODO: Data preprocessing
│
├── 📁 evaluation/                 # Evaluation framework
│   ├── tournament.py            # TODO: Round-robin tournament
│   ├── metrics.py               # TODO: Performance metrics
│   └── visualize.py             # TODO: Results plotting
│
├── 📁 utils/                      # Helper functions
│   ├── go_board.py              # TODO: Go game logic
│   └── sgf_parser.py            # TODO: SGF parsing
│
├── 📁 experiments/                # Results storage
│   ├── checkpoints/             # Model weights
│   ├── logs/                    # Training logs
│   └── results/                 # Tournament results
│
├── 📁 report/                     # Documentation
│   ├── figures/                 # Plots and diagrams
│   └── drafts/                  # Report versions
│
├── 📁 presentation/               # Slide deck materials
│
└── 📁 docs/                       # Additional docs
    ├── DAILY_TASKS.md          # Team task tracker
    └── RESOURCES.md            # Useful links

```

---

## 🚀 IMMEDIATE NEXT STEPS (TONIGHT)

### 1. Initialize Git Repository (5 minutes)

```bash
cd "G:\School\CSCI323\Group Project\CSCI323 FT44"
git init
git add .
git commit -m "Initial project structure"
```

**Optional:** Create GitHub repository and push:
```bash
git remote add origin <your-github-url>
git push -u origin main
```

### 2. Set Up Python Environment (15 minutes)

```bash
# Create virtual environment
conda create -n go_ai python=3.9
conda activate go_ai

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup_check.py
```

**Expected output:**
```
✅ Python Version
✅ Required Packages  
⚠️  CUDA/GPU (optional)
✅ Directory Structure

🎉 All checks passed!
```

### 3. Assign Team Roles (10 minutes)

Update `docs/DAILY_TASKS.md` with actual member names:
- Member 1 (YOU - GPU owner): Training Lead
- Member 2: Literature Review Lead
- Member 3: MCTS Implementation
- Member 4: Evaluation Framework
- Member 5: Neural Networks Support
- Member 6: Report Writing Lead

### 4. Start First Tasks (30 minutes)

**Member 1 (YOU):**
```bash
# Test GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Download sample SGF files
# Visit: https://u-go.net/gamerecords/
# Download ~100 9x9 games to test with
```

**Member 2:**
- Read AlphaGo paper pages 1-2
- Start Background Theory outline

**Member 3:**
- Research basic MCTS algorithm
- Find GNU Go installation guide

**Member 4:**
- Design evaluation metrics spreadsheet
- List statistics to track

**Member 5:**
- Review PyTorch CNN tutorial
- Study `models/policy_net.py`

**Member 6:**
- Create Google Doc for report
- Draft Introduction paragraph

---

## 📋 TONIGHT'S MEETING CHECKLIST

- [ ] Everyone has repository access
- [ ] Everyone ran `setup_check.py` successfully
- [ ] Roles confirmed and documented
- [ ] First tasks assigned with deadlines
- [ ] Communication channel established (WhatsApp/Discord/Slack)
- [ ] Next meeting scheduled

---

## 🎯 KEY MILESTONES

| Date | Milestone | Owner |
|------|-----------|-------|
| **Nov 8** | Policy network code complete | Member 5 |
| **Nov 8** | Dataset downloaded & preprocessed | Member 1 |
| **Nov 8** | MCTS implementation started | Member 3 |
| **Nov 9 11:59PM** | **PRESENTATION DUE** | Member 6 |
| **Nov 10** | Training complete, model integrated | Member 1 |
| **Nov 11** | Evaluation tournament finished | Member 4 |
| **Nov 13** | First report draft complete | Member 6 |
| **Nov 16 11:59PM** | **FINAL REPORT DUE** | All |

---

## ⚡ QUICK COMMANDS

```bash
# Check Python environment
python setup_check.py

# Test policy network
python models/policy_net.py

# View training config
python -c "from training.configs import *; print(TRAINING_CONFIG)"

# Start TensorBoard
tensorboard --logdir experiments/logs/

# Run tests (once implemented)
pytest tests/

# Format code
black models/ training/ evaluation/ utils/
```

---

## 📚 ESSENTIAL READING (BEFORE TOMORROW)

1. **AlphaGo Paper** - Located in project files:
   - Pages 1-2: Introduction and Method Overview
   - Focus on Figure 1: Training Pipeline

2. **Project Plan** - In main README.md:
   - Timeline (Day 1-10)
   - Role descriptions
   - Expected results

3. **This Guide** - You're reading it! 

---

## 🆘 TROUBLESHOOTING

### "setup_check.py fails"
- Make sure you're in the project directory
- Activate conda environment: `conda activate go_ai`
- Reinstall: `pip install -r requirements.txt`

### "CUDA not available"
- Don't worry! CPU training works (just slower)
- Member 1's GPU will handle main training

### "Can't find dataset"
- Download instructions in `data/README.md`
- Will be addressed tomorrow (Day 2)

### "Git issues"
- Optional for tonight
- Can push to GitHub tomorrow

---

## ✅ SUCCESS CRITERIA FOR TONIGHT

By end of tonight, you should have:

1. ✅ Repository set up and accessible by all
2. ✅ Python environment working on your machine
3. ✅ Team roles assigned and documented
4. ✅ First individual tasks started
5. ✅ Communication channel active
6. ✅ Tomorrow's priorities clear

---

## 🎉 YOU'RE READY!

The repository is fully set up with:
- ✅ Complete directory structure
- ✅ Starter code (Policy Network)
- ✅ Configuration files
- ✅ Documentation templates
- ✅ Git setup
- ✅ Development tools

**Now it's time to build something amazing! Good luck team! 🚀**

---

## 📞 NEED HELP?

- Check `docs/RESOURCES.md` for useful links
- Review `docs/DAILY_TASKS.md` for task tracker
- Ask Claude (me!) for technical guidance
- Consult course materials on Moodle

**Remember:** You have 9 days and a solid plan. Stay focused, communicate well, and you'll succeed!

---

*Created: Nov 7, 2025, 8:30 PM*  
*Team: CSCI323 FT44*  
*Project: AlphaGo Implementation (9×9 Go)*
