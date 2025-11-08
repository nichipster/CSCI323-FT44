# Experiments Directory

This directory stores all experimental results, model checkpoints, and training logs.

## Structure

- `checkpoints/` - Saved model weights (.pth files)
- `logs/` - TensorBoard logs and training logs
- `results/` - Tournament results, statistics, and game records

## Checkpoints

Model checkpoints are saved as:
- `best_policy.pth` - Best model based on validation accuracy
- `policy_epoch_XX.pth` - Checkpoint after epoch XX
- `final_policy.pth` - Final model after training

## Logs

TensorBoard logs can be viewed with:
```bash
tensorboard --logdir experiments/logs/
```

## Results

Tournament results are saved as CSV files:
- `tournament_results.csv` - Win/loss matrix
- `game_logs.csv` - Individual game records
- `statistics.json` - Summary statistics
