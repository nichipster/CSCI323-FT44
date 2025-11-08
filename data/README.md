# Data Directory

This directory contains all training and test data for the project.

## Structure

- `sgf_games/` - Raw SGF files downloaded from online Go servers
- `processed/` - Preprocessed training data in PyTorch tensor format
- `raw/` - Temporary storage for downloads

## Dataset Sources

### 9×9 KGS Games
- **Source:** https://u-go.net/gamerecords/
- **Alternative:** http://www.badukmovies.com/
- **Target:** 20,000-50,000 games
- **Filter:** Dan-level players (6d-9d)

### Download Instructions

1. Visit the KGS archives
2. Filter for 9×9 games
3. Download games from recent years (2015-2023)
4. Place SGF files in `sgf_games/` directory

Or use the automated script:
```bash
python utils/download_data.py --board_size 9 --output data/sgf_games/
```

## Data Statistics

After processing, you should have:
- Training set: ~45,000 positions
- Test set: ~5,000 positions
- Input features: 9×9×4 (board state + history)
- Output labels: 82 classes (81 moves + pass)

## Preprocessing

Convert SGF files to training format:
```bash
python training/data_loader.py --input data/sgf_games/ --output data/processed/
```

This will create:
- `train_data.pt` - Training dataset
- `test_data.pt` - Test dataset
- `data_stats.json` - Dataset statistics
