"""
Enhanced Training Script for AlphaGo Baseline
Supports both synthetic and real game data (SGF files)
"""

import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_pipeline import TrainingPipeline, generate_synthetic_data
from training.sgf_data_loader import SGFDataLoader
from models.alphago_baseline import AlphaGoBaseline


def main():
    parser = argparse.ArgumentParser(description='Train AlphaGo Baseline with Real Game Data')
    
    # Data source
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real SGF game data instead of synthetic')
    parser.add_argument('--sgf_dir', type=str, default='data/sgf_games',
                       help='Directory containing SGF files')
    parser.add_argument('--max_games', type=int, default=None,
                       help='Maximum number of games to load (None for all)')
    parser.add_argument('--use_processed', action='store_true',
                       help='Load previously processed data')
    parser.add_argument('--processed_data', type=str, default='data/processed/train_data.pt',
                       help='Path to processed data file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Synthetic data parameters
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of training samples (for synthetic data)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, or auto')
    parser.add_argument('--checkpoint_dir', type=str, default='experiments/checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--mcts_sims', type=int, default=400,
                       help='MCTS simulations for final system')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("\n" + "="*70)
    print("AlphaGo Baseline Training")
    print("Policy Network + MCTS + Neural Rollouts")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Data source: {'Real SGF games' if args.use_real_data else 'Synthetic'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  MCTS simulations: {args.mcts_sims}")
    
    # Load training data
    if args.use_real_data:
        if args.use_processed:
            print(f"\nLoading processed data from {args.processed_data}...")
            try:
                loader = SGFDataLoader()
                all_states, all_moves = loader.load_processed_data(args.processed_data)
            except FileNotFoundError:
                print(f"\n⚠ Processed data not found!")
                print("Run preprocessing first or use --use_real_data without --use_processed")
                return
        else:
            print(f"\nLoading SGF games from {args.sgf_dir}...")
            loader = SGFDataLoader(sgf_dir=args.sgf_dir)
            all_states, all_moves = loader.load_all_games(max_games=args.max_games)
            
            if len(all_states) == 0:
                print("\n⚠ No game data loaded!")
                print("\n📥 Please download 9×9 Go games:")
                print("   1. Visit: https://u-go.net/gamerecords/")
                print("   2. Filter for 9×9 board size")
                print("   3. Save SGF files to:", args.sgf_dir)
                return
            
            loader.save_processed_data(all_states, all_moves)
        
        train_states, train_moves, val_states, val_moves = \
            loader.create_train_val_split(all_states, all_moves, 
                                        val_split=args.val_split, 
                                        shuffle=True)
    else:
        print(f"\nGenerating {args.num_samples} synthetic training samples...")
        train_states, train_moves = generate_synthetic_data(num_samples=args.num_samples)
        val_states, val_moves = generate_synthetic_data(num_samples=args.num_samples // 5)
        
        print(f"  Training samples: {len(train_states)}")
        print(f"  Validation samples: {len(val_states)}")
    
    # Create training pipeline
    pipeline = TrainingPipeline(device=device, checkpoint_dir=args.checkpoint_dir)
    
    print("\n" + "="*70)
    print("Training Policy Network (Supervised Learning)")
    print("="*70)
    
    # Train policy network
    policy_history = pipeline.train_policy_supervised(
        train_data=(train_states, train_moves),
        val_data=(val_states, val_moves),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n✓ Policy network training completed!")
    
    # Save training history
    pipeline.save_history()
    
    # Create complete AlphaGo baseline
    print("\n" + "="*70)
    print("Creating Complete AlphaGo Baseline System")
    print("="*70)
    
    alphago = AlphaGoBaseline(
        policy_net=pipeline.policy_net,
        num_simulations=args.mcts_sims,
        device=device
    )
    
    # Show system info
    info = alphago.get_info()
    print("\nFinal System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Save complete system
    alphago.save(args.checkpoint_dir)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print("\nYour baseline system includes:")
    print("  ✓ Policy Network (trained on" + 
          (" real game data)" if args.use_real_data else " synthetic data)"))
    print("  ✓ MCTS (configured)")
    print("  ✓ Neural Rollouts (using policy network)")
    print("\nYou can now use the trained system:")
    print(f"  from models.alphago_baseline import AlphaGoBaseline")
    print(f"  alphago = AlphaGoBaseline.load('{args.checkpoint_dir}')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
