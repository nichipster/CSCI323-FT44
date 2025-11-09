"""
Training Script for AlphaGo Baseline

Policy Network + MCTS + Neural Rollouts
"""

import torch
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_pipeline import TrainingPipeline, generate_synthetic_data
from models.alphago_baseline import AlphaGoBaseline


def main():
    parser = argparse.ArgumentParser(description='Train AlphaGo Baseline')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of training samples (for synthetic data)')
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
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Training samples: {args.num_samples}")
    print(f"  MCTS simulations: {args.mcts_sims}")
    
    # Create training pipeline
    pipeline = TrainingPipeline(device=device, checkpoint_dir=args.checkpoint_dir)
    
    print("\n" + "="*70)
    print("Training Policy Network (Supervised Learning)")
    print("="*70)
    
    # Generate training data
    print(f"\nGenerating {args.num_samples} synthetic training samples...")
    train_states, train_moves = generate_synthetic_data(num_samples=args.num_samples)
    val_states, val_moves = generate_synthetic_data(num_samples=args.num_samples // 5)
    
    print(f"  Training samples: {len(train_states)}")
    print(f"  Validation samples: {len(val_states)}")
    
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
    print("  ✓ Policy Network (trained)")
    print("  ✓ MCTS (configured)")
    print("  ✓ Neural Rollouts (using policy network)")
    print("\nYou can now use the trained system:")
    print(f"  from models.alphago_baseline import AlphaGoBaseline")
    print(f"  alphago = AlphaGoBaseline.load('{args.checkpoint_dir}')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
