"""
Visualize training history for policy network
Creates publication-quality plots for presentation and report
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_training_history(filepath):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_results(history, save_path='training_analysis.png'):
    """Create comprehensive visualization of training results"""
    
    # Extract data
    epochs = [entry['epoch'] for entry in history['policy_loss']]
    train_loss = [entry['train'] for entry in history['policy_loss']]
    val_loss = [entry['val'] for entry in history['policy_loss']]
    train_acc = [entry['train'] for entry in history['policy_accuracy']]
    val_acc = [entry['val'] for entry in history['policy_accuracy']]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Policy Network Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add loss improvement annotations
    loss_improvement = train_loss[0] - train_loss[-1]
    ax1.text(0.5, 0.95, f'Loss Reduction: {loss_improvement:.3f}', 
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Policy Network Move Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add accuracy annotations
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    ax2.text(0.5, 0.95, f'Final Train: {final_train_acc:.1f}% | Val: {final_val_acc:.1f}%', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax2.text(0.5, 0.85, f'Overfitting Gap: {overfitting_gap:.1f}%', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training visualization to {save_path}")
    plt.show()

def print_training_summary(history):
    """Print comprehensive training summary"""
    
    # Extract final metrics
    final_epoch = history['policy_loss'][-1]
    initial_epoch = history['policy_loss'][0]
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY - 50 EPOCHS")
    print("="*60)
    
    print("\n📊 LOSS METRICS:")
    print(f"  Initial Training Loss:    {initial_epoch['train']:.4f}")
    print(f"  Final Training Loss:      {final_epoch['train']:.4f}")
    print(f"  Loss Reduction:           {initial_epoch['train'] - final_epoch['train']:.4f}")
    print(f"  ")
    print(f"  Initial Validation Loss:  {initial_epoch['val']:.4f}")
    print(f"  Final Validation Loss:    {final_epoch['val']:.4f}")
    print(f"  Val Loss Reduction:       {initial_epoch['val'] - final_epoch['val']:.4f}")
    
    # Extract accuracy metrics
    final_acc = history['policy_accuracy'][-1]
    initial_acc = history['policy_accuracy'][0]
    best_val_acc_entry = max(history['policy_accuracy'], key=lambda x: x['val'])
    
    print("\n🎯 ACCURACY METRICS:")
    print(f"  Initial Training Accuracy:   {initial_acc['train']:.2f}%")
    print(f"  Final Training Accuracy:     {final_acc['train']:.2f}%")
    print(f"  Accuracy Improvement:        +{final_acc['train'] - initial_acc['train']:.2f}%")
    print(f"  ")
    print(f"  Initial Validation Accuracy: {initial_acc['val']:.2f}%")
    print(f"  Final Validation Accuracy:   {final_acc['val']:.2f}%")
    print(f"  Best Validation Accuracy:    {best_val_acc_entry['val']:.2f}% (Epoch {best_val_acc_entry['epoch']})")
    
    overfitting_gap = final_acc['train'] - final_acc['val']
    print(f"\n⚠️  OVERFITTING GAP:")
    print(f"  Train-Val Accuracy Gap:      {overfitting_gap:.2f}%")
    
    if overfitting_gap > 10:
        print(f"  ⚠️  SEVERE OVERFITTING DETECTED!")
        print(f"     Model is memorizing training data")
    elif overfitting_gap > 5:
        print(f"  ⚠️  Moderate overfitting")
    else:
        print(f"  ✅ Good generalization")
    
    # Check for plateau
    last_10_val_acc = [entry['val'] for entry in history['policy_accuracy'][-10:]]
    val_acc_variance = np.std(last_10_val_acc)
    
    print(f"\n📈 CONVERGENCE STATUS:")
    print(f"  Last 10 epochs val accuracy std: {val_acc_variance:.2f}%")
    
    if val_acc_variance < 0.5:
        print(f"  ⚠️  VALIDATION PLATEAU - Model stopped improving")
    else:
        print(f"  ✅ Still improving")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if overfitting_gap > 10:
        print("1. ⚠️  Implement regularization (dropout, L2)")
        print("2. 🔄 Add data augmentation (rotations, flips)")
        print("3. 📉 Reduce model complexity or increase dataset size")
    
    if val_acc_variance < 0.5:
        print("4. 🛑 Training converged - no need for more epochs")
        print("5. 🎯 Focus on model evaluation and comparison")
    
    if final_acc['val'] < 25:
        print("6. 💡 Consider ensemble methods or MCTS integration")
    
    print("="*60 + "\n")

def compare_to_alphago_paper():
    """Compare results to AlphaGo paper benchmarks"""
    print("\n📄 COMPARISON TO ALPHAGO PAPER:")
    print("-" * 60)
    print("AlphaGo Paper (19×19 board):")
    print("  - SL Policy Network: 57% move prediction accuracy")
    print("  - Used 30M positions from KGS")
    print("  - 13-layer CNN with 192 filters")
    print()
    print("Your Implementation (9×9 board):")
    print("  - Validation accuracy: 20.06%")
    print("  - Smaller dataset (SGF files)")
    print("  - Simpler architecture")
    print()
    print("Note: Direct comparison is difficult because:")
    print("  ✓ Different board sizes (9×9 vs 19×19)")
    print("  ✓ Different dataset sizes")
    print("  ✓ Different network architectures")
    print("  ✓ 20% accuracy on 9×9 is reasonable for baseline")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    # Load training history
    history_path = "experiments/checkpoints/training_history.json"
    
    print("📊 Loading training history...")
    history = load_training_history(history_path)
    
    # Print summary
    print_training_summary(history)
    
    # Compare to paper
    compare_to_alphago_paper()
    
    # Create visualization
    print("📈 Creating visualizations...")
    plot_training_results(history, save_path='experiments/results/training_analysis.png')
    
    print("\n✅ Analysis complete!")
    print("Use 'training_analysis.png' in your presentation!")
