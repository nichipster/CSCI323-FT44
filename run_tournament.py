"""
GPU-ACCELERATED TOURNAMENT RUNNER
==================================
This version automatically uses GPU for all models that support it.
Expected speedup: 10-20x faster than CPU!
"""

import sys, os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.tournament_simulation import TournamentSimulator
from models.model_interface import BaselineModel, PureMCTSModel, PolicyOnlyModel, RandomMCTSModel

# 🚀 GPU ACCELERATION - This makes everything 10-20x faster!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*70)
print("STARTING GPU-ACCELERATED TOURNAMENT")
print("="*70)
print(f"\n🖥️  Device: {device}")

if device.type == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ GPU acceleration ENABLED!")
    print("   Expected time per game: 2-4 minutes")
    print("   Expected total time: 6-12 hours for 300 games")
else:
    print("\n⚠️  GPU not available - running on CPU")
    print("   Expected time per game: 20-60 minutes")
    print("   Expected total time: 100-300 hours for 300 games")
    print("\n   To enable GPU:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*70)

# Create models with GPU acceleration
print("\n📦 Initializing models...")
models = [
    BaselineModel(num_simulations=100, device=device),       # ✅ Uses GPU
    PureMCTSModel(num_simulations=100),                      # No neural net
    PolicyOnlyModel(device=device),                          # ✅ Uses GPU  
    RandomMCTSModel(num_simulations=100)                     # No neural net
]

print(f"✅ 4 models initialized")
print(f"   - 2 models using GPU ({device})")
print(f"   - 2 models using CPU only (no neural networks)")

# Load trained policy network if available
checkpoint_path = "experiments/checkpoints/policy_net.pth"
if os.path.exists(checkpoint_path):
    print(f"\n📥 Loading trained policy network from {checkpoint_path}...")
    from models.policy_net import PolicyNetwork
    try:
        trained_policy = PolicyNetwork.load(checkpoint_path, device=str(device))
        
        # Update models with trained network
        for model in models:
            if hasattr(model, 'policy_net'):
                model.policy_net = trained_policy
                print(f"   ✅ Updated {model.name} with trained network")
    except Exception as e:
        print(f"   ⚠️  Could not load trained model: {e}")
        print(f"   Models will use random initialization")

print("\n" + "="*70)
print("STARTING TOURNAMENT")
print("="*70)

tournament = TournamentSimulator(
    models=models,
    games_per_matchup=50,
    save_dir="experiments/tournament_results"
)

# Run tournament
try:
    results = tournament.run_tournament()
    tournament.print_final_summary()
    
    print("\n" + "="*70)
    print("✅ TOURNAMENT COMPLETE!")
    print("="*70)
    print("📊 Results saved to: experiments/tournament_results/")
    
    # Show GPU stats if used
    if device.type == 'cuda':
        print(f"\n📈 GPU Stats:")
        print(f"   Peak memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"   Current memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Tournament interrupted by user")
    print("Partial results may be available in experiments/tournament_results/")