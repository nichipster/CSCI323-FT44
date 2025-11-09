"""
Quick GPU Check
===============
Run this before your tournament to verify GPU is working.
"""

import torch

print("="*70)
print("GPU STATUS CHECK")
print("="*70)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {cuda_available}")

if cuda_available:
    print(f"\n🎮 GPU DETECTED:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
    
    # Quick speed test
    print(f"\n⚡ Speed Test:")
    from models.policy_net import PolicyNetwork
    import time
    
    model = PolicyNetwork().to('cuda')
    dummy_input = torch.randn(10, 4, 9, 9).to('cuda')
    
    # Warm-up
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    gpu_time = (time.time() - start) / 100 * 1000  # ms
    print(f"   GPU inference: {gpu_time:.2f} ms per batch")
    print(f"   GPU inference: {gpu_time/10:.2f} ms per position")
    
    # Estimate tournament time
    moves_per_game = 40
    num_simulations = 100
    games = 300
    
    time_per_move = (gpu_time / 10) * num_simulations / 1000  # seconds
    time_per_game = time_per_move * moves_per_game / 60  # minutes
    total_time = time_per_game * games / 60  # hours
    
    print(f"\n📊 Tournament Estimate (with GPU):")
    print(f"   Time per move: ~{time_per_move:.1f} seconds")
    print(f"   Time per game: ~{time_per_game:.1f} minutes")
    print(f"   Full tournament (300 games): ~{total_time:.1f} hours")
    
    print(f"\n✅ GPU IS READY TO USE!")
    print(f"   Run: python run_tournament_gpu.py")
    
else:
    print(f"\n❌ GPU NOT AVAILABLE")
    print(f"\nTo fix:")
    print(f"1. Uninstall current PyTorch:")
    print(f"   pip uninstall torch torchvision torchaudio")
    print(f"\n2. Reinstall with CUDA support:")
    print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print(f"\n3. Run this script again to verify")

print("\n" + "="*70)