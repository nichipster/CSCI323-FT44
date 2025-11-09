"""
Verification Script for Real Game Data Training
Tests that your system is ready for training with real data
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*70)
    print("Testing Imports")
    print("="*70)
    
    tests = []
    
    try:
        import torch
        tests.append(("✓", "torch", torch.__version__))
    except ImportError:
        tests.append(("✗", "torch", "NOT INSTALLED"))
    
    try:
        import sgfmill
        tests.append(("✓", "sgfmill", "installed"))
    except ImportError:
        tests.append(("✗", "sgfmill", "NOT INSTALLED - pip install sgfmill"))
    
    try:
        from tqdm import tqdm
        tests.append(("✓", "tqdm", "installed"))
    except ImportError:
        tests.append(("✗", "tqdm", "NOT INSTALLED - pip install tqdm"))
    
    try:
        from training.sgf_data_loader import SGFDataLoader
        tests.append(("✓", "SGFDataLoader", "ready"))
    except ImportError as e:
        tests.append(("✗", "SGFDataLoader", f"ERROR: {e}"))
    
    try:
        from models.alphago_baseline import AlphaGoBaseline
        tests.append(("✓", "AlphaGoBaseline", "ready"))
    except ImportError as e:
        tests.append(("✗", "AlphaGoBaseline", f"ERROR: {e}"))
    
    try:
        from training.train_pipeline import TrainingPipeline
        tests.append(("✓", "TrainingPipeline", "ready"))
    except ImportError as e:
        tests.append(("✗", "TrainingPipeline", f"ERROR: {e}"))
    
    print("\nImport Test Results:")
    for status, name, version in tests:
        print(f"  {status} {name:20s} {version}")
    
    all_passed = all(status == "✓" for status, _, _ in tests)
    
    if all_passed:
        print("\n✅ All imports successful!")
    else:
        print("\n⚠ Some imports failed. Install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    return all_passed


def test_sgf_directory():
    """Check if SGF directory exists and contains files"""
    print("\n" + "="*70)
    print("Testing SGF Directory")
    print("="*70)
    
    sgf_dir = 'data/sgf_games'
    
    if not os.path.exists(sgf_dir):
        print(f"\n⚠ Directory not found: {sgf_dir}")
        print("   Creating directory...")
        os.makedirs(sgf_dir, exist_ok=True)
        print(f"   ✓ Created: {sgf_dir}")
        print("\n📥 Next step: Download SGF files to this directory")
        return False
    
    import glob
    sgf_files = glob.glob(os.path.join(sgf_dir, '**', '*.sgf'), recursive=True)
    
    print(f"\nDirectory: {os.path.abspath(sgf_dir)}")
    print(f"SGF files found: {len(sgf_files)}")
    
    if len(sgf_files) == 0:
        print("\n⚠ No SGF files found!")
        print("\n📥 Download Instructions:")
        print("   1. Visit: https://u-go.net/gamerecords/")
        print("   2. Filter for 9×9 games")
        print("   3. Download 100-500 games")
        print(f"   4. Save to: {os.path.abspath(sgf_dir)}")
        return False
    else:
        print(f"\n✅ Found {len(sgf_files)} SGF files!")
        if len(sgf_files) < 50:
            print(f"   ⚠ Recommended: 100+ games for good training")
        
        print("\nSample files:")
        for f in sgf_files[:5]:
            size = os.path.getsize(f)
            print(f"   - {os.path.basename(f)} ({size} bytes)")
        
        if len(sgf_files) > 5:
            print(f"   ... and {len(sgf_files) - 5} more")
        
        return True


def test_sgf_loading():
    """Test loading a sample SGF file"""
    print("\n" + "="*70)
    print("Testing SGF Loading")
    print("="*70)
    
    try:
        from training.sgf_data_loader import SGFDataLoader
        
        loader = SGFDataLoader('data/sgf_games')
        
        print("\nAttempting to load first 5 games...")
        states, moves = loader.load_all_games(max_games=5, min_moves=10)
        
        if len(states) == 0:
            print("\n⚠ Could not load any games!")
            return False
        
        print(f"\n✅ Successfully loaded {len(states)} positions!")
        print(f"\nData Details:")
        print(f"   Positions: {len(states)}")
        print(f"   State shape: {states[0].shape}")
        print(f"   Sample moves: {moves[:10]}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading SGF files: {e}")
        return False


def test_training_pipeline():
    """Test that training pipeline can be initialized"""
    print("\n" + "="*70)
    print("Testing Training Pipeline")
    print("="*70)
    
    try:
        from training.train_pipeline import TrainingPipeline
        
        print("\nInitializing training pipeline...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = TrainingPipeline(device=device)
        
        print(f"✅ Training pipeline ready!")
        print(f"   Device: {device}")
        
        param_count = sum(p.numel() for p in pipeline.policy_net.parameters())
        print(f"   Policy network parameters: {param_count:,}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("REAL GAME DATA TRAINING - VERIFICATION SCRIPT")
    print("="*70)
    
    results = {}
    
    results['imports'] = test_imports()
    results['directory'] = test_sgf_directory()
    
    if results['directory']:
        results['loading'] = test_sgf_loading()
    else:
        results['loading'] = False
        print("\n⏭ Skipping SGF loading test (no data)")
    
    results['pipeline'] = test_training_pipeline()
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your system is ready for training!")
        print("\nNext steps:")
        print("1. Ensure you have 100+ SGF files in data/sgf_games/")
        print("2. Run training:")
        print("   python train_with_real_data.py --use_real_data --epochs 20")
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("\nPlease fix the issues above before training.")
        if not results['imports']:
            print("  → Install dependencies: pip install -r requirements.txt")
        if not results['directory']:
            print("  → Download SGF files to data/sgf_games/")


if __name__ == "__main__":
    main()
