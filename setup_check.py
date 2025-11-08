#!/usr/bin/env python3
"""
Quick Setup Script for CSCI323 AlphaGo Project

This script helps you verify your environment is set up correctly.
Run this after installing requirements.txt
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    print("=" * 60)
    print("CHECKING PYTHON VERSION")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ ERROR: Python 3.9 or higher required!")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_packages():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED PACKAGES")
    print("=" * 60)
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'tensorboard',
        'sgfmill',
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:20} installed")
        except ImportError:
            print(f"❌ {package:20} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("CHECKING CUDA/GPU")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available (will use CPU)")
            print("   Training will be slower but still functional")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_directory_structure():
    """Check if directory structure exists"""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    import os
    
    required_dirs = [
        'data',
        'data/sgf_games',
        'data/processed',
        'models',
        'training',
        'evaluation',
        'utils',
        'experiments',
        'experiments/checkpoints',
        'experiments/logs',
        'experiments/results',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path:30} exists")
        else:
            print(f"❌ {dir_path:30} NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("CSCI323 ALPHAGO PROJECT - ENVIRONMENT SETUP CHECK")
    print("=" * 60)
    
    results = []
    
    # Run checks
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_packages()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("Directory Structure", check_directory_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    all_passed = all(status for _, status in results)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("1. Download 9×9 SGF games to data/sgf_games/")
        print("2. Run: python training/data_loader.py")
        print("3. Run: python training/train_policy.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("pip install -r requirements.txt")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
