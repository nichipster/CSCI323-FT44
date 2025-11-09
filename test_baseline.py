"""
Test Script for AlphaGo Baseline Components

Quick tests to verify all components are working correctly
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_policy_network():
    """Test policy network"""
    print("\n" + "="*60)
    print("Testing Policy Network")
    print("="*60)
    
    from models.policy_net import PolicyNetwork
    
    model = PolicyNetwork()
    dummy_input = torch.randn(2, 4, 9, 9)
    output = model(dummy_input)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output sum: {output.sum(dim=1)}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    assert output.shape == (2, 82), "Output shape incorrect"
    assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-5), "Output not a probability distribution"
    
    print("✓ Policy Network: PASSED")
    return True

def test_go_game():
    """Test Go game environment"""
    print("\n" + "="*60)
    print("Testing Go Game Environment")
    print("="*60)
    
    from utils.go_game import GoGame
    
    game = GoGame()
    
    # Test initial state
    print("✓ Game initialized")
    print(f"✓ Board size: {game.BOARD_SIZE}x{game.BOARD_SIZE}")
    
    # Test state representation
    state = game.get_state()
    print(f"✓ State shape: {state.shape}")
    assert state.shape == (4, 9, 9), "State shape incorrect"
    
    # Test legal moves
    legal_moves = game.get_legal_moves()
    print(f"✓ Legal moves: {len(legal_moves)}")
    assert len(legal_moves) == 82, "Should have 82 legal moves initially (81 + pass)"
    
    # Test making moves
    game.make_move(40)  # Center
    print("✓ Made move at center")
    
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) == 81, "Should have 81 legal moves after first move"
    
    print("✓ Go Game Environment: PASSED")
    return True

def test_mcts():
    """Test MCTS"""
    print("\n" + "="*60)
    print("Testing MCTS")
    print("="*60)
    
    from models.policy_net import PolicyNetwork
    from utils.mcts_neural_rollouts import MCTS
    from utils.go_game import GoGame
    
    policy_net = PolicyNetwork()
    
    print("✓ Policy network created")
    
    mcts = MCTS(policy_net, num_simulations=10)
    print("✓ MCTS initialized with 10 simulations")
    
    game = GoGame()
    
    print("Running MCTS search...")
    move, move_probs = mcts.get_best_move(game, temperature=0)
    
    print(f"✓ Best move: {move}")
    print(f"✓ Move probability: {move_probs[move]:.4f}")
    
    assert 0 <= move < 82, "Invalid move"
    assert torch.allclose(move_probs.sum(), torch.tensor(1.0), atol=1e-5), "Move probs not normalized"
    
    print("✓ MCTS: PASSED")
    return True

def test_alphago_baseline():
    """Test complete AlphaGo baseline"""
    print("\n" + "="*60)
    print("Testing AlphaGo Baseline System")
    print("="*60)
    
    from models.alphago_baseline import AlphaGoBaseline
    from utils.go_game import GoGame
    
    alphago = AlphaGoBaseline(num_simulations=10)
    print("✓ AlphaGo baseline created")
    
    # Get system info
    info = alphago.get_info()
    print(f"✓ Policy network parameters: {info['policy_network_params']:,}")
    print(f"✓ Total parameters: {info['total_params']:,}")
    print(f"✓ Evaluation method: {info['evaluation_method']}")
    
    # Test move selection
    game = GoGame()
    move, move_probs = alphago.select_move(game, temperature=0)
    print(f"✓ Selected move: {move}")
    
    # Test move prediction (policy network only)
    pred_probs = alphago.predict_move(game)
    print(f"✓ Policy prediction shape: {pred_probs.shape}")
    
    print("✓ AlphaGo Baseline: PASSED")
    return True

def test_training_pipeline():
    """Test training pipeline"""
    print("\n" + "="*60)
    print("Testing Training Pipeline")
    print("="*60)
    
    from training.train_pipeline import TrainingPipeline, generate_synthetic_data
    
    pipeline = TrainingPipeline(device='cpu')
    print("✓ Training pipeline created")
    
    # Generate small dataset
    print("Generating synthetic data...")
    states, moves = generate_synthetic_data(num_samples=50)
    print(f"✓ Generated {len(states)} samples")
    
    # Test short training (1 epoch)
    print("Testing policy network training (1 epoch)...")
    pipeline.train_policy_supervised(
        train_data=(states[:40], moves[:40]),
        val_data=(states[40:], moves[40:]),
        epochs=1,
        batch_size=8,
        learning_rate=0.001
    )
    print("✓ Policy training completed")
    
    print("✓ Training Pipeline: PASSED")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "ALPHAGO BASELINE TESTS")
    print("="*70)
    
    tests = [
        ("Policy Network", test_policy_network),
        ("Go Game Environment", test_go_game),
        ("MCTS", test_mcts),
        ("AlphaGo Baseline", test_alphago_baseline),
        ("Training Pipeline", test_training_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ {test_name}: FAILED")
            print(f"  Error: {str(e)}")
            results.append((test_name, "FAILED"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for _, result in results if result == "PASSED")
    total = len(results)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
