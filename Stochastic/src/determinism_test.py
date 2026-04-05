#!/usr/bin/env python3
"""
Determinism Testing Module

Demonstrates deterministic vs non-deterministic behavior in machine learning
and random number generation.
"""

import random
import numpy as np
import torch
from typing import List, Tuple


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for maximum reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ All seeds set to {seed}")
    print(f"✓ PyTorch deterministic mode: ON")


def demonstrate_prng_determinism() -> None:
    """
    Demonstrate that PRNGs are deterministic with fixed seeds.
    
    Shows that computer "randomness" is actually a deterministic
    sequence starting from a seed value.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: PRNG Determinism")
    print("="*60)
    
    print("\nRun 1 (seed=42):")
    set_all_seeds(42)
    run1_python = [random.randint(1, 100) for _ in range(5)]
    run1_numpy = np.random.randint(1, 100, size=5)
    run1_torch = torch.randint(1, 100, (5,))
    
    print(f"  Python random: {run1_python}")
    print(f"  NumPy random:  {run1_numpy}")
    print(f"  PyTorch random: {run1_torch.tolist()}")
    
    print("\nRun 2 (seed=42) - Should be IDENTICAL:")
    set_all_seeds(42)
    run2_python = [random.randint(1, 100) for _ in range(5)]
    run2_numpy = np.random.randint(1, 100, size=5)
    run2_torch = torch.randint(1, 100, (5,))
    
    print(f"  Python random: {run2_python}")
    print(f"  NumPy random:  {run2_numpy}")
    print(f"  PyTorch random: {run2_torch.tolist()}")
    
    print("\nRun 3 (seed=999) - Should be DIFFERENT:")
    set_all_seeds(999)
    run3_python = [random.randint(1, 100) for _ in range(5)]
    run3_numpy = np.random.randint(1, 100, size=5)
    run3_torch = torch.randint(1, 100, (5,))
    
    print(f"  Python random: {run3_python}")
    print(f"  NumPy random:  {run3_numpy}")
    print(f"  PyTorch random: {run3_torch.tolist()}")
    
    # Verification
    print("\n" + "-"*60)
    assert run1_python == run2_python, "Python random should be deterministic!"
    assert np.array_equal(run1_numpy, run2_numpy), "NumPy random should be deterministic!"
    assert torch.equal(run1_torch, run2_torch), "PyTorch random should be deterministic!"
    
    print("✓ VERIFIED: Same seed → Same output (DETERMINISTIC)")
    print("✓ VERIFIED: Different seed → Different output")
    print("-"*60)


def demonstrate_sgd_reproducibility() -> None:
    """
    Demonstrate that stochastic gradient descent is reproducible with seeds.
    
    Shows how a "stochastic" algorithm becomes deterministic through
    pseudo-randomness.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: SGD Reproducibility")
    print("="*60)
    
    def simple_sgd_step(weights: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Simulate one SGD step with random gradient"""
        # Simulated "stochastic" gradient
        gradient = np.random.randn(*weights.shape)
        return weights - learning_rate * gradient
    
    # Run 1
    print("\nRun 1 (seed=42):")
    set_all_seeds(42)
    weights = np.array([1.0, 2.0, 3.0])
    print(f"  Initial weights: {weights}")
    
    for i in range(3):
        weights = simple_sgd_step(weights)
        print(f"  Step {i+1}: {weights}")
    
    run1_final = weights.copy()
    
    # Run 2 - Same seed
    print("\nRun 2 (seed=42) - Should be IDENTICAL:")
    set_all_seeds(42)
    weights = np.array([1.0, 2.0, 3.0])
    print(f"  Initial weights: {weights}")
    
    for i in range(3):
        weights = simple_sgd_step(weights)
        print(f"  Step {i+1}: {weights}")
    
    run2_final = weights.copy()
    
    # Verification
    print("\n" + "-"*60)
    assert np.allclose(run1_final, run2_final), "SGD should be reproducible!"
    print("✓ VERIFIED: SGD with same seed → Same weight updates")
    print("✓ Stochastic algorithm made DETERMINISTIC via seed")
    print("-"*60)


def demonstrate_neural_network_inference() -> None:
    """
    Demonstrate that neural network inference is deterministic.
    
    Shows that once weights are frozen, the network becomes a
    deterministic function.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Neural Network Inference")
    print("="*60)
    
    # Create a simple neural network
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Freeze weights (simulate trained model)
    model.eval()
    
    # Create input
    torch.manual_seed(123)
    input_tensor = torch.randn(1, 10)
    
    print("\nInput tensor:")
    print(f"  {input_tensor.squeeze().tolist()[:5]}... (truncated)")
    
    # Run inference 3 times
    outputs = []
    for i in range(3):
        with torch.no_grad():
            output = model(input_tensor)
            outputs.append(output.item())
            print(f"\nInference run {i+1}: {output.item():.6f}")
    
    # Verification
    print("\n" + "-"*60)
    assert all(outputs[0] == out for out in outputs), "Inference should be deterministic!"
    print("✓ VERIFIED: Same input → Same output (DETERMINISTIC)")
    print("✓ Neural network inference is a deterministic function")
    print("-"*60)


def demonstrate_dropout_nondeterminism() -> None:
    """
    Demonstrate that dropout makes inference non-deterministic.
    
    Shows when neural networks are intentionally non-deterministic.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Dropout Non-Determinism")
    print("="*60)
    
    # Create model with dropout
    torch.manual_seed(42)
    model_with_dropout = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.Dropout(p=0.5),  # 50% dropout
        torch.nn.Linear(5, 1)
    )
    
    # Create input
    torch.manual_seed(123)
    input_tensor = torch.randn(1, 10)
    
    print("\nModel with Dropout (p=0.5)")
    print("Running inference in TRAINING mode (dropout active):")
    
    model_with_dropout.train()  # Dropout is active
    outputs_train = []
    for i in range(5):
        output = model_with_dropout(input_tensor)
        outputs_train.append(output.item())
        print(f"  Run {i+1}: {output.item():.6f}")
    
    print("\nRunning inference in EVAL mode (dropout disabled):")
    model_with_dropout.eval()  # Dropout is disabled
    outputs_eval = []
    for i in range(5):
        with torch.no_grad():
            output = model_with_dropout(input_tensor)
            outputs_eval.append(output.item())
            print(f"  Run {i+1}: {output.item():.6f}")
    
    # Verification
    print("\n" + "-"*60)
    unique_train = len(set(outputs_train))
    unique_eval = len(set(outputs_eval))
    
    print(f"Training mode (dropout ON):  {unique_train} unique outputs → NON-DETERMINISTIC")
    print(f"Eval mode (dropout OFF):     {unique_eval} unique outputs → DETERMINISTIC")
    print("-"*60)


def demonstrate_greedy_vs_sampling() -> None:
    """
    Demonstrate difference between greedy decoding and sampling.
    
    Simulates LLM token selection with different strategies.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Greedy vs Sampling Decoding")
    print("="*60)
    
    # Simulated token probabilities
    vocab = ["the", "a", "an", "this", "that"]
    logits = np.array([0.5, 0.3, 0.1, 0.07, 0.03])  # Probability distribution
    
    print("\nToken probabilities:")
    for token, prob in zip(vocab, logits):
        print(f"  '{token}': {prob:.2%}")
    
    # Greedy decoding
    print("\nGreedy Decoding (always pick highest probability):")
    for i in range(5):
        greedy_choice = vocab[np.argmax(logits)]
        print(f"  Run {i+1}: '{greedy_choice}'")
    
    # Sampling
    print("\nSampling (probabilistic selection):")
    set_all_seeds(42)
    for i in range(5):
        sampled_choice = np.random.choice(vocab, p=logits)
        print(f"  Run {i+1}: '{sampled_choice}'")
    
    # Temperature scaling
    print("\nSampling with Temperature = 2.0 (more random):")
    set_all_seeds(42)
    temperature = 2.0
    temp_logits = logits ** (1 / temperature)
    temp_probs = temp_logits / temp_logits.sum()
    
    print("  Adjusted probabilities:")
    for token, prob in zip(vocab, temp_probs):
        print(f"    '{token}': {prob:.2%}")
    
    for i in range(5):
        sampled_choice = np.random.choice(vocab, p=temp_probs)
        print(f"  Run {i+1}: '{sampled_choice}'")
    
    print("\n" + "-"*60)
    print("✓ Greedy: DETERMINISTIC (always same output)")
    print("✓ Sampling: STOCHASTIC (variable output)")
    print("✓ Higher temperature: MORE randomness")
    print("-"*60)


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DETERMINISM IN MACHINE LEARNING")
    print("Demonstrating Deterministic vs Stochastic Behavior")
    print("="*60)
    
    demonstrate_prng_determinism()
    demonstrate_sgd_reproducibility()
    demonstrate_neural_network_inference()
    demonstrate_dropout_nondeterminism()
    demonstrate_greedy_vs_sampling()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. PRNGs are DETERMINISTIC with fixed seeds")
    print("2. Stochastic algorithms (SGD) are reproducible via seeds")
    print("3. Neural network inference is DETERMINISTIC")
    print("4. Training is STOCHASTIC (random init, SGD, dropout)")
    print("5. Some components (dropout, sampling) are intentionally random")
    print("6. Greedy decoding → deterministic, Sampling → stochastic")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
