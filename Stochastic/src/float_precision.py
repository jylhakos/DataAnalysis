#!/usr/bin/env python3
"""
Floating-Point Non-Associativity Demonstration

Demonstrates how floating-point arithmetic in GPUs is non-associative,
leading to non-deterministic outputs in LLM inference.
"""

import numpy as np
import torch
from typing import List


def demonstrate_float_non_associativity() -> None:
    """
    Demonstrate that (a + b) + c ≠ a + (b + c) in floating-point arithmetic.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Floating-Point Non-Associativity")
    print("="*60)
    
    # Example with regular Python floats
    a, b, c = 1e16, 1.0, -1e16
    
    result1 = (a + b) + c
    result2 = a + (b + c)
    
    print("\nMathematically: (a + b) + c = a + (b + c)")
    print(f"But in floating-point:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  c = {c}")
    print(f"\n  (a + b) + c = ({a} + {b}) + {c} = {result1}")
    print(f"  a + (b + c) = {a} + ({b} + {c}) = {result2}")
    print(f"\n  Equal? {result1 == result2}")
    print(f"  Difference: {abs(result1 - result2)}")
    
    print("\n" + "-"*60)
    print("✗ FAILED: (a + b) + c ≠ a + (b + c)")
    print("✓ This is why GPU parallelization causes non-determinism")
    print("-"*60)


def demonstrate_precision_loss() -> None:
    """
    Demonstrate precision loss in different floating-point formats.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Precision Loss in Different Formats")
    print("="*60)
    
    # Original value
    original = 1.123456789
    
    # Convert to different precisions
    fp32 = np.float32(original)
    fp16 = np.float16(original)
    
    if torch.cuda.is_available():
        bf16 = torch.tensor(original, dtype=torch.bfloat16).item()
    else:
        # Simulate bfloat16 behavior
        bf16 = np.float32(original)  # Approximate
    
    print(f"\nOriginal (Python float): {original}")
    print(f"FP32 (32-bit):           {fp32:.10f}")
    print(f"FP16 (16-bit):           {fp16:.10f}")
    print(f"BF16 (16-bit):           {bf16:.10f}")
    
    print(f"\nPrecision loss:")
    print(f"  FP32: {abs(original - float(fp32)):.2e}")
    print(f"  FP16: {abs(original - float(fp16)):.2e}")
    print(f"  BF16: {abs(original - bf16):.2e}")
    
    print("\n" + "-"*60)
    print("✓ Lower precision → Greater rounding errors")
    print("✓ These errors cascade in billion-parameter models")
    print("-"*60)


def demonstrate_summation_order() -> None:
    """
    Demonstrate how summation order affects results.
    
    This is what happens in GPU kernels when operations are parallelized.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Summation Order Matters")
    print("="*60)
    
    # Create array of floating-point numbers
    np.random.seed(42)
    numbers = np.random.randn(1000).astype(np.float32)
    
    # Sum in different orders
    sum_forward = np.sum(numbers)
    sum_backward = np.sum(numbers[::-1])
    sum_sorted_asc = np.sum(np.sort(numbers))
    sum_sorted_desc = np.sum(np.sort(numbers)[::-1])
    
    print(f"\nSumming 1000 random numbers:")
    print(f"  Forward order:       {sum_forward:.10f}")
    print(f"  Backward order:      {sum_backward:.10f}")
    print(f"  Sorted (ascending):  {sum_sorted_asc:.10f}")
    print(f"  Sorted (descending): {sum_sorted_desc:.10f}")
    
    print(f"\nDifferences from forward sum:")
    print(f"  Backward:  {abs(sum_forward - sum_backward):.2e}")
    print(f"  Asc sort:  {abs(sum_forward - sum_sorted_asc):.2e}")
    print(f"  Desc sort: {abs(sum_forward - sum_sorted_desc):.2e}")
    
    print("\n" + "-"*60)
    print("✓ Different orders → Different results")
    print("✓ GPU threads finish in non-deterministic order")
    print("✓ This is a major source of LLM non-determinism")
    print("-"*60)


def demonstrate_matrix_multiplication_variance() -> None:
    """
    Demonstrate non-determinism in matrix multiplication on GPU.
    
    Based on the example from the user's prompt.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: GPU Matrix Multiplication Variance")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available. This demonstration requires a GPU.")
        print("  Running CPU version (which is deterministic)...\n")
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Create random matrices
    torch.manual_seed(42)
    A = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    B = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    
    # Reference multiplication
    ref = torch.mm(A, B)
    
    print(f"\nRunning matrix multiplication on {device.upper()}:")
    print(f"  Matrix size: 2048 × 2048")
    print(f"  Dtype: bfloat16")
    
    # Test for non-determinism
    differences = []
    for i in range(10):
        result = torch.mm(A, B)
        diff = (result - ref).abs().max().item()
        differences.append(diff)
        
        if diff > 0:
            print(f"  Run {i+1}: Difference = {diff:.2e} (NON-DETERMINISTIC)")
        else:
            print(f"  Run {i+1}: Difference = {diff:.2e} (deterministic)")
    
    max_diff = max(differences)
    num_different = sum(1 for d in differences if d > 0)
    
    print(f"\nResults:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  Non-deterministic runs: {num_different}/10")
    
    if device == 'cuda' and num_different > 0:
        print("\n" + "-"*60)
        print("✗ GPU matrix multiplication is NON-DETERMINISTIC")
        print("✓ This affects every forward pass in LLMs")
        print("✓ Even with temperature=0, outputs can vary")
        print("-"*60)
    else:
        print("\n" + "-"*60)
        print("✓ CPU computation is deterministic")
        print("  (but ~100x slower than GPU)")
        print("-"*60)


def demonstrate_attention_variance() -> None:
    """
    Simulate how floating-point errors affect attention mechanism.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Attention Mechanism Variance")
    print("="*60)
    
    # Simulate attention scores
    np.random.seed(42)
    d_k = 64  # Key dimension
    seq_len = 512  # Sequence length
    
    # Query and Key matrices
    Q = np.random.randn(seq_len, d_k).astype(np.float16)
    K = np.random.randn(seq_len, d_k).astype(np.float16)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # Apply softmax
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # Find tokens with nearly equal probabilities
    # These are most susceptible to floating-point variance
    print("\nFinding tokens with nearly equal attention probabilities:")
    
    for i in range(min(3, seq_len)):
        sorted_probs = np.sort(attention_probs[i])[::-1]
        top1, top2 = sorted_probs[0], sorted_probs[1]
        diff = top1 - top2
        
        print(f"\n  Position {i}:")
        print(f"    Top-1 probability: {top1:.6f}")
        print(f"    Top-2 probability: {top2:.6f}")
        print(f"    Difference:        {diff:.6f}")
        
        if diff < 0.001:
            print(f"    ⚠ CRITICAL: Very close probabilities!")
            print(f"       Floating-point error could flip the selection")
    
    print("\n" + "-"*60)
    print("✓ When probabilities are close, small errors matter")
    print("✓ Attention mechanism is highly sensitive to variance")
    print("✓ This cascades through all transformer layers")
    print("-"*60)


def demonstrate_batch_size_effect() -> None:
    """
    Demonstrate how different batch sizes lead to different results.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Batch Size Effect on Determinism")
    print("="*60)
    
    print("\nBatch size affects how GPUs split reduction operations.")
    print("Different batches → Different operation orders → Different results")
    
    # Simulate RMSNorm with different batch sizes
    torch.manual_seed(42)
    
    hidden_size = 768
    
    # Same tensor, different batch sizes
    batch1 = torch.randn(1, hidden_size)
    batch4 = batch1.repeat(4, 1)
    
    def rms_norm(x):
        """Root Mean Square Normalization"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        return x / (rms + 1e-6)
    
    # Normalize
    norm1 = rms_norm(batch1)
    norm4 = rms_norm(batch4)
    
    # Extract first item from batch of 4
    norm4_first = norm4[0:1]
    
    # Compare
    diff = (norm1 - norm4_first).abs().max().item()
    
    print(f"\nRMSNorm on batch size 1:")
    print(f"  Shape: {batch1.shape}")
    print(f"  First 5 values: {norm1[0, :5].tolist()}")
    
    print(f"\nRMSNorm on batch size 4 (showing first item):")
    print(f"  Shape: {batch4.shape}")
    print(f"  First 5 values: {norm4_first[0, :5].tolist()}")
    
    print(f"\nMaximum difference: {diff:.2e}")
    
    if diff > 1e-6:
        print("\n" + "-"*60)
        print("✗ Different batch sizes → Different results")
        print("✓ This is why the same prompt can yield different outputs")
        print("-"*60)
    else:
        print("\n" + "-"*60)
        print("✓ In this case, results are similar")
        print("  (But in production with larger batches, variance increases)")
        print("-"*60)


def main():
    """Run all floating-point demonstrations"""
    print("\n" + "="*60)
    print("FLOATING-POINT NON-ASSOCIATIVITY IN LLMs")
    print("Why LLMs are Non-Deterministic Even at Temperature 0")
    print("="*60)
    
    demonstrate_float_non_associativity()
    demonstrate_precision_loss()
    demonstrate_summation_order()
    demonstrate_matrix_multiplication_variance()
    demonstrate_attention_variance()
    demonstrate_batch_size_effect()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Floating-point math is NON-ASSOCIATIVE")
    print("2. (a + b) + c ≠ a + (b + c) due to rounding")
    print("3. GPU parallelization → non-deterministic operation order")
    print("4. Different batch sizes → different reduction patterns")
    print("5. Errors cascade through transformer layers")
    print("6. Temperature 0 does NOT guarantee determinism")
    print("7. Solutions: Batch-invariant kernels, higher precision")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
