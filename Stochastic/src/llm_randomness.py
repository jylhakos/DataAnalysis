#!/usr/bin/env python3
"""
LLM Randomness Psychology Demonstration

This module demonstrates how LLMs exhibit human-like biases when generating
"random" numbers, based on research from arXiv:2502.19965.

Key findings:
- LLMs favor certain numbers (3, 7) over uniform distribution
- Different languages produce different biases
- Temperature affects determinism but not bias elimination
- LLMs are "stochastic parrots" for randomness tasks
"""

import random
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed. Install with: pip install ollama")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"✓ Random seed set to {seed}")


def generate_true_random_numbers(n: int = 100, low: int = 1, high: int = 10) -> List[int]:
    """
    Generate truly (pseudo-)random numbers using Python's PRNG.
    
    This serves as a baseline for comparison with LLM-generated numbers.
    
    Args:
        n: Number of random numbers to generate
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        
    Returns:
        List of random integers
    """
    return [random.randint(low, high) for _ in range(n)]


def calculate_statistics(numbers: List[int]) -> Dict:
    """
    Calculate statistical properties of a number distribution.
    
    Args:
        numbers: List of integers
        
    Returns:
        Dictionary with statistical metrics
    """
    counter = Counter(numbers)
    total = len(numbers)
    
    # Expected frequency for uniform distribution
    unique_values = len(set(numbers))
    expected_freq = total / unique_values
    
    # Chi-square test for uniformity
    observed = [counter[i] for i in sorted(counter.keys())]
    expected = [expected_freq] * len(observed)
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    stats_dict = {
        'mean': np.mean(numbers),
        'median': np.median(numbers),
        'std': np.std(numbers),
        'min': min(numbers),
        'max': max(numbers),
        'distribution': dict(counter),
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'is_uniform': p_value > 0.05,  # 5% significance level
        'most_common': counter.most_common(3),
        'least_common': counter.most_common()[-3:] if len(counter) >= 3 else counter.most_common()
    }
    
    return stats_dict


def generate_llm_random_number(
    model: str = "llama3.2",
    temperature: float = 0.7,
    language: str = "English",
    low: int = 1,
    high: int = 10
) -> Tuple[int, str]:
    """
    Prompt an LLM to generate a "random" number.
    
    Args:
        model: Ollama model name
        temperature: Sampling temperature (0.0 = deterministic, >1.0 = creative)
        language: Prompt language (affects bias)
        low: Lower bound
        high: Upper bound
        
    Returns:
        Tuple of (number, raw_response)
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama package not installed")
    
    prompts = {
        "English": f"Generate a random number between {low} and {high}. Reply with ONLY the number, nothing else.",
        "Japanese": f"{low}から{high}までのランダムな数字を生成してください。数字のみを返してください。",
        "German": f"Generiere eine zufällige Zahl zwischen {low} und {high}. Antworte NUR mit der Zahl, sonst nichts.",
        "Spanish": f"Genera un número aleatorio entre {low} y {high}. Responde SOLO con el número, nada más."
    }
    
    prompt = prompts.get(language, prompts["English"])
    
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': temperature,
                'num_predict': 10  # Limit output length
            }
        )
        
        raw_response = response['response'].strip()
        
        # Extract number from response
        # Try to parse as integer
        for word in raw_response.split():
            try:
                number = int(''.join(c for c in word if c.isdigit()))
                if low <= number <= high:
                    return number, raw_response
            except ValueError:
                continue
        
        # If we can't extract a valid number, return None
        return None, raw_response
        
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return None, str(e)


def run_llm_randomness_experiment(
    model: str = "llama3.2",
    n_trials: int = 100,
    temperature: float = 0.7,
    language: str = "English",
    low: int = 1,
    high: int = 10
) -> List[int]:
    """
    Run multiple trials of LLM random number generation.
    
    Args:
        model: Ollama model name
        n_trials: Number of trials
        temperature: Sampling temperature
        language: Prompt language
        low: Lower bound
        high: Upper bound
        
    Returns:
        List of generated numbers
    """
    print(f"\n{'='*60}")
    print(f"Running LLM Randomness Experiment")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Language: {language}")
    print(f"Trials: {n_trials}")
    print(f"Range: [{low}, {high}]")
    print(f"{'='*60}\n")
    
    numbers = []
    failed_trials = 0
    
    for i in range(n_trials):
        number, response = generate_llm_random_number(
            model=model,
            temperature=temperature,
            language=language,
            low=low,
            high=high
        )
        
        if number is not None:
            numbers.append(number)
            if (i + 1) % 10 == 0:
                print(f"Trial {i+1}/{n_trials}: {number}")
        else:
            failed_trials += 1
            print(f"Trial {i+1}/{n_trials}: FAILED - {response[:50]}")
    
    print(f"\nCompleted: {len(numbers)} successful, {failed_trials} failed")
    return numbers


def visualize_distributions(
    llm_numbers: List[int],
    true_random: List[int],
    title: str = "LLM vs True Random Distribution"
) -> None:
    """
    Create visualization comparing LLM and true random distributions.
    
    Args:
        llm_numbers: Numbers generated by LLM
        true_random: True random numbers
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LLM distribution
    axes[0].hist(llm_numbers, bins=range(min(llm_numbers), max(llm_numbers) + 2),
                 alpha=0.7, color='coral', edgecolor='black')
    axes[0].set_title('LLM-Generated Numbers')
    axes[0].set_xlabel('Number')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Add mean line
    axes[0].axvline(np.mean(llm_numbers), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(llm_numbers):.2f}')
    axes[0].legend()
    
    # True random distribution
    axes[1].hist(true_random, bins=range(min(true_random), max(true_random) + 2),
                 alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_title('True Random Numbers (Python PRNG)')
    axes[1].set_xlabel('Number')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Add mean line
    axes[1].axvline(np.mean(true_random), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(true_random):.2f}')
    axes[1].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('llm_randomness_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'llm_randomness_comparison.png'")
    plt.show()


def test_determinism(
    model: str = "llama3.2",
    temperature: float = 0.0,
    n_trials: int = 10
) -> None:
    """
    Test if LLM is deterministic at temperature 0.
    
    Args:
        model: Ollama model name
        temperature: Should be 0 for deterministic test
        n_trials: Number of repetitions
    """
    print(f"\n{'='*60}")
    print(f"Testing Determinism (Temperature = {temperature})")
    print(f"{'='*60}\n")
    
    prompt = "Generate a random number between 1 and 10. Reply with ONLY the number."
    
    responses = []
    for i in range(n_trials):
        number, response = generate_llm_random_number(
            model=model,
            temperature=temperature,
            language="English"
        )
        responses.append(number)
        print(f"Trial {i+1}: {number}")
    
    unique_responses = len(set(responses))
    
    print(f"\n{'='*60}")
    if unique_responses == 1:
        print(f"✓ DETERMINISTIC: All {n_trials} trials produced the same output")
        print(f"  Output: {responses[0]}")
    else:
        print(f"✗ NON-DETERMINISTIC: {unique_responses} different outputs across {n_trials} trials")
        print(f"  Distribution: {Counter(responses)}")
    print(f"{'='*60}\n")


def print_statistics_report(stats: Dict, title: str = "Statistical Analysis") -> None:
    """
    Print formatted statistical report.
    
    Args:
        stats: Statistics dictionary from calculate_statistics()
        title: Report title
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Mean:                {stats['mean']:.2f}")
    print(f"Median:              {stats['median']:.2f}")
    print(f"Standard Deviation:  {stats['std']:.2f}")
    print(f"Range:               [{stats['min']}, {stats['max']}]")
    print(f"\nChi-Square Test for Uniformity:")
    print(f"  χ² statistic:      {stats['chi2_statistic']:.4f}")
    print(f"  p-value:           {stats['p_value']:.4f}")
    print(f"  Uniform?           {'Yes' if stats['is_uniform'] else 'No'} (α=0.05)")
    print(f"\nMost Common Numbers:")
    for num, count in stats['most_common']:
        print(f"  {num}: {count} times ({count/sum(stats['distribution'].values())*100:.1f}%)")
    print(f"\nLeast Common Numbers:")
    for num, count in stats['least_common'][::-1]:
        print(f"  {num}: {count} times ({count/sum(stats['distribution'].values())*100:.1f}%)")
    print(f"{'='*60}\n")


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("LLM Psychology: Random Number Generation")
    print("Based on research: arXiv:2502.19965")
    print("="*60)
    
    if not OLLAMA_AVAILABLE:
        print("\n⚠ Ollama not available. Demonstrating with true random only.\n")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Generate true random numbers
        true_random = generate_true_random_numbers(n=100, low=1, high=10)
        
        # Calculate and display statistics
        stats = calculate_statistics(true_random)
        print_statistics_report(stats, "True Random Numbers (Python PRNG)")
        
        print("\nTo run LLM experiments:")
        print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("2. Install Python package: pip install ollama")
        print("3. Pull a model: ollama pull llama3.2")
        print("4. Run this script again")
        
        return
    
    # Check if Ollama is running
    try:
        ollama.list()
    except Exception as e:
        print(f"\n⚠ Error: Ollama service not running: {e}")
        print("\nStart Ollama with: ollama serve")
        return
    
    # Set seed for reproducibility
    set_seed(42)
    
    # ========================================
    # Experiment 1: LLM vs True Random
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: LLM vs True Random")
    print("="*60)
    
    llm_numbers = run_llm_randomness_experiment(
        model="llama3.2",
        n_trials=50,
        temperature=0.7,
        language="English",
        low=1,
        high=10
    )
    
    true_random = generate_true_random_numbers(n=50, low=1, high=10)
    
    # Statistics
    llm_stats = calculate_statistics(llm_numbers)
    random_stats = calculate_statistics(true_random)
    
    print_statistics_report(llm_stats, "LLM-Generated Numbers")
    print_statistics_report(random_stats, "True Random Numbers")
    
    # Visualize
    if len(llm_numbers) > 0:
        visualize_distributions(llm_numbers, true_random)
    
    # ========================================
    # Experiment 2: Test Determinism
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Testing Determinism")
    print("="*60)
    
    test_determinism(model="llama3.2", temperature=0.0, n_trials=10)
    test_determinism(model="llama3.2", temperature=0.7, n_trials=10)
    
    # ========================================
    # Experiment 3: Language Impact
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Language Impact on Bias")
    print("="*60)
    
    for language in ["English", "Japanese", "German"]:
        print(f"\n--- {language} ---")
        lang_numbers = run_llm_randomness_experiment(
            model="llama3.2",
            n_trials=30,
            temperature=0.7,
            language=language,
            low=1,
            high=10
        )
        
        if len(lang_numbers) > 0:
            lang_stats = calculate_statistics(lang_numbers)
            print(f"Most common: {lang_stats['most_common'][0]}")
            print(f"Mean: {lang_stats['mean']:.2f}")
            print(f"Uniform? {lang_stats['is_uniform']}")
    
    print("\n" + "="*60)
    print("Experiments Complete!")
    print("="*60)
    print("\nKey Findings:")
    print("• LLMs exhibit human-like biases (central tendency)")
    print("• Not truly random, even with high temperature")
    print("• Different languages → different biases")
    print("• Temperature 0 ≠ guaranteed determinism")
    print("• LLMs are 'stochastic parrots' for randomness tasks")


if __name__ == "__main__":
    main()
