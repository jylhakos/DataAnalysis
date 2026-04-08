"""
Self-Refinement Loop using Monte Carlo Sampling

Demonstrates iterative improvement of LLM responses through
Monte Carlo sampling and selection.
"""

import numpy as np
from typing import List, Tuple, Dict
from llm_client import OllamaClient
import json
from pathlib import Path
import matplotlib.pyplot as plt


class SelfRefinementLoop:
    """Implements self-refinement using Monte Carlo methods."""
    
    def __init__(self, llm_client: OllamaClient):
        """
        Initialize self-refinement loop.
        
        Args:
            llm_client: Ollama client for LLM interaction
        """
        self.llm = llm_client
    
    def generate_candidates(self, current_answer: str, problem: str,
                          n_candidates: int = 5, temperature: float = 1.0) -> List[str]:
        """
        Generate multiple candidate improvements.
        
        Args:
            current_answer: Current best answer
            problem: Original problem statement
            n_candidates: Number of candidates to generate
            temperature: Sampling temperature
            
        Returns:
            List[str]: Candidate improved answers
        """
        refinement_prompt = f"""Problem: {problem}

Current answer: {current_answer}

Please provide an improved version of this answer that is more accurate, clear, and complete. 
Focus on fixing any errors and adding relevant details."""

        candidates = self.llm.generate_multiple(
            refinement_prompt,
            n_samples=n_candidates,
            temperature=temperature,
            max_tokens=300
        )
        
        return candidates
    
    def evaluate_candidates(self, candidates: List[str], problem: str) -> List[float]:
        """
        Evaluate quality of candidate answers.
        
        Args:
            candidates: List of candidate answers
            problem: Original problem for context
            
        Returns:
            List[float]: Scores for each candidate (0-100)
        """
        scores = []
        
        print(f"Evaluating {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            scoring_prompt = f"""Problem: {problem}

Answer to evaluate: {candidate}

Rate this answer on a scale of 0-100 based on:
- Accuracy and correctness
- Clarity and completeness
- Relevance to the problem

Provide ONLY a number between 0 and 100.

Score:"""
            
            result = self.llm.generate(scoring_prompt, temperature=0.2, max_tokens=10)
            
            if result:
                try:
                    import re
                    match = re.search(r'\d+', result)
                    if match:
                        score = float(match.group())
                        score = min(max(score, 0), 100)
                        scores.append(score)
                        print(f"  Candidate {i+1}: score = {score:.1f}")
                        continue
                except ValueError:
                    pass
            
            # Fallback: use length as crude heuristic
            score = min(len(candidate) / 5, 100)
            scores.append(score)
            print(f"  Candidate {i+1}: score = {score:.1f} (fallback)")
        
        return scores
    
    def select_best(self, candidates: List[str], scores: List[float]) -> Tuple[str, float]:
        """
        Select the best candidate based on scores.
        
        Args:
            candidates: List of candidate answers
            scores: Corresponding scores
            
        Returns:
            Tuple[str, float]: (best_candidate, best_score)
        """
        if not candidates or not scores:
            return "", 0.0
        
        best_idx = np.argmax(scores)
        return candidates[best_idx], scores[best_idx]
    
    def refine(self, initial_answer: str, problem: str,
               n_iterations: int = 5, n_candidates: int = 5,
               temperature: float = 1.0) -> Dict:
        """
        Run the self-refinement loop.
        
        Args:
            initial_answer: Starting answer
            problem: Problem statement
            n_iterations: Number of refinement iterations
            n_candidates: Candidates per iteration
            temperature: Sampling temperature
            
        Returns:
            Dict: Results including final answer and improvement history
        """
        current_answer = initial_answer
        history = {
            'iterations': [],
            'scores': [],
            'answers': [initial_answer]
        }
        
        print(f"Starting self-refinement loop ({n_iterations} iterations)...")
        print()
        
        # Evaluate initial answer
        initial_score = self.evaluate_candidates([initial_answer], problem)[0]
        history['scores'].append(initial_score)
        print(f"Initial score: {initial_score:.2f}")
        print()
        
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")
            print("-" * 40)
            
            # Generate candidate improvements
            candidates = self.generate_candidates(
                current_answer,
                problem,
                n_candidates,
                temperature
            )
            
            if not candidates:
                print("  No candidates generated, stopping.")
                break
            
            # Evaluate candidates
            scores = self.evaluate_candidates(candidates, problem)
            
            # Select best candidate
            best_candidate, best_score = self.select_best(candidates, scores)
            
            # Update history
            history['iterations'].append(iteration + 1)
            history['scores'].append(best_score)
            history['answers'].append(best_candidate)
            
            print(f"  Best score this iteration: {best_score:.2f}")
            
            # Check for improvement
            current_score = history['scores'][-2] if len(history['scores']) > 1 else 0
            improvement = best_score - current_score
            print(f"  Improvement: {improvement:+.2f}")
            
            # Update current answer
            current_answer = best_candidate
            print()
        
        return {
            'initial_answer': initial_answer,
            'final_answer': current_answer,
            'initial_score': history['scores'][0],
            'final_score': history['scores'][-1],
            'improvement': history['scores'][-1] - history['scores'][0],
            'history': history
        }


def visualize_refinement(results: Dict, save_path: str = "results/refinement_progress.png"):
    """
    Visualize the refinement progress.
    
    Args:
        results: Results dictionary from refine()
        save_path: Path to save figure
    """
    history = results['history']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scores over iterations
    iterations = [0] + history['iterations']
    scores = history['scores']
    
    ax.plot(iterations, scores, 'bo-', linewidth=2, markersize=8, label='Quality Score')
    ax.axhline(y=results['initial_score'], color='r', linestyle='--', 
               label=f'Initial Score: {results["initial_score"]:.1f}', alpha=0.7)
    ax.axhline(y=results['final_score'], color='g', linestyle='--',
               label=f'Final Score: {results["final_score"]:.1f}', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Quality Score (0-100)', fontsize=12)
    ax.set_title('Self-Refinement Progress via Monte Carlo Sampling', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def demonstrate_self_refinement():
    """Demonstration of self-refinement loop."""
    print("="*60)
    print("Self-Refinement Loop with Monte Carlo Sampling")
    print("="*60)
    print()
    
    # Initialize LLM client
    client = OllamaClient(model="llama2")
    
    # Check connection
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama server")
        print("Please start Ollama and pull a model:")
        print("  docker start ollama")
        print("  docker exec -it ollama ollama pull llama2")
        return
    
    print("Connected to Ollama server")
    print(f"Using model: {client.model}")
    print()
    
    # Problem to solve
    problem = "Explain what Monte Carlo simulation is and why it's useful in AI, in 2-3 sentences."
    
    print(f"Problem: {problem}")
    print()
    
    # Generate initial answer
    print("Generating initial answer...")
    initial_answer = client.generate(problem, temperature=0.8, max_tokens=150)
    
    if not initial_answer:
        print("ERROR: Failed to generate initial answer")
        return
    
    print(f"Initial answer: {initial_answer}")
    print()
    print("="*60)
    print()
    
    # Run self-refinement
    refiner = SelfRefinementLoop(client)
    
    results = refiner.refine(
        initial_answer=initial_answer,
        problem=problem,
        n_iterations=4,
        n_candidates=3,
        temperature=0.9
    )
    
    # Display results
    print()
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    print()
    print(f"Initial Score: {results['initial_score']:.2f}")
    print(f"Final Score:   {results['final_score']:.2f}")
    print(f"Improvement:   {results['improvement']:+.2f}")
    print()
    print("Final Answer:")
    print(results['final_answer'])
    print()
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON (make history serializable)
    save_results = {
        'problem': problem,
        'initial_answer': results['initial_answer'],
        'final_answer': results['final_answer'],
        'initial_score': results['initial_score'],
        'final_score': results['final_score'],
        'improvement': results['improvement'],
        'scores_by_iteration': results['history']['scores']
    }
    
    with open(results_dir / "self_refinement_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"Results saved to: results/self_refinement_results.json")
    
    # Visualize progress
    visualize_refinement(results)
    
    print()
    print("="*60)


if __name__ == "__main__":
    np.random.seed(42)
    demonstrate_self_refinement()
