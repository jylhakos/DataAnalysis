"""
Monte Carlo Tree Search for Large Language Models

Implements MCTS algorithm to improve LLM reasoning by exploring
multiple answer paths and selecting the best one.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from llm_client import OllamaClient
import json
from pathlib import Path


@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree."""
    state: str  # Current text/reasoning state
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def average_reward(self) -> float:
        """Calculate average reward (Q-value)."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate Upper Confidence Bound for Trees (UCT) score.
        
        UCT = Q(v')/N(v') + C * sqrt(ln(N(v)) / N(v'))
        
        Args:
            exploration_constant (float): Balance between exploration and exploitation
            
        Returns:
            float: UCT score
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None:
            return self.average_reward()
        
        # UCT formula
        exploitation = self.average_reward()
        exploration = exploration_constant * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0


class MCTS:
    """Monte Carlo Tree Search implementation for LLMs."""
    
    def __init__(self, llm_client: OllamaClient, exploration_constant: float = 1.414):
        """
        Initialize MCTS.
        
        Args:
            llm_client: Ollama client for LLM interaction
            exploration_constant: UCT exploration parameter
        """
        self.llm = llm_client
        self.exploration_constant = exploration_constant
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCT until reaching a leaf.
        
        Args:
            node: Starting node
            
        Returns:
            MCTSNode: Selected leaf node
        """
        current = node
        
        while not current.is_leaf():
            # Choose child with highest UCT score
            current = max(
                current.children,
                key=lambda c: c.uct_score(self.exploration_constant)
            )
        
        return current
    
    def expand(self, node: MCTSNode, prompt: str, n_expansions: int = 3) -> List[MCTSNode]:
        """
        Expansion phase: generate new child nodes.
        
        Args:
            node: Node to expand
            prompt: Prompt template for generating children
            n_expansions: Number of children to generate
            
        Returns:
            List[MCTSNode]: New child nodes
        """
        expansion_prompt = f"""Given the current reasoning step:
{node.state}

{prompt}

Provide one possible next step or refinement. Be concise and focus on improvement."""

        # Generate multiple candidate next steps
        responses = self.llm.generate_multiple(
            expansion_prompt,
            n_samples=n_expansions,
            temperature=1.0,
            max_tokens=200
        )
        
        children = []
        for response in responses:
            child = MCTSNode(state=response, parent=node)
            node.children.append(child)
            children.append(child)
        
        return children
    
    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: evaluate the quality of a node.
        
        Args:
            node: Node to evaluate
            
        Returns:
            float: Reward score (0-100)
        """
        score = self.llm.score_response(node.state, criteria="correctness and clarity")
        
        if score is None:
            # Fallback: use response length as a proxy (crude heuristic)
            score = min(len(node.state) / 10, 100)
        
        return score
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: update node statistics up the tree.
        
        Args:
            node: Starting node
            reward: Reward value to propagate
        """
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def search(self, initial_state: str, prompt: str, 
               n_iterations: int = 10, n_expansions: int = 3) -> MCTSNode:
        """
        Run MCTS to find the best reasoning path.
        
        Args:
            initial_state: Starting text/answer
            prompt: Task prompt for expansion
            n_iterations: Number of MCTS iterations
            n_expansions: Number of children per expansion
            
        Returns:
            MCTSNode: Best node found
        """
        root = MCTSNode(state=initial_state)
        
        print(f"Starting MCTS with {n_iterations} iterations...")
        print()
        
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")
            
            # 1. Selection
            leaf = self.select(root)
            print(f"  Selected node (visits: {leaf.visits}, reward: {leaf.average_reward():.2f})")
            
            # 2. Expansion
            if leaf.visits > 0 or iteration == 0:  # Expand visited nodes or root
                children = self.expand(leaf, prompt, n_expansions)
                print(f"  Expanded {len(children)} children")
                
                # Select one child to simulate
                if children:
                    leaf = children[0]
            
            # 3. Simulation (evaluation)
            reward = self.simulate(leaf)
            print(f"  Simulated reward: {reward:.2f}")
            
            # 4. Backpropagation
            self.backpropagate(leaf, reward)
            print(f"  Backpropagated to root")
            print()
        
        # Return best child of root (most visits or highest average reward)
        if root.children:
            best_node = max(root.children, key=lambda c: c.average_reward())
            print(f"Best node found: avg_reward={best_node.average_reward():.2f}, visits={best_node.visits}")
            return best_node
        
        return root
    
    def get_best_path(self, node: MCTSNode) -> List[str]:
        """
        Extract the path from root to a given node.
        
        Args:
            node: Target node
            
        Returns:
            List[str]: Path of states from root to node
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.state)
            current = current.parent
        
        return list(reversed(path))


def demonstrate_mcts():
    """Demonstration of MCTS for LLM reasoning."""
    print("="*60)
    print("Monte Carlo Tree Search for LLM Reasoning")
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
    problem = "What is 15 * 24?"
    
    print(f"Problem: {problem}")
    print()
    
    # Get initial answer
    print("Generating initial answer...")
    initial_answer = client.generate(problem, temperature=0.8, max_tokens=100)
    
    if not initial_answer:
        print("ERROR: Failed to generate initial answer")
        return
    
    print(f"Initial answer: {initial_answer}")
    print()
    
    # Run MCTS to refine the answer
    mcts = MCTS(client, exploration_constant=1.414)
    
    refinement_prompt = f"Original problem: {problem}\nProvide an improved or verified answer."
    
    best_node = mcts.search(
        initial_state=initial_answer,
        prompt=refinement_prompt,
        n_iterations=5,
        n_expansions=2
    )
    
    # Display results
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print()
    print(f"Best answer (avg_reward: {best_node.average_reward():.2f}):")
    print(best_node.state)
    print()
    
    # Show reasoning path
    path = mcts.get_best_path(best_node)
    print("Reasoning path:")
    for i, state in enumerate(path):
        print(f"\nStep {i}:")
        print(f"  {state[:200]}...")  # Truncate long states
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        "problem": problem,
        "initial_answer": initial_answer,
        "best_answer": best_node.state,
        "best_reward": best_node.average_reward(),
        "visits": best_node.visits,
        "path": path
    }
    
    with open(results_dir / "mcts_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Results saved to: results/mcts_results.json")


if __name__ == "__main__":
    np.random.seed(42)
    demonstrate_mcts()
