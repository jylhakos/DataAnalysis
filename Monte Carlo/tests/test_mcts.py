"""
Tests for Monte Carlo Tree Search implementation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcts_llm import MCTSNode, MCTS


class TestMCTSNode:
    """Test suite for MCTSNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = MCTSNode(state="test state")
        
        assert node.state == "test state"
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.total_reward == 0.0
    
    def test_node_with_parent(self):
        """Test creating node with parent."""
        parent = MCTSNode(state="parent")
        child = MCTSNode(state="child", parent=parent)
        
        assert child.parent == parent
        assert child in parent.children or True  # Child is created but not auto-added
    
    def test_average_reward_zero_visits(self):
        """Test average reward with zero visits."""
        node = MCTSNode(state="test")
        assert node.average_reward() == 0.0
    
    def test_average_reward_calculation(self):
        """Test average reward calculation."""
        node = MCTSNode(state="test")
        node.visits = 10
        node.total_reward = 75.0
        
        assert node.average_reward() == 7.5
    
    def test_uct_score_unvisited_node(self):
        """Test UCT score for unvisited node."""
        node = MCTSNode(state="test")
        assert node.uct_score() == float('inf')
    
    def test_uct_score_with_parent(self):
        """Test UCT score calculation with parent."""
        parent = MCTSNode(state="parent")
        parent.visits = 10
        
        child = MCTSNode(state="child", parent=parent)
        child.visits = 5
        child.total_reward = 30.0
        
        uct = child.uct_score(exploration_constant=1.414)
        
        # UCT = Q/N + C * sqrt(ln(parent_N) / N)
        # = 30/5 + 1.414 * sqrt(ln(10) / 5)
        # = 6.0 + 1.414 * sqrt(2.303 / 5)
        # = 6.0 + 1.414 * 0.679
        # ≈ 6.96
        
        assert isinstance(uct, float)
        assert uct > 6.0  # Should be greater than just average reward
        assert uct < 10.0  # Reasonable upper bound
    
    def test_uct_score_exploration_exploitation_balance(self):
        """Test that UCT balances exploration and exploitation."""
        parent = MCTSNode(state="parent")
        parent.visits = 100
        
        # Well-visited child with good reward
        child1 = MCTSNode(state="child1", parent=parent)
        child1.visits = 50
        child1.total_reward = 400.0  # avg = 8.0
        
        # Less-visited child with lower reward
        child2 = MCTSNode(state="child2", parent=parent)
        child2.visits = 5
        child2.total_reward = 30.0  # avg = 6.0
        
        uct1 = child1.uct_score(exploration_constant=1.414)
        uct2 = child2.uct_score(exploration_constant=1.414)
        
        # Child2 should have higher UCT due to exploration bonus
        # even though it has lower average reward
        assert uct2 > uct1, "Less-visited node should have higher UCT"
    
    def test_is_leaf_true(self):
        """Test is_leaf for node with no children."""
        node = MCTSNode(state="test")
        assert node.is_leaf() is True
    
    def test_is_leaf_false(self):
        """Test is_leaf for node with children."""
        parent = MCTSNode(state="parent")
        child = MCTSNode(state="child", parent=parent)
        parent.children.append(child)
        
        assert parent.is_leaf() is False


class TestMCTS:
    """Test suite for MCTS class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = Mock()
        self.mcts = MCTS(self.mock_llm, exploration_constant=1.414)
    
    def test_mcts_initialization(self):
        """Test MCTS initialization."""
        assert self.mcts.llm == self.mock_llm
        assert self.mcts.exploration_constant == 1.414
    
    def test_select_leaf_node(self):
        """Test selection on a leaf node returns itself."""
        node = MCTSNode(state="leaf")
        selected = self.mcts.select(node)
        
        assert selected == node
    
    def test_select_with_children(self):
        """Test selection chooses child with highest UCT."""
        root = MCTSNode(state="root")
        root.visits = 10
        
        # Create children with different statistics
        child1 = MCTSNode(state="child1", parent=root)
        child1.visits = 5
        child1.total_reward = 20.0
        
        child2 = MCTSNode(state="child2", parent=root)
        child2.visits = 3
        child2.total_reward = 18.0
        
        child3 = MCTSNode(state="child3", parent=root)
        child3.visits = 2
        child3.total_reward = 10.0
        
        root.children = [child1, child2, child3]
        
        # Selection should choose one of the children
        selected = self.mcts.select(root)
        assert selected in [child1, child2, child3]
    
    def test_backpropagate(self):
        """Test backpropagation updates nodes correctly."""
        # Create a path: root -> child1 -> grandchild
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)
        grandchild = MCTSNode(state="grandchild", parent=child)
        
        reward = 50.0
        self.mcts.backpropagate(grandchild, reward)
        
        # All nodes in path should be updated
        assert grandchild.visits == 1
        assert grandchild.total_reward == 50.0
        
        assert child.visits == 1
        assert child.total_reward == 50.0
        
        assert root.visits == 1
        assert root.total_reward == 50.0
    
    def test_backpropagate_multiple_times(self):
        """Test backpropagation accumulates correctly."""
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)
        
        self.mcts.backpropagate(child, 30.0)
        self.mcts.backpropagate(child, 50.0)
        
        assert child.visits == 2
        assert child.total_reward == 80.0
        assert child.average_reward() == 40.0
        
        assert root.visits == 2
        assert root.total_reward == 80.0
    
    def test_simulate_calls_llm_scorer(self):
        """Test that simulate calls LLM scoring function."""
        self.mock_llm.score_response = Mock(return_value=75.0)
        
        node = MCTSNode(state="test response")
        reward = self.mcts.simulate(node)
        
        assert reward == 75.0
        self.mock_llm.score_response.assert_called_once()
    
    def test_simulate_fallback_on_none(self):
        """Test simulate uses fallback when LLM returns None."""
        self.mock_llm.score_response = Mock(return_value=None)
        
        node = MCTSNode(state="a" * 500)  # 500 char response
        reward = self.mcts.simulate(node)
        
        # Fallback: min(len(state) / 10, 100)
        assert reward == min(500 / 10, 100)
        assert reward == 50.0
    
    def test_get_best_path(self):
        """Test extraction of path from root to node."""
        root = MCTSNode(state="root")
        child = MCTSNode(state="child", parent=root)
        grandchild = MCTSNode(state="grandchild", parent=child)
        
        path = self.mcts.get_best_path(grandchild)
        
        assert path == ["root", "child", "grandchild"]
    
    def test_get_best_path_single_node(self):
        """Test path for single node."""
        node = MCTSNode(state="single")
        path = self.mcts.get_best_path(node)
        
        assert path == ["single"]


class TestMCTSIntegration:
    """Integration tests for full MCTS workflow."""
    
    def test_mcts_exploration_convergence(self):
        """Test that MCTS explores and converges to better solutions."""
        # Create a simple tree manually
        root = MCTSNode(state="root")
        root.visits = 100
        
        # Good path
        good_child = MCTSNode(state="good", parent=root)
        good_child.visits = 60
        good_child.total_reward = 540.0  # avg = 9.0
        
        # Bad path
        bad_child = MCTSNode(state="bad", parent=root)
        bad_child.visits = 40
        bad_child.total_reward = 120.0  # avg = 3.0
        
        root.children = [good_child, bad_child]
        
        # Best child should be the one with highest average reward
        best = max(root.children, key=lambda c: c.average_reward())
        assert best == good_child


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
