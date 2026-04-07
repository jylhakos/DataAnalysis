"""
Embeddings Demonstration

This script demonstrates how word embeddings work and how to
compute similarity between vectors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class SimpleEmbedding:
    """
    A simple word embedding system for demonstration purposes.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
    
    def get_embedding(self, word_id: int) -> torch.Tensor:
        """Get embedding for a word ID."""
        return self.embedding(torch.tensor([word_id]))[0]
    
    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        similarity = (a · b) / (|a| * |b|)
        """
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        similarity = dot_product / (norm1 * norm2)
        return similarity.item()
    
    def euclidean_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        distance = sqrt(sum((a_i - b_i)^2))
        """
        distance = torch.dist(vec1, vec2, p=2)
        return distance.item()


def demonstrate_embeddings():
    """
    Demonstrate how embeddings work.
    """
    print("=" * 60)
    print("WORD EMBEDDINGS DEMONSTRATION")
    print("=" * 60)
    
    # Create simple vocabulary
    vocab = {
        "king": 0,
        "queen": 1,
        "man": 2,
        "woman": 3,
        "cat": 4,
        "dog": 5,
        "paris": 6,
        "france": 7
    }
    
    vocab_size = len(vocab)
    embedding_dim = 50
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Words: {list(vocab.keys())}")
    
    # Create embedding system
    emb_system = SimpleEmbedding(vocab_size, embedding_dim)
    
    # Get embeddings for some words
    king_emb = emb_system.get_embedding(vocab["king"])
    queen_emb = emb_system.get_embedding(vocab["queen"])
    man_emb = emb_system.get_embedding(vocab["man"])
    woman_emb = emb_system.get_embedding(vocab["woman"])
    
    print("\n" + "-" * 60)
    print("Sample embedding vector for 'king' (first 10 dimensions):")
    print(king_emb[:10].detach().numpy())
    
    print("\nEmbedding properties:")
    print(f"- Each word is represented as a {embedding_dim}-dimensional vector")
    print(f"- Vector values are continuous (not just 0s and 1s)")
    print(f"- Similar words have similar vectors")


def demonstrate_similarity():
    """
    Demonstrate similarity computation between word vectors.
    """
    print("\n" + "=" * 60)
    print("VECTOR SIMILARITY DEMONSTRATION")
    print("=" * 60)
    
    # Create vocabulary
    vocab = {
        "king": 0,
        "queen": 1,
        "man": 2,
        "woman": 3,
        "cat": 4,
        "dog": 5,
        "paris": 6,
        "london": 7
    }
    
    # Create embeddings
    torch.manual_seed(42)
    emb_system = SimpleEmbedding(len(vocab), 100)
    
    # Define word pairs to compare
    word_pairs = [
        ("king", "queen"),
        ("king", "man"),
        ("cat", "dog"),
        ("cat", "paris"),
        ("paris", "london"),
        ("man", "woman")
    ]
    
    print("\nCosine Similarity (higher = more similar):")
    print("-" * 60)
    
    for word1, word2 in word_pairs:
        emb1 = emb_system.get_embedding(vocab[word1])
        emb2 = emb_system.get_embedding(vocab[word2])
        
        similarity = emb_system.cosine_similarity(emb1, emb2)
        
        # Create visual bar
        bar_length = int(abs(similarity) * 30)
        bar = "█" * bar_length
        
        print(f"{word1:8s} - {word2:8s}: {similarity:6.3f} {bar}")
    
    print("\nInterpretation:")
    print("- Values close to 1.0 indicate high similarity")
    print("- Values close to 0.0 indicate low similarity")
    print("- Negative values indicate opposite meanings")


def demonstrate_vector_arithmetic():
    """
    Demonstrate vector arithmetic: king - man + woman ≈ queen
    """
    print("\n" + "=" * 60)
    print("VECTOR ARITHMETIC DEMONSTRATION")
    print("=" * 60)
    
    print("\nThe famous example: king - man + woman ≈ queen")
    print("-" * 60)
    
    # Create vocabulary
    vocab = {
        "king": 0,
        "queen": 1,
        "man": 2,
        "woman": 3,
        "prince": 4,
        "princess": 5
    }
    
    # For demonstration, create embeddings with specific properties
    # In practice, these would be learned from data
    torch.manual_seed(123)
    emb_system = SimpleEmbedding(len(vocab), 50)
    
    # Get embeddings
    king = emb_system.get_embedding(vocab["king"])
    queen = emb_system.get_embedding(vocab["queen"])
    man = emb_system.get_embedding(vocab["man"])
    woman = emb_system.get_embedding(vocab["woman"])
    
    # Perform vector arithmetic
    result = king - man + woman
    
    print("\nVector operation: king - man + woman")
    print("\nComparing result to all words:")
    print("-" * 60)
    
    # Compare result to all words
    similarities = {}
    for word, idx in vocab.items():
        word_emb = emb_system.get_embedding(idx)
        sim = emb_system.cosine_similarity(result, word_emb)
        similarities[word] = sim
    
    # Sort by similarity
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    for word, sim in sorted_words:
        bar = "█" * int(sim * 40)
        print(f"{word:10s}: {sim:6.3f} {bar}")
    
    print("\nNote: In real trained embeddings, 'queen' would rank highest")
    print("This demonstrates that embeddings capture semantic relationships")


def demonstrate_one_hot_vs_dense():
    """
    Compare one-hot encoding vs dense embeddings.
    """
    print("\n" + "=" * 60)
    print("ONE-HOT ENCODING vs DENSE EMBEDDINGS")
    print("=" * 60)
    
    vocab = ["cat", "dog", "bird", "fish"]
    vocab_size = len(vocab)
    
    print(f"\nVocabulary: {vocab}")
    print(f"Vocabulary size: {vocab_size}")
    
    # One-hot encoding
    print("\n" + "-" * 60)
    print("ONE-HOT ENCODING:")
    print("-" * 60)
    
    for i, word in enumerate(vocab):
        one_hot = torch.zeros(vocab_size)
        one_hot[i] = 1
        print(f"{word:6s}: {one_hot.numpy()}")
    
    print("\nProperties:")
    print("- Dimension: 4 (same as vocabulary size)")
    print("- Sparse: Only one 1, rest are 0s")
    print("- No semantic meaning")
    print("- All words equally distant from each other")
    
    # Dense embedding
    print("\n" + "-" * 60)
    print("DENSE EMBEDDING (8 dimensions):")
    print("-" * 60)
    
    torch.manual_seed(42)
    embedding = nn.Embedding(vocab_size, 8)
    
    for i, word in enumerate(vocab):
        dense_emb = embedding(torch.tensor([i]))[0]
        print(f"{word:6s}: [{', '.join([f'{x:.2f}' for x in dense_emb[:5].detach().numpy()])}...]")
    
    print("\nProperties:")
    print("- Dimension: 8 (can be any size < vocabulary)")
    print("- Dense: All values non-zero")
    print("- Captures semantic meaning")
    print("- Similar words have similar vectors")
    
    # Calculate similarity between cat and dog
    cat_emb = embedding(torch.tensor([0]))
    dog_emb = embedding(torch.tensor([1]))
    
    similarity = torch.cosine_similarity(cat_emb, dog_emb, dim=1)[0].item()
    
    print(f"\nCosine similarity between 'cat' and 'dog': {similarity:.3f}")


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("WORD EMBEDDINGS AND VECTOR REPRESENTATIONS")
    print("Understanding how words become vectors")
    print("*" * 60)
    
    # Run demonstrations
    demonstrate_embeddings()
    demonstrate_similarity()
    demonstrate_vector_arithmetic()
    demonstrate_one_hot_vs_dense()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nKey Concepts:")
    print("1. Embeddings convert words to dense vectors")
    print("2. Similar words have similar vectors")
    print("3. Vector arithmetic captures semantic relationships")
    print("4. Dense embeddings are more efficient than one-hot encoding")
    print()
