"""
Naive Bayes Classifier for Spam Filtering
==========================================

This module implements a Naive Bayes classifier from scratch for email spam detection.
It demonstrates the practical application of Bayes' theorem in text classification.

Key Concepts:
- Bayes' Theorem: P(Spam|Words) = P(Words|Spam) * P(Spam) / P(Words)
- Naive Assumption: All words are conditionally independent
- Laplace Smoothing: Handles words not seen during training

Author: Data Analysis with Bayesian Methods
Date: 2026
"""

from typing import Set, List, Tuple, Dict, Iterable, NamedTuple
import re
import math
from collections import defaultdict

# ============================================================================
# SECTION 1: Text Tokenization
# ============================================================================

def tokenize(text: str) -> Set[str]:
    """
    Convert text into a set of unique lowercase words.
    
    Args:
        text (str): Input text to tokenize
    
    Returns:
        Set[str]: Set of unique lowercase words
    
    Example:
        >>> tokenize("Data Science is science")
        {"data", "science", "is"}
    """
    text = text.lower()                         # Convert to lowercase
    all_words = re.findall("[a-z0-9']+", text)  # Extract alphanumeric words
    return set(all_words)                       # Remove duplicates

# Test tokenization
assert tokenize("Data Science is science") == {"data", "science", "is"}

# ============================================================================
# SECTION 2: Data Structures
# ============================================================================

class Message(NamedTuple):
    """
    Represents a training or test message.
    
    Attributes:
        text (str): The message content
        is_spam (bool): True if spam, False if ham (legitimate)
    """
    text: str
    is_spam: bool

# ============================================================================
# SECTION 3: Naive Bayes Classifier Implementation
# ============================================================================

class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for binary text classification (spam vs ham).
    
    The classifier uses:
    - Laplace (additive) smoothing to handle unseen words
    - Log probabilities to prevent numerical underflow
    - Conditional independence assumption between words
    
    Attributes:
        k (float): Smoothing factor (default 0.5)
        tokens (Set[str]): Vocabulary of all unique words seen during training
        token_spam_counts (Dict[str, int]): Word counts in spam messages
        token_ham_counts (Dict[str, int]): Word counts in ham messages
        spam_messages (int): Total number of spam messages
        ham_messages (int): Total number of ham messages
    """
    
    def __init__(self, k: float = 0.5) -> None:
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            k (float): Laplace smoothing parameter (default 0.5)
                      Prevents zero probabilities for unseen words
        """
        self.k = k  # Smoothing factor for Laplace smoothing

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        """
        Train the classifier on a collection of labeled messages.
        
        This method:
        1. Counts spam vs ham messages (for prior probabilities)
        2. Builds vocabulary from all messages
        3. Counts word occurrences in spam and ham (for likelihoods)
        
        Args:
            messages (Iterable[Message]): Training data with labels
        """
        for message in messages:
            # Increment message counts for prior probabilities
            # P(Spam) = spam_messages / total_messages
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Build vocabulary and count word occurrences
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """
        Calculate conditional probabilities of a word given spam/ham.
        
        Uses Laplace smoothing to handle unseen words:
        P(word|spam) = (count_in_spam + k) / (total_spam + 2*k)
        
        Args:
            token (str): A word from the vocabulary
        
        Returns:
            Tuple[float, float]: (P(token|spam), P(token|ham))
        """
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        # Laplace smoothing prevents zero probabilities
        # Adding k to numerator and 2*k to denominator
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        """
        Predict the probability that a message is spam.
        
        Uses Bayes' theorem with the naive independence assumption:
        P(Spam|Words) ∝ P(Spam) * ∏ P(Word|Spam)
        
        Log probabilities are used to prevent numerical underflow.
        
        Args:
            text (str): The message to classify
        
        Returns:
            float: Probability that the message is spam (0.0 to 1.0)
                  Values > 0.5 typically indicate spam
        """
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary
        # Calculate P(Message|Spam) and P(Message|Ham)
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If word appears in the message:
            # Multiply by P(word|class)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # If word doesn't appear:
            # Multiply by P(not word|class) = 1 - P(word|class)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        # Convert back from log space
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        
        # Return P(Spam|Message) using Bayes' theorem
        return prob_if_spam / (prob_if_spam + prob_if_ham)

# ============================================================================
# SECTION 4: Testing and Validation
# ============================================================================

def run_basic_tests():
    """Run basic unit tests to verify the classifier works correctly."""
    
    print("=" * 70)
    print("Running Basic Tests")
    print("=" * 70)
    
    # Create simple training data
    messages = [
        Message("spam rules", is_spam=True),
        Message("ham rules", is_spam=False),
        Message("hello ham", is_spam=False)
    ]

    # Initialize and train model
    model = NaiveBayesClassifier(k=0.5)
    model.train(messages)

    # Verify training results
    assert model.tokens == {"spam", "ham", "rules", "hello"}
    assert model.spam_messages == 1
    assert model.ham_messages == 2
    assert model.token_spam_counts == {"spam": 1, "rules": 1}
    assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}
    
    print("\nVocabulary correctly built")
    print(f"  Tokens: {model.tokens}")
    print(f"  Spam messages: {model.spam_messages}")
    print(f"  Ham messages: {model.ham_messages}")
    
    return model

def test_prediction(model: NaiveBayesClassifier):
    """Test the prediction functionality with manual calculation."""
    
    print("\n" + "=" * 70)
    print("Testing Prediction")
    print("=" * 70)
    
    text = "hello spam"
    
    # Manual calculation of probabilities for verification
    probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
        1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
        1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
        (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
    ]

    probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
        1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
        1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
        (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
    ]

    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))
    expected_prob = p_if_spam / (p_if_spam + p_if_ham)
    
    # Test prediction
    predicted_prob = model.predict(text)
    
    print(f"\n  Test message: '{text}'")
    print(f"  Predicted spam probability: {predicted_prob:.4f}")
    print(f"  Expected probability: {expected_prob:.4f}")
    
    assert abs(predicted_prob - expected_prob) < 0.0001
    print("\nPrediction test passed!")

# ============================================================================
# SECTION 5: Realistic Spam Filtering Example
# ============================================================================

def create_spam_dataset() -> List[Message]:
    """
    Create a realistic email spam dataset for demonstration.
    
    Returns:
        List[Message]: Training data with spam and ham examples
    """
    return [
        # Spam messages - typical spam patterns
        Message("Winner! You've won a FREE prize! Click here NOW!", is_spam=True),
        Message("Get rich quick! Make money fast from home!", is_spam=True),
        Message("URGENT: Claim your FREE gift card today!", is_spam=True),
        Message("Congratulations! You won the lottery! Send payment details!", is_spam=True),
        Message("Limited time offer! Buy now and save! Act fast!", is_spam=True),
        Message("Free credit check! Click here for your report!", is_spam=True),
        Message("Earn money online! Work from home! Easy money!", is_spam=True),
        Message("You have been selected! Claim your prize now!", is_spam=True),
        Message("Lowest prices guaranteed! Buy pills online cheap!", is_spam=True),
        Message("Click here to unsubscribe from winning notifications", is_spam=True),
        
        # Ham messages - legitimate emails
        Message("Meeting scheduled for tomorrow at 2pm in room 301", is_spam=False),
        Message("Project deadline reminder: submission due Friday", is_spam=False),
        Message("Team lunch next week, please confirm attendance", is_spam=False),
        Message("Code review requested for pull request 42", is_spam=False),
        Message("Quarterly report attached for your review", is_spam=False),
        Message("Can we reschedule our meeting to next Monday?", is_spam=False),
        Message("Documentation update completed for the API", is_spam=False),
        Message("Thank you for your presentation yesterday", is_spam=False),
        Message("Please review the updated project timeline", is_spam=False),
        Message("Conference call scheduled for next Thursday", is_spam=False),
        Message("New employee orientation scheduled for next week", is_spam=False),
        Message("Server maintenance scheduled for this weekend", is_spam=False),
        Message("Monthly team meeting agenda attached", is_spam=False),
        Message("Thanks for helping with the debugging session", is_spam=False),
        Message("Please update your contact information in the system", is_spam=False),
    ]

def demonstrate_spam_filtering():
    """Demonstrate the spam filter on realistic email data."""
    
    print("\n" + "=" * 70)
    print("Realistic Spam Filtering Demonstration")
    print("=" * 70)
    
    # Create dataset
    messages = create_spam_dataset()
    
    print(f"\nDataset: {len(messages)} emails")
    spam_count = sum(1 for m in messages if m.is_spam)
    ham_count = len(messages) - spam_count
    print(f"  Spam: {spam_count} emails")
    print(f"  Ham: {ham_count} emails")
    
    # Train model
    model = NaiveBayesClassifier(k=1.0)
    model.train(messages)
    
    print(f"\nModel trained!")
    print(f"  Vocabulary size: {len(model.tokens)} unique words")
    
    # Test on new messages
    test_messages = [
        "Congratulations! You won a free prize!",
        "Let's schedule a meeting for tomorrow",
        "Click here for amazing offers and prizes",
        "Please review the project documentation",
        "URGENT: Free money! Act now!",
        "Team standup at 10am today"
    ]
    
    print("\n" + "-" * 70)
    print("Testing on new messages:")
    print("-" * 70)
    
    for msg in test_messages:
        prob = model.predict(msg)
        classification = "SPAM" if prob > 0.5 else "HAM"
        confidence = prob if prob > 0.5 else (1 - prob)
        
        print(f"\nMessage: '{msg}'")
        print(f"  Classification: {classification}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Spam probability: {prob:.4f}")

def analyze_spammiest_words(model: NaiveBayesClassifier, n: int = 10):
    """Analyze which words are most indicative of spam."""
    
    print("\n" + "=" * 70)
    print("Word Analysis: Spam Indicators")
    print("=" * 70)
    
    def spam_probability_for_word(token: str) -> float:
        """Calculate P(Spam|word) for a single word."""
        prob_if_spam, prob_if_ham = model._probabilities(token)
        return prob_if_spam / (prob_if_spam + prob_if_ham)
    
    # Sort words by their spam probability
    sorted_words = sorted(
        model.tokens,
        key=lambda t: spam_probability_for_word(t),
        reverse=True
    )
    
    print(f"\nTop {n} Spammiest Words:")
    for i, word in enumerate(sorted_words[:n], 1):
        prob = spam_probability_for_word(word)
        print(f"  {i:2d}. '{word}' - {prob:.2%} spam probability")
    
    print(f"\nTop {n} Hammiest Words:")
    for i, word in enumerate(sorted_words[-n:], 1):
        prob = spam_probability_for_word(word)
        print(f"  {i:2d}. '{word}' - {prob:.2%} spam probability")

# ============================================================================
# SECTION 6: Main Execution
# ============================================================================

def main():
    """
    Main execution function demonstrating all features of the classifier.
    """
    
    print("\n" + "=" * 70)
    print(" NAIVE BAYES SPAM CLASSIFIER DEMONSTRATION")
    print("=" * 70)
    print("\nBased on Bayes' Theorem:")
    print("  P(Spam|Words) = P(Words|Spam) × P(Spam) / P(Words)")
    print("\nUsing the naive assumption that all words are independent.")
    
    # Run basic tests
    basic_model = run_basic_tests()
    test_prediction(basic_model)
    
    # Run realistic spam filtering demo
    demonstrate_spam_filtering()
    
    # Analyze spam indicators
    spam_messages = create_spam_dataset()
    analysis_model = NaiveBayesClassifier(k=1.0)
    analysis_model.train(spam_messages)
    analyze_spammiest_words(analysis_model)
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print("\nThe classifier demonstrates:")
    print("  - Bayes' theorem application")
    print("  - Laplace smoothing for unseen words")
    print("  - Log probabilities to prevent underflow")
    print("  - Realistic spam filtering capability")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
