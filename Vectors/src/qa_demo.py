"""
Question Answering Demonstration

This script demonstrates a simple question-answering system using
transformer models. It shows how vectors and attention mechanisms
enable the model to answer questions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import warnings

warnings.filterwarnings('ignore')


class SimpleQASystem:
    """
    A simple question-answering system using pre-trained transformers.
    """
    
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad'):
        """
        Initialize the QA system with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        print(f"Loading model: {model_name}")
        print("This may take a moment...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        print("Model loaded successfully!\n")
    
    def answer_question(self, question, context):
        """
        Answer a question based on the given context.
        
        Args:
            question: The question to answer
            context: The context containing the answer
            
        Returns:
            answer: The extracted answer
            confidence: Confidence score
        """
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get answer span
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Extract answer
        answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence
        start_score = torch.max(outputs.start_logits).item()
        end_score = torch.max(outputs.end_logits).item()
        confidence = (start_score + end_score) / 2
        
        return answer, confidence


def demo_simple_qa():
    """
    Demonstrate simple question answering.
    """
    print("=" * 60)
    print("SIMPLE QUESTION ANSWERING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize QA system
    qa_system = SimpleQASystem()
    
    # Test question about France
    question1 = "What is the capital of France?"
    context1 = "France is a country in Western Europe. Paris is the capital and most populous city of France."
    
    print(f"Question: {question1}")
    print(f"Context: {context1}")
    print("\nProcessing...")
    
    answer1, confidence1 = qa_system.answer_question(question1, context1)
    
    print(f"\nAnswer: {answer1}")
    print(f"Confidence: {confidence1:.2f}")
    
    # Test more questions
    print("\n" + "-" * 60)
    
    test_cases = [
        {
            "question": "Where is France located?",
            "context": "France is a country in Western Europe. Paris is the capital and most populous city of France."
        },
        {
            "question": "What is the most populous city?",
            "context": "France is a country in Western Europe. Paris is the capital and most populous city of France."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young star-crossed lovers."
        },
        {
            "question": "What type of work is Romeo and Juliet?",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young star-crossed lovers."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Question: {test['question']}")
        print(f"Context: {test['context']}")
        
        answer, confidence = qa_system.answer_question(
            test['question'],
            test['context']
        )
        
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 60)


def demonstrate_vector_representation():
    """
    Show how questions and context are represented as vectors.
    """
    print("\n" + "=" * 60)
    print("VECTOR REPRESENTATION DEMONSTRATION")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Example question
    question = "What is the capital of France?"
    
    print(f"\nOriginal question: '{question}'")
    
    # Tokenize
    tokens = tokenizer.tokenize(question)
    print(f"\nTokens: {tokens}")
    
    # Convert to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")
    
    print("\nExplanation:")
    print("- Each word/subword is converted to a token")
    print("- Each token is mapped to a unique ID (integer)")
    print("- These IDs are used to look up embedding vectors")
    print("- The embeddings are high-dimensional vectors (e.g., 768 dimensions)")
    print("- These vectors capture semantic meaning")


def demonstrate_attention_in_qa():
    """
    Explain how attention helps in question answering.
    """
    print("\n" + "=" * 60)
    print("HOW ATTENTION ENABLES QUESTION ANSWERING")
    print("=" * 60)
    
    question = "What is the capital of France?"
    context = "France is in Europe. Paris is the capital of France."
    
    print(f"\nQuestion: {question}")
    print(f"Context: {context}")
    
    print("\nHow the model processes this:")
    print("-" * 60)
    
    print("\n1. EMBEDDING PHASE:")
    print("   - Convert question tokens to vectors: [What, is, capital, France]")
    print("   - Convert context tokens to vectors: [France, Europe, Paris, capital, ...]")
    
    print("\n2. ATTENTION PHASE:")
    print("   - Query vectors from question attend to context")
    print("   - 'capital' in question attends strongly to 'Paris' and 'capital' in context")
    print("   - 'France' in question attends to 'France' in context")
    
    print("\n3. ATTENTION WEIGHTS (Conceptual):")
    print("   When processing 'capital of France':")
    print("     - 'Paris': 0.65 (high attention)")
    print("     - 'capital': 0.20")
    print("     - 'France': 0.10")
    print("     - 'Europe': 0.05 (low attention)")
    
    print("\n4. ANSWER EXTRACTION:")
    print("   - Model identifies start token: 'Paris'")
    print("   - Model identifies end token: 'Paris'")
    print("   - Extracted answer: 'Paris'")
    
    print("\nKey Insight:")
    print("The attention mechanism allows the model to focus on relevant")
    print("parts of the context that contain the answer to the question.")


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "*" * 60)
    print("QUESTION ANSWERING WITH TRANSFORMERS")
    print("Demonstrating how attention and vectors enable Q&A")
    print("*" * 60)
    
    try:
        # Run demonstrations
        demo_simple_qa()
        demonstrate_vector_representation()
        demonstrate_attention_in_qa()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        
        print("\nSummary:")
        print("1. Questions and context are converted to vectors (embeddings)")
        print("2. Attention mechanism identifies relevant context")
        print("3. Model extracts answer span from context")
        print("4. This demonstrates practical application of transformers")
        print()
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nNote: This demo requires the transformers library.")
        print("Install with: pip install transformers")
        print()


if __name__ == "__main__":
    main()
