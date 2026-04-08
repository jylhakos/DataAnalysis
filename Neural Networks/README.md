# Neural Networks: An Implementation of Transformer Architecture for Large Language Models

> **Quick Links:**
> - [Getting Started](#development-environment-setup) - Quick setup guide

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
   - 2.1 [Neural Networks Fundamentals](#neural-networks-fundamentals)
   - 2.2 [Large Language Models](#large-language-models)
   - 2.3 [Motivation](#motivation)
3. [Mathematical Foundations](#mathematical-foundations)
   - 3.1 [Linear Algebra](#linear-algebra)
   - 3.2 [Calculus and Optimization](#calculus-and-optimization)
   - 3.3 [Probability Theory](#probability-theory)
4. [Transformer Architecture](#transformer-architecture)
   - 4.1 [Self-Attention Mechanism](#self-attention-mechanism)
   - 4.2 [Multi-Head Attention](#multi-head-attention)
   - 4.3 [Positional Encoding](#positional-encoding)
   - 4.4 [Encoder Architecture](#encoder-architecture)
   - 4.5 [Decoder Architecture](#decoder-architecture)
   - 4.6 [Feed-Forward Networks](#feed-forward-networks)
5. [Activation Functions](#activation-functions)
6. [Loss Functions and Training](#loss-functions-and-training)
7. [Implementation Details](#implementation-details)
8. [Project Structure](#project-structure)
9. [Development Environment Setup](#development-environment-setup)
10. [Usage and Workflow](#usage-and-workflow)
11. [DevOps and Deployment](#devops-and-deployment)
12. [Inference and Testing](#inference-and-testing)
13. [Frequently Asked Questions](#frequently-asked-questions)
14. [References](#references)
15. [License](#license)

---

## Abstract

This work presents an implementation of the Transformer architecture for Large Language Models (LLMs), built from scratch using Python libraries including NumPy and PyTorch. Following the work "Attention Is All You Need" (Vaswani et al., 2017), we implement a complete encoder-decoder Transformer model with multi-head self-attention mechanisms. This implementation serves as an educational resource for understanding the mathematical foundations and architectural components that power modern LLMs such as GPT, BERT, and other transformer-based models.

## Introduction

### Neural Networks Fundamentals

Neural networks are computational systems inspired by biological neural structures, consisting of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts during learning. The fundamental operation of a neural network can be expressed as:

$$
\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})
$$

where:
- $\mathbf{x}$ is the input vector
- $\mathbf{W}$ is the weight matrix
- $\mathbf{b}$ is the bias vector
- $f$ is a non-linear activation function
- $\mathbf{y}$ is the output vector

Neural networks learn through backpropagation, adjusting weights based on prediction errors to minimize a loss function. Deep neural networks contain multiple hidden layers, each capable of learning increasingly abstract representations of the input data.

### Large Language Models

Large Language Models (LLMs) are specialized deep neural networks designed to process, understand, and generate human-like text. They are typically based on the Transformer architecture and can contain billions of parameters. LLMs convert text into numerical representations (embeddings) and process these through multiple layers that calculate relationships between words using self-attention mechanisms.

Key characteristics of LLMs include:

1. **Massive Scale**: Billions to trillions of parameters
2. **Transformer Architecture**: Parallel processing via self-attention
3. **Self-Supervised Learning**: Pre-training on vast text corpora
4. **Transfer Learning**: Fine-tuning for specific tasks
5. **Emergent Capabilities**: Few-shot learning and reasoning

### Motivation

This implementation aims to demystify the Transformer architecture by providing a complete, working implementation from first principles. Unlike black-box implementations, this project exposes the mathematical foundations and architectural decisions that enable LLMs to achieve remarkable language understanding and generation capabilities.

## Mathematical Foundations

### Linear Algebra

Neural networks heavily rely on linear algebra operations. Data is represented as tensors (multi-dimensional arrays), and transformations are performed via matrix operations.

**Matrix Multiplication**: The core operation in neural networks:

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
$$

where $C_{ij} = \sum_{k} A_{ik}B_{kj}$

**Vector Spaces**: Word embeddings map discrete tokens to continuous vector spaces where semantic relationships are preserved:

$$
\mathbf{e}_w \in \mathbb{R}^{d_{model}}
$$

### Calculus and Optimization

Training neural networks requires computing gradients of the loss function with respect to all parameters.

**Gradient Descent**: Iterative optimization algorithm:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

where:
- $\theta$ represents model parameters
- $\eta$ is the learning rate
- $\mathcal{L}$ is the loss function
- $\nabla_\theta \mathcal{L}$ is the gradient

**Backpropagation**: Uses the chain rule to compute gradients efficiently:

$$
\frac{\partial \mathcal{L}}{\partial w_i} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial w_i}
$$

**Adam Optimizer**: Adaptive learning rate method combining momentum and RMSprop:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

### Probability Theory

Neural networks use probability distributions for uncertainty estimation and regularization.

**Softmax Function**: Converts logits to probability distribution:

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Cross-Entropy Loss**: Measures the difference between predicted and true distributions:

$$
\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

## Transformer Architecture

The Transformer architecture, introduced in "Attention Is All You Need" (2017), revolutionized sequence modeling by replacing recurrent connections with self-attention mechanisms, enabling parallel processing and better capture of long-range dependencies.

### Self-Attention Mechanism

Self-attention allows each position in a sequence to attend to all positions in the previous layer. For an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$, we compute three matrices:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V
$$

where $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d_{model} \times d_k}$ are learned projection matrices.

**Scaled Dot-Product Attention**:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from becoming too large, which would push the softmax into regions with extremely small gradients.

**Attention Scores**: The softmax operation produces attention weights:

$$
\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{k=1}^{n} \exp(q_i \cdot k_k / \sqrt{d_k})}
$$

where $\alpha_{ij}$ represents how much position $i$ attends to position $j$.

### Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
$$

Parameters:
- $\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{model}}$

Typically, $d_k = d_v = d_{model}/h$ where $h$ is the number of heads.

### Positional Encoding

Since the Transformer has no recurrence or convolution, positional information must be injected. We use sinusoidal functions:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

where:
- $pos$ is the position
- $i$ is the dimension

This allows the model to easily learn to attend by relative positions, as for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

### Encoder Architecture

The encoder consists of $N$ identical layers. Each layer has two sub-layers:

1. **Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network**

Each sub-layer is followed by layer normalization and residual connection:

$$
\text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))
$$

**Layer Normalization**:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where $\mu$ and $\sigma^2$ are mean and variance computed across features, and $\gamma$, $\beta$ are learnable parameters.

### Decoder Architecture

The decoder also consists of $N$ identical layers. Each layer has three sub-layers:

1. **Masked Multi-Head Self-Attention**: Prevents positions from attending to subsequent positions
2. **Multi-Head Cross-Attention**: Attends to encoder outputs
3. **Position-wise Feed-Forward Network**

**Masked Attention**: Ensures predictions for position $i$ depend only on known outputs at positions less than $i$:

$$
\text{Mask}_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

Applied before softmax: $\text{softmax}(\mathbf{QK}^T/\sqrt{d_k} + \text{Mask})$

### Feed-Forward Networks

Each position's representation is processed independently through the same feed-forward network:

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

Parameters:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$

Typically $d_{ff} = 4 \times d_{model}$.

## Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(x) = \max(0, x)
$$

Advantages: Computationally efficient, mitigates vanishing gradient problem.

Gradient:

$$
\frac{d}{dx}\text{ReLU}(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

### GELU (Gaussian Error Linear Unit)

Often used in Transformers:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

Approximation:

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

### Softmax

Normalizes output to probability distribution:

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}
$$

Properties:
- Output values in $(0, 1)$
- Sum equals 1
- Differentiable

## Loss Functions and Training

### Cross-Entropy Loss

For language modeling, we use cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

where:
- $N$ is the number of samples
- $C$ is the number of classes (vocabulary size)
- $y_{ic}$ is the true label (one-hot encoded)
- $\hat{y}_{ic}$ is the predicted probability

### Label Smoothing

Prevents the model from becoming over-confident:

$$
y'_{ic} = (1 - \epsilon)y_{ic} + \frac{\epsilon}{C}
$$

where $\epsilon$ is the smoothing parameter (typically 0.1).

### Training Procedure

1. **Forward Pass**: Compute predictions
2. **Loss Computation**: Calculate cross-entropy loss
3. **Backward Pass**: Compute gradients via backpropagation
4. **Parameter Update**: Apply optimizer (e.g., Adam)
5. **Learning Rate Scheduling**: Warm-up and decay

**Learning Rate Schedule** (from original paper):

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})
$$

## Implementation Details

This implementation uses:

- **NumPy**: For basic tensor operations and educational clarity
- **PyTorch**: For efficient computation and automatic differentiation
- **Pure Python**: No high-level abstractions, exposing all mechanisms

### Model Hyperparameters

Default configuration (similar to Transformer-Base):

| Parameter | Value |
|-----------|-------|
| $d_{model}$ | 512 |
| $N$ (layers) | 6 |
| $h$ (attention heads) | 8 |
| $d_k = d_v$ | 64 |
| $d_{ff}$ | 2048 |
| Dropout | 0.1 |
| Vocabulary Size | 10000 |

### Computational Complexity

**Self-Attention**: $O(n^2 \cdot d)$ where $n$ is sequence length and $d$ is model dimension.

**Feed-Forward**: $O(n \cdot d^2)$

For typical values, self-attention dominates when sequences are long, but feed-forward layers contain more parameters.

## Project Structure

```
Neural Networks/
│
├── README.md                      # This file - Documentation
├── .gitignore                     # Git ignore rules for virtual env and binaries
├── requirements.txt               # Python dependencies
│
├── src/                          # Source code directory
│   ├── __init__.py
│   ├── transformer.py            # Main Transformer implementation
│   ├── attention.py              # Self-attention mechanisms
│   ├── encoder.py                # Encoder architecture
│   ├── decoder.py                # Decoder architecture
│   ├── embeddings.py             # Token and positional embeddings
│   ├── feedforward.py            # Position-wise feed-forward network
│   ├── utils.py                  # Utility functions
│   └── config.py                 # Model configuration
│
├── train.py                      # Training script
├── inference.py                  # Inference and text generation
├── server.py                     # Simple inference server
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   └── test_utils.py
│
├── examples/                     # Example usage scripts
│   ├── simple_training.py
│   └── interactive_demo.py
│
└── data/                         # Data directory (excluded from git)
    └── .gitkeep
```

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- VS Code (recommended)
- Linux environment
- Git

### Virtual Environment Setup

Creating and activating a virtual environment isolates dependencies and prevents conflicts with system packages.

#### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Neural\ Networks

# Create virtual environment
python3 -m venv venv
```

#### Step 2: Activate Virtual Environment

```bash
# On Linux/macOS
source venv/bin/activate

# Your prompt should now show (venv) prefix
```

#### Step 3: Verify Virtual Environment

```bash
# Check Python path (should point to venv)
which python

# Check pip path
which pip

# Verify Python version
python --version
```

#### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### VS Code Integration

1. Open Command Palette: `Ctrl+Shift+P`
2. Select: "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python`

VS Code will automatically activate the virtual environment in integrated terminals.

### Deactivating Virtual Environment

```bash
deactivate
```

## Usage and Workflow

### Training a Model

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run training with default configuration
python train.py

# With custom parameters
python train.py --epochs 10 --batch-size 32 --d-model 512 --num-layers 6
```

### Generating Text (Inference)

```bash
# Interactive mode
python inference.py --model-path checkpoints/model.pt --interactive

# Single prompt
python inference.py --model-path checkpoints/model.pt --prompt "Hello, how are"

# With temperature control
python inference.py --model-path checkpoints/model.pt --prompt "Once upon a time" --temperature 0.8
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_attention.py

# With verbose output
pytest -v tests/
```

### Starting Inference Server

```bash
# Start server on default port (5000)
python server.py --model-path checkpoints/model.pt

# Custom port
python server.py --model-path checkpoints/model.pt --port 8080
```

## DevOps and Deployment

## 🔧 Technical Specifications

### Model Architecture
```
Transformer Encoder-Decoder
├── Parameters: 9,375,624
├── Encoder Layers: 3
├── Decoder Layers: 3
├── Attention Heads: 4
├── Model Dimension: 256
├── FFN Dimension: 1024
└── Vocabulary Size: 5000
```

### Dependencies
```
✓ PyTorch 2.11.0+cu130 (CUDA)
✓ NumPy 2.4.4
✓ Pandas 3.0.2
✓ Flask 2.3.0
✓ pytest 9.0.3
✓ Jupyter 1.1.1
✓ tqdm, matplotlib, pandas
```
---

### How to Use

### 1. Train the Model
```bash
source venv/bin/activate
python train.py --epochs 10 --batch-size 32 --model-size small
```

### 2. Run Inference
```bash
python inference.py \
  --model-path checkpoints/best_model.pt \
  --prompt "Your text here" \
  --interactive
```

### 3. Start REST API Server
```bash
python server.py --model-path checkpoints/best_model.pt --port 5000
```

### 4. Test with curl
```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "max_length": 50,
    "temperature": 0.8,
    "top_k": 50
  }'
```

### File Locations
- **Source Code:** `./src/`
- **Scripts:** `./train.py`, `./inference.py`, `./server.py`
- **Examples:** `./examples/`
- **Tests:** `./tests/`
- **Checkpoints:** `./checkpoints/`
- **Virtual Env:** `./venv/`

### Key Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v

# Train model
python train.py --epochs 10 --batch-size 32

# Start server
python server.py --model-path checkpoints/best_model.pt

# Run inference
python inference.py --model-path checkpoints/best_model.pt --interactive
```

---

### Containerization with Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY server.py .
COPY checkpoints/ ./checkpoints/

EXPOSE 5000

CMD ["python", "server.py", "--model-path", "checkpoints/model.pt", "--host", "0.0.0.0"]
```

Build and run:

```bash
# Build image
docker build -t transformer-llm .

# Run container
docker run -p 5000:5000 transformer-llm
```

### Continuous Integration

Example `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt
      - name: Run tests
        run: |
          source venv/bin/activate
          pytest tests/
```

### Monitoring and Logging

The implementation includes logging for:
- Training progress (loss, accuracy, perplexity)
- Model checkpointing
- Inference latency
- Error tracking

Logs are written to `logs/` directory with timestamps.

### Scaling Considerations

1. **Distributed Training**: Use PyTorch DistributedDataParallel for multi-GPU training
2. **Model Parallelism**: Split large models across GPUs
3. **Gradient Accumulation**: Simulate larger batch sizes with limited memory
4. **Mixed Precision**: Use FP16 for faster training

## Inference and Testing

### Using the Inference Server

Once the server is running, test it with curl:

```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox",
    "max_length": 50,
    "temperature": 0.7
  }'

# Encode text to embeddings
curl -X POST http://localhost:5000/encode \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!"
  }'
```

### Testing with Python Requests

```python
import requests
import json

url = "http://localhost:5000/generate"
payload = {
    "prompt": "Neural networks are",
    "max_length": 100,
    "temperature": 0.8,
    "top_k": 50
}

response = requests.post(url, json=payload)
result = response.json()
print(result["generated_text"])
```

### Ollama Compatibility

While this implementation provides its own inference server, it can potentially be integrated with Ollama by:

1. **Converting Model Format**: Export the trained model to GGUF format
2. **Creating Modelfile**: Define architecture and parameters
3. **Importing to Ollama**: Use `ollama create` command

However, for full Ollama compatibility, the model should follow the exact architecture and quantization schemes that Ollama supports. The provided inference server is simpler and more directly integrated with this implementation.

### Performance Benchmarking

```bash
# Measure inference speed
python examples/benchmark.py --model-path checkpoints/model.pt --num-samples 100

# Profile memory usage
python -m memory_profiler inference.py --model-path checkpoints/model.pt
```

## Frequently Asked Questions

### Q1: What are Large Language Models?

Large Language Models are deep neural networks trained on vast amounts of text data to understand and generate human-like language. They typically use the Transformer architecture with billions of parameters, enabling tasks like text generation, translation, question-answering, and more.

### Q2: How do neural networks actually work?

Neural networks process information through layers of interconnected neurons. Each neuron applies a weighted sum to its inputs, adds a bias, and passes the result through an activation function. During training, these weights are adjusted via backpropagation to minimize prediction errors.

### Q3: What's the difference between LLMs and standard neural networks?

LLMs are specialized neural networks with:
- Massive scale (billions vs thousands of parameters)
- Transformer architecture (parallel processing vs sequential RNNs)
- Self-supervised learning (no labeled data required)
- General-purpose language understanding vs task-specific models

### Q4: Why Transformers over RNNs?

Transformers offer several advantages:
- **Parallelization**: Process entire sequences simultaneously
- **Long-range dependencies**: Self-attention captures relationships across long distances
- **Scalability**: Better GPU utilization for large-scale training
- **No vanishing gradients**: Direct connections through attention

### Q5: How does self-attention differ from traditional attention?

Self-attention computes attention weights between all positions within the same sequence, allowing each position to gather information from all other positions. Traditional attention (in seq2seq models) computes attention between two different sequences (e.g., source and target in translation).

### Q6: Can this model run on CPU only?

Yes, though training will be significantly slower. The implementation automatically detects GPU availability and falls back to CPU if necessary. For production inference on CPU, consider model quantization and optimization techniques.

### Q7: How much data is needed to train this model?

For meaningful language understanding, millions of sentences are typically required. This implementation can work with smaller datasets for demonstration, but production LLMs are trained on billions or trillions of tokens.

### Q8: Can I use this for production applications?

For production, consider:
- Using optimized frameworks (Hugging Face Transformers, vLLM)
- Quantization and model compression
- Proper inference optimization
- Robust error handling and monitoring

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 30.

2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *arXiv preprint arXiv:1810.04805*.

3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Blog*.

4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *Advances in Neural Information Processing Systems*, 33.

5. Jurafsky, D., & Martin, J. H. (2023). "Speech and Language Processing" (3rd ed.). Draft chapters available at: https://web.stanford.edu/~jurafsky/slp3/

6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

7. PyTorch Documentation. https://pytorch.org/docs/stable/index.html

8. Hugging Face Documentation. https://huggingface.co/docs

9. Google Machine Learning Crash Course. https://developers.google.com/machine-learning/crash-course

10. Amazon Web Services. "What is a Large Language Model?" https://aws.amazon.com/what-is/large-language-model/

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Acknowledgments**: This project draws inspiration from the work "Attention Is All You Need".


**Citation**: If you use this implementation in your research or projects, please cite this repository and the original Transformer paper.

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

