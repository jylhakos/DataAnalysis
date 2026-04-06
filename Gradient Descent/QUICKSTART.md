# Quick Start

## Getting Started with Gradient Descent and LLMs

This guide will help you set up and run the project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Set up the virtual environment

Run the automated setup script:

```bash
cd "/home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Gradient Descent"
source setup_venv.sh
```

This script will:
- Create a virtual environment named `llm_env`
- Activate the environment automatically
- Install PyTorch and all dependencies
- Verify the installation

For future sessions, activate the existing environment:

```bash
source activate_env.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv llm_env

# Activate it
source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

When you're done working, deactivate the environment:

```bash
deactivate
```

### 2. Verify installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Running Examples

### Basic Gradient Descent

```bash
cd src
python gradient_descent.py
```

This demonstrates:
- Batch gradient descent on quadratic and Rosenbrock functions
- Visualization of optimization trajectories
- Effect of different learning rates

### Neural Network Training

```bash
python neural_network.py
```

This demonstrates:
- Training a neural network with gradient descent
- Backpropagation implementation
- Decision boundary visualization

### Attention Mechanisms

```bash
python attention.py
```

This demonstrates:
- Multi-head attention
- Grouped-query attention (GQA)
- Causal self-attention with RoPE
- Parameter comparison

### Transformer Architecture

```bash
python transformer.py
```

This demonstrates:
- Complete GPT-style language model
- Transformer blocks with RMSNorm
- Text generation capabilities

### LLM Training with Gradient Descent

```bash
python training.py
```

This demonstrates:
- Training a language model from scratch
- Gradient clipping in action
- Learning rate scheduling
- Training metrics visualization

## Interactive Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Then explore:
1. `notebooks/01_gradient_descent_basics.ipynb` - Fundamentals
2. `notebooks/03_transformer_attention.ipynb` - Attention mechanisms

## Project Structure

```
.
├── README.md                 # Comprehensive documentation
├── requirements.txt          # Python dependencies
├── setup.sh                 # Setup script
├── src/                     # Source code
│   ├── gradient_descent.py  # Gradient descent algorithms
│   ├── neural_network.py    # Neural network with GD
│   ├── attention.py         # Attention mechanisms
│   ├── transformer.py       # Transformer architecture
│   └── training.py          # LLM training loop
├── notebooks/               # Jupyter notebooks
│   ├── 01_gradient_descent_basics.ipynb
│   └── 03_transformer_attention.ipynb
└── visualizations/          # Output plots (created automatically)
```

## Key Concepts Covered

### Gradient Descent

- **Batch GD**: Uses entire dataset
- **Stochastic GD (SGD)**: Uses single samples
- **Mini-batch GD**: Uses small batches (standard in LLMs)

Update rule: `w ← w - η∇L(w)`

### Gradient Clipping

Prevents exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Formula: If `||g|| > c`, then `g ← c * g / ||g||`

### Attention Mechanism

Scaled dot-product attention:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### LLM Training

1. Tokenization
2. Forward pass through transformer
3. Compute loss (cross-entropy)
4. Backward pass (compute gradients)
5. Gradient clipping
6. Parameter update (Adam optimizer)

## Troubleshooting

### Import Errors

Make sure virtual environment is activated:
```bash
source llm_env/bin/activate
```

### CUDA Out of Memory

Reduce batch size or model size in training configuration.

### Slow Training

- Enable mixed precision: `use_amp=True`
- Reduce sequence length or model size
- Use GPU if available


## Resources

- **D2L.ai**: https://d2l.ai/chapter_optimization/gd.html
- **Attention Is All You Need**: Original transformer paper
- **PyTorch Docs**: https://pytorch.org/docs/
- **Hugging Face**: https://huggingface.co/docs/transformers/

## Common Commands

```bash
# Activate environment
source llm_env/bin/activate

# Deactivate environment
deactivate

# Run all demos
cd src
python gradient_descent.py
python neural_network.py
python attention.py
python transformer.py
python training.py

# Generate visualizations
python ../visualizations/plots.py

# Start Jupyter
jupyter notebook

# Install new package
pip install package_name
pip freeze > requirements.txt
```
