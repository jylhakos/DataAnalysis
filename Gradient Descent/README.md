# Gradient Descent

Using Gradient Descent and large language models (LLMs).

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Setup environment (automated)
source setup_venv.sh

# 2. Run a training example
cd src
python training.py

# 3. Explore notebooks
jupyter notebook
```

For detailed instructions, see [Practical Implementation](#practical-implementation).

## Table of Contents

1. [Introduction](#introduction)
   - [What are Large Language Models?](#what-are-large-language-models)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Types of Gradient Descent](#types-of-gradient-descent)
4. [Gradient Descent in Neural Networks](#gradient-descent-in-neural-networks)
5. [Challenges in Deep Learning](#challenges-in-deep-learning)
6. [Gradient Clipping](#gradient-clipping)
7. [Transformers: The Foundation of Modern LLMs](#transformers-the-foundation-of-modern-llms)
8. [Attention Mechanisms and Transformers](#attention-mechanisms-and-transformers)
9. [Gradient Descent in LLM Training](#gradient-descent-in-llm-training)
10. [Fine-Tuning Large Language Models](#fine-tuning-large-language-models)
11. [Modern LLM Architectures](#modern-llm-architectures)
12. [Practical Implementation](#practical-implementation)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
    - [Project Structure](#project-structure)
    - [Running the Code](#running-the-code)
    - [Fine-Tuning and Training Workflow](#fine-tuning-and-training-workflow)
    - [Troubleshooting](#troubleshooting)
13. [References](#references)

## Introduction

Gradient descent is a fundamental optimization algorithm in machine learning and deep learning, particularly crucial for training large language models (LLMs). The algorithm iteratively minimizes a cost function by moving in the direction opposite to the gradient of the function at the current point.

### What are Large Language Models?

Large language models (LLMs) are a subclass of neural language models that typically refer to language models trained on very huge datasets and text corpora using the transformer architecture with a large number of parameters. In essence, large LLMs are scaled versions of neural language models where the model size and data size are exponentially increased. LLMs are also the first models for which fine-tuning for specific tasks and domains became prevalent.

These models can contain billions or even trillions of parameters, enabling them to:
- Understand and generate human-like text
- Perform complex reasoning and problem-solving
- Adapt to new tasks through fine-tuning
- Process and generate content across multiple modalities (text, code, etc.)

In the context of LLMs, gradient descent plays a critical role in:
- Pre-training transformer-based architectures with billions of parameters on massive datasets
- Fine-tuning pre-trained models on specific tasks and domain-specific data
- Optimizing attention mechanisms and position embeddings
- Handling exploding and vanishing gradient problems in deep networks
- Enabling parameter-efficient fine-tuning methods like LoRA and QLoRA

## Mathematical Foundations

### Univariate Gradient Descent

For a real-valued differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$, the Taylor expansion provides the foundation:

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + O(\epsilon^2)$$

To minimize $f(x)$, we move in the opposite direction of the derivative:

$$x \leftarrow x - \eta f'(x)$$

where $\eta > 0$ is the learning rate hyperparameter.

### Multivariate Gradient Descent

For a function $f: \mathbb{R}^d \rightarrow \mathbb{R}$, the gradient is a vector of partial derivatives:

$$\nabla f(\mathbf{x}) = \left[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\right]^T$$

The update rule becomes:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$

### Learning Rate Selection

The learning rate $\eta$ is critical:
- Too large: algorithm may diverge or oscillate
- Too small: convergence is slow and may get stuck in local minima

## Types of Gradient Descent

### Batch Gradient Descent

Computes the gradient using the entire training dataset:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{n}\sum_{i=1}^{n} \nabla L(f(\mathbf{x}^{(i)}; \mathbf{w}), y^{(i)})$$

**Advantages:**
- Stable convergence
- Guaranteed to converge to global minimum for convex functions

**Disadvantages:**
- Computationally expensive for large datasets
- Memory intensive
- Slow updates

### Stochastic Gradient Descent (SGD)

Updates parameters using one sample at a time:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(f(\mathbf{x}^{(i)}; \mathbf{w}), y^{(i)})$$

**Advantages:**
- Fast updates
- Memory efficient
- Can escape local minima due to noise

**Disadvantages:**
- High variance in updates
- Noisy convergence
- May not converge to exact minimum

### Mini-batch Gradient Descent

Balances batch and stochastic approaches using small batches:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{m}\sum_{i=1}^{m} \nabla L(f(\mathbf{x}^{(i)}; \mathbf{w}), y^{(i)})$$

where $m$ is the batch size (typically 32, 64, 128, 256, etc.).

**Advantages:**
- Efficient use of vectorized operations
- Balance between speed and stability
- Better convergence than pure SGD
- Works well with GPU parallelization

This is the **standard approach in LLM training**.

### Advanced Optimizers

Modern LLM training typically uses adaptive learning rate methods:

**Adam (Adaptive Moment Estimation):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $m_t$ and $v_t$ are first and second moment estimates.

## Gradient Descent in Neural Networks

### Forward Propagation

For a neural network with $L$ layers:

$$\mathbf{a}^{[0]} = \mathbf{x}$$
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

where $g^{[l]}$ is the activation function.

### Backpropagation

The gradient is computed backwards through the network using the chain rule:

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}}$$

$$\frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot g'^{[l]}(\mathbf{z}^{[l]})$$

where $\odot$ denotes element-wise multiplication.

### Loss Functions for LLMs

**Cross-Entropy Loss** (standard for language modeling):

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

For next-token prediction in LLMs:

$$L = -\sum_{t=1}^{T} \log P(w_t | w_{\lt t})$$

## Challenges in Deep Learning

### Vanishing Gradients

In deep networks, gradients can become exponentially small during backpropagation:

$$\frac{\partial L}{\partial \mathbf{W}^{[1]}} = \frac{\partial L}{\partial \mathbf{z}^{[L]}} \prod_{l=2}^{L} \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{z}^{[l-1]}}$$

If each term $< 1$, the product becomes vanishingly small.

**Solutions:**
- ReLU and variants (LeakyReLU, GELU, SwiGLU)
- Residual connections (skip connections)
- Layer normalization / RMSNorm
- Careful weight initialization

### Exploding Gradients

Conversely, gradients can grow exponentially large, causing:
- Numerical instability (NaN values)
- Drastic parameter updates
- Inability to converge

This is particularly problematic in:
- Recurrent Neural Networks (RNNs)
- Very deep networks without normalization
- LLMs with long context lengths

**Key observation:** The loss landscape can have steep "cliffs" where gradients spike.

## Gradient Clipping

### Theory

Gradient clipping constrains the norm of gradients to prevent exploding gradients. If the gradient norm exceeds threshold $c$:

$$\mathbf{g} \leftarrow \frac{c \cdot \mathbf{g}}{\|\mathbf{g}\|}$$

This preserves the gradient direction but limits its magnitude.

### Mathematical Formulation

Given gradients $\mathbf{g}$ and threshold $c$:

$$\tilde{\mathbf{g}} = \begin{cases} 
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq c \\
\frac{c \cdot \mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > c
\end{cases}$$

### PyTorch Implementation

```python
import torch.nn.utils as nn_utils

# After loss.backward(), before optimizer.step()
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Empirical Results

In training GPT-2 small models (from referenced experiments):
- **Without gradient clipping:** Test loss = 3.692
- **With gradient clipping (max_norm=1.0):** Test loss = 3.678

Though the improvement seems modest (0.014), gradient clipping prevents catastrophic divergence and stabilizes training, especially crucial in the early stages of LLM training.

### When to Use Gradient Clipping

Essential for:
- Training RNNs and LSTMs
- Training transformers with very deep architectures
- Fine-tuning LLMs on new domains
- Training with high learning rates

Typical values: $c \in [0.5, 5.0]$, with 1.0 being common for LLMs.

## Transformers: The Foundation of Modern LLMs

### Transformer Architecture Overview

Transformers were first introduced in the seminal paper "Attention is All You Need" by Vaswani et al. in 2017. Unlike earlier neural network architectures that processed text sequentially (like RNNs and LSTMs), transformers process entire sequences simultaneously through self-attention mechanisms. This parallel processing capability has made transformers the dominant architecture for modern LLMs.

<p align="center">
  <img src="transformers.svg" alt="Transformer Architecture" width="800">
</p>

*The standard architecture for a Transformer model with the encoder shown on the left and a decoder shown on the right.*

**Source:** [AWS - What are Transformers in Artificial Intelligence](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)

The transformer architecture consists of two main components:

**Encoder (Left):**
- Processes the input sequence
- Contains multiple identical layers (stacked)
- Each layer has:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward network
  - Residual connections and layer normalization
- Outputs contextual representations of the input

**Decoder (Right):**
- Generates the output sequence
- Contains multiple identical layers (stacked)
- Each layer has:
  - Masked multi-head self-attention (prevents looking ahead)
  - Encoder-decoder attention (attends to encoder outputs)
  - Position-wise feed-forward network
  - Residual connections and layer normalization
- Uses previous outputs to predict the next token

### Key Components

**Positional Encoding:** Since transformers process sequences in parallel, positional information must be explicitly added to embeddings to preserve word order.

**Residual Connections:** Enable better gradient flow in deep networks by providing shortcuts around layers.

**Layer Normalization:** Stabilizes training and improves convergence by normalizing activations across features.

### Why Transformers Matter for LLMs

1. **Parallelization:** Unlike RNNs, transformers can process all tokens simultaneously, enabling efficient training on GPUs and TPUs.

2. **Long-Range Dependencies:** Self-attention allows the model to directly connect any two positions in a sequence, capturing long-range dependencies more effectively than recurrent architectures.

3. **Scalability:** The architecture scales efficiently to billions of parameters, enabling modern LLMs like GPT-4, LLaMA, and Claude.

4. **Transfer Learning:** Pre-trained transformer models can be fine-tuned for specific tasks with relatively small amounts of task-specific data.

### Modern Transformer Variants

Modern LLMs often use decoder-only transformers (like GPT) rather than the full encoder-decoder architecture:
- **Decoder-only models (GPT, LLaMA):** Focus on autoregressive text generation
- **Encoder-only models (BERT):** Focus on understanding and classification
- **Encoder-decoder models (T5, BART):** Best for translation and summarization

## Attention Mechanisms and Transformers

### Scaled Dot-Product Attention

The foundation of transformer architectures (Vaswani et al., 2017):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q$ (queries): $n \times d_k$
- $K$ (keys): $m \times d_k$
- $V$ (values): $m \times d_v$
- $d_k$: dimension of keys (scaling prevents softmax saturation)

### Multi-Head Attention

Projects Q, K, V multiple times with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Purpose:** Allows the model to attend to information from different representation subspaces.

### Self-Attention in Transformers

In self-attention, $Q$, $K$, and $V$ are all derived from the same input sequence:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

### Gradient Flow in Attention

The attention mechanism provides relatively smooth gradient flow compared to RNNs:

$$\frac{\partial \text{Attention}}{\partial Query} \propto \text{softmax}(\text{scores}) \cdot V$$

Residual connections around attention layers further improve gradient flow:

$$\text{Output} = \text{LayerNorm}(X + \text{Attention}(X))$$

## Gradient Descent in LLM Training

### Training Pipeline

1. **Tokenization:** Convert text to token IDs
2. **Forward Pass:** Compute predictions through transformer layers
3. **Loss Computation:** Cross-entropy loss on next-token predictions
4. **Backward Pass:** Compute gradients via backpropagation
5. **Gradient Clipping:** Constrain gradient norms
6. **Parameter Update:** Apply optimizer (Adam, AdamW)

### Gradient Accumulation

For very large models that don't fit in GPU memory with desired batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

Uses both FP16 and FP32 to speed up training while maintaining stability:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### Training Interventions Analysis

Based on empirical experiments with GPT-2 small:

| Intervention | Test Loss | Improvement |
|-------------|-----------|-------------|
| Baseline | 3.692 | - |
| + Gradient Clipping | 3.678 | -0.014 |
| + Remove Dropout | 3.641 | -0.051 |
| + Add QKV Bias | 3.669 | -0.023 |

### Layer Normalization Variants

**Pre-Norm (modern standard):**
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Better gradient flow, essential for very deep transformers.

**RMSNorm** (used in LLaMA, Gemma):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \odot \gamma$$

Simpler than LayerNorm, removes mean-centering for efficiency.

## Fine-Tuning Large Language Models

Fine-tuning is the process of adapting a pre-trained language model to perform better on specific tasks or domains. This technique has become a cornerstone of modern LLM deployment, enabling organizations to customize general-purpose models for specialized applications while requiring significantly less data and computational resources than training from scratch.

### What is LLM Fine-Tuning?

Fine-tuning a large language model involves continuing the training of a pre-trained LLM on a targeted dataset to improve its performance on a specific task or within a particular domain. This approach builds on the model's existing knowledge base, reducing the time and resources required compared to training a model from scratch.

**Mathematically**, fine-tuning is an optimization problem where we start with a pre-trained model's weights (θ₀) and search for the smallest adjustment (Δθ) that minimizes the model's errors on specific data (D):

$$\min_{\Delta\theta} L(D; \theta_0 + \Delta\theta)$$

where:
- θ₀ represents all weights the base model learned during pre-training
- Δθ is the adjustment to those original weights
- L is the loss function measuring prediction errors
- D is the fine-tuning dataset

### When to Use Fine-Tuning vs. Other Approaches

There are different methods to adapt LLMs to your use case:

**In-Context Learning (ICL):** Providing task examples within the prompt. Quick but limited by context window.

**Zero/Few-Shot Inference:** Using the model as-is with minimal examples. Works for simple tasks but may not achieve optimal performance.

**Fine-Tuning:** Best for:
- Consistent behavior across many similar queries
- Domain-specific terminology and knowledge
- Improved accuracy on specialized tasks
- Adapting model tone and style
- Tasks requiring very low latency (smaller fine-tuned models can outperform larger general models)

### How Gradient Descent Powers Fine-Tuning

#### Understanding Model Weights in Transformers

In transformer models, learnable parameters (θ) are organized into massive matrices stored throughout the network:

**Attention Mechanism Weights:**
- Query matrix (W_Q): Determines "what information am I looking for?"
- Key matrix (W_K): Identifies "what information is available?"
- Value matrix (W_V): Contains the actual information to be shared

Given an input X, the attention mechanism computes:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Feed-Forward Network Weights:**
After attention, each position passes through a two-layer network:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

where W₁ expands the representation to a higher dimension (typically 4× larger), σ is a non-linear activation (GELU or SwiGLU), and W₂ compresses back to the original dimension.

#### The Gradient Descent Cycle in Fine-Tuning

Fine-tuning operates through a four-step cycle that gradually adjusts billions of parameters:

**1. Forward Pass: Computing Predictions**

The model processes input text through dozens of transformer layers, using current weight matrices (W_Q, W_K, W_V, W₁, W₂) to produce predictions.

**2. Loss Calculation: Measuring Error**

The model's prediction is compared to the correct answer using a loss function (typically cross-entropy):

$$L = -\sum_{t=1}^{T} \log P(w_t | w_{\lt t})$$

Higher loss indicates worse predictions.

**3. Backward Pass: Computing Gradients**

Using backpropagation and the chain rule, we compute the gradient of the loss with respect to every parameter:

$$\nabla_{W^Q} L, \quad \nabla_{W^K} L, \quad \nabla_{W^V} L, \quad \nabla_{W_1} L, \quad \nabla_{W_2} L$$

Each gradient answers: "If I increase this weight slightly, how much will the loss increase?"

**4. Weight Update: Applying the Gradients**

Finally, update each weight by taking a small step in the direction that reduces loss:

$$W_{\text{new}} = W_{\text{old}} - \eta \nabla_W L$$

The learning rate η (typically 0.00001 for fine-tuning) controls step size to avoid "forgetting" pre-trained knowledge.

### Types of Fine-Tuning Methods

#### 1. Supervised Fine-Tuning (SFT)

The most common approach, where the model is updated using labeled examples that demonstrate the desired behavior:

**Process:**
1. Prepare labeled dataset of (input, desired_output) pairs
2. Compute loss between model predictions and desired outputs
3. Update weights via gradient descent
4. Validate on held-out data

**Use Cases:** Task-specific adaptation (classification, question answering, summarization)

#### 2. Parameter-Efficient Fine-Tuning (PEFT)

Training full LLMs is computationally intensive. PEFT methods only update a small subset of parameters while "freezing" the rest:

**LoRA (Low-Rank Adaptation):**
- Adds small trainable matrices alongside frozen weights
- Reduces trainable parameters by 10,000× while maintaining quality
- Updates: $$W = W_0 + BA$$ where B and A are low-rank matrices

**Advantages:**
- Dramatically reduced memory requirements
- Faster training
- Multiple task-specific adapters can share the same base model
- Less risk of catastrophic forgetting

#### 3. Instruction Fine-Tuning

Training the model using examples that demonstrate how it should respond to queries:

**Dataset Format:**
```json
{
  "instruction": "Summarize this weather report",
  "input": "[weather report text]",
  "output": "[concise summary]"
}
```

**Purpose:** Improves the model's ability to follow instructions and maintain consistent output format.

#### 4. Task-Specific Fine-Tuning

Adapting a pre-trained model to excel at particular tasks using domain-specific labeled data:

**Examples:**
- Legal document analysis
- Weather forecast from meteorological data
- Code generation for specific programming languages
- Scientific literature summarization

### Gradient Descent Hyperparameters for Fine-Tuning

To successfully fine-tune LLMs, you must carefully set these parameters:

**Learning Rate (η):**
- Typically set very low (1e-5 to 1e-6) to avoid "catastrophic forgetting"
- Pre-trained weights encode valuable general knowledge
- Small learning rates preserve this knowledge while adapting to new tasks

**Epochs:**
- Number of times the model sees the entire fine-tuning dataset
- Typically 3-10 epochs for small specialized datasets
- Too many epochs risk overfitting to the fine-tuning data

**Batch Size:**
- Trade-off between memory and training stability
- Larger batches provide more stable gradients
- Typical values: 8, 16, 32 for LLM fine-tuning

**Gradient Clipping:**
- Limits maximum gradient value to prevent "exploding gradients"
- Essential for stable fine-tuning
- Typical value: max_norm = 1.0

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Mixed Precision (FP16/BF16):**
- Using 16-bit precision reduces memory usage
- Enables fine-tuning larger models
- Requires gradient scaling to maintain numerical stability

### Quantization for Efficient Fine-Tuning

Quantization is the process of reducing the precision of model weights and activations, enabling fine-tuning of large models on limited hardware:

#### From 32-bit to Lower Precision

**Standard Precision:** Models typically use 32-bit floating-point (FP32) weights, which:
- Provide high precision (~16.8 million discrete values)
- Consume significant memory (4 bytes per parameter)
- A 7B parameter model requires ~28GB just for weights

**Quantization Approaches:**

**1. FP16/BF16 (16-bit floating-point):**
- Reduces memory by 50%
- Straightforward conversion from FP32
- Requires careful handling of very small/large values
- Accumulation typically still in FP32 for numerical stability

**2. INT8 (8-bit integer):**
- Reduces memory by 75% compared to FP32
- Requires calibration to map continuous FP32 values to discrete INT8 range
- Uses affine quantization scheme:

$$x = S \cdot (x_q - Z)$$

where:
- x_q is the quantized INT8 value
- S is the scale (positive float)
- Z is the zero-point (INT8 value corresponding to 0 in FP32)

**3. 4-bit Quantization (QLoRA):**
- Reduces memory by 87.5%
- Enables fine-tuning 65B+ models on consumer GPUs
- Uses NormalFloat (NF4) format optimized for normally distributed weights
- Combines with LoRA for parameter-efficient fine-tuning

#### Quantization Techniques

**Post-Training Quantization:**
1. Start with trained FP32 model
2. Calibrate using representative data
3. Convert weights and activations to lower precision
4. Minimal accuracy loss for many tasks

**Quantization-Aware Training (QAT):**
1. Simulate quantization during training
2. Model learns to be robust to quantization errors
3. Better accuracy than post-training quantization
4. Requires more computation during fine-tuning

**Example - QLoRA Fine-Tuning:**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Practical Fine-Tuning Workflow

**Step 1: Prepare Training Data**

Divide data into training, validation, and test splits:
```python
from datasets import load_dataset, DatasetDict

dataset = load_dataset("your_dataset")
dataset = dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test'].train_test_split(test_size=0.5)['train'],
    'test': dataset['test'].train_test_split(test_size=0.5)['test']
})
```

**Step 2: Configure Training**

Set hyperparameters based on your compute budget and data size:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-5,
    fp16=True,  # Mixed precision
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,  # Reduce memory at cost of speed
)
```

**Step 3: Fine-Tune with Gradient Descent**

During the fine-tuning phase:
1. Model sees labeled data and makes predictions
2. Loss is calculated (cross-entropy for language modeling)
3. Gradients are computed via backpropagation
4. Gradient clipping prevents instability
5. Optimizer (AdamW) updates weights

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Training loop internally performs:
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         outputs = model(batch)
#         loss = criterion(outputs, labels)
#         loss.backward()  # Compute gradients
#         clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()  # Apply gradient descent
#         optimizer.zero_grad()

trainer.train()
```

**Step 4: Evaluate and Deploy**

Assess performance on held-out test data and deploy the fine-tuned model:
```python
results = trainer.evaluate(dataset["test"])
trainer.save_model("./fine_tuned_model")
```

### Gradient Descent as an LLM Optimizer

Recent research explores using LLMs themselves as optimizers. Instead of computing mathematical gradients, these approaches use LLMs to analyze errors and suggest improvements:

**APO (Adversarial Prompt Optimization):**
- Treats prompt editing as "semantic gradient descent"
- LLM analyzes prediction errors
- Generates improved prompts that move toward better performance
- Analogous to following gradients in prompt space

This represents a fascinating direction where LLMs optimize their own behavior through language rather than numerical gradients.

### Challenges in Fine-Tuning

**1. Catastrophic Forgetting:**
- The model can lose general capabilities when fine-tuned too aggressively
- Mitigation: Use low learning rates, early stopping, regularization

**2. Overfitting:**
- Model memorizes training data rather than learning generalizable patterns
- Mitigation: Data augmentation, dropout, validation-based early stopping

**3. Data Quality:**
- Fine-tuning quality depends heavily on labeled data quality
- Poor data leads to poor fine-tuned models
- Mitigation: Careful data curation, human review

**4. Computational Cost:**
- Full fine-tuning of billion-parameter models requires significant compute
- Mitigation: PEFT methods like LoRA, quantization (QLoRA)

### Why Small Changes Have Large Effects

Transformer models use distributed representations where knowledge is spread across thousands of parameters. Even changing just 0.1% of weights can substantially affect behavior because:

1. **Network Effects:** Small changes propagate through layers, altering information flow throughout the model

2. **Emergent Behaviors:** Slight weight adjustments can cause the model to exhibit new capabilities (understanding jargon, maintaining tone, specialized reasoning)

3. **Compositional Power:** The model combines many small pieces of information, so tweaking their interactions leads to different results

This is the power of fine-tuning: targeted updates to a small fraction of parameters can adapt a general-purpose LLM to highly specialized tasks while retaining broad knowledge.

## Modern LLM Architectures

### Architecture Comparison

| Model | Attention Type | Normalization | Activation | Position Encoding | MoE |
|-------|---------------|---------------|------------|-------------------|-----|
| GPT-3 | MHA | LayerNorm | GELU | Learned | No |
| LLaMA 2/3 | GQA | RMSNorm | SwiGLU | RoPE | No |
| Gemma 2/3 | GQA | RMSNorm | GELU/GELU | RoPE | No |
| Qwen 2.5/3 | GQA | RMSNorm | SwiGLU | RoPE | No |
| DeepSeek V3 | MLA | RMSNorm | SwiGLU | RoPE | Yes |
| Mixtral 8x7B | MHA | RMSNorm | SwiGLU | RoPE | Yes (8 experts) |
| OLMo 2 | GQA | LayerNorm | SwiGLU | RoPE | No |

### Grouped-Query Attention (GQA)

Reduces memory and computation compared to Multi-Head Attention:
- Multiple query heads share the same key and value heads
- Trade-off between MHA (all separate) and MQA (single shared)

### Multi-Head Latent Attention (MLA)

Used in DeepSeek V3:
- Compresses key-value cache through latent representations
- Reduces memory footprint for long context
- Maintains model quality with efficiency gains

### Mixture of Experts (MoE)

Activates only a subset of parameters per token:

$$y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

where:
- $G(x)$: gating network (determines which experts to use)
- $E_i(x)$: expert network
- Typically top-2 or top-8 experts activated per token

**Benefits:**
- Larger model capacity without proportional compute increase
- Better scaling properties
- Specialized experts for different token types

### Rotary Position Embeddings (RoPE)

Encodes position information through rotation:

$$f(x, m) = x \cdot e^{im\theta}$$

where $m$ is the position index and $\theta$ depends on the dimension.

**Advantages over sinusoidal embeddings:**
- Relative position encoding naturally emerges
- Better extrapolation to longer sequences
- Used in most modern LLMs (LLaMA, Qwen, Gemma)

## Practical Implementation

### Prerequisites

Before setting up the environment, ensure you have:

- **Python 3.8 or higher** installed on your system
- **Git** (for cloning the repository)
- **4GB+ RAM** (8GB+ recommended for training)
- **CUDA-capable GPU** (optional, but recommended for larger experiments)

Check your Python version:
```bash
python3 --version
```

### Environment Setup

#### Virtual Environment Setup

**How to Use:**
```bash
# Initial setup (creates and activates venv, installs all dependencies)
source setup_venv.sh

# Subsequent sessions (just activates existing venv)
source activate_env.sh

# When done
deactivate
```

#### Option 1: Automated Setup (Recommended)

Use the provided setup script for automatic environment configuration:

```bash
# Clone or navigate to the repository
cd "Gradient Descent"

# Run automated setup script
source setup_venv.sh
```

This script will:
1. Create a virtual environment named `llm_env`
2. Activate the environment
3. Upgrade pip to the latest version
4. Install PyTorch with CPU support
5. Install all dependencies from `requirements.txt`

**For subsequent sessions**, activate the environment with:
```bash
source activate_env.sh
```

#### Option 2: Manual Setup

If you prefer manual setup or need GPU support:

**Step 1: Create Virtual Environment**
```bash
# Create a new virtual environment
python3 -m venv llm_env

# Activate the environment
# On Linux/macOS:
source llm_env/bin/activate
# On Windows:
llm_env\Scripts\activate
```

**Step 2: Upgrade pip**
```bash
pip install --upgrade pip
```

**Step 3: Install PyTorch**

For CPU-only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For GPU (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install Required Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Project Structure

```
Gradient Descent/
├── src/
│   ├── gradient_descent.py      # Basic gradient descent implementations
│   ├── neural_network.py        # Simple neural network with GD
│   ├── attention.py             # Attention mechanism implementation
│   ├── transformer.py           # Transformer architecture (RMSNorm, RoPE)
│   └── training.py              # LLM training loop with gradient clipping
├── notebooks/
│   ├── 01_gradient_descent_basics.ipynb
│   └── 03_transformer_attention.ipynb
├── visualizations/
│   └── plots.py                 # Matplotlib visualization utilities
├── transformers.svg             # Transformer architecture diagram
├── requirements.txt             # Python dependencies
├── setup_venv.sh               # Automated environment setup
├── activate_env.sh             # Environment activation helper
└── README.md
```

### Running the Code

#### Step 1: Activate Virtual Environment

Always activate the environment before running any code:
```bash
source activate_env.sh
# or manually: source llm_env/bin/activate
```

#### Step 2: Run Individual Scripts

**A. Gradient Descent Basics**

Run the basic gradient descent implementations:
```bash
cd src
python gradient_descent.py
```

This demonstrates:
- Univariate gradient descent
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Visualizations of convergence

**B. Neural Network Training**

Train a simple feedforward neural network:
```bash
python neural_network.py
```

Features:
- Multi-layer perceptron with gradient descent
- Backpropagation implementation
- Loss and accuracy tracking
- Training/validation split

**C. Attention Mechanisms**

Explore attention mechanisms:
```bash
python attention.py
```

Includes:
- Scaled dot-product attention
- Multi-head attention
- Grouped-query attention (GQA)
- Attention weight visualizations

**D. Transformer Architecture**

Run the complete transformer implementation:
```bash
python transformer.py
```

Implements:
- RMSNorm (modern layer normalization)
- Rotary Position Embeddings (RoPE)
- Multi-head attention
- Feed-forward networks
- Complete transformer block

**E. LLM Training Loop**

Run a complete training example with gradient descent:
```bash
python training.py
```

Features:
- Mini-batch gradient descent
- Adam optimizer with weight decay (AdamW)
- Gradient clipping (preventing exploding gradients)
- Learning rate scheduling
- Mixed precision training (FP16)
- Loss tracking and visualization

#### Step 3: Run Jupyter Notebooks

Launch Jupyter for interactive exploration:

```bash
# From the project root directory
jupyter notebook
```

Then open:
- `notebooks/01_gradient_descent_basics.ipynb` - Interactive gradient descent examples
- `notebooks/03_transformer_attention.ipynb` - Attention mechanisms step-by-step

Navigate through cells using Shift+Enter to execute each code block.

### Fine-Tuning and Training Workflow

#### Basic Training Example

Here's how to train a small language model using the provided code:

```bash
# Activate environment
source activate_env.sh

# Navigate to source directory
cd src

# Run the training script
python training.py
```

The `training.py` script demonstrates:
1. **Data preparation** - Creating training and validation datasets
2. **Model initialization** - Small transformer with configurable parameters
3. **Optimizer setup** - AdamW with learning rate scheduling
4. **Training loop** - Mini-batch gradient descent with:
   - Forward pass through transformer
   - Loss calculation (cross-entropy)
   - Backward pass (compute gradients)
   - Gradient clipping (max norm = 1.0)
   - Weight updates
5. **Validation** - Periodic evaluation on held-out data
6. **Checkpointing** - Save best model weights

#### Customizing Training Parameters

To modify training hyperparameters, edit the configuration in `training.py`:

```python
# Learning rate (typical range for fine-tuning: 1e-5 to 1e-4)
learning_rate = 3e-4

# Batch size (adjust based on available memory)
batch_size = 32

# Number of training epochs
epochs = 10

# Gradient clipping threshold
max_grad_norm = 1.0

# Enable mixed precision (faster training, less memory)
use_amp = True
```

#### Fine-Tuning Pre-trained Models

For fine-tuning pre-trained models with Hugging Face transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,  # Gradient clipping
)

# Create trainer with gradient descent optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start fine-tuning with gradient descent
trainer.train()
```

### Troubleshooting

**Problem: `ModuleNotFoundError: No module named 'torch'`**

Solution: Ensure virtual environment is activated and PyTorch is installed:
```bash
source llm_env/bin/activate
pip install torch torchvision torchaudio
```

**Problem: CUDA out of memory errors**

Solutions:
- Reduce batch size in training configuration
- Enable gradient checkpointing
- Use gradient accumulation
- Switch to CPU or use quantization (FP16, INT8)

**Problem: Very slow training on CPU**

Solutions:
- Install CUDA-enabled PyTorch for GPU acceleration
- Reduce model size (fewer layers, smaller hidden dimension)
- Use smaller batch sizes
- Enable mixed precision training

**Problem: Gradients exploding (NaN losses)**

Solutions:
- Enable gradient clipping (already in `training.py`)
- Reduce learning rate
- Check for numerical instabilities in data
- Use gradient scaling with mixed precision

**Problem: Model not converging**

Solutions:
- Increase learning rate (if loss not changing)
- Decrease learning rate (if loss oscillating)
- Add learning rate warm-up
- Check data preprocessing and normalization
- Increase training epochs

**Problem: Jupyter kernel dies when running notebooks**

Solutions:
- Reduce batch size or model size
- Increase system memory
- Close other applications
- Use CPU instead of GPU if memory constrained

### Example Code Files

Implementation files in the `src/` directory:
- **gradient_descent.py** - Implementations of batch, SGD, and mini-batch gradient descent
- **neural_network.py** - Neural network trained with gradient descent
- **attention.py** - Scaled dot-product and multi-head attention mechanisms
- **transformer.py** - Complete transformer block with RMSNorm, RoPE, gradient clipping
- **training.py** - LLM-style training loop with Adam optimizer, learning rate scheduling
- **visualizations/plots.py** - Loss landscapes, convergence plots, gradient flow visualization

## References

### Papers

1. Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
2. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization". ICLR.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the difficulty of training recurrent neural networks". ICML.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition". CVPR.
5. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization". arXiv:1607.06450.

### Online Resources

1. D2L.ai - Dive into Deep Learning: https://d2l.ai/chapter_optimization/gd.html
2. D2L.ai - Transformer Architecture: https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
3. AWS - What are Transformers in AI: https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/
4. Hugging Face - Quantization Guide: https://huggingface.co/docs/optimum/concept_guides/quantization
5. Google Research - Fine-tuning LLMs with Differential Privacy: https://research.google/blog/fine-tuning-llms-with-user-level-differential-privacy/
6. Michael Brenndoerfer - Mathematics of LLM Fine-Tuning: https://mbrenndoerfer.com/writing/mathematics-llm-fine-tuning-how-and-why-it-works-explained
7. Medium - How Gradient Descent Trains GPT-Like Models: https://medium.com/@andreagomar18/how-gradient-descent-really-trains-gpt-like-models-1fc9699b626f
8. Giles Thomas - Building LLMs from Scratch: https://gilesthomas.com
9. Sebastian Raschka - LLM Architecture Comparison: Understanding LLM Evolution
10. PyTorch Documentation: https://pytorch.org/docs/
11. Hugging Face Transformers: https://huggingface.co/docs/transformers/

### Books

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.
2. Zhang, A., et al. (2023). "Dive into Deep Learning". Cambridge University Press.

### Summary

- **Gradient descent** is the backbone of neural network training, including LLMs
- **Types of gradient descent** (batch, SGD, mini-batch) offer different trade-offs between speed and stability
- **Exploding and vanishing gradients** remain key challenges in deep learning
- **Gradient clipping** is essential for stable LLM training
- **Transformers** revolutionized NLP by enabling parallel processing and better long-range dependencies
- **Modern transformer architectures** use advanced attention mechanisms (GQA, MLA) and efficiency improvements
- **Fine-tuning** enables adaptation of pre-trained LLMs to specific domains with minimal data
- **Parameter-efficient methods** (LoRA, QLoRA) make fine-tuning accessible on consumer hardware
- **Quantization** (FP16, INT8, 4-bit) dramatically reduces memory requirements
- **Adaptive optimizers** (Adam, AdamW) and architectural innovations (residual connections, normalization) improve gradient flow
- **Small weight changes** in distributed neural representations can have large effects on model behavior
- **Mixture of Experts** and efficiency improvements enable training and deploying larger models

---

For practical implementations and code examples, see the `src/` directory and Jupyter notebooks in `notebooks/`.

