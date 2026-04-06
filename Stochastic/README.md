# Stochastic Features in Large Language Models

A study of deterministic and Stochastic approaches in machine learning, with a focus on large language models (LLMs) and their probabilistic nature.

## Table of Contents

- [Introduction](#introduction)
- [Deterministic vs. Stochastic in Machine Learning](#deterministic-vs-stochastic-in-machine-learning)
- [Stochastic Mathematics is Deterministic in ML](#stochastic-mathematics-is-deterministic-in-ml)
- [Mathematical Foundations](#mathematical-foundations)
- [Probabilistic Text Generation in LLMs](#probabilistic-text-generation-in-llms)
- [Why LLMs Produce Non-Deterministic Outputs](#why-llms-produce-non-deterministic-outputs)
- [Temperature 0: Myth vs. Reality](#temperature-0-myth-vs-reality)
- [Differences Between Stochastic and Deterministic Sampling](#differences-between-stochastic-and-deterministic-sampling)
- [Are Neural Networks Deterministic?](#are-neural-networks-deterministic)
- [Understanding Non-Determinism in LLMs](#understanding-non-determinism-in-llms)
- [The Psychology of LLMs as Random Number Generators](#the-psychology-of-llms-as-random-number-generators)
- [Stochastic Parrots or Intelligent Systems?](#stochastic-parrots-or-intelligent-systems)
- [Testing if LLMs are Stochastic Parrots](#testing-if-llms-are-stochastic-parrots)
- [How Randomness Prevents Parroting](#how-randomness-prevents-parroting)
- [Impact on Coding Assistants and AI Agents](#impact-on-coding-assistants-and-ai-agents)
- [Reinforcement Learning and Non-Determinism](#reinforcement-learning-and-non-determinism)
- [Floating-Point Non-Associativity](#floating-point-non-associativity)
- [Practical Implementation](#practical-implementation)
- [Resources](#resources)

---

## Introduction

In the field of Data Analysis and Machine Learning, understanding the distinction between **deterministic** and **stochastic** processes is crucial for building reliable, reproducible systems. This document explores how stochastic mathematics underpins modern large language models (LLMs) while examining when and why these probabilistic systems can behave deterministically.

**Stochastic** processes involve inherent randomness and uncertainty, commonly seen in techniques like stochastic gradient descent (SGD), dropout regularization, and probabilistic sampling in LLMs. However, through careful control of random seeds and sampling parameters, these seemingly random processes can be made reproducible and deterministic.

---

## Deterministic vs. Stochastic in Machine Learning

### What is Deterministic?

A **deterministic** process always produces the same output for a given input. In machine learning:

- **Inference with fixed weights**: Once a neural network is trained, the same input always yields the same output
- **Sorting algorithms**: Always produce the same ordered result
- **Mathematical functions**: $f(x) = 2x + 3$ always returns the same value for the same $x$

### What is Stochastic?

**Stochastic results** are generally not considered deterministic, as they involve inherent randomness. Examples include:

- **Stochastic Gradient Descent (SGD)**: Uses random mini-batches of data
- **Dropout**: Randomly deactivates neurons during training
- **Random initialization**: Weights are initialized with random values
- **Data shuffling**: Training data is randomly shuffled each epoch

However, in machine learning, **stochastic processes become effectively deterministic through pseudo-randomness**. By setting a fixed seed for random number generators, the same output is produced from the same input every time.

```python
import torch
import numpy as np

# Setting seeds makes stochastic processes deterministic
torch.manual_seed(42)
np.random.seed(42)

# Now random operations are reproducible
random_tensor = torch.randn(3, 3)  # Always produces the same "random" tensor
```

---

## Stochastic Mathematics is Deterministic in ML

### Pseudo-Random Generators (PRNGs)

Computer-generated randomness is typically **pseudo-random**, meaning it is a deterministic sequence of numbers starting from a seed value. The mathematical formula for a simple Linear Congruential Generator (LCG) is:

$$X_{n+1} = (aX_n + c) \mod m$$

Where:
- $X_n$ is the current random number
- $a$ is the multiplier
- $c$ is the increment
- $m$ is the modulus
- $X_0$ is the seed

If you fix the seed (e.g., `torch.manual_seed(42)`), the "stochastic" algorithm follows the exact same path every time.

### Reproducibility in Practice

By controlling the initialization of weights, dropout layers, and shuffling, developers can make experiments repeatable (deterministic):

```python
def set_seed(seed=42):
    """Ensure reproducibility across runs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Key principle**: While algorithms model uncertainty, setting a specific seed means the "random" numbers are calculated, not truly random, allowing for reproducibility.

---

## Mathematical Foundations

### Probability Distributions in LLMs

LLMs compute a probability distribution over the vocabulary $V$ for the next token:

$$P(w_t | w_1, w_2, \ldots, w_{t-1}) = \text{softmax}(z_t)$$

Where $z_t$ are the logits (raw scores) from the model, and softmax is defined as:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{|V|} e^{z_j}}$$

### Temperature Scaling

Temperature $\tau$ controls the randomness of predictions:

$$P_\tau(w_i) = \frac{e^{z_i/\tau}}{\sum_{j=1}^{|V|} e^{z_j/\tau}}$$

Where:
- $\tau \to 0$: Distribution becomes peaked (deterministic, greedy decoding)
- $\tau = 1$: Original distribution
- $\tau > 1$: Distribution becomes flatter (more random)

### Stochastic Gradient Descent

The gradient update rule in SGD is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; x^{(i)}, y^{(i)})$$

Where:
- $\theta$ represents model parameters
- $\eta$ is the learning rate
- $(x^{(i)}, y^{(i)})$ is a randomly sampled mini-batch
- $\mathcal{L}$ is the loss function

The stochastic nature comes from using a random subset of data rather than the full dataset.

### Entropy and Information Theory

The entropy of a probability distribution measures uncertainty:

$$H(P) = -\sum_{i=1}^{|V|} P(w_i) \log P(w_i)$$

- High entropy → High uncertainty, more diverse outputs
- Low entropy → Low uncertainty, more predictable outputs

---

## Probabilistic Text Generation in LLMs

Large language models are fundamentally **probabilistic systems**, not calculators. While the underlying neural network computes mathematical probabilities, generation involves sampling from these distributions using parameters like **temperature** and **top-p**, intentionally adding randomness to produce more creative, human-like text rather than repetitive answers.

### Key Stochastic Components

1. **Token Probability Distribution**: The model assigns a probability to every possible next token
2. **Sampling Methods**: Convert probabilities into actual token selection
3. **Variational Parameters**: Temperature, top-k, top-p control randomness
4. **Hardware-Level Variance**: GPU parallelization introduces subtle non-determinism

### Mathematical Representation

Given input sequence $X = (x_1, \ldots, x_n)$, an LLM models:

$$P(x_{n+1}, \ldots, x_m | x_1, \ldots, x_n) = \prod_{t=n+1}^{m} P(x_t | x_1, \ldots, x_{t-1})$$

This auto-regressive generation is inherently probabilistic, as each token is sampled from a distribution.

---

## Why LLMs Produce Non-Deterministic Outputs

LLM outputs are **non-deterministic**, yielding different answers to the same query, because they are designed as probabilistic systems. Several factors contribute to this:

### 1. Stochastic Sampling

LLMs predict the next word (token) by assigning probabilities to all possible candidates. Sampling methods select from these candidates:

- **Greedy Decoding**: Always pick the highest probability token (most deterministic)
- **Top-k Sampling**: Sample from the top $k$ most likely tokens
- **Top-p (Nucleus) Sampling**: Sample from the smallest set of tokens whose cumulative probability ≥ $p$

$$P_{\text{nucleus}}(w) = \begin{cases} 
\frac{P(w)}{\sum_{w' \in V_p} P(w')} & \text{if } w \in V_p \\
0 & \text{otherwise}
\end{cases}$$

Where $V_p$ is the smallest set such that $\sum_{w \in V_p} P(w) \geq p$.

### 2. Temperature Scaling

This parameter controls the "randomness." A higher temperature flattens the probability distribution, making low-probability tokens more likely to be chosen, resulting in diverse outputs.

### 3. Floating-Point Inconsistencies

Because LLMs run on GPUs using parallel processing, tiny differences in floating-point calculations, hardware acceleration, or parallel order of operations can lead to different results, particularly over long auto-regressive generation.

Due to the non-associativity of floating-point arithmetic:

$$(a + b) + c \neq a + (b + c)$$

### 4. Load Balancing

Production environments often send requests to different model replicas, which may be running on slightly different hardware configurations, leading to varied outputs.

### 5. Dynamic Batching

GPU operations are batched for efficiency. Different batch sizes change the order of reduction operations, introducing variance even with identical inputs.

---

## Temperature 0: Myth vs. Reality

### Is Temperature 0 Truly Deterministic?

**Common belief**: Setting temperature to 0 (greedy decoding) makes LLMs deterministic by forcing the model to always select the most likely next token.

**Reality**: Even with temperature 0, **a small amount of variation is still possible** due to:

1. **Floating-Point Errors**: Can affect probability calculations, causing different tokens to be deemed "most likely"
2. **Batch Size Variations**: Different batch sizes alter reduction splitting in GPU kernels
3. **Hardware Differences**: Running on different GPUs (H100 vs. A100) yields different numerical results
4. **Quantization**: FP16 vs. BF16 vs. INT8 precision differences

### Mathematical Explanation

Even with greedy decoding (temperature → 0), if two tokens have probabilities:

$$P(w_1) = 0.1847, \quad P(w_2) = 0.1846$$

Floating-point rounding across billions of operations can flip which token is selected.

**Conclusion**: Temperature 0 provides **near-deterministic** behavior but cannot guarantee absolute determinism in production environments.

---

## Differences Between Stochastic and Deterministic Sampling

| Aspect | Deterministic Sampling | Stochastic Sampling |
|--------|------------------------|---------------------|
| **Output** | Same output every time for given input | Variable outputs for same input |
| **Algorithm** | Fixed rules (e.g., greedy selection) | Probabilistic selection |
| **Reproducibility** | Guaranteed with same inputs | Requires seed control for reproducibility |
| **Use Cases** | Unit testing, auditing, compliance | Creative generation, exploration |
| **Examples** | Systematic sampling, greedy decoding | Monte Carlo, nucleus sampling, SGD |
| **Randomness** | None | Controlled via PRNGs |

### Deterministic Sampling

Follows fixed rules or algorithms, ensuring the same output is produced every time for a given input. For example, selecting every 10th item from a dataset guarantees identical results across runs.

**Ideal for**: Tasks requiring reproducibility, such as unit testing, regulatory compliance, and debugging.

### Stochastic Sampling

Incorporates randomness, meaning results can vary even with identical inputs. Examples include:

- **Monte Carlo Simulations**: Model complex systems (e.g., financial risk) by generating many possible outcomes
- **Stochastic Gradient Descent**: Uses random mini-batches to escape local minima
- **Dropout**: Randomly deactivates neurons to prevent overfitting

### Making Stochastic Methods Deterministic

In algorithms like Stochastic Gradient Descent (SGD), setting a fixed seed (e.g., `seed=42`) ensures that:
- Random weight initializations occur in the same way
- Data shuffling follows the same order
- The learning path becomes deterministic

During inference, models that can behave stochastically (e.g., using sampling) are forced to be deterministic by using methods like **greedy decoding** (choosing only the highest probability next token), ensuring the same input always produces the same output.

---

## Are Neural Networks Deterministic?

### Short Answer

Neural networks are **generally deterministic at inference time** but **stochastic during training**.

### Detailed Explanation

#### During Inference

Once a network is trained and the weights are frozen, it behaves as a **deterministic function**. The same input vector $\mathbf{x}$ will map to the same output $\mathbf{y}$:

$$\mathbf{y} = f_\theta(\mathbf{x})$$

Where $\theta$ represents the fixed, trained parameters.

#### During Training

Training is rarely deterministic:

1. **Stochastic Gradient Descent (SGD)**: Uses random mini-batches
2. **Random Weight Initialization**: $W \sim \mathcal{N}(0, \sigma^2)$
3. **Data Shuffling**: Training data order is randomized
4. **Dropout**: Randomly masks neurons with probability $p$:

$$\mathbf{h}_{\text{dropout}} = \mathbf{h} \odot \mathbf{m}, \quad \mathbf{m} \sim \text{Bernoulli}(1-p)$$

### When Neural Networks Are Non-Deterministic

#### 1. Generative Models

- **VAEs (Variational Autoencoders)**: Sample from a learned distribution
  
  $$\mathbf{z} \sim \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$$

- **GANs (Generative Adversarial Networks)**: Generator creates new samples from noise

  $$G(\mathbf{z}), \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

#### 2. Explicit Noise Injection

Networks using techniques like dropout at inference time will produce different results.

#### 3. Hardware Influences

Even if an algorithm is designed to be deterministic, numerical inaccuracies (e.g., floating-point errors) on different GPUs can cause subtle differences in outputs.

**Summary**: Neural networks are deterministic mathematical functions, but their training is stochastic, and certain architectures are designed to be probabilistic.

---

## Understanding Non-Determinism in LLMs

**Non-determinism in Large Language Models (LLMs)** means identical prompts can yield different outputs. This stems from intentional sampling, hardware-level calculations, and infrastructure choices.

### Key Sources of Non-Determinism

#### 1. Token Sampling Parameters

When **temperature**, **top-k**, or **top-p** (nucleus sampling) are used, the model does not simply choose the most likely next word. Instead, it selects from a probability distribution, introducing intentional randomness to improve response diversity.

#### 2. Floating-Point Non-Associativity

GPUs optimize speed by changing the order of calculations. Since floating-point math is not associative:

$$(a + b) + c \neq a + (b + c)$$

due to rounding. Different operation orders produce slightly different numbers, which can cascade into completely different token selections.

#### 3. Dynamic Batching and Server Load

Most APIs use **dynamic batching**, where your request is grouped with others. If your prompt is batched with different requests on different runs, the GPU handles calculations differently, resulting in non-deterministic output.

#### 4. GPU Parallelization

Operations are split across thousands of threads. If the order of operations depends on which thread finishes first, the computation result changes.

#### 5. Model Quantization and Infrastructure

Running the same model in different precision (e.g., FP16 vs. BF16) or using different hardware types (e.g., H100 vs. A100) leads to different numerical results.

### Why This Matters

#### 1. Irreproducible Bugs

A user reports an issue. You can't reproduce it because the system generates different output now.

#### 2. Inconsistent User Experiences

The same user asks the same question twice and gets contradictory responses.

#### 3. Testing Challenges

Traditional unit tests expecting exact output matches fail with LLMs.

### Technical Deep Dive

The primary cause of non-determinism is **varying batch sizes** during inference. Even when a user repeatedly submits the same prompt, the output can vary across runs, since the request may be batched together with other users' requests.

Different batch sizes influence the reduction splitting process of kernels (RMSNorm, matrix multiplication, attention). This leads to varying order and size for each reduction block, causing non-deterministic outputs due to floating-point non-associativity.

### Solutions for Deterministic Inference

Recent research has developed **batch-invariant implementations** that replace reduction kernels with versions that produce consistent results regardless of batch size. This ensures:

- Same prompt → Same output
- Reproducible debugging
- Consistent user experiences

---

## The Psychology of LLMs as Random Number Generators

### Research Overview

Recent research ([arXiv:2502.19965](https://arxiv.org/pdf/2502.19965)) reveals that **large language models are fundamentally probabilistic models that often exhibit deterministic, biased behaviors** when tasked with acting as random number generators.

### Key Findings

While based on stochastic transformer architectures, LLMs mimic **human cognitive biases** such as:

- Favoring specific numbers (e.g., 3, 7)
- Preferring middle-range values
- Avoiding extreme values ("tail aversion")
- Reproducing patterns from training data rather than true randomness

### The "Stochastic Parrots" Syndrome

**LLMs are not true random number generators**; they often produce biased, predictable, and deterministic outputs that mirror human cognitive limitations.

They tend to reproduce **overrepresented sequences from training data**, creating "stylistic regularities" rather than pure random outputs.

### Factors Influencing "Random" Numbers

#### 1. Architectural and Model Differences

Different models (GPT-4, Claude, Llama) show unique biases when generating numbers. They often favor certain numbers instead of a uniform distribution:

$$P_{\text{uniform}}(n) = \frac{1}{N}, \quad \text{but actual: } P_{\text{LLM}}(n) \neq \text{uniform}$$

#### 2. Prompt Language Variations

The language used in a prompt significantly affects output:

- Prompting in **Japanese** can create a preference for "1"
- Other languages favor "3"
- This indicates **language-specific cultural biases** within training data

#### 3. Inherent Bias vs. Randomness

LLMs frequently fall into predictable patterns when asked to generate a "random" number, rather than producing statistically random results.

#### 4. Token Probability

LLMs are designed to predict the **most likely next token**, which makes them inherently biased against true, uniform randomness.

The probability distribution is:

$$P(w) \propto \exp(\text{score}(w))$$

This naturally favors high-frequency tokens from training data.

### Human-Like Biases

When prompted for a "random" number, LLMs often:

- Avoid "tails" (extreme values like 1, 100)
- Over-represent central numbers (3, 7, 50)
- Reflect a **"central tendency bias"** found in human psychology

### Pseudo-Reasoning in Advanced Models

Models with **Chain-of-Thought** capabilities (e.g., DeepSeek-R1) may propose complex methods like:
- Taking the modulo of current time
- Using environmental variables
- Applying mathematical transformations

But often conclude with the same biased, deterministic outputs.

### Implications for Software Development

To build reliable AI systems, developers must **bridge the gap between probabilistic creativity and deterministic logic**:

#### The Probabilistic Component (LLM)

- **Use for**: Interpreting unstructured intent or generating creative drafts
- **Implementation**: Use OpenAI or Ollama library for Python
- **Control**: Adjust temperature (0.0 for near-deterministic, >1.0 for high randomness)

#### The Deterministic Component (Scripting)

- **Use for**: Tasks requiring absolute consistency (mathematical calculations, data validation)
- **Implementation**: Use Python's built-in `random` module or NumPy's random generator for genuine (pseudo)randomness
- **Reliability**: Deterministic code ensures identical inputs always yield same outputs, vital for auditability and compliance

---

## Stochastic Parrots or Intelligent Systems?

The term **"stochastic parrots"** was coined to describe LLMs as systems that merely repeat patterns from training data without true understanding. However, modern research suggests a more nuanced picture.

### What LLM Training Actually Does

LLM training relies on **stochasticity** in the form of stochastic gradient descent to build statistical representations of text-based language.

**"Parroting" is central to how LLMs learn**:
1. Given a bunch of text, the model follows each piece in sequence
2. The model is rewarded for correctly predicting the next piece
3. The result is a model where each piece of text is associated with numeric information about the sequences it tends to occur in

When prompted to generate, LLMs draw on learned representations to **parrot smaller spans of text** based on whether they're a probable continuation of what came before—that's yet another form of stochasticity.

### The Key Conceptual Point

LLM training works by consuming **text-based descriptions** of people's thoughts, intents, and experiences. They do not participate in these experiences themselves, but can produce descriptions of them as if they were there.

### Beyond Simple Parroting

Modern chatbots do something to get from a user's language-based input to a linguistically well-formed output (response). The processes within that arc are increasingly referred to as **"reasoning"**, though computationally-grounded terminology such as **"goal-based processing"** might be more appropriate.

Recent research highlights that sufficiently large LLMs (like GPT-4) move beyond simple pattern matching into:
- **In-context learning**: Solving tasks based on examples in the prompt
- **Compositional generalization**: Combining learned concepts in novel ways
- **Self-correction**: Identifying and fixing mistakes
- **Causal reasoning**: Understanding cause-and-effect relationships

### Distinguishing Metrics

| Stochastic Parrot | Intelligent System |
|-------------------|-------------------|
| High $P$ (copying patterns) | Low direct copying |
| Deterministic repetition | Adaptive responses |
| Low entropy | Variable entropy based on task |
| Fails novel constraints | Handles novel scenarios |
| Cannot self-correct | Can identify and fix errors |

---

## Testing if LLMs are Stochastic Parrots

Testing whether LLMs are "stochastic parrots" requires moving beyond simple text generation tasks. Here are scientifically rigorous approaches:

### 1. Test for Novelty and Synthesis (Zero-Shot Learning)

If an LLM is a parrot, it can only repeat training patterns. Present tasks unlikely to exist in training data:

#### Novel Creative Writing
Request a story in a highly specific, restricted style:
"Write a poem about quantum mechanics in the style of Shakespeare using only words that start with 'S'."

#### Invented Languages/Constraints
Ask the LLM to translate into a language you make up with new grammar rules. If it follows constraints, it's generating, not repeating.

### 2. Test for Spatial and Logical Reasoning

Stochastic parrots fail at tasks requiring a mental model of the world.

#### Physical Understanding
Test with novel physical scenarios using abstract input (like grid-based layouts) to check understanding of underlying phenomena.

#### Spatial Reasoning
"A is to the left of B, B is on top of C... if I rotate 90 degrees, where is A relative to C?"

#### Code Interpretation
Ask to analyze a new, non-functional code snippet and explain what it would do, or fix it based on custom constraints.

### 3. Self-Correction and Causal Reasoning

#### Self-Refine/Chain-of-Thought
Ask the model to solve a difficult math problem without giving the answer, then ask it to review its work, identify mistakes, and correct them. A parrot cannot identify its own mistakes.

#### Causal Queries
"What happens if X occurs?" where X breaks normal causal relationships (e.g., "What if gravity worked in reverse for only objects made of plastic?").

### 4. "Unlikely" Constraints

Parrots excel at predicting high-probability sequences. Force them into low-probability scenarios:

#### Forced Vocabulary
Ask for a summary of a topic while forbidding the 10 most common words associated with that topic.

#### Complex Formatting
Ask for output in a very specific, rare format (e.g., specific LaTeX structure combined with JSON) unlikely to be found together in training data.

### Evaluation Criteria

**Parrot indicators**:
- $P(\text{copying}) > 0.8$
- Deterministic, predictable responses
- Low entropy
- Fails when constraints are novel

**Not-a-parrot indicators**:
- High capability for interpolation/extrapolation
- Self-correction abilities
- Handles novel constraints
- Variable, context-appropriate responses

---

## How Randomness Prevents Parroting

Counter-intuitively, **randomness in LLMs actually prevents verbatim parroting** of training data, rather than causing it.

### The Mechanism

#### Without Randomness (Temperature 0, Greedy Decoding)

- Model always selects the highest probability token
- More likely to reproduce exact training sequences
- Output becomes deterministic and repetitive
- Higher risk of memorization

#### With Randomness (Temperature > 0, Sampling)

- Model samples from probability distribution
- Forces exploration of alternative phrasings
- Prevents exact reproduction of training text
- Encourages novel combinations

### Mathematical Explanation

With greedy decoding, if training data contains the exact sequence "The quick brown fox":

$$P(\text{"quick"} | \text{"The"}) = 0.9 \rightarrow \text{always select "quick"}$$

With sampling (temperature > 0):

$$P_\tau(\text{"quick"} | \text{"The"}) = 0.6 \rightarrow \text{might select "fast", "swift", etc.}$$

### Research Evidence

Studies show that:
1. **Lower temperature** → Higher verbatim reproduction from training data
2. **Higher temperature** → More novel combinations and paraphrasing
3. **Optimal temperature** (0.7-1.0) → Balance between coherence and creativity

### The "Stochastic Parrot" Paradox

The term "stochastic parrot" suggests randomness causes parroting, but:
- **Reality**: Stochasticity breaks memorization
- **Determinism**: More likely to parrot exact sequences
- **Controlled randomness**: Enables creative, non-verbatim responses

---

## Impact on Coding Assistants and AI Agents

### How Non-Determinism Affects Code Generation

AI coding assistants like GitHub Copilot (powered by models like Claude Sonnet) face unique challenges due to LLM non-determinism:

#### 1. Inconsistent Suggestions

The same prompt may generate:
- Different variable names
- Alternative implementations
- Varying code structures

This can be **beneficial** (exploring multiple solutions) or **problematic** (inconsistent team conventions).

#### 2. Debugging Challenges

When debugging AI-generated code:
- Cannot always reproduce the exact code that caused an issue
- May receive different "fixes" for the same problem
- Requires understanding the _pattern_ of errors, not specific instances

#### 3. Testing and Verification

Traditional unit testing assumes:

$$f(\text{input}) = \text{same output}$$

But with LLM-based code generation:

$$f_{\text{LLM}}(\text{prompt}, \text{context}) \neq \text{deterministic}$$

### Mitigation Strategies

#### For Developers

1. **Explicit constraints**: Specify exact requirements in prompts
2. **Code review**: Always review AI-generated code
3. **Style guides**: Use linters to enforce consistent formatting
4. **Seed control**: When possible, use deterministic settings for critical code

#### For AI Systems

1. **Lower temperature**: Use temperature 0-0.3 for code generation
2. **Structured output**: Enforce JSON schemas or type systems
3. **Multi-shot prompting**: Provide examples of desired output format
4. **Verification layers**: Add automated testing before code acceptance

### The Balancing Act

**Determinism wanted for**:
- Production code
- Mathematical calculations
- Security-critical functions
- Database operations

**Stochasticity valued for**:
- Creative problem-solving
- Multiple implementation approaches
- Exploratory coding
- Refactoring suggestions

---

## Reinforcement Learning and Non-Determinism

### Does RLHF Make LLMs Non-Deterministic?

**Reinforcement Learning from Human Feedback (RLHF)** does not inherently make LLMs non-deterministic, but it encourages behavior that can appear that way.

### What is Reinforcement Learning?

Imagine training a dog to sit. You say "Sit!" and when the dog sits, you give it a treat (positive reward). Over time, the dog associates sitting with rewards and is more likely to sit on command.

In RL:
- **Dog** → Agent (LLM)
- **You** → Environment
- **Treat** → Reward signal

The mathematical formulation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

Where:
- $s$ = state (current context)
- $a$ = action (generated token)
- $r$ = reward (human preference score)
- $\alpha$ = learning rate
- $\gamma$ = discount factor

### Aspects of RL-Induced Variability

#### 1. Behavioral Flexibility

RL increases the model's ability to choose from a wider range of high-quality responses, which may reduce consistency compared to strictly greedy, supervised pre-training.

#### 2. Reward Model Subjectivity

In methods like RLHF or Direct Preference Optimization (DPO), the reward model aligns the LLM with human preferences, which can be nuanced and lead to varied outputs depending on the prompt.

#### 3. On-Policy Learning

The need for true on-policy learning in RL requires exploring different outputs during training, though this happens during training, not necessarily inference.

### Non-RL Sources of Non-Determinism

Research indicates that even without RL, modern LLMs are rarely fully deterministic due to:

1. **Batch Invariance Failure**: The primary cause is varying batch sizes during inference
2. **Floating-Point Arithmetic**: Non-associativity in parallel hardware
3. **GPU Kernel Variance**: Different reduction splitting processes

Therefore, while **RL trains models to produce varied, human-preferred responses**, the lack of true determinism in production often stems from **how GPUs process queries**, not just the learning algorithm itself.

---

## Floating-Point Non-Associativity

### The Core Problem

Floating-point arithmetic in GPUs exhibits **non-associativity**, meaning:

$$(a + b) + c \neq a + (b + c)$$

due to finite precision and rounding errors. This property directly impacts computation of attention scores and logits in the transformer architecture.

### Mathematical Example

Consider adding three numbers in different orders:

```python
import torch

A = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
B = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)

ref = torch.mm(A, B)

for _ in range(1000):
    result = torch.mm(A, B)
    diff = (result - ref).abs().max().item()
    if diff != 0:
        print(f"Difference detected: {diff}")
```

### Why This Happens

#### 1. Finite Precision

Floating-point numbers have limited precision:
- **FP32**: 23 bits for mantissa
- **FP16**: 10 bits for mantissa
- **BF16**: 7 bits for mantissa

#### 2. Rounding Errors

Each operation introduces small rounding errors:

$$\text{fl}(a + b) = (a + b)(1 + \epsilon), \quad |\epsilon| \leq \epsilon_{\text{machine}}$$

#### 3. Order Dependency

In parallel processing, the order of operations depends on which thread finishes first:

$$\text{Thread 1: } (a_1 + a_2) + (a_3 + a_4)$$
$$\text{Thread 2: } ((a_1 + a_3) + a_2) + a_4$$

These can yield different results.

### Impact on Transformer Attention

The attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q, K, V$ are Query, Key, Value matrices
- Matrix multiplications ($QK^T$, result $\times V$) involve massive parallel summations
- Parallel operations across thousands of GPU threads introduce order dependency

### Impact on Logits and Token Selection

The final logits are computed as:

$$z = Wh + b$$

Where $W$ is a large weight matrix and $h$ is the hidden state. Even tiny differences in $z$ can change which token has the highest probability:

$$\arg\max_i z_i$$

### Solutions

#### 1. Batch-Invariant Kernels

Replace standard reduction operations with versions that:
- Use consistent operation ordering
- Maintain deterministic reduction paths
- Ensure identical results regardless of batch size

#### 2. Deterministic Algorithms

PyTorch offers:

```python
torch.use_deterministic_algorithms(True)
```

This forces deterministic implementations but may reduce performance.

#### 3. Higher Precision

Use FP32 instead of FP16/BF16 for critical calculations (at the cost of speed).

---

## Practical Implementation

This repository includes a Python implementation demonstrating the concepts covered in this document.

### Project Structure

```
📁 Stochastic/
├── 📄 README.md              # This file
├── 📄 QUICKSTART.md          # Setup instructions
├── 📄 .gitignore             # Python/venv exclusions
├── 📁 src/
│   ├── 📄 llm_randomness.py  # Main demonstration
│   ├── 📄 determinism_test.py # Reproducibility tests
│   └── 📄 float_precision.py  # Floating-point examples
├── 📁 venv/                  # Virtual environment (excluded from Git)
└── 📄 requirements.txt       # Python dependencies
```

### Use Case: The Psychology of LLMs as Random Number Generators

Based on research from [arXiv:2502.19965](https://arxiv.org/pdf/2502.19965), this implementation demonstrates how LLMs exhibit human-like biases when generating "random" numbers.

### What the Code Does

1. **Prompts LLMs** to generate random numbers
2. **Collects statistics** on number distributions
3. **Compares** to true uniform randomness
4. **Visualizes biases** (preference for certain numbers)
5. **Tests determinism** with temperature 0 vs. temperature > 0

### Requirements

- Python 3.8+
- Virtual environment (venv)
- Ollama (for local LLM inference)
- Docker (optional, for containerized Ollama)

### Setup Instructions

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions including:
- Virtual environment creation
- Ollama installation
- Docker setup (optional)
- Running the demonstrations

### Expected Output

The demonstration will show:

1. **Distribution Analysis**:
   - Histogram of generated numbers
   - Statistical tests for uniformity (Chi-square test)
   - Identification of biased numbers

2. **Determinism Tests**:
   - Identical outputs with temperature 0 and fixed seed
   - Varying outputs with temperature > 0
   - Impact of different prompts/languages

3. **Model Comparison**:
   - Different LLMs exhibit different biases
   - Architectural differences in randomness
   - Language-specific patterns

### Testing Determinism

The code includes tests to verify:

```python
# Test 1: Same seed, same output
set_seed(42)
output1 = generate_random_number()

set_seed(42)
output2 = generate_random_number()

assert output1 == output2  # Should be True

# Test 2: Different seeds, likely different output
set_seed(42)
output1 = generate_random_number()

set_seed(99)
output2 = generate_random_number()

assert output1 != output2  # Likely True
```

---

## Resources

### Research Papers

1. **Deterministic or probabilistic? The psychology of LLMs as random number generators**
   - arXiv: https://arxiv.org/pdf/2502.19965
   - Hugging Face: https://huggingface.co/papers/2502.19965

2. **Non-determinism in LLMs**
   - https://www.pamelatoman.net/blog/2023/08/nondeterminism-in-llms/

3. **Does Temperature 0 Guarantee Deterministic LLM Outputs?**
   - https://www.vincentschmalbach.com/does-temperature-0-guarantee-deterministic-llm-outputs/

4. **Defeating Non-determinism in LLM Inference**
   - OpenAI Community: https://community.openai.com/t/defeating-nondeterminism-in-llm-inference/1358623
   - Thinking Machines: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

5. **Deterministic Inference in SGLang**
   - https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/

### Articles

1. **LLMs are not Stochastic Parrots: How randomness prevents parroting**
   - https://pub.towardsai.net/llms-are-not-stochastic-parrots-how-randomness-prevents-parroting-not-causes-it-c44be684f890

2. **Reinforcement Learning from Human Feedback**
   - https://huggingface.co/learn/llm-course/chapter12/2

### Tools and Frameworks

- **Ollama**: https://ollama.ai/
- **Docker**: https://www.docker.com/
- **PyTorch**: https://pytorch.org/
- **NumPy**: https://numpy.org/

### Further Reading

- **Attention Is All You Need** (Transformer paper): https://arxiv.org/abs/1706.03762
- **On the Dangers of Stochastic Parrots** (Original paper): https://dl.acm.org/doi/10.1145/3442188.3445922
- **Monte Carlo Methods**: Understanding stochastic simulation
- **Pseudo-Random Number Generation**: Theory and practice

---

## Conclusion

The interplay between **deterministic** and **stochastic** processes in machine learning, particularly in large language models, is subtle and nuanced:

1. **LLMs are probabilistic systems** built on deterministic neural networks
2. **Stochasticity can be controlled** through seeds and parameters
3. **True randomness doesn't exist** in computers—only pseudo-randomness
4. **Non-determinism persists** even at temperature 0 due to hardware factors
5. **Randomness prevents parroting**, not causes it
6. **Understanding these principles** is crucial for building reliable AI systems

By understanding both the mathematical foundations and practical implications of stochastic vs. deterministic behavior, developers can build robust, reproducible, and reliable machine learning systems.

---

**License**: MIT
