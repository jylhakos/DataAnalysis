# Quick Start

This document will help you set up the development environment and run the stochastic demonstrations on Linux with VS Code.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Installing Dependencies](#installing-dependencies)
- [Ollama Setup](#ollama-setup)
- [Docker Setup (Optional)](#docker-setup-optional)
- [Running the Demonstrations](#running-the-demonstrations)
- [Verifying Determinism](#verifying-determinism)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **VS Code**: Latest version
- **Git**: For version control

Check your Python version:

```bash
python3 --version
```

---

## Virtual Environment Setup

**Virtual environments are essential** for Python development to isolate project dependencies.

### Step 1: Navigate to Project Directory

```bash
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Stochastic
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
```

This creates a `venv/` directory containing the isolated Python environment.

### Step 3: Activate Virtual Environment

```bash
source venv/bin/activate
```

After activation, your prompt should change to show `(venv)`:

```
(venv) user@laptop:~/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Stochastic$
```

### Step 4: Verify Activation

```bash
which python
# Should output: /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Stochastic/venv/bin/python
```

**Important**: Always activate the virtual environment before running scripts or installing packages.

### Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

---

## Installing Dependencies

With the virtual environment **activated**:

### Install All Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
pip list
```

You should see packages like:
- numpy
- scipy
- matplotlib
- torch
- ollama

### Individual Package Installation (if needed)

```bash
# Core packages
pip install numpy scipy matplotlib

# PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch (GPU version with CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ollama Python client
pip install ollama
```

---

## Ollama Setup

Ollama allows you to run LLMs locally for experimentation.

### Installation on Linux

#### Method 1: One-line Install Script

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Method 2: Manual Installation

1. Download the binary:

```bash
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/local/bin/
```

2. Start the service:

```bash
ollama serve
```

### Verify Installation

```bash
ollama --version
```

### Pull a Model

Download a lightweight model for testing:

```bash
# Small model (~2GB)
ollama pull llama3.2

# Alternative: Larger model for better results (~4GB)
ollama pull llama3.2:3b
```

### Start Ollama Service

In a **separate terminal**:

```bash
ollama serve
```

**Keep this terminal running** while executing the demonstrations.

### Test Ollama

In another terminal:

```bash
ollama run llama3.2 "Generate a random number between 1 and 10"
```

---

## Docker Setup (Optional)

Running Ollama in Docker provides isolation and easier management.

### Install Docker

```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Verify Docker Installation

```bash
sudo docker run hello-world
```

### Run Ollama in Docker

```bash
# Pull Ollama Docker image
docker pull ollama/ollama

# Run Ollama container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull a model inside the container
docker exec -it ollama ollama pull llama3.2
```

### Access Ollama from Docker

The service is now available at `http://localhost:11434`.

---

## Running the Demonstrations

Ensure:
1. Virtual environment is **activated** (`source venv/bin/activate`)
2. Ollama service is **running** (`ollama serve` in separate terminal or Docker)

### Demonstration 1: LLM Random Number Psychology

This demonstrates how LLMs exhibit biases when generating "random" numbers.

```bash
python src/llm_randomness.py
```

**Expected Output**:
- Statistical analysis of LLM-generated vs. true random numbers
- Visualization saved as `llm_randomness_comparison.png`
- Chi-square test results showing non-uniform distribution
- Language-specific bias patterns

**Duration**: ~5-10 minutes (depending on number of trials)

### Demonstration 2: Determinism Testing

Tests reproducibility with different random seeds.

```bash
python src/determinism_test.py
```

**Expected Output**:
- Proof that PRNGs are deterministic with fixed seeds
- SGD reproducibility demonstration
- Neural network inference determinism
- Dropout non-determinism examples

**Duration**: ~1-2 minutes

### Demonstration 3: Floating-Point Precision

Shows why temperature 0 doesn't guarantee determinism.

```bash
python src/float_precision.py
```

**Expected Output**:
- Floating-point non-associativity examples
- Precision loss in different formats (FP32, FP16, BF16)
- GPU matrix multiplication variance
- Attention mechanism sensitivity

**Duration**: ~2-3 minutes

**Note**: GPU demonstrations require CUDA. On CPU-only systems, some tests will show deterministic behavior.

---

## Verifying Determinism

### Test 1: Same Seed Reproducibility

Run the same script twice with the same seed:

```bash
python src/determinism_test.py > output1.txt
python src/determinism_test.py > output2.txt
diff output1.txt output2.txt
```

**Expected**: No output (files are identical) = Deterministic

### Test 2: LLM Temperature 0

Edit `src/llm_randomness.py` to set temperature to 0.0 and run multiple times:

```python
test_determinism(model="llama3.2", temperature=0.0, n_trials=10)
```

**Expected**: Most outputs are identical, but small variations may occur due to floating-point errors.

### Test 3: GPU vs CPU

If you have a GPU:

```bash
# Run on GPU
python src/float_precision.py

# Run on CPU (slower but deterministic)
CUDA_VISIBLE_DEVICES="" python src/float_precision.py
```

Compare the matrix multiplication results.

---

## Troubleshooting

### Problem: Virtual environment not activating

**Solution**:
```bash
# Make sure you're in the project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Stochastic

# Try activating again
source venv/bin/activate

# If it doesn't exist, create it
python3 -m venv venv
```

### Problem: `ModuleNotFoundError: No module named 'ollama'`

**Solution**:
```bash
# Ensure venv is activated
source venv/bin/activate

# Install ollama
pip install ollama
```

### Problem: Ollama connection error

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not, start it
ollama serve

# Or restart Docker container
docker restart ollama
```

### Problem: `RuntimeError: Ollama package not installed`

**Solution**:
```bash
# Activate venv
source venv/bin/activate

# Install ollama
pip install ollama

# Verify
python -c "import ollama; print(ollama.__version__)"
```

### Problem: CUDA out of memory

**Solution**:
```bash
# Use a smaller model
ollama pull llama3.2:1b

# Or reduce batch size in scripts
# Edit src/llm_randomness.py: n_trials=20 (instead of 100)
```

### Problem: Plots not displayed

**Solution**:

If running over SSH without X11 forwarding:

```bash
# Install non-GUI backend
pip install matplotlib
```

Edit scripts to use:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Problem: Permission denied when installing packages

**Solution**:

Never use `sudo` with pip in a virtual environment:

```bash
# Wrong
sudo pip install numpy

# Correct
source venv/bin/activate
pip install numpy
```

---

## VS Code Integration

### Python Extension

Install the Python extension in VS Code:
1. Press `Ctrl+Shift+X`
2. Search for "Python"
3. Install the Microsoft Python extension

### Select Virtual Environment

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the venv interpreter: `./venv/bin/python`

### Running Scripts in VS Code

1. Open a Python file (e.g., `src/llm_randomness.py`)
2. Press `Ctrl+F5` to run without debugging
3. Or use the integrated terminal:
   ```bash
   source venv/bin/activate
   python src/llm_randomness.py
   ```

### Debugging

1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Use Debug Console to inspect variables

---

## Project Structure

```
📁 Stochastic/
├── 📄 README.md              # Comprehensive theory documentation
├── 📄 QUICKSTART.md          # This file
├── 📄 .gitignore             # Git exclusions
├── 📄 requirements.txt       # Python dependencies
├── 📁 venv/                  # Virtual environment (not in Git)
├── 📁 src/
│   ├── 📄 llm_randomness.py  # LLM psychology demonstration
│   ├── 📄 determinism_test.py # Reproducibility tests
│   └── 📄 float_precision.py  # Floating-point examples
└── 📄 llm_randomness_comparison.png  # Generated visualization
```

---

## Next Steps

1. **Read the Theory**: Open [README.md](README.md) for in-depth explanations
2. **Run Demonstrations**: Execute all three Python scripts
3. **Experiment**: Modify parameters (temperature, model, language)
4. **Compare Results**: Test different LLM models for bias patterns
5. **Visualize**: Analyze the generated plots

---

## Additional Resources

### Ollama Commands

```bash
# List installed models
ollama list

# Remove a model
ollama rm llama3.2

# Update a model
ollama pull llama3.2:latest

# View model info
ollama show llama3.2
```

### Docker Commands

```bash
# View running containers
docker ps

# Stop Ollama container
docker stop ollama

# Start Ollama container
docker start ollama

# View logs
docker logs ollama

# Remove container
docker rm ollama
```

### Virtual Environment Commands

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Deactivate
deactivate

# Remove venv
rm -rf venv
```

---

## Testing Your Setup

Run this comprehensive test:

```bash
#!/bin/bash

echo "Testing Stochastic Project Setup..."

# 1. Check Python
echo "1. Python version:"
python --version

# 2. Check venv
echo "2. Virtual environment:"
which python

# 3. Check packages
echo "3. Installed packages:"
pip list | grep -E "numpy|torch|ollama|scipy|matplotlib"

# 4. Check Ollama
echo "4. Ollama status:"
curl -s http://localhost:11434/api/version || echo "Ollama not running"

# 5. Test imports
echo "5. Testing imports:"
python -c "import numpy, torch, scipy, matplotlib; print('All imports successful')"

# 6. Test Ollama (optional)
echo "6. Testing Ollama (if running):"
python -c "import ollama; print(ollama.list())" 2>/dev/null || echo "Ollama not available"

echo "Setup test complete!"
```

Save as `test_setup.sh`, make executable, and run:

```bash
chmod +x test_setup.sh
./test_setup.sh
```

---

## FAQ

**Q: Do I need a GPU?**
A: No, but demonstrations run faster with GPU. Some floating-point variance tests specifically show GPU non-determinism behavior.

**Q: Which Python version should I use?**
A: Python 3.8 or higher. We recommend Python 3.10 for best compatibility.

**Q: Can I use a different LLM model?**
A: Yes! Try `ollama pull mistral` or `ollama pull codellama` and modify the scripts.

**Q: Why is the virtual environment in .gitignore?**
A: Virtual environments should not be committed to Git. They're system-specific and can be recreated from `requirements.txt`.

**Q: How do I update dependencies?**
A: Run `pip install --upgrade -r requirements.txt`

**Q: Can I run this on Windows or macOS?**
A: Yes, but activation command differs:
- Windows: `venv\Scripts\activate`
- macOS: `source venv/bin/activate` (same as Linux)

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review README.md for theoretical background
3. Check Ollama documentation: https://ollama.com/

---

## License

MIT License - See README.md for details

---
