# 🚀 QUICK START

## Get Running

### Step 1: Activate Virtual Environment
```bash
cd "/home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Neural Networks"
source venv/bin/activate
```

### Step 2: Verify Installation
```bash
python -c "from src.transformer import Transformer; print('✓ All modules loaded')"
```

### Step 3: Run Example
```bash
python examples/simple_training.py
```

**Expected output:**
```
✓ Model initialized (9.3M parameters)
✓ Forward pass successful
✓ Generation working
```

---

## Common Tasks

### Train a Model
```bash
python train.py \
  --epochs 10 \
  --batch-size 32 \
  --model-size small \
  --checkpoint-dir checkpoints
```

### Run Inference
```bash
# Interactive mode
python inference.py \
  --model-path checkpoints/best_model.pt \
  --interactive

# Single prompt
python inference.py \
  --model-path checkpoints/best_model.pt \
  --prompt "Your text here" \
  --max-length 50
```

### Start REST API Server
```bash
# Start server
python server.py \
  --model-path checkpoints/best_model.pt \
  --port 5000
```

### Test API with curl
```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 50,
    "temperature": 0.8,
    "top_k": 50
  }'

# Encode text
curl -X POST http://localhost:5000/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_attention.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## File Locations

| Purpose | Location |
|---------|----------|
| Source code | `src/` |
| Training script | `train.py` |
| Inference script | `inference.py` |
| API server | `server.py` |
| Examples | `examples/` |
| Tests | `tests/` |
| Checkpoints | `checkpoints/` |
| Documentation | `README.md` |
| Validation results | `VALIDATION_REPORT.md` |
| Completion summary | `COMPLETION_SUMMARY.md` |

---

## Model Configurations

### Small (Default)
```python
vocab_size: 5000
d_model: 256
n_heads: 4
n_layers: 3
d_ff: 1024
parameters: 9.3M
```

### Base
```python
vocab_size: 32000
d_model: 512
n_heads: 8
n_layers: 6
d_ff: 2048
parameters: ~65M
```

### Large
```python
vocab_size: 50000
d_model: 1024
n_heads: 16
n_layers: 12
d_ff: 4096
parameters: ~335M
```

---

## Training Options

```bash
python train.py \
  --model-size small \          # or 'base', 'custom'
  --epochs 10 \                 # number of epochs
  --batch-size 32 \             # batch size
  --warmup-steps 4000 \         # warmup steps for lr scheduler
  --train-samples 10000 \       # training samples
  --val-samples 1000 \          # validation samples
  --checkpoint-dir models \     # checkpoint directory
  --save-every 1 \              # save every N epochs
  --no-cuda                     # disable CUDA (CPU only)
```

---

## Inference Options

```bash
python inference.py \
  --model-path checkpoints/best_model.pt \
  --prompt "Your prompt" \
  --max-length 100 \            # max generation length
  --temperature 0.8 \           # sampling temperature (0.1-2.0)
  --top-k 50 \                  # top-k sampling
  --top-p 0.9 \                 # nucleus sampling
  --interactive \               # interactive mode
  --no-cuda                     # use CPU
```

---

## Server Options

```bash
python server.py \
  --model-path checkpoints/best_model.pt \
  --port 5000 \                 # server port
  --host 0.0.0.0 \              # server host
  --no-cuda                     # use CPU
```

---

## Troubleshooting

### Module not found error
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Verify Python path
echo $PYTHONPATH
```

### CUDA out of memory
```bash
# Reduce batch size
python train.py --batch-size 8

# Or use CPU
python train.py --no-cuda
```

### Import errors
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Server not starting
```bash
# Check if port is in use
lsof -i :5000

# Use different port
python server.py --port 8000
```

---

## Development Workflow

### 1. Make Changes
Edit files in `src/` directory

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Test Changes
```bash
python examples/simple_training.py
```

### 4. Train Model
```bash
python train.py --epochs 5 --batch-size 16
```

### 5. Validate
```bash
python inference.py --model-path checkpoints/best_model.pt --interactive
```

---

## Performance Tips

### Training
- Use CUDA if available (30x faster)
- Increase batch size for better GPU utilization
- Use mixed precision training with `--fp16` (if implemented)
- Enable gradient accumulation for large batches

### Inference
- Reduce `max_length` for faster generation
- Lower `temperature` for more deterministic output
- Use `top_k` or `top_p` for better quality
- Batch multiple requests when possible

### API Server
- Use Gunicorn for production: `gunicorn -w 4 server:app`
- Enable caching for frequently used prompts
- Implement rate limiting for public APIs
- Use load balancing for multiple instances

---

## Useful Commands

```bash
# Check model size
ls -lh checkpoints/*.pt

# Count parameters
python -c "from src.transformer import Transformer; from src.config import TransformerConfig; \
  config = TransformerConfig.small(); \
  model = Transformer(config); \
  print(f'{sum(p.numel() for p in model.parameters()):,} parameters')"

# Monitor training
tail -f train.log

# Find process using port
lsof -i :5000

# Check GPU usage
nvidia-smi

# Test API availability
curl http://localhost:5000/health
```

---

## Next Steps

1. **Train on your data**: Prepare your dataset and start training
2. **Experiment with hyperparameters**: Try different model sizes and configurations
3. **Fine-tune**: Use pre-trained checkpoint for downstream tasks
4. **Deploy**: Set up production server with Gunicorn/nginx
5. **Optimize**: Implement caching, quantization, or distillation

---

## Help

- **Documentation**: See [README.md](README.md) for complete documentation
- **Code Examples**: Look in `examples/` directory
- **Tests**: Check `tests/` for usage patterns

---

**Start with:**
```bash
source venv/bin/activate && python examples/simple_training.py
```
