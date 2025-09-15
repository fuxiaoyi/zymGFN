# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate zymgfn

# Or using pip
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python test_hydra_setup.py
```

### 3. Run Example

```bash
python example_usage.py
```

### 4. Start Training

```bash
# Basic training
python train_hydra.py

# With custom parameters
python train_hydra.py training.learning_rate=1e-4 data.enzyme_class=3.1.1.1
```

### 5. Generate Sequences

```bash
# Basic generation
python generate_sequences_hydra.py

# With custom parameters
python generate_sequences_hydra.py data.enzyme_class=4.2.1.1
```

## 🔧 Common Commands

### Training Commands

```bash
# Use different training methods
python train_hydra.py training=grpo
python train_hydra.py training=dpo

# Use different models
python train_hydra.py model=esm2
python train_hydra.py model=esm2_large

# Use CPU instead of GPU
python train_hydra.py hardware=cpu

# Override multiple parameters
python train_hydra.py \
    data.enzyme_class=4.2.1.1 \
    training.learning_rate=1e-4 \
    training.batch_size=8
```

### Configuration Commands

```bash
# Show current configuration
python train_hydra.py --cfg job

# Show help
python train_hydra.py --help

# Override specific config groups
python train_hydra.py model=esm2_large training=dpo hardware=cpu
```

## 📁 Directory Structure

```
zymGFN/
├── config/           # Hydra configurations
├── src/             # Source code
├── data/            # Data directory
├── outputs/         # Training outputs
├── logs/            # Log files
└── model/           # Model checkpoints
```

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've activated the conda environment
   ```bash
   conda activate zymgfn
   ```

2. **CUDA errors**: Use CPU mode
   ```bash
   python train_hydra.py hardware=cpu
   ```

3. **Configuration errors**: Check the config files in `config/` directory

### Getting Help

- Run the test script: `python test_hydra_setup.py`
- Check logs in `logs/` directory
- Use `--help` flag for command-line help

## 📚 Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the configuration files in `config/` directory
3. Check the source code in `src/` directory
4. Run experiments with different parameters

Happy coding! 🎉
