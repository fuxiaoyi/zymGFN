# zymGFN: Protein Language Model for Enzyme Design

A reinforcement learning framework for designing enzymes using protein language models, featuring GRPO (Group Relative Policy Optimization) and DPO (Direct Preference Optimization) training methods.

## 🚀 Quick Start

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd zymGFN

# Create and activate conda environment
conda env create -f environment.yml
conda activate zymgfn

# Verify installation
python -c "import hydra; print('Hydra installed successfully!')"
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd zymGFN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Training with GRPO

```bash
# Train with default configuration
python train_hydra.py

# Train with specific enzyme class
python train_hydra.py data.enzyme_class=4.2.1.1

# Train with different model
python train_hydra.py model=esm2_large

# Train with custom parameters
python train_hydra.py training.learning_rate=1e-4 training.batch_size=8
```

#### 2. Sequence Generation

```bash
# Generate sequences with default configuration
python generate_sequences_hydra.py

# Generate sequences for specific enzyme class
python generate_sequences_hydra.py data.enzyme_class=3.1.1.1

# Generate with custom parameters
python generate_sequences_hydra.py data.max_length=512
```

#### 3. Using Different Training Methods

```bash
# Use DPO instead of GRPO
python train_hydra.py training=dpo

# Use CPU instead of GPU
python train_hydra.py hardware=cpu
```

## 📁 Project Structure

```
zymGFN/
├── config/                     # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── model/                 # Model configurations
│   │   ├── esm2.yaml
│   │   └── esm2_large.yaml
│   ├── training/              # Training configurations
│   │   ├── grpo.yaml
│   │   └── dpo.yaml
│   ├── data/                  # Data configurations
│   │   └── enzyme_dataset.yaml
│   └── hardware/              # Hardware configurations
│       ├── gpu.yaml
│       └── cpu.yaml
├── src/                       # Source code
│   ├── GRPO/                  # GRPO implementation
│   ├── s-FT/                  # Supervised fine-tuning
│   └── wDPO/                  # Weighted DPO implementation
├── data/                      # Data directory
├── model/                     # Model checkpoints
├── outputs/                   # Training outputs
├── logs/                      # Log files
├── train_hydra.py            # Main training script
├── generate_sequences_hydra.py # Sequence generation script
├── environment.yml            # Conda environment
├── requirements.txt           # pip dependencies
└── README.md
```

## ⚙️ Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. All configuration files are located in the `config/` directory.

### Key Configuration Files

- **`config/config.yaml`**: Main configuration file
- **`config/model/`**: Model-specific configurations (ESM2 variants)
- **`config/training/`**: Training method configurations (GRPO, DPO)
- **`config/data/`**: Dataset configurations
- **`config/hardware/`**: Hardware-specific settings (GPU/CPU)

### Customizing Configurations

You can override any configuration parameter from the command line:

```bash
# Override multiple parameters
python train_hydra.py \
    data.enzyme_class=4.2.1.1 \
    training.learning_rate=1e-4 \
    training.batch_size=16 \
    model=esm2_large

# Use different data directory
python train_hydra.py paths.data_dir=/path/to/your/data

# Enable experiment tracking
python train_hydra.py experiment.use_wandb=true experiment.entity=your_entity
```

## 🔬 Methods

### GRPO (Group Relative Policy Optimization)
- Group-based reinforcement learning for protein design
- Optimizes sequences based on structural similarity and functional properties
- Uses TM-score and clustering for evaluation

### DPO (Direct Preference Optimization)
- Direct preference optimization for protein sequences
- Trains on preference data without explicit reward modeling
- Suitable for fine-tuning on specific enzyme classes

### iterative Supervised Fine-tuning (s-FT)
- Traditional supervised learning approach
- Fine-tunes pre-trained models on specific tasks
- Good baseline for comparison

## 📊 Data Requirements

The framework expects the following data structure:

```
data/
├── GRPO/
│   ├── dataset/              # Training datasets
│   ├── generated_sequences/  # Generated sequences
│   ├── PDB/                  # PDB structures
│   └── TMscores/            # TM-score results
├── sFT/
│   ├── dataset/
│   ├── generated_sequences/
│   └── PDB/
└── wDPO/
    ├── dataset/
    ├── generated_sequences/
    └── PDB/
```

## 🛠️ Advanced Usage

### Custom Model Configuration

Create a new model configuration file in `config/model/`:

```yaml
# config/model/custom_model.yaml
# @package model
_target_: transformers.AutoModelForCausalLM

model_name: "your-model-name"
pretrained: true
# ... other parameters
```

Then use it with:
```bash
python train_hydra.py model=custom_model
```

### Multi-GPU Training

For multi-GPU training, modify the hardware configuration:

```yaml
# config/hardware/multi_gpu.yaml
device: "cuda"
gpu_ids: [0, 1, 2, 3]
distributed:
  enabled: true
  world_size: 4
```

### Experiment Tracking

The framework supports Weights & Biases for experiment tracking:

```bash
# Enable wandb logging
python train_hydra.py experiment.use_wandb=true experiment.entity=your_entity
```

## 📈 Monitoring and Logging

- **Logs**: Stored in `logs/` directory
- **Outputs**: Training outputs saved in `outputs/`
- **Checkpoints**: Model checkpoints saved in `model/`
- **Wandb**: Optional experiment tracking (configure in `experiment` section)

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
   ```bash
   python train_hydra.py training.batch_size=2 hardware.gradient_checkpointing=true
   ```

2. **Missing dependencies**: Ensure all packages are installed
   ```bash
   conda env update -f environment.yml
   ```

3. **Configuration errors**: Check Hydra configuration syntax
   ```bash
   python train_hydra.py --cfg job
   ```

### Getting Help

- Check the logs in `logs/` directory for detailed error messages
- Use `--help` flag for command-line help:
  ```bash
  python train_hydra.py --help
  ```

## 📚 References

- [ProtRL Paper](https://arxiv.org/abs/2412.12979)
- [ESM2 Paper](https://www.science.org/doi/full/10.1126/science.ade2574)
- [GRPO Method](https://arxiv.org/abs/2402.03300)
- [DPO Method](https://dl.acm.org/doi/abs/10.5555/3666122.3668460)
- [Hydra Documentation](https://hydra.cc/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📧 Contact

For questions and support, please open an issue on GitHub or contact the maintainers.