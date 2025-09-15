#!/usr/bin/env python3
"""
Main training script using Hydra for configuration management.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    logger.info("Starting training with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create output directories
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Set up experiment tracking
    if cfg.experiment.use_wandb:
        import wandb
        wandb.init(
            project=cfg.experiment.project,
            entity=cfg.experiment.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.experiment.tags
        )
    
    # Import and run the appropriate training module
    if cfg.training._target_ == "grpo":
        from src.GRPO.train import train_grpo
        train_grpo(cfg)
    elif cfg.training._target_ == "dpo":
        from src.wDPO.train import train_dpo
        train_dpo(cfg)
    else:
        raise ValueError(f"Unknown training method: {cfg.training._target_}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
