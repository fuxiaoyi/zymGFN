#!/usr/bin/env python3
"""
Sequence generation script using Hydra for configuration management.
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
    Main sequence generation function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    logger.info("Starting sequence generation with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create output directories
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Import and run the appropriate sequence generation module
    from src.GRPO.seq_gen import generate_sequences
    generate_sequences(cfg)
    
    logger.info("Sequence generation completed successfully!")

if __name__ == "__main__":
    main()
