#!/usr/bin/env python3
"""
Example script demonstrating how to use Hydra configuration in zymGFN.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def example_usage(cfg: DictConfig) -> None:
    """
    Example function showing how to access Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    logger.info("=== zymGFN Configuration Example ===")
    
    # Access configuration values
    logger.info(f"Project: {cfg.project_name}")
    logger.info(f"Version: {cfg.version}")
    logger.info(f"Model: {cfg.model.model_name}")
    logger.info(f"Learning Rate: {cfg.training.learning_rate}")
    logger.info(f"Batch Size: {cfg.training.batch_size}")
    logger.info(f"Enzyme Class: {cfg.data.enzyme_class}")
    logger.info(f"Device: {cfg.hardware.device}")
    
    # Access nested configuration
    logger.info(f"GRPO Beta: {cfg.training.grpo.beta}")
    logger.info(f"Data Directory: {cfg.paths.data_dir}")
    
    # Show how to modify configuration programmatically
    cfg.training.learning_rate = 2e-5
    logger.info(f"Modified Learning Rate: {cfg.training.learning_rate}")
    
    # Print full configuration
    logger.info("\n=== Full Configuration ===")
    logger.info(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    example_usage()
