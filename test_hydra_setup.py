#!/usr/bin/env python3
"""
Test script to verify Hydra setup is working correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import hydra
        print("✓ Hydra imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Hydra: {e}")
        return False
    
    try:
        import omegaconf
        print("✓ OmegaConf imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OmegaConf: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra
        with initialize(config_path="config"):
            cfg = compose(config_name="config")
            
        print("✓ Main configuration loaded successfully")
        print(f"  Project: {cfg.project_name}")
        print(f"  Model: {cfg.model.model_name}")
        print(f"  Training method: {cfg.training._target_}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def test_config_overrides():
    """Test that configuration overrides work."""
    print("\nTesting configuration overrides...")
    
    try:
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Test override
        with initialize(config_path="config"):
            cfg = compose(config_name="config", overrides=["training.learning_rate=1e-4"])
            
        if cfg.training.learning_rate == 1e-4:
            print("✓ Configuration override successful")
            return True
        else:
            print(f"✗ Configuration override failed: expected 1e-4, got {cfg.training.learning_rate}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test configuration overrides: {e}")
        return False

def test_script_execution():
    """Test that the main scripts can be executed."""
    print("\nTesting script execution...")
    
    try:
        # Test example script
        result = subprocess.run([
            sys.executable, "example_usage.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Example script executed successfully")
            return True
        else:
            print(f"✗ Example script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test script execution: {e}")
        return False

def main():
    """Run all tests."""
    print("=== zymGFN Hydra Setup Test ===\n")
    
    tests = [
        test_imports,
        test_config_loading,
        test_config_overrides,
        test_script_execution,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Hydra setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
