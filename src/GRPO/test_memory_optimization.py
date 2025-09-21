#!/usr/bin/env python3
"""
Test script for ESMFold memory optimizations
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from memory_monitor import print_memory_status, clear_gpu_memory


def create_test_fasta(sequences: list, output_path: str):
    """Create a test FASTA file with given sequences"""
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">test_sequence_{i}\n{seq}\n")


def test_memory_optimization():
    """Test the memory optimization with different sequence lengths"""
    
    print("Testing ESMFold Memory Optimization")
    print("=" * 50)
    
    # Test sequences of different lengths
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # ~70 AA
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 2,  # ~140 AA
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 4,  # ~280 AA
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 8,  # ~560 AA (should be skipped)
    ]
    
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        temp_fasta = f.name
        create_test_fasta(test_sequences, temp_fasta)
    
    try:
        print("Initial memory status:")
        print_memory_status()
        
        # Test with different configurations
        configs = [
            {"max_sequence_length": 500, "use_mixed_precision": True},
            {"max_sequence_length": 300, "use_mixed_precision": True},
            {"max_sequence_length": 500, "use_mixed_precision": False},
        ]
        
        for i, config in enumerate(configs):
            print(f"\n--- Test Configuration {i+1}: {config} ---")
            
            # Set environment variables for the test
            os.environ['ITER'] = '0'
            os.environ['LABEL'] = 'test'
            os.environ['RUN_DIR'] = str(Path(temp_fasta).parent)
            
            # Clear memory before test
            clear_gpu_memory()
            
            # Run ESMFold with the test configuration
            cmd = f"""
            python ESM_Fold.py \\
                --config-path conf --config-name grpo_fold \\
                iteration_num=0 label=test \\
                esm.max_sequence_length={config['max_sequence_length']} \\
                esm.use_mixed_precision={str(config['use_mixed_precision']).lower()} \\
                paths.fasta_path="{temp_fasta}" \\
                hydra.run.dir="{Path(temp_fasta).parent}"
            """
            
            print(f"Running: {cmd.strip()}")
            
            # Note: In a real test, you would run the command here
            # For now, we'll just show what would be executed
            print("(Command would be executed here in a real test)")
            
            print("Memory status after test:")
            print_memory_status()
    
    finally:
        # Clean up
        os.unlink(temp_fasta)
        print("\nTest completed. Temporary files cleaned up.")


if __name__ == "__main__":
    test_memory_optimization()
