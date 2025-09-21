#!/usr/bin/env python3
"""
GPU Memory Monitor for ESMFold
Helps debug CUDA memory issues by monitoring GPU usage
"""

import torch
import time
import psutil
import os
from typing import Dict, Tuple


def get_gpu_memory_info() -> Dict[str, float]:
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    free = total - allocated
    
    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free,
        "utilization_percent": (allocated / total) * 100
    }


def get_system_memory_info() -> Dict[str, float]:
    """Get system memory information"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / 1024**3,
        "available_gb": memory.available / 1024**3,
        "used_gb": memory.used / 1024**3,
        "utilization_percent": memory.percent
    }


def print_memory_status():
    """Print current memory status"""
    print("=" * 60)
    print(f"Memory Status at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # GPU Memory
    gpu_info = get_gpu_memory_info()
    if gpu_info["available"]:
        print("GPU Memory:")
        print(f"  Total: {gpu_info['total_gb']:.2f} GB")
        print(f"  Allocated: {gpu_info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {gpu_info['reserved_gb']:.2f} GB")
        print(f"  Free: {gpu_info['free_gb']:.2f} GB")
        print(f"  Utilization: {gpu_info['utilization_percent']:.1f}%")
    else:
        print("GPU Memory: Not available")
    
    # System Memory
    sys_info = get_system_memory_info()
    print("\nSystem Memory:")
    print(f"  Total: {sys_info['total_gb']:.2f} GB")
    print(f"  Used: {sys_info['used_gb']:.2f} GB")
    print(f"  Available: {sys_info['available_gb']:.2f} GB")
    print(f"  Utilization: {sys_info['utilization_percent']:.1f}%")
    
    # Process Memory
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    print(f"\nProcess Memory: {process_memory.rss / 1024**3:.2f} GB")
    print("=" * 60)


def monitor_memory_usage(duration_seconds: int = 60, interval_seconds: int = 5):
    """Monitor memory usage for a specified duration"""
    print(f"Monitoring memory for {duration_seconds} seconds (interval: {interval_seconds}s)")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            print_memory_status()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("Final memory status:")
    print_memory_status()


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    else:
        print("No GPU available to clear")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Memory Monitor")
    parser.add_argument("--monitor", type=int, default=0, 
                       help="Monitor for N seconds (0 = single snapshot)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Monitoring interval in seconds")
    parser.add_argument("--clear", action="store_true",
                       help="Clear GPU memory cache")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_gpu_memory()
    
    if args.monitor > 0:
        monitor_memory_usage(args.monitor, args.interval)
    else:
        print_memory_status()
