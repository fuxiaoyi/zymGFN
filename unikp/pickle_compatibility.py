#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility layer for loading scikit-learn models saved with older versions.
This handles the dtype incompatibility issue in tree-based models between 0.24.x and 1.3.0.
"""

import pickle
import numpy as np
import warnings
from sklearn.tree._tree import Tree


class CompatibleUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles scikit-learn version compatibility issues.
    Specifically fixes the dtype mismatch in tree node arrays.
    """
    
    def find_class(self, module, name):
        """
        Override to intercept Tree class loading and apply compatibility fixes.
        """
        # Get the original class
        cls = super().find_class(module, name)
        
        # If this is a Tree class, wrap it with our compatibility layer
        if module == 'sklearn.tree._tree' and name == 'Tree':
            return self._create_compatible_tree_class(cls)
        
        return cls
    
    def _create_compatible_tree_class(self, original_tree_class):
        """
        Create a Tree class that handles the dtype compatibility issue.
        """
        class CompatibleTree(original_tree_class):
            def __setstate__(self, state):
                """
                Override __setstate__ to fix node array dtype before setting state.
                """
                if 'nodes' in state and isinstance(state['nodes'], np.ndarray):
                    # Expected dtype for scikit-learn 1.3.0
                    expected_dtype = np.dtype([
                        ('left_child', '<i8'),
                        ('right_child', '<i8'), 
                        ('feature', '<i8'),
                        ('threshold', '<f8'),
                        ('impurity', '<f8'),
                        ('n_node_samples', '<i8'),
                        ('weighted_n_node_samples', '<f8'),
                        ('missing_go_to_left', 'u1')
                    ])
                    
                    # Check if we need to fix the dtype
                    if state['nodes'].dtype != expected_dtype:
                        print(f"Fixing tree nodes dtype from {state['nodes'].dtype} to {expected_dtype}")
                        
                        # Create new array with correct dtype
                        new_nodes = np.zeros(state['nodes'].shape, dtype=expected_dtype)
                        
                        # Copy existing fields
                        for field in state['nodes'].dtype.names:
                            if field in expected_dtype.names:
                                new_nodes[field] = state['nodes'][field]
                        
                        # Set missing_go_to_left to 0 (default behavior)
                        new_nodes['missing_go_to_left'] = 0
                        
                        # Replace the nodes array
                        state['nodes'] = new_nodes
                
                # Call the original __setstate__
                return super().__setstate__(state)
        
        return CompatibleTree


def safe_model_load(file_path):
    """
    Safely load a scikit-learn model with compatibility fixes.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        The loaded model object
        
    Raises:
        Exception: If all loading methods fail
    """
    try:
        # First try normal loading
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except (ValueError, TypeError) as e:
        if 'incompatible dtype' in str(e) or 'node array' in str(e):
            print(f"Detected compatibility issue, trying compatibility fix...")
            try:
                # Use our custom unpickler
                with open(file_path, 'rb') as f:
                    unpickler = CompatibleUnpickler(f)
                    model = unpickler.load()
                return model
            except Exception as e2:
                print(f"Compatibility fix failed: {e2}")
                # Try with warnings suppressed as last resort
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(file_path, 'rb') as f:
                            unpickler = CompatibleUnpickler(f)
                            model = unpickler.load()
                    return model
                except Exception as e3:
                    print(f"All compatibility fixes failed: {e3}")
                    raise e2
        else:
            raise e


def load_with_downgrade_fix(file_path):
    """
    Alternative approach: Load the pickle file and manually fix the tree structures.
    This is a more aggressive approach that works by loading the raw pickle data.
    """
    try:
        # Load the raw pickle data
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Try to load with our custom unpickler
        import io
        f = io.BytesIO(raw_data)
        unpickler = CompatibleUnpickler(f)
        model = unpickler.load()
        return model
        
    except Exception as e:
        print(f"Downgrade fix failed: {e}")
        raise e

