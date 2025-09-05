#!/usr/bin/env python3
"""
Utility functions for PDW data input/output and processing.

This module provides helper functions for loading, saving, and processing
Pulse Descriptor Word (PDW) data in various formats.
"""

import numpy as np
import os
import json
import h5py  # Add h5py for HDF5 support
from typing import Dict, List, Tuple, Optional, Union

def load_pdw_data(file_path: str, format: str = 'npy', limit: Optional[int] = None) -> np.ndarray:
    """
    Load PDW data from file with an optional limit on the number of data points.
    
    Args:
        file_path: Path to PDW data file
        format: File format ('npy', 'csv', 'txt', 'hdf5')
        limit: Maximum number of data points to load (None for all data)
    
    Returns:
        pdw_data: Array of shape (n_pulses, 5) containing PDW features
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format == 'npy':
        data = np.load(file_path)
    elif format == 'csv':
        data = np.loadtxt(file_path, delimiter=',')
    elif format == 'txt':
        data = np.loadtxt(file_path)
    elif format == 'hdf5':
        with h5py.File(file_path, 'r') as f:
            data = f['pdw_data'][:]
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: npy, csv, txt, hdf5")
    
    if limit is not None:
        return data[:limit]
    return data