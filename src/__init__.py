"""
PINN3D: Physics-Informed Neural Networks for 3D Advection-Diffusion Problems
==============================================================================

A comprehensive Python package for solving 3D advection-diffusion problems 
using Physics-Informed Neural Networks (PINNs).

Main Components:
    - DataGenerator: Generate training data for collocation, boundary, and initial conditions
    - Network3D: Neural network architectures for 3D problems
    - AdvectionDiffusion3D: PDE residual computation for 3D advection-diffusion
    - PINNTrainer: Training engine with sequence-to-sequence methodology

Examples:
    >>> import torch
    >>> from pinn3D import Network3D, DataGenerator, AdvectionDiffusion3D, PINNTrainer
    >>> 
    >>> # Create network
    >>> network = Network3D.create_from_config({'layer_sizes': [64, 64, 64]}, device)
    >>> 
    >>> # Create data generator
    >>> data_gen = DataGenerator(domain=(0,10,0,10,0,5), time_horizon=3600, ...)
    >>> 
    >>> # Train model
    >>> trainer = PINNTrainer(network, pde, data_gen, kx=0.1, ky=0.1, kz=0.1)
    >>> trainer.train_seq2seq(N_segments=10, N_col=5000, N_epochs=100)
"""

# Import version info
from ._version import __version__, __author__, __email__, __license__

# Import core classes
from .DataGenerator import DataGenerator
from .Network3D import Network3D
from .AdvectionDiffusion3D import AdvectionDiffusion3D
from .PINNTrainer import PINNTrainer

# Define public API
__all__ = [
    # Core classes
    'DataGenerator',
    'Network3D', 
    'AdvectionDiffusion3D',
    'PINNTrainer',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

# Package metadata
__title__ = "pinn3D"
__description__ = "Physics-Informed Neural Networks for 3D Advection-Diffusion Problems"
__url__ = "https://github.com/yourusername/pinn3D" 