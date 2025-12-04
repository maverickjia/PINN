"""
Physics-Informed Neural Network (PINN) for 3D Advection-Diffusion PDEs.

This module provides a neural network architecture specifically designed for solving
3D advection-diffusion partial differential equations using the PINN approach.
The network takes spatial coordinates (x, y, z) and time t as input and outputs
the concentration field c(x, y, z, t).

Classes:
    Network3D: A fully connected neural network for 3D PDE solving with PINN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Dict, Any, List, Union
import warnings


class Network3D(nn.Module):
    """
    Physics-Informed Neural Network for 3D Advection-Diffusion PDEs.
    
    This network is specifically designed for solving 3D advection-diffusion
    partial differential equations using the PINN approach. It takes spatial
    coordinates (x, y, z) and time t as input and outputs the concentration
    field c(x, y, z, t).
    
    The network architecture is a fully connected feedforward neural network
    with configurable hidden layers and activation functions.
    
    Output Activation Options:
    - 'linear': Returns network_output * scale (raw output with learnable scaling)
    - 'exp': Returns exp(network_output * scale) for non-negative outputs with learnable scaling
    
    Attributes:
        input_dim (int): Input dimension (4 for x, y, z, t)
        output_dim (int): Output dimension (1 for concentration c)
        hidden_activation (callable): Activation function for hidden layers
        output_activation (callable): Activation function for output layer
        output_activation_type (str): Type of output activation ('linear' or 'exp')
        init_method (str): Weight initialization method
        device (torch.device): Device to run the network on
        layers (nn.ModuleList): List of linear layers
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = 'tanh',
        init_method: str = 'glorot',
        load_path: Optional[str] = None,
        freeze_hidden: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        output_activation: str = 'exp'  
    ) -> None:
        """
        Initialize the PINN network.
        
        Args:
            layer_sizes: List of neurons per hidden layer, e.g. [50, 50, 50, 50]
            hidden_activation: Activation function for hidden layers ('tanh', 'relu', 'sigmoid')
            init_method: Weight initialization method ('glorot', 'he', 'normal', 'zeros')
            load_path: Path to pretrained weights file (optional)
            freeze_hidden: Whether to freeze hidden layers for transfer learning
            device: Device to run the network on (auto-detected if None)
            dtype: torch dtype for all layers and computations (default: torch.float32)
            output_activation: Output activation function ('exp' [default] for non-negativity with learnable scaling, or 'linear' for raw output with learnable scaling)
        Raises:
            ValueError: If hidden_activation or init_method is unsupported
            FileNotFoundError: If load_path is provided but file doesn't exist
            RuntimeError: If there's a mismatch between saved and current architecture
        """
        super(Network3D, self).__init__()
        
        # Validate inputs
        if not isinstance(layer_sizes, list) or len(layer_sizes) == 0:
            raise ValueError("layer_sizes must be a non-empty list of integers")
        
        if not all(isinstance(size, int) and size > 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers")
        
        # Set network dimensions
        self.input_dim = 4  # x, y, z, t
        self.output_dim = 1  # c (concentration)
        self.layer_sizes = layer_sizes.copy()  # Store for reference
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set initialization method
        self.init_method = init_method.lower()
        
        # Set hidden activation function
        self._set_hidden_activation(hidden_activation)
        
        # Set output activation
        self._set_output_activation(output_activation)
        
        # Set dtype
        self.dtype = dtype
        
        # Build network layers
        self._build_layers()
        
        # Add a learnable parameter for output scaling
        # Using log_scale ensures the final scale = exp(log_scale) is always positive,
        # allowing unconstrained optimization while maintaining numerical stability
        self.log_output_scale = nn.Parameter(torch.tensor(0.0, dtype=self.dtype))
        
        # Load pretrained weights or initialize from scratch
        if load_path is not None:
            self._load_pretrained_weights(load_path, freeze_hidden)
        else:
            # Initialize weights from scratch
            self.apply(self._initialize_weights)
            if freeze_hidden:
                self._freeze_hidden_layers()
        
        # Move network to device
        self.to(device=self.device, dtype=self.dtype)
    
    def _set_hidden_activation(self, hidden_activation: str) -> None:
        """
        Set the activation function for hidden layers.
        
        Args:
            hidden_activation: Hidden activation function name
            
        Raises:
            ValueError: If hidden_activation is not supported
        """
        if hidden_activation == 'tanh':
            self.hidden_activation = torch.tanh
        elif hidden_activation == 'relu':
            self.hidden_activation = F.relu
        elif hidden_activation == 'sigmoid':
            self.hidden_activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported hidden_activation: {hidden_activation}. "
                           f"Supported: 'tanh', 'relu', 'sigmoid'")
    
    def _set_output_activation(self, output_activation: str) -> None:
        """
        Set the activation function for the output layer.
        Args:
            output_activation: Output activation function name ('linear' or 'exp')
        Raises:
            ValueError: If output_activation is not supported
        """
        if output_activation == 'linear':
            self.output_activation = lambda x: x
            self.output_activation_type = 'linear'
        elif output_activation == 'exp':
            self.output_activation = torch.exp
            self.output_activation_type = 'exp'
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}. Supported: 'linear', 'exp'")
    
    def _build_layers(self) -> None:
        """Build the neural network layers."""
        layers = []
        prev_dim = self.input_dim
        
        # Add hidden layers
        for hidden_dim in self.layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim, dtype=self.dtype))
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, self.output_dim, dtype=self.dtype))
        self.layers = nn.ModuleList(layers)
    
    def _load_pretrained_weights(self, load_path: str, freeze_hidden: bool) -> None:
        """
        Load pretrained weights from file.
        
        Args:
            load_path: Path to the pretrained weights file
            freeze_hidden: Whether to freeze hidden layers after loading
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If there's an architecture mismatch
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            # Load state dict with weights_only=True for security
            state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
            # Support loading from a checkpoint file (as saved by PINNTrainer.save_checkpoint)
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'network_state_dict' in state_dict:
                    state_dict = state_dict['network_state_dict']
            # Load weights
            load_result = self.load_state_dict(state_dict, strict=False)
            # Report loading results
            if load_result.missing_keys:
                warnings.warn(f"Missing keys in checkpoint: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                warnings.warn(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("✅ Checkpoint loaded successfully - all parameters matched")
            else:
                print("⚠️  Checkpoint loaded with some parameter mismatches")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {load_path}: {str(e)}")
        if freeze_hidden:
            self._freeze_hidden_layers()
    
    def _initialize_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for a single module.
        
        Args:
            module: The module to initialize
        """
        if isinstance(module, nn.Linear):
            if self.init_method in ['glorot', 'xavier']:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif self.init_method == 'he':
                nonlinearity = 'relu' if self.hidden_activation == F.relu else 'tanh'
                nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
                nn.init.zeros_(module.bias)
            elif self.init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                nn.init.zeros_(module.bias)
            elif self.init_method == 'zeros':
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            else:
                raise ValueError(f"Unsupported init_method: {self.init_method}. "
                               f"Supported: 'glorot', 'he', 'normal', 'zeros'")
    
    def _freeze_hidden_layers(self) -> None:
        """
        Freeze all hidden layers (all layers except the last one).
        
        This is useful for transfer learning where you want to keep
        the learned features from pretrained weights and only train
        the output layer.
        """
        for layer in self.layers[:-1]:  # All except the last layer
            for param in layer.parameters():
                param.requires_grad = False
        print("🔒 Hidden layers frozen (only output layer trainable)")
    
    def _unfreeze_all_layers(self) -> None:
        """
        Unfreeze all layers.
        
        This makes all parameters trainable again after freezing.
        """
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("🔓 All layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4) containing (x, y, z, t) coordinates
            
        Returns:
            Output tensor of shape (batch_size, 1) containing concentration values.
            For 'linear' output_activation: returns network_output * scale where scale = exp(log_output_scale)
            For 'exp' output_activation: returns exp(network_output * scale) where scale = exp(log_output_scale)
            
        Raises:
            ValueError: If input tensor has wrong shape
        """
        # Validate input shape
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), "
                           f"got {x.shape}")
        
        # Forward pass through layers
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply hidden activation only to hidden layers, not the output layer
            if i < num_layers - 1:
                x = self.hidden_activation(x)
        
        # Apply output activation (linear or exp)
        if self.output_activation_type == 'linear':
            scale = torch.exp(self.log_output_scale)
            x = x * scale
        elif self.output_activation_type == 'exp':
            scale = torch.exp(self.log_output_scale)
            x = torch.exp(x * scale)
        
        return x
    
    def _get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of trainable and total parameters.
        
        Returns:
            Dictionary with 'trainable' and 'total' parameter counts
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable_params,
            'total': total_params
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the network state to a file.
        
        Args:
            filepath: Path where to save the model
            
        Raises:
            RuntimeError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save state dict
            torch.save(self.state_dict(), filepath)
            print(f"✅ Model saved to {filepath}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {filepath}: {str(e)}")
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Create Network3D instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary with network parameters
            device: Device to create network on
            
        Returns:
            Network3D instance
        """
        # Extract parameters from config
        layer_sizes = config.get('layer_sizes', [64, 64, 64])
        hidden_activation = config.get('hidden_activation', 'tanh')
        output_activation = config.get('output_activation', 'exp')
        init_method = config.get('init_method', 'glorot')
        dtype = config.get('dtype', torch.float32)
        
        return cls(
            layer_sizes=layer_sizes,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            init_method=init_method,
            device=device,
            dtype=dtype
        )
    
    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> 'Network3D':
        """
        Load a network from a saved file.
        
        Args:
            filepath: Path to the saved model file
            config: Configuration dictionary (must match saved architecture)
            device: Device to run the network on
            
        Returns:
            Network instance with loaded weights
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If there's an architecture mismatch
        """
        return cls(
            layer_sizes=config['layer_sizes'],
            hidden_activation=config.get('hidden_activation', 'tanh'),
            output_activation=config.get('output_activation', 'exp'),
            init_method=config.get('init_method', 'glorot'),
            dtype=config.get('dtype', torch.float32),
            load_path=filepath,
            device=device
        )
    
    @classmethod
    def create_or_load(
        cls,
        network_config: Dict[str, Any],
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> 'Network3D':
        """
        Create a new network or load from file if it exists.
        
        Args:
            network_config: Configuration dictionary
            model_path: Path to saved model (optional)
            device: Device to run the network on
            
        Returns:
            Network3D instance (either new or loaded)
        """
        if model_path is not None and os.path.exists(model_path):
            try:
                return cls.load_from_file(model_path, network_config, device)
            except Exception as e:
                warnings.warn(f"Failed to load model from {model_path}: {e}. "
                            f"Creating new network instead.")
        
        return cls.create_from_config(network_config, device)
    
    def __repr__(self) -> str:
        """String representation of the network."""
        param_counts = self._get_parameter_count()
        return (f"Network(layer_sizes={self.layer_sizes}, "
                f"hidden_activation={self.hidden_activation.__name__}, "
                f"output_activation={self.output_activation_type}, "
                f"init_method='{self.init_method}', "
                f"parameters={param_counts['total']}, "
                f"trainable={param_counts['trainable']})") 
