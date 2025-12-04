from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import logging
import random
import numpy as np
import time
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PINNTrainer:
    """
    Physics-Informed Neural Network (PINN) trainer for 3D advection-diffusion problems.
    
    This class implements sequence-to-sequence training with adaptive collocation point sampling
    using Residual-based Rejection Resampling (R3). The training is split into time segments
    to handle long time horizons efficiently.
    
    Attributes:
        network: Neural network model (Network3D)
        pde: PDE residual computation class (AdvectionDiffusion3D)
        data_gen: Data generator for sampling points (DataGenerator)
        optimizer: PyTorch optimizer (Adam by default)
        pde_weight: Weight for PDE residual loss
        bc_weight: Weight for boundary condition loss
        ic_segment_weight: Weight for initial condition loss for each segment
        ic_t0_weight: Weight for initial condition loss at t=0
        device: Computation device (CPU/GPU)
        T: Total time horizon [s]
        kx: Diffusion coefficient in x (float or None)
        ky: Diffusion coefficient in y (float or None)
        kz: Diffusion coefficient in z (float or None)
        k_net: Neural network for kx, ky, kz if any is None
        training_history: Dictionary to store training metrics
    """
    
    def __init__(
        self,
        network: 'Network',
        pde: 'AdvectionDiffusion3D',
        data_gen: 'DataGenerator',
        kx: Optional[float] = None,
        ky: Optional[float] = None,
        kz: Optional[float] = None,
        optimizer_type: str = "Adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        initial_lr: float = 1e-3,  # Initial learning rate for scheduler
        use_scheduler: bool = True,  # Whether to use learning rate scheduler
        pde_weight: float = 1.0,
        bc_weight: float = 1.0,
        ic_segment_weight: float = 1.0,
        ic_t0_weight: float = 1.0,
        device: Optional[torch.device] = None,
        max_grad_norm: Optional[float] = None,
        random_seed: Optional[int] = None,
        k_net_layers: Optional[List[int]] = None,  # k_net architecture: [hidden1, hidden2, ...]
        checkpoint_path: Optional[str] = None  # Unified checkpoint path for all operations
    ) -> None:
        """
        Initialize the PINN trainer.
        
        Args:
            network: Neural network model for approximating the solution
            pde: PDE class that computes residuals
            data_gen: Data generator for sampling collocation, boundary, and initial points
            kx: Diffusion coefficient in x (float or None)
            ky: Diffusion coefficient in y (float or None)
            kz: Diffusion coefficient in z (float or None)
            optimizer_type: Type of optimizer ("Adam", default: "Adam")
            optimizer_kwargs: Dictionary of optimizer parameters (default: None)
            initial_lr: Initial learning rate for scheduler (default: 1e-3)
            use_scheduler: Whether to use learning rate scheduler (default: True)
            pde_weight: Weight for PDE residual loss (default: 1.0)
            bc_weight: Weight for boundary condition loss (default: 1.0)
            ic_segment_weight: Weight for initial condition loss for each segment (default: 1.0)
            ic_t0_weight: Weight for initial condition loss at t=0 (default: 1.0)
            device: Computation device, auto-detected if None
            max_grad_norm: Maximum gradient norm for clipping (default: None)
            random_seed: If provided, sets the random seed for reproducibility.
            k_net_layers: Hidden layer sizes for k_net (default: [32, 32]). Input: 4, Output: 3.
            checkpoint_path: Unified checkpoint path for all operations (default: None, will use "./" if None)
        """
        
        self.network = network
        self.pde = pde
        self.data_gen = data_gen
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k_net = None
        
        self.device = device or next(network.parameters()).device
        
        # If any of kx, ky, kz is None, create a neural network to predict them
        if kx is None or ky is None or kz is None:
            # Default architecture: [32, 32] hidden layers
            if k_net_layers is None:
                k_net_layers = [32, 32]
            
            # Create custom k_net class similar to Network3D
            class KNet(nn.Module):
                def __init__(self, hidden_layers: List[int]):
                    super().__init__()
                    
                    # Build hidden layers
                    layers = []
                    input_size = 4  # x, y, z, t
                    for hidden_size in hidden_layers:
                        layers.extend([
                            nn.Linear(input_size, hidden_size),
                            nn.Tanh()
                        ])
                        input_size = hidden_size
                    
                    # Final output layer
                    layers.append(nn.Linear(input_size, 3))
                    
                    self.hidden_layers = nn.Sequential(*layers)
                    
                    # Trainable scale parameter for exponential output (similar to Network3D)
                    # Using log_output_scale to ensure positive scale: scale = exp(log_output_scale)
                    self.log_output_scale = nn.Parameter(torch.tensor(0.0))
                    
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # Forward pass through hidden layers
                    x = self.hidden_layers(x)
                    
                    # Apply exponential activation with trainable scale
                    # k = exp(network_output * scale) where scale = exp(log_output_scale)
                    scale = torch.exp(self.log_output_scale)
                    x = torch.exp(x * scale)
                    
                    return x
                
                def __repr__(self):
                    return f"KNet(hidden_layers={[layer.out_features for layer in self.hidden_layers if isinstance(layer, nn.Linear)][:-1]})"
            
            # Create k_net instance
            self.k_net = KNet(k_net_layers)
            logger.info(f"[PINNTrainer] k_net neural network built with architecture: 4 -> {' -> '.join(map(str, k_net_layers))} -> 3 (exponential output with trainable scale)")

        # dtype consistency check
        net_dtype = getattr(self.network, 'dtype', None)
        data_gen_dtype = getattr(self.data_gen, 'dtype', None)
        assert data_gen_dtype is not None, "DataGenerator must have a dtype attribute"
        assert net_dtype == data_gen_dtype, f"Network dtype ({net_dtype}) and DataGenerator dtype ({data_gen_dtype}) must be the same"
        # If assertion passes, use dtype as the variable name
        self.dtype = net_dtype
        
        # Move network to device and set proper dtype
        self.network = self.network.to(device=self.device, dtype=self.dtype)
        
        # Move k_net to device and set proper dtype if it exists
        if self.k_net is not None:
            self.k_net = self.k_net.to(device=self.device, dtype=self.dtype)
            logger.info(f"[PINNTrainer] k_net moved to device: {self.device}, dtype: {self.dtype}")
        
        # Set initial_lr before creating optimizer
        self.initial_lr = float(initial_lr)
        self.use_scheduler = bool(use_scheduler)
        
        # Create optimizer based on type
        self.optimizer = self._create_optimizer(optimizer_type, optimizer_kwargs)
            
        self.pde_weight = float(pde_weight)
        self.bc_weight = float(bc_weight)
        self.ic_segment_weight = float(ic_segment_weight)
        self.ic_t0_weight = float(ic_t0_weight)
        self.max_grad_norm = max_grad_norm
        
        # Set unified checkpoint path
        self.checkpoint_path = checkpoint_path or "./"
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_path, exist_ok=True)
        logger.info(f"[PINNTrainer] Using unified checkpoint path: {self.checkpoint_path}")
        
        # Total time horizon from data generator
        self.T = self.data_gen.T  # [s]
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'pde_loss': [],
            'bc_loss': [],
            'ic_segment_loss': [],
            'ic_t0_loss': [],
            'learning_rate': [],
            'test_total_loss': [],
            'test_pde_loss': [],
            'test_bc_loss': [],
            'test_ic_loss': [],
            'segment_training_epochs': 0,  # Total epochs after segment training
        }

        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")

        # Initialize test data attributes for later use 
        self.test_xyztuvw_col: Optional[torch.Tensor] = None
        self.test_xyzt_ic: Optional[torch.Tensor] = None
        self.test_xyzt_bc: Optional[torch.Tensor] = None

    def _reset_scheduler(self, n_epochs: int, initial_lr: float = 1e-3) -> None:
        """
        Reset and configure scheduler for training.
        Learning rate will be constant for the first half, then decay from initial_lr to final_lr over the second half.
        Args:
            n_epochs: Number of epochs for training
            initial_lr: Initial learning rate (default: 1e-3)
        """
        # Set optimizer learning rate to initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

        if self.use_scheduler:
            # Use the provided initial learning rate (don't read from optimizer which might have decayed)
            final_lr = initial_lr / 10

            self._lr_phase_split = n_epochs // 2  # Number of epochs for constant LR
            self._lr_initial = initial_lr
            self._lr_final = final_lr
            self._lr_total_epochs = n_epochs

            # Calculate gamma for ExponentialLR to reach final_lr from initial_lr in n_decay_epochs steps
            n_decay_epochs = n_epochs - self._lr_phase_split
            if n_decay_epochs > 1:
                gamma = (final_lr / initial_lr) ** (1.0 / n_decay_epochs)
            else:
                gamma = 1.0

            # Store scheduler for phase 2
            self._phase2_scheduler = ExponentialLR(self.optimizer, gamma=gamma)
            self._phase2_started = False

            logger.info(f"Scheduler: constant lr={initial_lr} for {self._lr_phase_split} epochs, then decay to {final_lr} over {n_decay_epochs} epochs (gamma={gamma:.6f})")
            logger.info(f"Expected final LR: {initial_lr * (gamma ** n_decay_epochs):.2e}")
        else:
            # No scheduler - keep constant learning rate
            logger.info(f"No scheduler: constant lr={initial_lr} for all {n_epochs} epochs")

    def _validate_training_params(
        self,
        N_segments: int,
        N_col: int,
        N_bc: int,
        N_ic_segment: int,
        N_ic_t0: int,
        N_epochs: Optional[int] = None,
        loss_threshold: Optional[float] = None,
        max_epochs_per_segment: Optional[int] = None
    ) -> None:
        """
        Validate training parameters to provide clear error messages.
        
        Args:
            N_segments: Number of time segments
            N_col: Number of collocation points
            N_bc: Number of boundary condition points
            N_ic_segment: Number of initial condition points for segments
            N_ic_t0: Number of initial condition points at t=0
            N_epochs: Total epochs (optional)
            loss_threshold: Loss threshold for early stopping (optional)
            max_epochs_per_segment: Max epochs per segment (optional)
            
        Raises:
            ValueError: If any parameters are invalid
        """
        if N_segments <= 0:
            raise ValueError("N_segments must be positive")
        if N_col < 0:
            raise ValueError("N_col must be non-negative")
        if N_bc < 0:
            raise ValueError("N_bc must be non-negative")
        if N_ic_segment < 0:
            raise ValueError("N_ic_segment must be non-negative")
        if N_ic_t0 < 0:
            raise ValueError("N_ic_t0 must be non-negative")
        if N_epochs is not None and N_epochs <= 0:
            raise ValueError("N_epochs must be positive if provided")
        if loss_threshold is not None and loss_threshold <= 0:
            raise ValueError("loss_threshold must be positive if provided")
        if max_epochs_per_segment is not None and max_epochs_per_segment <= 0:
            raise ValueError("max_epochs_per_segment must be positive if provided")
        if N_epochs is None and (loss_threshold is None or max_epochs_per_segment is None):
            raise ValueError("If N_epochs is None, both loss_threshold and max_epochs_per_segment must be provided")

    def train_seq2seq(
        self,
        N_segments: int,
        N_col: int,
        N_bc: int,
        N_ic_segment: int,
        N_ic_t0: int,
        N_epochs: Optional[int] = None,
        normalize: bool = True,
        verbose: int = 1,
        load_checkpoint_path: Optional[str] = None,
        loss_threshold: Optional[float] = None,  # Stop segment if loss < this
        max_epochs_per_segment: Optional[int] = None,  # Max epochs per segment if using loss_threshold stopping
        N_test_col: Optional[int] = None,  # Test collocation data size
        N_test_bc: Optional[int] = None,   # Test boundary condition data size
        N_test_ic: Optional[int] = None,   # Test initial condition data size
        test_xyztuvw_col: Optional[torch.Tensor] = None,  # Optionally pre-generated test collocation data
        test_xyzt_bc: Optional[torch.Tensor] = None,      # Optionally pre-generated test boundary condition data
        test_xyzt_ic: Optional[torch.Tensor] = None,      # Optionally pre-generated test initial condition data
        test_every: int = 1  # Compute test loss every N epochs (default: 1, set to 0 to disable)
    ) -> None:
        """
        Train the PINN using sequence-to-sequence approach with adaptive collocation sampling.
        
        The training is split into time segments. For each segment:
        1. Generate collocation, boundary, and initial condition data
        2. Train for the allocated epochs with R3 adaptive sampling
        3. Use the network prediction at segment end as initial condition for next segment
        
        CHECKPOINT AND WEIGHT MANAGEMENT FLOW:
        =====================================
        
        1. INITIALIZATION:
           - If load_checkpoint_path is provided:
             * Load checkpoint metadata (paths, start_segment, best_total_loss)
             * Load actual weights for the starting segment
             * Resume from start_segment
           - If no checkpoint: start from segment 0 with initialized networks
        
        2. SEGMENT TRAINING:
           - For segment k > 0: Load best weights from segment k-1
           - For segment k = 0: Use current network state (from checkpoint or initialization)
           - During training: Save best weights for current segment when loss improves
           - After each segment: best weights are saved as:
             * cnet_weights_segment_{k}_best.pth
             * knet_weights_segment_{k}_best.pth (if k_net exists)
             * checkpoint_segment_{k}_best.pth (metadata)
        
        3. FINAL SAVING:
           - After all segments: Copy best weights from last segment to final paths:
             * cnet_weights_final.pth
             * knet_weights_final.pth (if k_net exists)
             * checkpoint_final.pth (metadata)
        
        Args:
            N_segments: Number of time segments to split the training
            N_col: Total number of collocation points across all segments
            N_bc: Total number of boundary condition points across all segments
            N_ic_segment: Total number of initial condition within segment points across all segments
            N_ic_t0: Total number of initial condition points at t=0 across all segments
            N_epochs: Total number of training epochs across all segments (default: None; if None, use loss_threshold and max_epochs_per_segment)
            normalize: Whether to normalize data (default: True)
            verbose: Print frequency for training progress (default: 1)
            load_checkpoint_path: Path to load checkpoint from (default: None)
            loss_threshold: Stop segment if loss < this (default: None)
            max_epochs_per_segment: Max epochs per segment if using threshold stopping (default: None)
            N_test_col: Number of test collocation points (optional, if test_xyztuvw_col not provided)
            N_test_bc: Number of test boundary condition points (optional, if test_xyzt_bc not provided)
            N_test_ic: Number of test initial points (optional, if test_xyzt_ic not provided)
            test_xyztuvw_col: Optionally pre-generated test collocation data (x, y, z, t, u, v, w)
            test_xyzt_bc: Optionally pre-generated test boundary condition data (x, y, z, t)
            test_xyzt_ic: Optionally pre-generated test initial condition data (x, y, z, t)
            test_every: Compute test loss every N epochs (default: 1, set to 0 to disable)
        Note:
            All data must be normalized and have the same dtype as the network parameters.
            Checkpoints are saved using the unified checkpoint path set during initialization.
        """
        logger.info(f"[PINNTrainer] Training started on device: {self.device}, dtype: {self.dtype}")
        start_time = time.time()
        
        

        

        # Validate input parameters
        self._validate_training_params(N_segments, N_col, N_bc, N_ic_segment, N_ic_t0, N_epochs, loss_threshold, max_epochs_per_segment)
        
        # Clear training history only if not loading from checkpoint
        if load_checkpoint_path is None:
            self.training_history = {key: [] for key in self.training_history}
            self.training_history['segment_training_epochs'] = 0
        
        # Create time grid for segmenting the training
        t_grid = torch.linspace(0, self.T, N_segments + 1, device=self.device)  # [s]
        
        # Distribute epochs and data points evenly across segments
        if N_epochs is not None:
            epochs_per_segment = self._split_across_segments(N_epochs, N_segments)
        else:
            epochs_per_segment = [max_epochs_per_segment] * N_segments  
        N_col_per_segment = self._split_across_segments(N_col, N_segments)
        N_bc_per_segment = self._split_across_segments(N_bc, N_segments)
        N_ic_segment_per_segment = self._split_across_segments(N_ic_segment, N_segments)
        N_ic_t0_per_segment = self._split_across_segments(N_ic_t0, N_segments)

        # Generate or use provided test data
        if test_xyztuvw_col is not None and test_xyzt_bc is not None and test_xyzt_ic is not None:
            pass  # Use provided
        elif N_test_col is not None and N_test_bc is not None and N_test_ic is not None:
            test_xyztuvw_col, test_xyzt_ic, test_xyzt_bc = self.data_gen.generate_full_domain_data(N_test_col, N_test_ic, N_test_bc, normalize=normalize)
        else:
            test_xyztuvw_col = None
            test_xyzt_bc = None
            test_xyzt_ic = None

        self.test_xyztuvw_col = test_xyztuvw_col
        self.test_xyzt_bc = test_xyzt_bc
        self.test_xyzt_ic = test_xyzt_ic  

        # Track total epochs for checkpoint saving
        total_epochs_completed = 0

        # Initialize best weights paths for saving during training
        best_total_loss = float('inf')
        start_segment = 0
        best_cnet_weights_path = None
        best_knet_weights_path = None

        # Load checkpoint if specified (extract paths for the starting segment)
        if load_checkpoint_path is not None:
            try:
                checkpoint = torch.load(load_checkpoint_path, map_location=self.device)
                best_cnet_weights_path = checkpoint.get('cnet_weights_path', None)
                best_knet_weights_path = checkpoint.get('knet_weights_path', None)
                start_segment = checkpoint.get('current_segment', 0)
                best_total_loss = checkpoint.get('best_total_loss', float('inf'))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {load_checkpoint_path}: {e}")
                logger.warning("Training is starting from scratch. Previous progress will be overwritten.")

        # Train each time segment sequentially
        for k in range(start_segment, N_segments):
            # For k > 0, load best weights from previous segment
            if k > 0:
                if best_cnet_weights_path is not None and os.path.exists(best_cnet_weights_path):
                    self.network.load_state_dict(torch.load(best_cnet_weights_path, map_location=self.device, weights_only=True))
                    logger.info(f"Loaded best cnet weights from segment {k-1} for segment {k}")
                else:
                    logger.warning(f"No best cnet weights found for segment {k-1}, training from current checkpoint.")
                
                if self.k_net is not None and best_knet_weights_path is not None and os.path.exists(best_knet_weights_path):
                    self.k_net.load_state_dict(torch.load(best_knet_weights_path, map_location=self.device, weights_only=True))
                    logger.info(f"Loaded best knet weights from segment {k-1} for segment {k}")
                elif self.k_net is not None:
                    logger.warning(f"No best knet weights found for segment {k-1}, training from current checkpoint.")
            elif k == 0:
                # Handle weight loading for the first segment
                if best_cnet_weights_path is not None and os.path.exists(best_cnet_weights_path):
                    self.network.load_state_dict(torch.load(best_cnet_weights_path, map_location=self.device, weights_only=True))
                    logger.info(f"Loaded cnet weights from checkpoint: {best_cnet_weights_path}")
                else:
                    logger.info(f"Training segment 0 from scratch.")
                
                if self.k_net is not None and best_knet_weights_path is not None and os.path.exists(best_knet_weights_path):
                    self.k_net.load_state_dict(torch.load(best_knet_weights_path, map_location=self.device, weights_only=True))
                    logger.info(f"Loaded k_net weights from checkpoint: {best_knet_weights_path}")
                
            
            # Reset best loss for each segment except the first one
            if k > start_segment:
                best_total_loss = float('inf')
            
            # Define time boundaries for current segment
            t_start, t_end = t_grid[k].item(), t_grid[k+1].item()  # [s]
            logger.info(f"\n=== Training segment {k+1}/{N_segments}: t in [{t_start:.2f}, {t_end:.2f}) ===\n")
            
            # Get allocated resources for this segment
            n_epochs = epochs_per_segment[k]
            n_col = N_col_per_segment[k]
            n_bc = N_bc_per_segment[k]
            n_ic_segment = N_ic_segment_per_segment[k]
            n_ic_t0 = N_ic_t0_per_segment[k]

            # Reset scheduler for this segment to decay lr from initial_lr to initial_lr/10
            if n_epochs is not None and n_epochs > 0:
                self._reset_scheduler(n_epochs, initial_lr=self.initial_lr)

            # Generate training data for current segment, handle N_col/N_bc/N_ic_segment/N_ic_t0 == 0
            try:
                xyztuvw_col = self.data_gen.generate_col_data(t_start, t_end, n_col, normalize=normalize) if n_col > 0 else None
                xyzt_bc = self.data_gen.generate_bc_data(t_start, t_end, n_bc, normalize=normalize) if n_bc > 0 else None
                xyzt_ic_segment = self.data_gen.generate_ic_data(t_start, n_ic_segment, normalize=normalize) if n_ic_segment > 0 else None
                xyzt_ic_t0 = self.data_gen.generate_ic_data(0.0, n_ic_t0, normalize=normalize) if n_ic_t0 > 0 else None
                # Enforce dtype and device consistency
                if xyztuvw_col is not None:
                    xyztuvw_col = xyztuvw_col.to(dtype=self.dtype, device=self.device)
                if xyzt_bc is not None:
                    xyzt_bc = xyzt_bc.to(dtype=self.dtype, device=self.device)
                if xyzt_ic_segment is not None:
                    xyzt_ic_segment = xyzt_ic_segment.to(dtype=self.dtype, device=self.device)
                if xyzt_ic_t0 is not None:
                    xyzt_ic_t0 = xyzt_ic_t0.to(dtype=self.dtype, device=self.device)
                  
            except Exception as e:
                raise RuntimeError(f"Failed to generate training data for segment {k+1}: {e}")
            
            # Set up initial condition for this segment
            with torch.no_grad():
                if t_start == 0.0:
                    ic_true = torch.zeros_like(xyzt_ic_segment[:, 0:1], device=self.device, dtype=self.dtype).detach() if xyzt_ic_segment is not None else None
                else:
                    try:
                        ic_true = self.network(xyzt_ic_segment).detach() if xyzt_ic_segment is not None else None
                    except Exception as e:
                        logger.warning(f"Failed to compute initial condition for segment {k+1}: {e}")
                        ic_true = torch.zeros_like(xyzt_ic_segment[:, 0:1], device=self.device, dtype=self.dtype).detach() if xyzt_ic_segment is not None else None

            # Train for allocated epochs in this segment
            # Ensure n_epochs is an int
            if n_epochs is None:
                raise ValueError("n_epochs must be set for the epoch loop.")
            for epoch in range(n_epochs):
                try:
                    # Compute losses for training
                    self.optimizer.zero_grad()
                    if xyztuvw_col is not None and xyztuvw_col.shape[0] > 0:
                        if self.k_net is not None:
                            # Input to k_net: x, y, z, t (normalized coordinates)
                            k_input = xyztuvw_col[:, :4]  # x, y, z, t
                            k_pred = self.k_net(k_input)
                            # Use tensor outputs for spatially varying diffusion coefficients
                            kx = k_pred[:, 0]
                            ky = k_pred[:, 1]
                            kz = k_pred[:, 2]
                            pde_residual = self.pde.compute_pde_values(xyztuvw_col, kx, ky, kz)[1]
                        else:
                            pde_residual = self.pde.compute_pde_values(xyztuvw_col, self.kx, self.ky, self.kz)[1]
                        pde_loss = torch.mean(pde_residual**2)
                    else:
                        pde_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    if xyzt_bc is not None and xyzt_bc.shape[0] > 0:
                        bc_val = self.pde.compute_bc_values(xyzt_bc)[1]
                        bc_loss = torch.mean(bc_val**2)
                    else:
                        bc_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    if xyzt_ic_segment is not None and xyzt_ic_segment.shape[0] > 0 and ic_true is not None:
                        ic_segment_pred = self.pde.compute_ic_values(xyzt_ic_segment)[1]
                        ic_segment_loss = torch.mean((ic_segment_pred - ic_true)**2)
                    else:
                        ic_segment_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    if xyzt_ic_t0 is not None and xyzt_ic_t0.shape[0] > 0:
                        ic_t0_pred = self.pde.compute_ic_values(xyzt_ic_t0)[1]
                        ic_t0_loss = torch.mean(ic_t0_pred ** 2)
                    else:
                        ic_t0_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    total_loss = (
                        self.pde_weight * pde_loss +
                        self.bc_weight * bc_loss +
                        self.ic_segment_weight * ic_segment_loss +
                        self.ic_t0_weight * ic_t0_loss
                    )
                    
                    # Validate that we have at least one non-zero loss component for gradient computation
                    if (pde_loss.item() == 0.0 and bc_loss.item() == 0.0 and 
                        ic_segment_loss.item() == 0.0 and ic_t0_loss.item() == 0.0):
                        logger.warning(f"All loss components are zero at epoch {epoch} in segment {k+1}. Skipping gradient computation.")
                        continue
                    
                    loss_dict = {
                        'pde': pde_loss.item(),
                        'bc': bc_loss.item(),
                        'ic_segment': ic_segment_loss.item(),
                        'ic_t0': ic_t0_loss.item()
                    }
                    
                    # Backpropagation and optimizer step
                    total_loss.backward()
                    # Apply gradient clipping if specified
                    if self.max_grad_norm is not None:
                        # Clip gradients for all parameters (network + k_net if exists)
                        all_params = []
                        for param_group in self.optimizer.param_groups:
                            all_params.extend(param_group['params'])
                        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Two-phase LR schedule
                    if self.use_scheduler and hasattr(self, '_lr_phase_split') and epoch >= self._lr_phase_split:
                        if not getattr(self, '_phase2_started', False):
                            self._phase2_started = True
                            logger.info(f"Starting LR decay phase at epoch {epoch}")
                        self._phase2_scheduler.step()
                    # else: keep LR constant
                    
                    # Update collocation data using R3 adaptive sampling
                    if xyztuvw_col is not None and xyztuvw_col.shape[0] > 0:
                        xyztuvw_col = self._update_collocation_data(xyztuvw_col, (pde_residual**2).squeeze(), t_start, t_end, normalize=normalize)
                        # Enforce dtype and device consistency after resampling
                        xyztuvw_col = xyztuvw_col.to(dtype=self.dtype, device=self.device)
                        assert xyztuvw_col.dtype == self.dtype, f"Collocation data dtype {xyztuvw_col.dtype} does not match network dtype {self.dtype} after resampling"
                    
                    # Store training history
                    self.training_history['total_loss'].append(total_loss.item())
                    self.training_history['pde_loss'].append(loss_dict['pde'])
                    self.training_history['bc_loss'].append(loss_dict['bc'])
                    self.training_history['ic_segment_loss'].append(loss_dict['ic_segment'])
                    self.training_history['ic_t0_loss'].append(loss_dict['ic_t0'])
                    self.training_history['learning_rate'].append(self._get_current_lr())
                    
                    # Check if this is the best loss for the segment
                    if total_loss.item() < best_total_loss:
                        best_total_loss = total_loss.item()
                        best_checkpoint_path = os.path.join(self.checkpoint_path, f"checkpoint_segment_{k}_best.pth")
                        best_cnet_weights_path = os.path.join(self.checkpoint_path, f"cnet_weights_segment_{k}_best.pth")
                        best_knet_weights_path = os.path.join(self.checkpoint_path, f"knet_weights_segment_{k}_best.pth") if self.k_net is not None else None
                        try:
                            # Save both cnet and knet weights during segment training
                            self.save_checkpoint(best_checkpoint_path, best_cnet_weights_path, best_knet_weights_path, best_total_loss, k)
                        except Exception as e:
                            logger.warning(f"Failed to save best model checkpoint for segment {k}: {e}")

                    # Compute and store test loss using helper method
                    if test_every > 0 and total_epochs_completed % test_every == 0:
                        test_total_loss_val, test_pde_loss_val, test_bc_loss_val, test_ic_loss_val = self._compute_test_losses(
                            test_xyztuvw_col, test_xyzt_bc, test_xyzt_ic
                        )
                    else:
                        test_total_loss_val = test_pde_loss_val = test_bc_loss_val = test_ic_loss_val = None
                    
                    self.training_history['test_total_loss'].append(test_total_loss_val)
                    self.training_history['test_pde_loss'].append(test_pde_loss_val)
                    self.training_history['test_bc_loss'].append(test_bc_loss_val)
                    self.training_history['test_ic_loss'].append(test_ic_loss_val)

                    total_epochs_completed += 1
                    
                    # Print training progress
                    if verbose and (epoch % verbose == 0 or (n_epochs is not None and epoch == n_epochs - 1)):
                        current_lr = self._get_current_lr()
                        logger.info(f"Segment {k+1}/{N_segments} | Epoch {epoch}: total_loss={total_loss.item():.4e} | "
                                   f"pde={loss_dict['pde']:.2e} bc={loss_dict['bc']:.2e} ic_segment={loss_dict['ic_segment']:.2e} ic_t0={loss_dict['ic_t0']:.2e} | "
                                   f"lr={current_lr:.2e}")
                        if test_total_loss_val is not None:
                            logger.info(f"[TEST] Segment {k+1}/{N_segments} | Epoch {epoch}: test_total_loss={test_total_loss_val:.4e} | test_pde={test_pde_loss_val:.2e} test_bc={test_bc_loss_val:.2e} test_ic={test_ic_loss_val:.2e}")

                    # Check for segment stopping condition
                    if loss_threshold is not None and total_loss.item() < loss_threshold:
                        logger.info(f"Stopping segment {k+1} at epoch {epoch} as total loss {total_loss.item():.4e} < {loss_threshold}")
                        break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU out of memory at epoch {epoch} in segment {k+1}. Try reducing batch size.")
                        raise
                    else:
                        logger.error(f"Runtime error at epoch {epoch} in segment {k+1}: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error at epoch {epoch} in segment {k+1}: {e}")
                    raise

            

        # After all segments, save the final best models
        try:
            final_checkpoint_path = os.path.join(self.checkpoint_path, "checkpoint_final.pth")
            final_cnet_weights_path = os.path.join(self.checkpoint_path, "cnet_weights_final.pth")
            final_knet_weights_path = os.path.join(self.checkpoint_path, "knet_weights_final.pth")
            import shutil
            
            # Copy concentration network weights from last segment's best
            if os.path.exists(best_cnet_weights_path):
                shutil.copy(best_cnet_weights_path, final_cnet_weights_path)
            
            # Copy k_net weights from last segment's best
            if self.k_net is not None and os.path.exists(best_knet_weights_path):
                shutil.copy(best_knet_weights_path, final_knet_weights_path)
                logger.info(f"Saved k_net weights for final model: {final_knet_weights_path}")
            
            # Save final checkpoint with k_net weights (weights already saved to final paths)
            self.save_checkpoint(final_checkpoint_path, final_cnet_weights_path, final_knet_weights_path, best_total_loss, N_segments - 1, save_current_weights=False)
                
            logger.info(f"Saved final best models to {final_checkpoint_path}, {final_cnet_weights_path}, and {final_knet_weights_path if self.k_net is not None else '[no k_net]'}")
            
            # Record total epochs after segment training
            self.training_history['segment_training_epochs'] = total_epochs_completed
        except Exception as e:
            logger.warning(f"Failed to save final best model: {e}")

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[PINNTrainer] Seq2Seq training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    def _create_optimizer(self, optimizer_type, optimizer_kwargs=None):
        """
        Create optimizer based on type and parameters.
        
        Args:
            optimizer_type: "Adam"
            optimizer_kwargs: Dictionary of optimizer parameters
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer_type = optimizer_type.upper()
        kwargs = optimizer_kwargs or {}
        
        if optimizer_type == "ADAM":
            # Default Adam parameters for PINNs
            default_kwargs = {
                "lr": self.initial_lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0
            }
            default_kwargs.update(kwargs)
            
            # Collect all parameters: network + k_net (if exists)
            params = list(self.network.parameters())
            if self.k_net is not None:
                params.extend(list(self.k_net.parameters()))
            
            return Adam(params, **default_kwargs)
            
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                           f"Supported option: Adam")


    def _get_current_lr(self):
        """Get the current learning rate."""
        try:
            # Check if we're in phase 2 and have a phase2 scheduler
            if hasattr(self, '_phase2_scheduler') and hasattr(self, '_phase2_started') and self._phase2_started:
                # Use the phase2 scheduler's last LR if available
                if hasattr(self._phase2_scheduler, 'get_last_lr'):
                    lr_from_scheduler = self._phase2_scheduler.get_last_lr()[0]
                    lr_from_optimizer = self.optimizer.param_groups[0]['lr']
                    if abs(lr_from_scheduler - lr_from_optimizer) > 1e-10:
                        logger.warning(f"LR mismatch: scheduler={lr_from_scheduler:.2e}, optimizer={lr_from_optimizer:.2e}")
                    return lr_from_scheduler
            # Fallback to optimizer parameter group (works for both phases)
            return self.optimizer.param_groups[0]['lr']
        except (IndexError, AttributeError):
            # Final fallback to optimizer parameter group
            return self.optimizer.param_groups[0]['lr']

    def get_training_stats(self):
        """
        Get training statistics.
        
        Returns:
            dict: Dictionary containing training statistics
        """
        if not self.training_history['total_loss']:
            return {}
        
        return {
            'final_loss': self.training_history['total_loss'][-1],
            'min_loss': min(self.training_history['total_loss']),
            'max_loss': max(self.training_history['total_loss']),
            'final_lr': self.training_history['learning_rate'][-1],
            'total_epochs': len(self.training_history['total_loss'])
        }

    def save_checkpoint(self, checkpoint_path: str, cnet_weights_path: str, knet_weights_path: Optional[str] = None, best_total_loss: Optional[float] = None, current_segment: Optional[int] = None, save_current_weights: bool = True) -> None:
        """
        Save training checkpoint.
        Args:
            checkpoint_path: Path to save the checkpoint
            cnet_weights_path: Path to save the concentration network weights
            knet_weights_path: Path to save k_net weights (optional, None to skip saving)
            best_total_loss: Best total loss so far
            current_segment: Current segment index
            save_current_weights: If True, save current model weights; if False, assume weights are already saved at the paths
        Note:
            Saves k_net weights only if knet_weights_path is provided and k_net exists.
        """
        # Save concentration network weights (only if requested)
        if save_current_weights:
            torch.save(self.network.state_dict(), cnet_weights_path)
        
        # Save k_net weights if path is provided and k_net exists (only if requested)
        if save_current_weights and knet_weights_path is not None and self.k_net is not None:
            torch.save(self.k_net.state_dict(), knet_weights_path)
            #logger.info(f"Saved k_net weights to: {knet_weights_path}")
        
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'pde_weight': self.pde_weight,
            'bc_weight': self.bc_weight,
            'ic_segment_weight': self.ic_segment_weight,
            'ic_t0_weight': self.ic_t0_weight,
            'device': self.device,
            'best_total_loss': best_total_loss,
            'current_segment': current_segment,
            'random_seed': self.random_seed,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state().cpu().numpy().tolist(),
            'cnet_weights_path': cnet_weights_path,
            'knet_weights_path': knet_weights_path,
        }
        # Save CUDA RNG state if training on CUDA
        if torch.cuda.is_available() and self.device.type == 'cuda':
            checkpoint['torch_cuda_random_state'] = torch.cuda.get_rng_state().cpu().numpy().tolist()
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load training checkpoint.
        Args:
            filepath: Path to the checkpoint file
        Note:
            Loads both concentration net and k_net weights if present.
            Scheduler state is NOT loaded since we reset it for each segment.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        # Only load weights if the file exists
        cnet_weights_path = checkpoint.get('cnet_weights_path', None)
        if cnet_weights_path is not None and os.path.exists(cnet_weights_path):
            self.network.load_state_dict(torch.load(cnet_weights_path, map_location=self.device, weights_only=True))
        # Load k_net weights if present (for resuming interrupted training, not segment transitions)
        knet_weights_path = checkpoint.get('knet_weights_path', None)
        if self.k_net is not None and knet_weights_path is not None and os.path.exists(knet_weights_path):
            self.k_net.load_state_dict(torch.load(knet_weights_path, map_location=self.device, weights_only=True))
            logger.info("k_net weights loaded from checkpoint (for training resumption)")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
        self.training_history = checkpoint['training_history']
        self.pde_weight = checkpoint['pde_weight']
        self.bc_weight = checkpoint['bc_weight']
        self.ic_segment_weight = checkpoint['ic_segment_weight']
        self.ic_t0_weight = checkpoint['ic_t0_weight']
        
        if 'best_total_loss' in checkpoint:
            # Note: This is for informational purposes, actual best loss tracking is handled separately
            logger.info(f"Checkpoint contains best_total_loss: {checkpoint['best_total_loss']}")
        if 'current_segment' in checkpoint:
            # Note: This is for informational purposes, actual segment tracking is handled separately
            logger.info(f"Checkpoint contains current_segment: {checkpoint['current_segment']}")
        if 'random_seed' in checkpoint and checkpoint['random_seed'] is not None:
            self.random_seed = checkpoint['random_seed']
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            logger.info(f"Random seed restored to {self.random_seed}")
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
        if 'torch_random_state' in checkpoint:
            torch.set_rng_state(torch.tensor(checkpoint['torch_random_state'], dtype=torch.uint8))
        # Restore CUDA RNG state if available and training on CUDA
        if 'torch_cuda_random_state' in checkpoint and torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.set_rng_state(torch.tensor(checkpoint['torch_cuda_random_state'], dtype=torch.uint8))
            logger.info("CUDA RNG state restored")

    @staticmethod
    def _split_across_segments(N: int, N_segments: int) -> List[int]:
        """
        Evenly split N (e.g. # total points, # total epochs) across N_segments, 
        distributing any remainder to the first segments.
        
        Args:
            N: Total quantity to split
            N_segments: Number of segments to split into
            
        Returns:
            list: List of length N_segments with distributed quantities
        """
        if N < 0:
            raise ValueError("N must be non-negative")
        if N_segments <= 0:
            raise ValueError("N_segments must be positive")
        
        base = N // N_segments
        rem = N % N_segments
        result = [base] * N_segments
        # Distribute remainder to first segments
        for i in range(rem):
            result[i] += 1
        return result

    @torch.no_grad()
    def _update_collocation_data(self, xyztuvw_col: torch.Tensor, pde_residual: torch.Tensor, t_start: float, t_end: float, normalize: bool = True) -> torch.Tensor:
        """
        Perform Residual-based Rejection Resampling (R3) to adaptively update collocation points.

        Args:
            xyztuvw_col (torch.Tensor): Current collocation points, shape (N, 7).
            pde_residual (torch.Tensor): PDE residuals at collocation points, shape (N,).
            t_start (float): Start of the time segment.
            t_end (float): End of the time segment.
            normalize (bool): Whether to normalize new samples (default: True).

        Returns:
            torch.Tensor: Updated collocation points, shape (N, 7), where points with above-average residuals are kept and the rest are resampled.
        """
        try:
            # Validate input shapes
            if xyztuvw_col.dim() != 2 or xyztuvw_col.shape[1] != 7:
                logger.warning(f"Invalid collocation data shape: {xyztuvw_col.shape}, expected (N, 7)")
                return xyztuvw_col
            if pde_residual.shape[0] != xyztuvw_col.shape[0]:
                logger.warning(f"Shape mismatch: residual {pde_residual.shape} vs collocation {xyztuvw_col.shape}")
                return xyztuvw_col
                
            # Detach pde_loss to avoid tracking gradients during resampling
            pde_residual_detached = pde_residual.detach()
            # Compute the average PDE residual
            avg_pde_residual = pde_residual_detached.mean()
            # Create a mask for points with residuals above the average
            keep_mask = pde_residual_detached >= avg_pde_residual
            # Keep collocation points with above-average residuals
            xyztuvw_keep = xyztuvw_col[keep_mask]
            # Number of new points to sample to maintain the original batch size
            N_new_sample = xyztuvw_col.shape[0] - xyztuvw_keep.shape[0]
            if N_new_sample > 0:
                # Resample new collocation points to replace those with high residuals
                xyztuvw_new = self.data_gen.generate_col_data(t_start, t_end, N_new_sample, normalize=normalize)
                # Ensure device and dtype consistency for new data
                xyztuvw_new = xyztuvw_new.to(dtype=self.dtype, device=self.device)
                # Validate new data shape and dtype
                if xyztuvw_new.shape[1] != 7:
                    logger.warning(f"Generated data has wrong shape: {xyztuvw_new.shape}, expected (N, 7)")
                    return xyztuvw_col
            else:
                # Use xyztuvw_col.shape[1] to ensure correct shape even if xyztuvw_keep is empty
                xyztuvw_new = xyztuvw_col.new_zeros((0, xyztuvw_col.shape[1]))
            # Concatenate kept and new points to form the updated batch
            xyztuvw_col_updated = torch.cat([xyztuvw_keep, xyztuvw_new], dim=0)
            # Verify final shape
            if xyztuvw_col_updated.shape != xyztuvw_col.shape:
                logger.warning(f"Shape mismatch after resampling: {xyztuvw_col_updated.shape} vs {xyztuvw_col.shape}")
                return xyztuvw_col
            # Shuffle the collocation points to randomize their order
            xyztuvw_col_updated = xyztuvw_col_updated[torch.randperm(xyztuvw_col_updated.shape[0])]
            return xyztuvw_col_updated
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"R3 resampling failed with {type(e).__name__}: {e}. Returning original collocation points.")
            return xyztuvw_col
        except Exception as e:
            logger.error(f"Unexpected error in R3 resampling: {type(e).__name__}: {e}. Returning original collocation points.")
            return xyztuvw_col

    def _compute_test_losses(
        self,
        test_xyztuvw_col: Optional[torch.Tensor] = None,
        test_xyzt_bc: Optional[torch.Tensor] = None,
        test_xyzt_ic: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute test losses for evaluation.
        
        Args:
            test_xyztuvw_col: Test collocation data
            test_xyzt_bc: Test boundary condition data
            test_xyzt_ic: Test initial condition data
            
        Returns:
            Tuple of (total_loss, pde_loss, bc_loss, ic_loss) or None if no test data
        """
        if test_xyztuvw_col is None and test_xyzt_bc is None and test_xyzt_ic is None:
            return None, None, None, None
            
        with torch.no_grad():
            test_pde_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            test_bc_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            test_ic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            if test_xyztuvw_col is not None and test_xyztuvw_col.shape[0] > 0:
                if self.k_net is not None:
                    # Input to k_net: x, y, z, t (normalized coordinates)
                    k_input = test_xyztuvw_col[:, :4]  # x, y, z, t
                    k_pred = self.k_net(k_input)
                    # Use tensor outputs for spatially varying diffusion coefficients
                    kx = k_pred[:, 0]
                    ky = k_pred[:, 1]
                    kz = k_pred[:, 2]
                    test_pde_residual = self.pde.compute_pde_values(test_xyztuvw_col, kx, ky, kz)[1]
                else:
                    test_pde_residual = self.pde.compute_pde_values(test_xyztuvw_col, self.kx, self.ky, self.kz)[1]
                test_pde_loss = torch.mean(test_pde_residual**2)
                
            if test_xyzt_bc is not None and test_xyzt_bc.shape[0] > 0:
                test_bc_val = self.pde.compute_bc_values(test_xyzt_bc)[1]
                test_bc_loss = torch.mean(test_bc_val**2)
                
            if test_xyzt_ic is not None and test_xyzt_ic.shape[0] > 0:
                test_ic_pred = self.pde.compute_ic_values(test_xyzt_ic)[1]
                test_ic_true = torch.zeros_like(test_ic_pred, device=self.device, dtype=self.dtype)
                test_ic_loss = torch.mean((test_ic_pred - test_ic_true) ** 2)
                
            test_total_loss = (
                self.pde_weight * test_pde_loss +
                self.bc_weight * test_bc_loss +
                self.ic_t0_weight * test_ic_loss
            )
            
            return (
                test_total_loss.item(),
                test_pde_loss.item(),
                test_bc_loss.item(),
                test_ic_loss.item()
            )

    def finetune_on_full_domain(
        self,
        N_col: int,
        N_bc: int,
        N_ic: int,
        N_epochs: int,
        loss_threshold: Optional[float] = None,
        verbose: int = 1,
        normalize: bool = True,
        test_xyztuvw_col: Optional[torch.Tensor] = None,
        test_xyzt_bc: Optional[torch.Tensor] = None,
        test_xyzt_ic: Optional[torch.Tensor] = None,
        reset_scheduler: bool = True,
        test_every: int = 1,  # Compute test loss every N epochs (default: 1, set to 0 to disable)
    ):
        """
        Fine-tune the model on the full domain using all-domain data.
        Loads the best model from the last segment and trains further.

        WEIGHT MANAGEMENT FLOW:
        ======================
        
        1. LOADING:
           - Load best weights from seq2seq training:
             * cnet_weights_final.pth (concentration network)
             * knet_weights_final.pth (diffusion network, if exists)
           - If weights don't exist: warn and use current model state
        
        2. TRAINING:
           - Train on full domain data for N_epochs
           - Save best weights during training when loss improves:
             * cnet_weights_final_final.pth
             * knet_weights_final_final.pth (if k_net exists)
             * checkpoint_final_final.pth (metadata)
           - Directly overwrite final_final results on each improvement
        
        3. COMPLETION:
           - Best weights are already saved at final_final paths
           - No additional copying needed

        Args:
            N_col: Number of collocation points for full domain
            N_bc: Number of boundary condition points for full domain
            N_ic: Number of initial condition points for full domain
            N_epochs: Number of epochs to train
            loss_threshold: Optional early stopping threshold
            verbose: Print frequency
            normalize: Whether to normalize data
            test_xyztuvw_col: Optional test collocation data (if None, uses test data from previous training)
            test_xyzt_bc: Optional test boundary condition data (if None, uses test data from previous training)
            test_xyzt_ic: Optional test initial condition data (if None, uses test data from previous training)
            reset_scheduler: Whether to reset scheduler to decay lr from 1e-3 to 5e-5 (default: True)
            test_every: Compute test loss every N epochs (default: 1, set to 0 to disable)
        Note:
            The optimizer and scheduler are reused from previous training unless reset_scheduler=True.
            Checkpoints are saved using the unified checkpoint path set during initialization.
        """
        # 1. Load best weights from last segment
        last_best_cnet_weights = os.path.join(self.checkpoint_path, "cnet_weights_final.pth")
        last_best_knet_weights = os.path.join(self.checkpoint_path, "knet_weights_final.pth")
        
        weights_loaded = False
        if os.path.exists(last_best_cnet_weights):
            self.network.load_state_dict(torch.load(last_best_cnet_weights, map_location=self.device, weights_only=True))
            logger.info(f"Loaded best cnet weights from last segment for fine-tuning: {last_best_cnet_weights}")
            weights_loaded = True
        else:
            logger.warning(f"No best cnet weights found for last segment at {last_best_cnet_weights}, using current model.")
        
        if self.k_net is not None and os.path.exists(last_best_knet_weights):
            self.k_net.load_state_dict(torch.load(last_best_knet_weights, map_location=self.device, weights_only=True))
            logger.info(f"Loaded best knet weights from last segment for fine-tuning: {last_best_knet_weights}")
            weights_loaded = True
        elif self.k_net is not None:
            logger.warning(f"No best knet weights found for last segment at {last_best_knet_weights}, using current model.")
        
        if not weights_loaded:
            logger.warning("No weights were loaded for fine-tuning. This may indicate that seq2seq training was not completed.")

        # 2. Generate full-domain training data
        xyztuvw_col, xyzt_ic, xyzt_bc = self.data_gen.generate_full_domain_data(N_col, N_ic, N_bc, normalize=normalize)
        xyztuvw_col = xyztuvw_col.to(dtype=self.dtype, device=self.device)
        xyzt_ic = xyzt_ic.to(dtype=self.dtype, device=self.device)
        xyzt_bc = xyzt_bc.to(dtype=self.dtype, device=self.device)

        # 3. Use test data from arguments if provided, otherwise from previous training
        if test_xyztuvw_col is None:
            test_xyztuvw_col = self.test_xyztuvw_col
        if test_xyzt_bc is None:
            test_xyzt_bc = self.test_xyzt_bc
        if test_xyzt_ic is None:
            test_xyzt_ic = self.test_xyzt_ic

        # 4. Optionally reset scheduler for fine-tuning
        if reset_scheduler:
            self._reset_scheduler(N_epochs, initial_lr=self.initial_lr)

        best_loss = float('inf')
        best_checkpoint_path = os.path.join(self.checkpoint_path, "checkpoint_final_final.pth")
        best_cnet_model_path = os.path.join(self.checkpoint_path, "cnet_weights_final_final.pth")
        best_knet_model_path = os.path.join(self.checkpoint_path, "knet_weights_final_final.pth")
        for epoch in range(N_epochs):
            self.optimizer.zero_grad()
            # Compute losses
            if self.k_net is not None:
                # Input to k_net: x, y, z, t (normalized coordinates)
                k_input = xyztuvw_col[:, :4]  # x, y, z, t
                k_pred = self.k_net(k_input)
                # Use tensor outputs for spatially varying diffusion coefficients
                kx = k_pred[:, 0]
                ky = k_pred[:, 1]
                kz = k_pred[:, 2]
                pde_residual = self.pde.compute_pde_values(xyztuvw_col, kx, ky, kz)[1]
            else:
                pde_residual = self.pde.compute_pde_values(xyztuvw_col, self.kx, self.ky, self.kz)[1]
            pde_loss = torch.mean(pde_residual**2)
            bc_val = self.pde.compute_bc_values(xyzt_bc)[1]
            bc_loss = torch.mean(bc_val**2)
            ic_pred = self.pde.compute_ic_values(xyzt_ic)[1]
            ic_true = torch.zeros_like(ic_pred, device=self.device, dtype=self.dtype)
            ic_loss = torch.mean((ic_pred - ic_true) ** 2)
            total_loss = self.pde_weight * pde_loss + self.bc_weight * bc_loss + self.ic_t0_weight * ic_loss
            total_loss.backward()
            if self.max_grad_norm is not None:
                # Clip gradients for all parameters (network + k_net if exists)
                all_params = []
                for param_group in self.optimizer.param_groups:
                    all_params.extend(param_group['params'])
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            self.optimizer.step()

            # Two-phase LR schedule
            if self.use_scheduler and hasattr(self, '_lr_phase_split') and epoch >= self._lr_phase_split:
                if not getattr(self, '_phase2_started', False):
                    self._phase2_started = True
                    logger.info(f"Starting LR decay phase at epoch {epoch}")
                self._phase2_scheduler.step()
            # else: keep LR constant

            # R3 adaptive sampling for collocation points
            xyztuvw_col = self._update_collocation_data(xyztuvw_col, (pde_residual**2).squeeze(), 0.0, self.T, normalize=normalize)
            xyztuvw_col = xyztuvw_col.to(dtype=self.dtype, device=self.device)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                # Save weights using the save_checkpoint method for consistency
                self.save_checkpoint(best_checkpoint_path, best_cnet_model_path, best_knet_model_path, best_loss, None)
            
            # Compute and store test loss using helper method (respecting test_every)
            if test_every > 0 and epoch % test_every == 0:
                test_total_loss_val, test_pde_loss_val, test_bc_loss_val, test_ic_loss_val = self._compute_test_losses(
                    test_xyztuvw_col, test_xyzt_bc, test_xyzt_ic
                )
            else:
                test_total_loss_val = test_pde_loss_val = test_bc_loss_val = test_ic_loss_val = None
                
            self.training_history['test_total_loss'].append(test_total_loss_val)
            self.training_history['test_pde_loss'].append(test_pde_loss_val)
            self.training_history['test_bc_loss'].append(test_bc_loss_val)
            self.training_history['test_ic_loss'].append(test_ic_loss_val)

            # Also log training losses for consistency
            self.training_history['total_loss'].append(total_loss.item())
            self.training_history['pde_loss'].append(pde_loss.item())
            self.training_history['bc_loss'].append(bc_loss.item())  # BC training included in finetune
            self.training_history['ic_segment_loss'].append(0.0)  # No IC segment training in finetune
            self.training_history['ic_t0_loss'].append(ic_loss.item())
            self.training_history['learning_rate'].append(self._get_current_lr())

            if verbose and (epoch % verbose == 0 or epoch == N_epochs - 1):
                current_lr = self._get_current_lr()
                logger.info(f"[Finetune] Epoch {epoch}: total_loss={total_loss.item():.4e} | pde={pde_loss.item():.2e} bc={bc_loss.item():.2e} ic={ic_loss.item():.2e} | lr={current_lr:.2e}")
                if test_total_loss_val is not None:
                    logger.info(f"[Finetune TEST] Epoch {epoch}: test_total_loss={test_total_loss_val:.4e} | test_pde={test_pde_loss_val:.2e} test_bc={test_bc_loss_val:.2e} test_ic={test_ic_loss_val:.2e}")
            if loss_threshold is not None and total_loss.item() < loss_threshold:
                logger.info(f"[Finetune] Early stopping at epoch {epoch} (loss < threshold)")
                break
            
        logger.info(f"[Finetune] Best loss: {best_loss:.4e}. Best model saved to {best_cnet_model_path}")

    def inference_on_grid(
        self,
        N_x: int,
        N_y: int, 
        N_z: int,
        N_t: Optional[int] = None,
        save_outputs: bool = True,
        output_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a grid and predict concentration, diffusion coefficients, and PDE residuals.
        
        Args:
            N_x: Number of grid points in x dimension
            N_y: Number of grid points in y dimension
            N_z: Number of grid points in z dimension
            N_t: Number of grid points in t dimension (default: length of wind vector)
            save_outputs: Whether to save outputs to numpy files
            output_dir: Directory to save outputs (default: checkpoint_path)
            
        Returns:
            Tuple containing:
            - C: Concentration array (N_x, N_y, N_z, N_t)
            - Kx: Diffusion coefficient array (N_x, N_y, N_z, N_t)
            - Ky: Diffusion coefficient array (N_x, N_y, N_z, N_t)
            - Kz: Diffusion coefficient array (N_x, N_y, N_z, N_t)
            - pde_residual: PDE residual array (N_x, N_y, N_z, N_t)
            - x: x-coordinate array
            - y: y-coordinate array
            - z: z-coordinate array
            - t: t-coordinate array
        """
        import numpy as np
        
        # Set N_t to wind vector length if not provided
        if N_t is None:
            N_t = len(self.data_gen.u)
            logger.info(f"Using N_t = {N_t} (length of wind vector)")

        logger.info(f"[PINNTrainer] Starting inference on {N_x}x{N_y}x{N_z}x{N_t} grid")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.checkpoint_path
        
        # Load final models
        self._load_final_models()
        
        # Generate grid data using data generator
        logger.info(f"Generating {N_x}x{N_y}x{N_z}x{N_t} grid using data generator...")
        
        # Validate data generator has required method
        if not hasattr(self.data_gen, 'generate_grid_data'):
            raise AttributeError("DataGenerator must have 'generate_grid_data' method")
        
        # Use data generator to create grid data
        xyztuvw, x, y, z, t = self.data_gen.generate_grid_data(N_x, N_y, N_z, N_t, normalize=True)
        
        logger.info(f"Grid ranges: x=[{x.min():.1f}, {x.max():.1f}], y=[{y.min():.1f}, {y.max():.1f}], z=[{z.min():.1f}, {z.max():.1f}], t=[{t.min():.1f}, {t.max():.1f}]")
        
        # Extract xyzt for k_net input
        xyzt = xyztuvw[:, :4]  # x, y, z, t
        
        logger.info(f"Grid shape: {xyzt.shape}")
        
        # Run inference
        with torch.no_grad():
            # Predict concentration
            concentration = self.network(xyzt)
            
            # Predict diffusion coefficients
            if self.k_net is not None:
                # k_net input: [x, y, z, t] - normalized coordinates                
                diffusion_coeffs = self.k_net(xyzt)
                Kx = diffusion_coeffs[:, 0]
                Ky = diffusion_coeffs[:, 1]
                Kz = diffusion_coeffs[:, 2]
            else:
                # Use constant diffusion coefficients
                N = xyzt.shape[0]
                Kx = torch.full((N,), self.kx, dtype=self.dtype, device=self.device)
                Ky = torch.full((N,), self.ky, dtype=self.dtype, device=self.device)
                Kz = torch.full((N,), self.kz, dtype=self.dtype, device=self.device)
        
        # Compute PDE residuals (requires gradients)
        # Temporarily enable gradients for PDE computation
        with torch.enable_grad():
            if self.k_net is not None:
                pde_residual = self.pde.compute_pde_values(xyztuvw, Kx, Ky, Kz)[1]
            else:
                pde_residual = self.pde.compute_pde_values(xyztuvw, self.kx, self.ky, self.kz)[1]
        
        # Reshape to 4D: (N_x, N_y, N_z, N_t)
        C = concentration.detach().cpu().numpy().reshape(N_x, N_y, N_z, N_t)
        pde_residual = pde_residual.detach().cpu().numpy().reshape(N_x, N_y, N_z, N_t)
        
        # Diffusion coefficients are 4D (spatial + time) since k_net takes [x, y, z, t]
        Kx = Kx.cpu().numpy().reshape(N_x, N_y, N_z, N_t)
        Ky = Ky.cpu().numpy().reshape(N_x, N_y, N_z, N_t)
        Kz = Kz.cpu().numpy().reshape(N_x, N_y, N_z, N_t)
        
        logger.info(f"✅ Inference completed!")
        logger.info(f"   Concentration shape: {C.shape}, range: [{C.min():.3e}, {C.max():.3e}]")
        logger.info(f"   Diffusion shapes: Kx={Kx.shape}, Ky={Ky.shape}, Kz={Kz.shape}")
        logger.info(f"   PDE residual shape: {pde_residual.shape}, range: [{pde_residual.min():.3e}, {pde_residual.max():.3e}]")
        
        # Save outputs if requested
        if save_outputs:
            self._save_inference_outputs(C, Kx, Ky, Kz, pde_residual, x, y, z, t, output_dir)
        
        return C, Kx, Ky, Kz, pde_residual, x, y, z, t
    
    def _load_final_models(self):
        """Load the final trained models."""
        # Try to load final_final models first, then final models
        final_final_cnet_path = os.path.join(self.checkpoint_path, "cnet_weights_final_final.pth")
        final_final_knet_path = os.path.join(self.checkpoint_path, "knet_weights_final_final.pth")
        final_cnet_path = os.path.join(self.checkpoint_path, "cnet_weights_final.pth")
        final_knet_path = os.path.join(self.checkpoint_path, "knet_weights_final.pth")
        
        # Load concentration network
        if os.path.exists(final_final_cnet_path):
            self.network.load_state_dict(torch.load(final_final_cnet_path, map_location=self.device, weights_only=True))
            logger.info(f"✅ Loaded final_final concentration network: {final_final_cnet_path}")
        elif os.path.exists(final_cnet_path):
            self.network.load_state_dict(torch.load(final_cnet_path, map_location=self.device, weights_only=True))
            logger.info(f"✅ Loaded final concentration network: {final_cnet_path}")
        else:
            raise FileNotFoundError(f"No concentration network weights found in {self.checkpoint_path}")
        
        # Load k_net if it exists
        if self.k_net is not None:
            if os.path.exists(final_final_knet_path):
                self.k_net.load_state_dict(torch.load(final_final_knet_path, map_location=self.device, weights_only=True))
                logger.info(f"✅ Loaded final_final k_net: {final_final_knet_path}")
            elif os.path.exists(final_knet_path):
                self.k_net.load_state_dict(torch.load(final_knet_path, map_location=self.device, weights_only=True))
                logger.info(f"✅ Loaded final k_net: {final_knet_path}")
            else:
                logger.warning(f"No k_net weights found, using constant diffusion coefficients")
        
        # Set networks to evaluation mode
        self.network.eval()
        if self.k_net is not None:
            self.k_net.eval()
    
    def _save_inference_outputs(
        self,
        C: np.ndarray,
        Kx: np.ndarray,
        Ky: np.ndarray,
        Kz: np.ndarray,
        pde_residual: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        output_dir: str
    ):
        """Save inference outputs to numpy files."""
        logger.info(f"💾 Saving inference outputs to: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save concentration
        np.save(os.path.join(output_dir, 'concentration_4d.npy'), C)
        logger.info(f"   Saved concentration: {C.shape}")
        
        # Save diffusion coefficients
        np.save(os.path.join(output_dir, 'kx_4d.npy'), Kx)
        np.save(os.path.join(output_dir, 'ky_4d.npy'), Ky)
        np.save(os.path.join(output_dir, 'kz_4d.npy'), Kz)
        logger.info(f"   Saved diffusion coefficients: Kx={Kx.shape}, Ky={Ky.shape}, Kz={Kz.shape}")
        
        # Save PDE residuals
        np.save(os.path.join(output_dir, 'pde_residual_4d.npy'), pde_residual)
        logger.info(f"   Saved PDE residuals: {pde_residual.shape}")
        
        # Save coordinates
        np.savez(os.path.join(output_dir, 'coordinates.npz'), x=x, y=y, z=z, t=t)
        logger.info(f"   Saved coordinates: x={x.shape}, y={y.shape}, z={z.shape}, t={t.shape}")
        
        # Save metadata
        metadata = {
            'concentration_shape': C.shape,
            'diffusion_shapes': {
                'kx': Kx.shape,
                'ky': Ky.shape,
                'kz': Kz.shape
            },
            'pde_residual_shape': pde_residual.shape,
            'coordinate_ranges': {
                'x': [float(x.min()), float(x.max())],
                'y': [float(y.min()), float(y.max())],
                'z': [float(z.min()), float(z.max())],
                't': [float(t.min()), float(t.max())]
            },
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_path': self.checkpoint_path
        }
        
        import json
        with open(os.path.join(output_dir, 'inference_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"   Saved metadata: inference_metadata.json")
        
        logger.info(f"✅ All inference outputs saved successfully!")

    
   