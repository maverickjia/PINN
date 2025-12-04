"""
Physics-Informed Neural Network (PINN) Data Generator for 3D Advection-Diffusion PDEs.

UNIT CONVENTIONS SUMMARY:
========================
INPUT PARAMETERS (all in SI units):
- domain: (x_min, x_max, y_min, y_max, z_min, z_max) in meters [m]
- time_horizon: total time T in seconds [s]  
- u, v, w: wind components in meters/second [m/s]
- source_xyz: source location(s) in meters [m]
- wind_times: time points for wind data in seconds [s]

OUTPUT DATA:
- xyzt coordinates: normalized to [-0.5, 0.5] range (dimensionless)
- wind components (u, v, w): remain in original units [m/s]

NORMALIZATION:
- Spatial coordinates: [x_min, x_max] -> [-0.5, 0.5], [y_min, y_max] -> [-0.5, 0.5], [z_min, z_max] -> [-0.5, 0.5]
- Time coordinate: [0, T] -> [-0.5, 0.5]
- Wind components: No normalization applied (kept in [m/s])

Example: Generating test data for the full domain and time horizon
---------------------------------------------------------------
# 1. Create DataGenerator instance with single source
from pinn3D import DataGenerator

data_gen = DataGenerator(
    domain=(0, 1, 0, 1, 0, 1),  # [m] - spatial domain in meters
    time_horizon=1.0,             # [s] - time horizon in seconds
    u=[1.0, 1.2, 0.8, 1.1],      # [m/s] - wind components in m/s (4 time points)
    v=[0.0, 0.1, 0.0, -0.1],     # [m/s]
    w=[0.0, 0.0, 0.1, 0.0],      # [m/s]
    source_xyz=(0.5, 0.5, 0.5),  # [m] - source location in meters
    dtype=torch.float32,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 1b. Create DataGenerator instance with multiple sources
data_gen_multi = DataGenerator(
    domain=(0, 1, 0, 1, 0, 1),  # [m] - spatial domain in meters
    time_horizon=1.0,             # [s] - time horizon in seconds
    u=[1.0, 1.2, 0.8, 1.1],      # [m/s] - wind components in m/s (4 time points)
    v=[0.0, 0.1, 0.0, -0.1],     # [m/s]
    w=[0.0, 0.0, 0.1, 0.0],      # [m/s]
    source_xyz=[(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)],  # [m] - source locations in meters
    dtype=torch.float32,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 1c. Create DataGenerator instance with custom wind time points
data_gen_custom = DataGenerator(
    domain=(0, 1, 0, 1, 0, 1),  # [m] - spatial domain in meters
    time_horizon=1.0,             # [s] - time horizon in seconds
    u=[1.0, 1.2, 0.8, 1.1],      # [m/s] - wind components in m/s
    v=[0.0, 0.1, 0.0, -0.1],     # [m/s]
    w=[0.0, 0.0, 0.1, 0.0],      # [m/s]
    wind_times=[0.0, 0.3, 0.7, 1.0],  # [s] - custom time points for wind data
    source_xyz=(0.5, 0.5, 0.5),  # [m] - source location in meters
    dtype=torch.float32,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 2. Generate test data (collocation, initial, and boundary condition points)
test_xyztuvw_col, test_xyzt_ic, test_xyzt_bc = data_gen.generate_full_domain_data(
    N_col=10000,  # number of collocation points
    N_ic=2000,    # number of initial points at t=0
    N_bc=1000,    # number of boundary condition points
    normalize=True  # Output xyzt coordinates normalized to [-0.5, 0.5]
)

# 3. Pass test data to PINNTrainer.train_seq2seq
trainer.train_seq2seq(
    ...,
    test_xyztuvw_col=test_xyztuvw_col,  # xyzt normalized, uvw in original units
    test_xyzt_bc=test_xyzt_bc,           # xyzt normalized
    test_xyzt_ic=test_xyzt_ic            # xyzt normalized
)
---------------------------------------------------------------
"""
import torch
from typing import Tuple, Union, List
import numpy as np

class DataGenerator:
    """
    Data generator for 3D advection-diffusion PINN problems.
    
    UNIT CONVENTIONS:
    =================
    INPUT UNITS:
    - domain: (x_min, x_max, y_min, y_max, z_min, z_max) spatial domain in meters [m]
    - time_horizon: total time T in seconds [s]
    - u, v, w: wind components in m/s [m/s] (must have same length)
    - source_xyz: source location(s) in meters [m]
    - wind_times: time points corresponding to wind values in seconds [s] (optional)
    
    OUTPUT UNITS:
    - xyzt coordinates: normalized to [-0.5, 0.5] range
    - wind components (u, v, w): remain in original units [m/s]
    
    Args:
        domain: (x_min, x_max, y_min, y_max, z_min, z_max) spatial domain in meters
        time_horizon: total time T in seconds
        u, v, w: wind components in m/s (must have same length)
        source_xyz: source location or list of source locations in meters
        wind_times: time points corresponding to wind values in seconds (optional, defaults to evenly spaced)
        dtype: torch dtype for output tensors
        device: torch device for output tensors

    All generated data is normalized to [-0.5, 0.5] using the full domain and [0, T] for time.
    Wind components are assigned based on closest time match to the provided wind data.
    """
    def __init__(
        self,
        domain: Tuple[float, float, float, float, float, float],
        time_horizon: float,
        u: list,
        v: list,
        w: list,
        source_xyz: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
        wind_times: Union[list, np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        # Input validation for domain: must be list or tuple of 6 numbers
        if not (isinstance(domain, (list,tuple)) and len(domain) == 6 and all(isinstance(x, (int, float)) for x in domain)):
            raise ValueError("domain must be a tuple of 6 numbers (x_min, x_max, y_min, y_max, z_min, z_max)")
        
        # Validate domain bounds
        x_min, x_max, y_min, y_max, z_min, z_max = domain
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")
        if z_min >= z_max:
            raise ValueError(f"z_min ({z_min}) must be less than z_max ({z_max})")
        
        # Input validation for time_horizon: must be positive number
        if not isinstance(time_horizon, (int, float)) or time_horizon <= 0:
            raise ValueError("time_horizon must be a positive number")
        # Input validation for u, v, and w: must be lists or numpy arrays of same length
        if not (isinstance(u, (list, np.ndarray)) and isinstance(v, (list, np.ndarray)) and isinstance(w, (list, np.ndarray))):
            raise ValueError("u, v, and w must be lists or numpy arrays")
        if len(u) != len(v) or len(u) != len(w):
            raise ValueError("u, v, and w must have the same length")
        if len(u) == 0:
            raise ValueError("Wind arrays u, v, and w cannot be empty")
        # Input validation for source_xyz: must be list or tuple of 3 numbers, or list of such tuples
        if isinstance(source_xyz, (list, tuple)) and len(source_xyz) == 3 and all(isinstance(x, (int, float)) for x in source_xyz):
            # Single source case - convert to list for consistency
            source_xyz_list: List[Tuple[float, float, float]] = [source_xyz]
        elif isinstance(source_xyz, list) and all(isinstance(s, (list, tuple)) and len(s) == 3 and all(isinstance(x, (int, float)) for x in s) for s in source_xyz):
            # Multiple sources case - already in correct format
            source_xyz_list: List[Tuple[float, float, float]] = source_xyz
        else:
            raise ValueError("source_xyz must be a tuple of 3 numbers (x, y, z) or a list of such tuples for multiple sources")
        
        # Spatial domain
        self.domain = domain  # [m]
        x_min, x_max, y_min, y_max, z_min, z_max = domain
        self.Lx = x_max - x_min
        self.Ly = y_max - y_min
        self.Lz = z_max - z_min
        
        # Validate source locations are within domain bounds
        for i, source in enumerate(source_xyz_list):
            sx, sy, sz = source
            if not (x_min <= sx <= x_max):
                raise ValueError(f"Source {i} x-coordinate ({sx}) must be within domain bounds [{x_min}, {x_max}]")
            if not (y_min <= sy <= y_max):
                raise ValueError(f"Source {i} y-coordinate ({sy}) must be within domain bounds [{y_min}, {y_max}]")
            if not (z_min <= sz <= z_max):
                raise ValueError(f"Source {i} z-coordinate ({sz}) must be within domain bounds [{z_min}, {z_max}]")
        
        # Source locations in both original and normalized coordinates
        self.source_xyz = [np.array(s) for s in source_xyz_list]  # [m] - list of arrays
        self.source_xyz_normalized = [
            np.array([
                (s[0] - (x_min + x_max) / 2) / self.Lx,
                (s[1] - (y_min + y_max) / 2) / self.Ly,
                (s[2] - (z_min + z_max) / 2) / self.Lz
            ]) for s in source_xyz_list
        ]  # normalized to [-0.5, 0.5] - list of arrays

        # Time horizon
        self.T = time_horizon  # [s]

        # Wind components as numpy arrays
        self.u = np.array(u)
        self.v = np.array(v)
        self.w = np.array(w)
        
        # wind_times: time points corresponding to wind values
        # Assume wind data is evenly distributed over the time horizon
        if wind_times is None:
            self.wind_times = np.linspace(0, self.T, len(self.u))
        else:
            self.wind_times = np.array(wind_times)
            # Validate wind_times has same length as wind arrays
            if len(self.wind_times) != len(self.u):
                raise ValueError(f"wind_times length ({len(self.wind_times)}) must match wind arrays length ({len(self.u)})")
        
        # Validate wind data covers the time horizon
        if len(self.wind_times) < 2:
            raise ValueError("Wind data must have at least 2 time points")
        if not np.allclose(self.wind_times[0], 0.0) or not np.allclose(self.wind_times[-1], self.T):
            raise ValueError(f"Wind data must cover the full time horizon [0, {self.T}]")
        if not np.all(np.diff(self.wind_times) >= 0):
            raise ValueError("Wind times must be monotonically increasing")

        # Data type and device for output tensors
        self.dtype = dtype
        self.device = device

    def _get_wind_at_times(self, t_vals: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get wind components (u, v, w) at given time points.
        
        Args:
            t_vals: Array of time values
            
        Returns:
            Tuple of (u_vals, v_vals, w_vals) as torch tensors
        """
        # Ensure t_vals are within bounds
        t_vals_clipped = np.clip(t_vals, 0.0, self.T)
        
        # Find closest wind time index for each t
        wind_idx = np.abs(self.wind_times - t_vals_clipped[:, None]).argmin(axis=1)
        
        # Ensure indices are within bounds
        wind_idx = np.clip(wind_idx, 0, len(self.wind_times) - 1)
        
        # Get wind values
        u_vals = torch.tensor(self.u[wind_idx], dtype=self.dtype, device=self.device)
        v_vals = torch.tensor(self.v[wind_idx], dtype=self.dtype, device=self.device)
        w_vals = torch.tensor(self.w[wind_idx], dtype=self.dtype, device=self.device)
        
        return u_vals, v_vals, w_vals

    def generate_bc_data(
        self, t_start: float, t_end: float, N_points: int, normalize: bool = True
    ) -> torch.Tensor:
        """
        Generate boundary data for the PDE on the z=z_min face
        within time segment [t_start, t_end).
        
        Args:
            t_start: start time in seconds [s]
            t_end: end time in seconds [s]
            N_points: number of points to sample
            normalize: whether to normalize output (default True)
            
        Returns:
            Tensor of shape (N_points, 4) with (x, y, z=z_min, t) sampled on the bottom boundary.
            If normalize is True (default), xyzt coordinates are normalized to [-0.5, 0.5].
            Output is always on self.device and self.dtype.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.domain
        # Sample points on the z=z_min face with varying x, y, t
        xyzt_bc = self._uniform_sampling(x_min, x_max, y_min, y_max, z_min, z_min, t_start, t_end, N_points)
        # Optionally normalize
        if normalize:
            xyzt_bc = self._normalize_xyzt(xyzt_bc)
        # Move to correct dtype and device
        return xyzt_bc.to(dtype=self.dtype, device=self.device)

    def generate_ic_data(
        self, t_start: float, N_points: int, normalize: bool = True
    ) -> torch.Tensor:
        """
        Generate initial data for the PDE within the domain at time t_start.
        
        Args:
            t_start: initial time in seconds [s]
            N_points: number of points to sample
            normalize: whether to normalize output (default True)
            
        Returns:
            Tensor of shape (N_points, 4) with (x, y, z, t) sampled on the domain.
            If normalize is True (default), xyzt coordinates are normalized to [-0.5, 0.5].
            Output is always on self.device and self.dtype.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.domain
        # Uniformly sample points in the domain at t_start
        xyzt_ic = self._uniform_sampling(x_min, x_max, y_min, y_max, z_min, z_max, t_start, t_start, N_points)
        if normalize:
            xyzt_ic = self._normalize_xyzt(xyzt_ic)
        return xyzt_ic.to(dtype=self.dtype, device=self.device)

    def generate_col_data(
        self, t_start: float, t_end: float, N_points: int, 
        normalize: bool = True, t_start_buffer: float = 1e-1
    ) -> torch.Tensor:
        """
        Generate collocation data for the PDE within the domain 
        within time segment [t_start, t_end).
        
        Args:
            t_start: start time in seconds [s]
            t_end: end time in seconds [s]
            N_points: number of points to sample
            normalize: whether to normalize output (default True)
            t_start_buffer: small buffer to avoid sampling exactly at t=0 in seconds [s] (default 1e-1)
            
        Returns:
            (N_points, 7) tensor of (x, y, z, t, u, v, w) where:
            - xyzt coordinates are normalized to [-0.5, 0.5] if normalize=True
            - wind components (u, v, w) remain in original units [m/s]
            Output is always on self.device and self.dtype.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.domain
        # Uniformly sample (x, y, z, t) in the domain and time segment
        t_start_eff = t_start + t_start_buffer if t_start == 0.0 else t_start
        xyzt_col = self._uniform_sampling(x_min, x_max, y_min, y_max, z_min, z_max, t_start_eff, t_end, N_points)
        
        # Extract t values for wind lookup
        t_col = xyzt_col[:, 3].cpu().numpy()  # (N_points,)
        
        # Get wind components using the helper method
        u_col, v_col, w_col = self._get_wind_at_times(t_col)
        
        # Optionally normalize
        if normalize:
            xyzt_col = self._normalize_xyzt(xyzt_col)
        xyzt_col = xyzt_col.to(dtype=self.dtype, device=self.device)
        
        # Concatenate (x, y, z, t, u, v, w)
        xyztuvw_col = torch.cat([xyzt_col, u_col.unsqueeze(1), v_col.unsqueeze(1), w_col.unsqueeze(1)], dim=1)
        return xyztuvw_col

    def generate_full_domain_data(
        self,
        N_col: int,
        N_ic: int,
        N_bc: int,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate collocation points over the full domain and time horizon,
        initial points at t=0, and boundary condition points at z=z_min 
        for use as a test or training set.

        Args:
            N_col: Number of collocation points (x, y, z, t, u, v, w) over [domain] x [0, T]
            N_ic: Number of initial points (x, y, z, t=0) over [domain] at t=0
            N_bc: Number of boundary condition points (x, y, z=z_min, t) over [domain] x [0, T] 
            normalize: Whether to normalize the output (default: True)

        Returns:
            xyztuvw_col: (N_col, 7) tensor of (x, y, z, t, u, v, w) where xyzt normalized, uvw in [m/s]
            xyzt_ic: (N_ic, 4) tensor of (x, y, z, t=0) with normalized xyzt coordinates
            xyzt_bc: (N_bc, 4) tensor of (x, y, z=z_min, t) with normalized xyzt coordinates
        """
        xyztuvw_col = self.generate_col_data(0.0, self.T, N_col, normalize=normalize)
        xyzt_ic = self.generate_ic_data(0.0, N_ic, normalize=normalize)
        xyzt_bc = self.generate_bc_data(0.0, self.T, N_bc, normalize=normalize)
        
        return xyztuvw_col, xyzt_ic, xyzt_bc

    def generate_grid_data(
        self,
        N_x: int,
        N_y: int,
        N_z: int,
        N_t: int,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a regular grid of points for inference.
        
        Args:
            N_x: Number of points in x dimension
            N_y: Number of points in y dimension
            N_z: Number of points in z dimension
            N_t: Number of points in t dimension
            normalize: Whether to normalize the output (default: True)
            
        Returns:
            xyztuvw: (N_x*N_y*N_z*N_t, 7) tensor of (x, y, z, t, u, v, w) where:
                     xyzt coordinates normalized to [-0.5, 0.5] if normalize=True, uvw in [m/s]
            x: (N_x,) array of x coordinates in original units [m]
            y: (N_y,) array of y coordinates in original units [m]
            z: (N_z,) array of z coordinates in original units [m]
            t: (N_t,) array of t coordinates in original units [s]
        """
        # Create coordinate arrays
        x = torch.linspace(self.domain[0], self.domain[1], N_x, dtype=self.dtype, device=self.device)
        y = torch.linspace(self.domain[2], self.domain[3], N_y, dtype=self.dtype, device=self.device)
        z = torch.linspace(self.domain[4], self.domain[5], N_z, dtype=self.dtype, device=self.device)
        t = torch.linspace(0, self.T, N_t, dtype=self.dtype, device=self.device)
        
        # Create meshgrid
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        
        # Flatten and stack
        xyzt = torch.stack([
            X.flatten(),
            Y.flatten(),
            Z.flatten(),
            T.flatten()
        ], dim=1)
        
        # Get wind components for each time point
        t_vals = xyzt[:, 3].cpu().numpy()
        
        # Get wind components using the helper method
        u_vals, v_vals, w_vals = self._get_wind_at_times(t_vals)
        
        # Create xyztuvw tensor
        xyztuvw = torch.cat([
            xyzt,  # x, y, z, t
            u_vals.unsqueeze(1),  # u
            v_vals.unsqueeze(1),  # v
            w_vals.unsqueeze(1)   # w
        ], dim=1)
        
        # Normalize if requested
        if normalize:
            xyztuvw[:, :4] = self._normalize_xyzt(xyztuvw[:, :4])
        
        return xyztuvw, x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), t.cpu().numpy()

    def _uniform_sampling(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float,
        t_min: float, t_max: float,
        N_points: int
    ) -> torch.Tensor:
        """
        Uniformly sample N_points in the 4D space (x, y, z, t).
        For each variable:
          - If min == max, all samples are set to that value.
          - Otherwise, samples are drawn from [min, max), i.e., min is included, max is excluded.
        If you want to include both endpoints, use torch.linspace instead of torch.rand.
        Returns: Tensor of shape (N_points, 4) with columns (x, y, z, t)
        """
        # Check for valid ranges
        if x_min > x_max or y_min > y_max or z_min > z_max or t_min > t_max:
            raise ValueError("For uniform sampling, min must be <= max for all dimensions.")
        # Sample x
        if x_min == x_max:
            x = torch.full((N_points,), x_min)
        else:
            x = torch.rand(N_points) * (x_max - x_min) + x_min
        # Sample y
        if y_min == y_max:
            y = torch.full((N_points,), y_min)
        else:
            y = torch.rand(N_points) * (y_max - y_min) + y_min
        # Sample z
        if z_min == z_max:
            z = torch.full((N_points,), z_min)
        else:
            z = torch.rand(N_points) * (z_max - z_min) + z_min
        # Sample t
        if t_min == t_max:
            t = torch.full((N_points,), t_min)
        else:
            t = torch.rand(N_points) * (t_max - t_min) + t_min
        # Stack into (N_points, 4)
        return torch.stack([x, y, z, t], dim=1)

    def _normalize_xyzt(self, xyzt: torch.Tensor) -> torch.Tensor:
        """
        Normalize the x, y, z, t coordinates to the range [-0.5, 0.5] using the full domain and [0, T] for time.
        
        Normalization formula:
        - x_norm = (x - (x_min + x_max)/2) / Lx
        - y_norm = (y - (y_min + y_max)/2) / Ly  
        - z_norm = (z - (z_min + z_max)/2) / Lz
        - t_norm = (t - (0 + T)/2) / T
        
        This maps [x_min, x_max] -> [-0.5, 0.5], [y_min, y_max] -> [-0.5, 0.5], etc.
        
        Args:
            xyzt: Tensor of shape (N, 4) with coordinates in original units [m, m, m, s]
            
        Returns:
            Normalized tensor of shape (N, 4) with coordinates in [-0.5, 0.5] range
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.domain
        t_min, t_max = 0, self.T
        # Check for zero-width domain or time
        if x_max == x_min or y_max == y_min or z_max == z_min or t_max == t_min:
            raise ValueError("Domain or time horizon has zero width, cannot normalize.")
        xyzt = xyzt.clone()  # Avoid in-place modification
        # Normalize x to [-0.5, 0.5]
        xyzt[:, 0] = (xyzt[:, 0] - (x_min + x_max) / 2) / self.Lx  
        # Normalize y to [-0.5, 0.5]
        xyzt[:, 1] = (xyzt[:, 1] - (y_min + y_max) / 2) / self.Ly
        # Normalize z to [-0.5, 0.5]
        xyzt[:, 2] = (xyzt[:, 2] - (z_min + z_max) / 2) / self.Lz
        # Normalize t to [-0.5, 0.5]
        xyzt[:, 3] = (xyzt[:, 3] - (t_min + t_max) / 2) / self.T
        return xyzt

