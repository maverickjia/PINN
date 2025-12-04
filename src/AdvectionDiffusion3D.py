import torch
from typing import Tuple, Any, List, Union

class AdvectionDiffusion3D:
    """
    Physics-Informed Neural Network (PINN) implementation for 3D Advection-Diffusion PDEs.
    
    This class implements the 3D advection-diffusion equation with Gaussian source terms:
    
        ∂c/∂t + u·∇c = ∇·(K∇c) + s(x,y,z)
    
    where:
    - c(x,y,z,t) is the concentration field (output of the PINN)
    - u = (u,v,w) is the velocity field (from data)
    - K = (kx,ky,kz) are diffusion coefficients
    - s(x,y,z) are Gaussian source terms
    
    IMPORTANT: All coordinates and parameters use NORMALIZED coordinates:
    - Input coordinates (x,y,z,t) are normalized to [-0.5, 0.5]
    - source_width is in normalized coordinates (proportion of domain size)
    - For example, source_width=0.025 means 2.5% of domain size in each direction
    """
    def __init__(
        self,
        pinn: torch.nn.Module,
        data_generator: Any,
        q: Union[float, List[float]],
        source_width: Union[float, List[float]] = 0.025,
    ):
        """
        pinn: instance of PINN (must support autograd and take normalized (x, y, z, t) as input)
        data_gen: instance of DataGenerator (must provide Lx, Ly, Lz, T attributes for normalization)
        q: emission rate (float) or list of emission rates for multiple sources
        source_width: width/spread of Gaussian source terms in NORMALIZED coordinates (proportion of domain extension).
               This is NOT in real meters. For example, source_width=0.025 means the Gaussian has a width of 
               2.5% of the domain size in each direction. Default is 0.025 (2.5% of domain size).
               Can be a float (same width for all sources) or list of floats (individual widths for each source).
               
               To convert from real-world width (in meters) to normalized width:
               normalized_width = real_width / domain_size
               For example, if your domain is 100m and you want a 2.5m wide source:
               source_width = 2.5 / 100 = 0.025
        """
        # Type checks for robustness
        if not hasattr(data_generator, 'Lx') or not hasattr(data_generator, 'Ly') or not hasattr(data_generator, 'Lz') or not hasattr(data_generator, 'T'):
            raise ValueError("data_gen must have Lx, Ly, Lz, and T attributes for normalization.")
        if not callable(pinn):
            raise ValueError("pinn must be callable (a neural network model)")
        
        # Handle single vs multiple sources
        if isinstance(q, (int, float)):
            # Single source case
            self.q = [float(q)]
            self.source_width = [float(source_width) if isinstance(source_width, (int, float)) else source_width[0]]
            self.x0 = [data_generator.source_xyz_normalized[0][0]]
            self.y0 = [data_generator.source_xyz_normalized[0][1]]
            self.z0 = [data_generator.source_xyz_normalized[0][2]]
        elif isinstance(q, list):
            # Multiple sources case
            if len(q) != len(data_generator.source_xyz_normalized):
                raise ValueError("Number of emission rates must match number of sources")
            
            # Handle source_width for multiple sources
            if isinstance(source_width, (int, float)):
                # Use the same source_width value for all sources
                self.source_width = [float(source_width)] * len(q)
            elif isinstance(source_width, list):
                # Use individual source_width values for each source
                if len(source_width) != len(q):
                    raise ValueError("If source_width is a list, it must have the same length as q")
                self.source_width = [float(s) for s in source_width]
            else:
                raise ValueError("source_width must be a float or list of floats")
            
            self.q = [float(qi) for qi in q]
            self.x0 = [data_generator.source_xyz_normalized[i][0] for i in range(len(q))]
            self.y0 = [data_generator.source_xyz_normalized[i][1] for i in range(len(q))]
            self.z0 = [data_generator.source_xyz_normalized[i][2] for i in range(len(q))]
        else:
            raise ValueError("q must be a float or list of floats")
        
        # Validate source_width values
        if any(s <= 0 for s in self.source_width):
            raise ValueError("All source_width values must be positive")
        
        self.pinn = pinn
        self.data_gen = data_generator

    def compute_pde_values(self, xyztuvw: torch.Tensor, kx: Union[float, torch.Tensor], ky: Union[float, torch.Tensor], kz: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyztuvw: tensor of shape (N, 7) where columns are (x, y, z, t, u, v, w)
        kx, ky, kz: diffusion coefficients (floats or tensors for spatially varying coefficients)
        All inputs must be in normalized coordinates (see DataGenerator normalization).
        Returns: tuple (xyztuvw, values), where values is tensor of shape (N, 1) containing PDE equation values
        """
        # Assert that normalization factors are nonzero
        Lx, Ly, Lz, T = self.data_gen.Lx, self.data_gen.Ly, self.data_gen.Lz, self.data_gen.T
        assert Lx != 0 and Ly != 0 and Lz != 0 and T != 0, "Domain and time normalization factors must be nonzero."
        # Optionally, check for normalization (cannot check numerically, so document clearly)
        # Comment: xyztuvw must be normalized to [-0.5, 0.5] for (x, y, z, t)
        xyztuvw = xyztuvw.clone().detach().requires_grad_(True)
        xyzt = xyztuvw[:, :4]
        u = xyztuvw[:, 4:5]
        v = xyztuvw[:, 5:6]
        w = xyztuvw[:, 6:7]
        c = self.pinn(xyzt)  # (N, 1)

        # Ensure the network output requires gradients
        if not c.requires_grad:
            raise RuntimeError("Network output does not require gradients. "
                             "Make sure the network is in training mode and parameters require gradients.")

        # Compute gradients
        grad = torch.autograd.grad(
            c, xyzt, grad_outputs=torch.ones_like(c),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # (N, 4)
        dc_dx = grad[:, 0:1]
        dc_dy = grad[:, 1:2]
        dc_dz = grad[:, 2:3]
        dc_dt = grad[:, 3:4]

        # Second derivatives
        d2c_dx2 = torch.autograd.grad(
            dc_dx, xyzt, grad_outputs=torch.ones_like(dc_dx),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0][:, 0:1]
        d2c_dy2 = torch.autograd.grad(
            dc_dy, xyzt, grad_outputs=torch.ones_like(dc_dy),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0][:, 1:2]
        d2c_dz2 = torch.autograd.grad(
            dc_dz, xyzt, grad_outputs=torch.ones_like(dc_dz),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0][:, 2:3]

        # Compute s(x, y, z) as a sum of sharp Gaussians for multiple sources
        x = xyzt[:, 0:1]
        y = xyzt[:, 1:2]
        z = xyzt[:, 2:3]
        
        # Initialize source term
        s = torch.zeros_like(x)
        
        # Add contribution from each source
        for i in range(len(self.q)):
            # Warn if source_width is extremely small (may cause numerical issues)
            if self.source_width[i] < 1e-4:
                import warnings
                warnings.warn(f"source_width[{i}] is very small; the source term may be numerically zero except at the center.")
            
            # 3D Gaussian source term: G(x,y,z) = (1/(2πσ²)^(3/2)) * exp(-r²/(2σ²))
            # where r² = (x-x₀)² + (y-y₀)² + (z-z₀)² and σ = source_width[i]
            sigma = self.source_width[i]
            
            # Compute squared distance from source center
            r_squared = ((x - self.x0[i]) ** 2 + (y - self.y0[i]) ** 2 + (z - self.z0[i]) ** 2)
            
            # Compute exponent with numerical stability
            exponent = -r_squared / (2 * sigma ** 2)
            exponent = torch.clamp(exponent, min=-50, max=50)  # Prevent overflow/underflow
            
            # 3D Gaussian normalization factor: 1/((2π)^(3/2) * σ³)
            normalization = 1.0 / ((2 * torch.pi) ** 1.5 * sigma ** 3)
            
            # Compute Gaussian and add to source term
            gauss = normalization * torch.exp(exponent)
            s += self.q[i] * gauss  # (N, 1)

        # Handle tensor diffusion coefficients to ensure proper broadcasting
        if isinstance(kx, torch.Tensor):
            kx = kx.unsqueeze(1)  # (N,) -> (N, 1)
        if isinstance(ky, torch.Tensor):
            ky = ky.unsqueeze(1)  # (N,) -> (N, 1)
        if isinstance(kz, torch.Tensor):
            kz = kz.unsqueeze(1)  # (N,) -> (N, 1)
        
        # PDE values (normalized form)
        residual = (
            (1 / T) * dc_dt
            + u * (1 / Lx) * dc_dx
            + v * (1 / Ly) * dc_dy
            + w * (1 / Lz) * dc_dz
            - kx * (1 / Lx**2) * d2c_dx2
            - ky * (1 / Ly**2) * d2c_dy2
            - kz * (1 / Lz**2) * d2c_dz2
            - s
        )
        
        return xyztuvw, residual  # (N, 7), (N, 1)

    def compute_bc_values(self, xyzt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyzt: tensor of shape (N, 4) where columns are (x, y, z, t)
        All inputs must be in normalized coordinates (see DataGenerator normalization).
        The z coordinate is already set to z_min by the DataGenerator.
        Returns: tuple (xyzt, values), where values is tensor of shape (N, 1)
        with the boundary condition values at each point 
        (Neumann: dc/dz values at z=z_min)
        """
        # Comment: xyzt must be normalized to [-0.5, 0.5] for (x, y, z, t)
        xyzt = xyzt.clone().detach().requires_grad_(True)
        c = self.pinn(xyzt)  # (N, 1)
        
        # Compute gradient to get dc/dz
        grad = torch.autograd.grad(
            c, xyzt, grad_outputs=torch.ones_like(c),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # (N, 4)
        dc_dz = grad[:, 2:3]  # Extract dc/dz (z is the 3rd coordinate, index 2)
        
        return xyzt, dc_dz

    def compute_ic_values(self, xyzt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyzt: tensor of shape (N, 4) where columns are (x, y, z, t)
        All inputs must be in normalized coordinates (see DataGenerator normalization).
        Returns: tuple (xyzt, values), where values is tensor of shape (N, 1)
        with the initial condition values at each point (concentration c at t=0)
        """
        # Comment: xyzt must be normalized to [-0.5, 0.5] for (x, y, z, t)
        xyzt = xyzt.clone().detach().requires_grad_(True)
        c = self.pinn(xyzt)  # (N, 1)
        return xyzt, c

