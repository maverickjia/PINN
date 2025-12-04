# tests/test_init.py

import pytest


def test_package_import():
    """Test that the pinn3D package imports successfully."""
    try:
        import pinn3D  
    except Exception as e:
        pytest.fail(f"Failed to import pinn3D: {e}")


def test_public_api_symbols():
    """Test that objects listed in __all__ are importable."""
    import pinn3D  # noqa: F401

    expected_symbols = [
        "DataGenerator",
        "Network3D",
        "AdvectionDiffusion3D",
        "PINNTrainer",
    ]

    for symbol in expected_symbols:
        assert hasattr(pinn3D, symbol), f"Missing symbol: {symbol}"


def test_core_class_instantiation():
    """
    Ensure the core physics classes can be instantiated with minimal, valid arguments.
    """
    from pinn3D import (
        DataGenerator,
        Network3D,
        AdvectionDiffusion3D,
    )
    import torch

    # ---- DataGenerator: use the actual signature ----
    # domain: (x_min, x_max, y_min, y_max, z_min, z_max)
    domain = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    time_horizon = 10.0

    # Simple wind time series: two time points with constant wind
    u = [1.0, 1.0]
    v = [0.0, 0.0]
    w = [0.0, 0.0]
    wind_times = [0.0, time_horizon]

    # A single source at the center of the domain
    source_xyz = (0.5, 0.5, 0.5)

    dg = DataGenerator(
        domain=domain,
        time_horizon=time_horizon,
        u=u,
        v=v,
        w=w,
        source_xyz=source_xyz,
        wind_times=wind_times,
    )
    assert dg is not None

    # ---- Network3D: match the __init__ signature (layer_sizes, not layers) ----
    # layer_sizes are hidden-layer sizes; input/output are handled inside Network3D.
    net = Network3D(
        layer_sizes=[50, 50],   # two hidden layers with 50 neurons each
        hidden_activation="tanh",
        init_method="glorot",
        output_activation="exp",
    )
    assert net is not None

    # Optional: sanity-check forward pass with a single (x, y, z, t) point
    x = torch.zeros(1, 4, dtype=torch.float32)
    y = net(x)
    assert y.shape[0] == 1

    # ---- AdvectionDiffusion3D: use the real signature ----
    pde = AdvectionDiffusion3D(
        pinn=net,
        data_generator=dg,
        q=1.0,              # simple scalar emission rate
        source_width=0.025, # default width
    )
    assert pde is not None


def test_trainer_interface():
    """
    Check that PINNTrainer is available and exposes the expected training API.
    """
    from pinn3D import PINNTrainer

    # Class is importable
    assert PINNTrainer is not None

    # It should at least provide the main training entry point used in docs/examples
    assert hasattr(PINNTrainer, "train_seq2seq"), "PINNTrainer is missing train_seq2seq()"
