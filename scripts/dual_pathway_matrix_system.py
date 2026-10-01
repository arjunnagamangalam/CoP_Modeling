"""
4th-order dual-pathway postural control model: two INDEPENDENT damped
oscillators (fast + slow), each representing a distinct physiological
control pathway, summed to produce the observed CoP position.

PHYSIOLOGICAL MOTIVATION (not just a math patch on the single-oscillator
model's known ceiling): postural control is well established in the
neuroscience literature to involve distinct pathways with different time
constants -- fast proprioceptive/spinal reflexes and slower supraspinal/
vestibular corrections. This directly matches the two-timescale structure
found very early in this project's velocity-ACF analysis (a fast ~15-40ms
component and a slower ~100-300ms component) that neither a single
oscillator, the earlier 4-state hidden-mode extension, nor SMM calibration
with reweighting could resolve.

State order: [x_fast, v_fast, x_slow, v_slow]. The two pathways are
DYNAMICALLY uncoupled (block-diagonal A -- no term in either pathway's
equation depends on the other's state) and coupled ONLY through the shared
observation y = x_fast + x_slow. Noise enters v_fast and v_slow
independently, each with its own intensity (D_fast, D_slow).
"""
import numpy as np

def build_continuous_system_dual_pathway(omega_fast_sq: float, gamma_fast: float,
                                            omega_slow_sq: float, gamma_slow: float):
    """State order: [x_fast, v_fast, x_slow, v_slow]. L has 2 columns --
    noise enters v_fast (column 0) and v_slow (column 1) independently."""
    A = np.array([
        # Fast pathway block (rows/cols 0-1): dx_fast/dt = v_fast;
        # dv_fast/dt = -omega_fast_sq * x_fast - gamma_fast * v_fast
        [0.0, 1.0, 0.0, 0.0],
        [-omega_fast_sq, -gamma_fast, 0.0, 0.0],
        # Slow pathway block (rows/cols 2-3): same damped-oscillator form,
        # independent parameters, no cross-terms with the fast block
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -omega_slow_sq, -gamma_slow],
    ])
    L = np.array([
        # Process noise enters only the velocity states (one column per
        # pathway), not the position states directly
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])

    return A, L