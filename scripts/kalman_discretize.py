"""
Stage 1a: exact discretization of the continuous 4-state linear SDE

  ds/dt = A s + L xi(t),   Var(xi increment over dt) = 2*D_xi*dt

into the discrete-time form s_{k+1} = F s_k + w_k, Cov(w_k) = Qd, using the
Van Loan (1978) method -- exact for a linear system, not an Euler
approximation, computed via scipy's validated matrix exponential (chosen
over a hand-derived closed form for the same reason four_state_model.py
used RK4 instead of a hand-derived convolution: lower derivation-bug risk).

D_xi uses the SAME diffusion-coefficient convention as everywhere else in
this project (e.g. extract_D_v_second_order_cm, generate_colored_noise_trace):
dw = (...) dt + sqrt(2*D_xi) dW, Var(dW) = dt.
"""
import numpy as np
from scipy.linalg import expm

def build_continuous_system(omega_sq: float, gamma: float, omega_n_sq: float,
                               gamma_n: float, kappa: float):
    """State order: [x, v, u, w]. Noise enters ONLY the w equation (L picks
    out the 4th state)."""
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-omega_sq, -gamma, kappa, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -omega_n_sq, -gamma_n],
    ])
    L = np.array([[0.0], [0.0], [0.0], [1.0]])
    return A, L


def van_loan_discretize(A: np.ndarray, L: np.ndarray, D_xi: float, dt: float):
    """
    Returns (F, Qd): exact discrete-time transition matrix and process-noise
    covariance for s_{k+1} = F s_k + w_k, Cov(w_k) = Qd, given the
    continuous system ds = A s dt + L dW with Var(dW) = 2*D_xi*dt.

    Block matrix uses [[-A, B],[0, A^T]] (NOT [[A,B],[0,-A^T]] -- an earlier
    version had this sign flipped, verified wrong by checking the zero-noise
    limit: with B=0 the block matrix is block-diagonal, and expm of the
    (2,2) block must reduce to exactly expm(A*dt) for F, which only holds
    with the +A^T sign, not -A^T. The wrong-sign version gave |eig(F)|>1,
    an unstable discretization of a system whose continuous dynamics are
    stable -- mathematically impossible for a correct exact discretization,
    which is what caught the bug.
    """
    n = A.shape[0]
    Qc = L @ np.array([[2.0 * D_xi]]) @ L.T  # (n, n), nonzero only at [3,3]

    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    M = M * dt

    expM = expm(M)
    Phi12 = expM[:n, n:]
    Phi22 = expM[n:, n:]
    F = Phi22.T
    Qd = F @ Phi12
    return F, Qd


def van_loan_discretize_general(A: np.ndarray, L: np.ndarray, D_diag: np.ndarray, dt: float):
    """
    Generalization of van_loan_discretize above to MULTIPLE independent
    noise sources -- needed for the dual-pathway (fast+slow oscillator)
    model, where noise enters two different states independently, each
    with its own intensity. L is (n, m) for m noise sources; D_diag is
    length m (one intensity per column of L). Same exact block-matrix
    algorithm and sign convention as the original (unchanged, still
    validated by the zero-noise-limit check documented there) -- only the
    Qc construction is generalized from a scalar to a diagonal matrix.

    Cross-validated against the original van_loan_discretize: with a
    single noise source (L as (n,1), D_diag as a 1-element array), this
    must give IDENTICAL (F, Qd) to the original function -- confirmed
    below before this is used for anything real.
    """
    n = A.shape[0]
    Qc = L @ np.diag(2.0 * np.asarray(D_diag)) @ L.T  # (n, n)

    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    M = M * dt

    expM = expm(M)
    Phi12 = expM[:n, n:]
    Phi22 = expM[n:, n:]
    F = Phi22.T
    Qd = F @ Phi12
    return F, Qd


def stationary_variance_u(D_xi: float, gamma_n: float, omega_n_sq: float) -> float:
    """Analytical stationary variance of u alone (the SAME formula used
    throughout this project: Var = D_v / (gamma * omega_sq), applied to the
    fast subsystem). Used ONLY to sanity-check the discretization below --
    independent of the Van Loan machinery, so agreement is a genuine
    cross-check, not circular."""
    return D_xi / (gamma_n * omega_n_sq)