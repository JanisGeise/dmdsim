"""
    Implements a thin numpy based wrapper around the flowTorch DMD analysis.
"""
import numpy as np
import torch as pt

from flowtorch.analysis import DMD

from .quantities import DMDResult

_PAIR_TOL = 1e-8


def _symmetrize_amplitudes(modes: np.ndarray, amplitudes: np.ndarray,
                           eigvals: np.ndarray) -> np.ndarray:
    """
    Enforce conjugate symmetry of the amplitudes of conjugate mode pairs.

    For a real snapshot matrix, the exact least-squares amplitudes of a
    conjugate eigenvalue pair must be conjugate themselves; flowTorch solves
    the amplitudes numerically from the first snapshot without enforcing this
    symmetry, and the solve noise grows with the mode count (closely spaced
    frequencies ill-condition the system). The real-valued reconstruction is
    invariant under the symmetrization — the pair contribution
    :math:`\\Re\\{\\phi_i b_i \\lambda_i^t\\} + \\Re\\{\\phi_j b_j \\lambda_j^t\\}`
    depends only on :math:`(b_i + \\bar{b}_j) / 2` — so the symmetry is
    restored at no cost, which keeps the pair detection of ``DMDResult.pairs``
    working for higher mode counts. Pairs are detected from the eigenvalues
    and the mode shapes (both exactly conjugate for real data); the
    amplitudes are deliberately excluded from the detection because they are
    the quantity being repaired.

    :param modes: complex mode shapes with one mode per column, shape ``(n_space, n_modes)``
    :type modes: np.ndarray
    :param amplitudes: complex spectral coefficients :math:`b_i`, shape ``(n_modes,)``
    :type amplitudes: np.ndarray
    :param eigvals: complex discrete-time eigenvalues :math:`\\lambda_i`, shape ``(n_modes,)``
    :type eigvals: np.ndarray
    :return: new amplitude array with conjugate symmetry enforced per pair, shape ``(n_modes,)``
    :rtype: np.ndarray
    """
    symmetrized = np.array(amplitudes, dtype=np.complex128, copy=True)
    n_modes = modes.shape[1]
    assigned: set[int] = set()
    for i in range(n_modes):
        if i in assigned:
            continue
        for j in range(i + 1, n_modes):
            if j in assigned:
                continue
            eig_match = (abs(eigvals[i] - np.conj(eigvals[j]))
                         <= _PAIR_TOL * max(1.0, abs(eigvals[i])))
            mode_scale = max(1.0, float(np.abs(modes[:, i]).max()))
            mode_match = (float(np.abs(modes[:, i] - np.conj(modes[:, j])).max())
                          <= _PAIR_TOL * mode_scale)
            if eig_match and mode_match:
                symmetric = 0.5 * (symmetrized[i] + np.conj(symmetrized[j]))
                symmetrized[i] = symmetric
                symmetrized[j] = np.conj(symmetric)
                assigned.update((i, j))
                break
    return symmetrized


def fit_dmd(samples: np.ndarray, dt: float, rank: int | None = None, optimal: bool = False,
            tlsq: bool = False) -> DMDResult:
    """
    Compute the DMD of a snapshot matrix and return the quantities as numpy arrays.

    The snapshot matrix is converted to a double precision ``torch`` tensor before
    the fit, so all returned quantities are accurate to double precision regardless
    of the flowTorch default dtype.

    :param samples: snapshot matrix with one snapshot per column, shape ``(n_space, n_time)``
    :type samples: np.ndarray
    :param dt: time step between two snapshots
    :type dt: float
    :param rank: SVD truncation rank; ``None`` uses the flowTorch optimal rank
        selection, defaults to None
    :type rank: int | None, optional
    :param optimal: compute the mode amplitudes based on a least-squares problem
        as in the spDMD method, defaults to False
    :type optimal: bool, optional
    :param tlsq: de-bias the linear operator by solving a total least-squares
        problem instead of a standard least-squares problem, defaults to False
    :type tlsq: bool, optional
    :return: DMD quantities with the flowTorch tensors converted to numpy arrays,
        including the flowTorch integral contribution of each mode
    :rtype: DMDResult
    """
    assert samples.ndim == 2, (
        f"The samples must be a 2D snapshot matrix, got shape: {samples.shape}")
    assert dt > 0.0, f"The time step must be positive, got: {dt}"
    matrix = pt.as_tensor(np.asarray(samples, dtype=np.float64))
    dmd = DMD(matrix, dt=dt, rank=rank, optimal=optimal, tlsq=tlsq)
    # flowTorch returns a list of tensors only for multiple input data matrices
    amplitudes = dmd.amplitude
    assert not isinstance(amplitudes, list), (
        "Expected a single data matrix, but flowTorch returned a list of amplitudes")
    modes = dmd.modes.numpy()
    eigvals = dmd.eigvals.numpy()
    return DMDResult(
        modes=modes,
        amplitudes=_symmetrize_amplitudes(modes, amplitudes.numpy(), eigvals),
        eigvals=eigvals,
        dt=dt,
        integral_contribution=dmd.integral_contribution.numpy(),
    )


if __name__ == "__main__":
    pass
