"""
    Implements a thin numpy based wrapper around the flowTorch DMD analysis.
"""
import numpy as np
import torch as pt

from flowtorch.analysis import DMD

from .quantities import DMDResult


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
    :return: DMD quantities with the flowTorch tensors converted to numpy arrays
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
    return DMDResult(
        modes=dmd.modes.numpy(),
        amplitudes=amplitudes.numpy(),
        eigvals=dmd.eigvals.numpy(),
        dt=dt,
    )


if __name__ == "__main__":
    pass
