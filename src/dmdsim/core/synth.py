"""
    Implements the synthesis of line fields from DMD quantities, the core of the eigenvalue playground.
"""
import numpy as np


def synthesize_window(modes: np.ndarray, amplitudes: np.ndarray, eigvals: np.ndarray,
                      dt: float, t_start: int, n_steps: int) -> np.ndarray:
    """
    Rebuild a real-valued line field for a window of integer time levels.

    Implements :math:`x(s, t) = \\Re\\left\\{\\sum_i \\phi_i(s)\\, b_i\\, \\lambda_i^t\\right\\}`
    for the integer time levels :math:`t = t_{start}, \\ldots, t_{start} + n_{steps} - 1`.
    A window starting beyond the fitted data range is the DMD prediction: the linear
    model is advanced with the same formula, so growing eigenvalues
    (:math:`|\\lambda| > 1`) amplify the field and shrinking ones damp it.

    :param modes: complex mode shapes with one mode per column, shape ``(n_space, n_modes)``
    :type modes: np.ndarray
    :param amplitudes: complex spectral coefficients :math:`b_i`, shape ``(n_modes,)``
    :type amplitudes: np.ndarray
    :param eigvals: complex discrete-time eigenvalues :math:`\\lambda_i`, shape ``(n_modes,)``
    :type eigvals: np.ndarray
    :param dt: time step between consecutive time levels; only used for consistency
        of the input quantities, the synthesis itself works with integer powers
    :type dt: float
    :param t_start: first integer time level of the window
    :type t_start: int
    :param n_steps: number of time levels of the window
    :type n_steps: int
    :return: real-valued rebuilt field with one snapshot per column, shape ``(n_space, n_steps)``
    :rtype: np.ndarray
    """
    assert t_start >= 0, f"The start time level must be non-negative, got: {t_start}"
    assert n_steps >= 1, f"The number of steps must be at least 1, got: {n_steps}"
    # time dynamics of each mode are the amplitude times the eigenvalue powers
    time_dynamics = amplitudes[:, None] * eigvals[:, None] ** np.arange(t_start,
                                                                        t_start + n_steps)
    return np.real(modes @ time_dynamics)


def synthesize(modes: np.ndarray, amplitudes: np.ndarray, eigvals: np.ndarray, dt: float,
               n_time: int) -> np.ndarray:
    """
    Rebuild a real-valued line field from mode shapes, amplitudes, and eigenvalues.

    Implements :math:`x(s, t) = \\Re\\left\\{\\sum_i \\phi_i(s)\\, b_i\\, \\lambda_i^t\\right\\}`
    with integer time levels :math:`t = 0, \\ldots, n_{time} - 1`. Modified eigenvalues
    or amplitudes, e.g., from dragging in the playground, directly change the rebuilt field.

    :param modes: complex mode shapes with one mode per column, shape ``(n_space, n_modes)``
    :type modes: np.ndarray
    :param amplitudes: complex spectral coefficients :math:`b_i`, shape ``(n_modes,)``
    :type amplitudes: np.ndarray
    :param eigvals: complex discrete-time eigenvalues :math:`\\lambda_i`, shape ``(n_modes,)``
    :type eigvals: np.ndarray
    :param dt: time step between consecutive time levels; only used for consistency
        of the input quantities, the synthesis itself works with integer powers
    :type dt: float
    :param n_time: number of time levels of the rebuilt field
    :type n_time: int
    :return: real-valued rebuilt field with one snapshot per column, shape ``(n_space, n_time)``
    :rtype: np.ndarray
    """
    return synthesize_window(modes, amplitudes, eigvals, dt, 0, n_time)


def mirror_conjugates(values: np.ndarray, pairs: list[tuple[int, int]],
                      sources: set[int] | None = None) -> np.ndarray:
    """
    Enforce conjugate symmetry on a quantity array according to given mode pairs.

    For each pair ``(i, j)``, the value at the target index is replaced by the
    conjugate of the value at the source index. By default, the first index of each
    pair is the source; a set of ``sources`` overrides this per pair, which is used
    after dragging a specific eigenvalue. The input array is not modified.

    :param values: complex quantity array, e.g., eigenvalues or amplitudes, shape ``(n_modes,)``
    :type values: np.ndarray
    :param pairs: index tuples ``(i, j)`` of conjugate pairs as reported by ``DMDResult.pairs``
    :type pairs: list[tuple[int, int]]
    :param sources: indices to treat as sources instead of the default first pair
        index; pairs without a source index keep their default orientation,
        defaults to None
    :type sources: set[int] | None, optional
    :return: new array with conjugate symmetry enforced, shape ``(n_modes,)``
    :rtype: np.ndarray
    """
    mirrored = np.array(values, dtype=np.complex128, copy=True)
    for i, j in pairs:
        # the dragged index, if any, acts as the source of the pair
        source = i
        if sources is not None and i not in sources and j in sources:
            source = j
        target = j if source == i else i
        mirrored[target] = np.conj(mirrored[source])
    return mirrored


if __name__ == "__main__":
    pass
