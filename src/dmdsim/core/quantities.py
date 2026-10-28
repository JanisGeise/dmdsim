"""
    Implements a container for DMD quantities with derived properties and conjugate pair bookkeeping.
"""
import numpy as np

_PAIR_TOL = 1e-8


class DMDResult:
    def __init__(self, modes: np.ndarray, amplitudes: np.ndarray, eigvals: np.ndarray,
                 dt: float, integral_contribution: np.ndarray | None = None) -> None:
        """
        Implement a class for holding and analyzing the quantities of a DMD fit.

        The quantities describe the reconstruction
        :math:`x(s, t) \\approx \\Re\\left\\{\\sum_i \\phi_i(s)\\, b_i\\, \\lambda_i^t\\right\\}`
        with the complex mode shapes :math:`\\phi_i`, the complex spectral coefficients
        :math:`b_i`, and the discrete-time eigenvalues :math:`\\lambda_i = \\exp(\\mu_i\\,dt)`.
        The radius of an eigenvalue encodes the growth or decay rate and its angle
        encodes the frequency of the corresponding mode.

        :param modes: complex mode shapes with one mode per column, shape ``(n_space, n_modes)``
        :type modes: np.ndarray
        :param amplitudes: complex spectral coefficients :math:`b_i`, shape ``(n_modes,)``
        :type amplitudes: np.ndarray
        :param eigvals: complex discrete-time eigenvalues :math:`\\lambda_i`, shape ``(n_modes,)``
        :type eigvals: np.ndarray
        :param dt: time step between two snapshots used in the DMD fit
        :type dt: float
        :param integral_contribution: integral contribution of each mode according to
            Kou et al. (2017) as provided by flowTorch; required for sorting by the
            ``"integral"`` criterion, defaults to None
        :type integral_contribution: np.ndarray | None, optional
        :return: None
        :rtype: None
        """
        self.modes = np.asarray(modes, dtype=np.complex128)
        self.amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        self.eigvals = np.asarray(eigvals, dtype=np.complex128)
        self.integral_contribution = (None if integral_contribution is None
                                      else np.asarray(integral_contribution, dtype=np.float64))
        self.dt = dt
        self._check_dimensions()

    def _check_dimensions(self) -> None:
        """
        Validate the dimensions and consistency of the stored quantities.

        :return: None
        :rtype: None
        """
        assert self.modes.ndim == 2, (
            f"The modes must be a 2D array with one mode per column, got shape: {self.modes.shape}")
        n_modes = self.modes.shape[1]
        assert self.amplitudes.shape == (n_modes,), (
            f"The amplitudes must have shape ({n_modes},), got: {self.amplitudes.shape}")
        assert self.eigvals.shape == (n_modes,), (
            f"The eigenvalues must have shape ({n_modes},), got: {self.eigvals.shape}")
        if self.integral_contribution is not None:
            assert self.integral_contribution.shape == (n_modes,), (
                f"The integral contributions must have shape ({n_modes},), "
                f"got: {self.integral_contribution.shape}")
        assert self.dt > 0.0, f"The time step must be positive, got: {self.dt}"

    def _is_conjugate_pair(self, i: int, j: int) -> bool:
        """
        Check whether two modes form a complex conjugate pair.

        :param i: index of the first mode
        :type i: int
        :param j: index of the second mode
        :type j: int
        :return: True if eigenvalues, amplitudes, and mode shapes are conjugate up to tolerance
        :rtype: bool
        """
        # eigenvalues and amplitudes are compared element-wise, modes column-wise
        eig_match = (abs(self.eigvals[i] - np.conj(self.eigvals[j]))
                     <= _PAIR_TOL * max(1.0, abs(self.eigvals[i])))
        amp_match = (abs(self.amplitudes[i] - np.conj(self.amplitudes[j]))
                     <= _PAIR_TOL * max(1.0, abs(self.amplitudes[i])))
        mode_scale = max(1.0, float(np.abs(self.modes[:, i]).max()))
        mode_match = float(np.abs(self.modes[:, i] - np.conj(self.modes[:, j])).max()) \
            <= _PAIR_TOL * mode_scale
        return bool(eig_match and amp_match and mode_match)

    def sort(self, by: str) -> None:
        """
        Reorder the modes in place according to a sorting criterion.

        Sorting only rearranges the stored quantities; the reconstruction is a
        sum over all modes and therefore invariant under the chosen order. The
        criteria follow flowTorch: the amplitude importance :math:`|b_i|` and the
        integral contribution of Kou et al. (2017),
        :math:`\\|\\phi_i\\|^2 \\sum_t |b_i \\lambda_i^t|`.

        :param by: sorting criterion; one of ``amplitude`` (descending
            :math:`|b|`), ``integral`` (descending integral contribution),
            or ``frequency`` (ascending :math:`|f|`, flowTorch uses frequency
            only as a filter)
        :type by: str
        :return: None
        :rtype: None
        """
        if by == "amplitude":
            key = np.abs(self.amplitudes)
            descending = True
        elif by == "integral":
            assert self.integral_contribution is not None, (
                "The integral criterion requires the integral contributions "
                "provided by flowTorch")
            key = self.integral_contribution
            descending = True
        elif by == "frequency":
            key = np.abs(self.frequencies)
            descending = False
        else:
            assert False, (f"Unknown sorting criterion {by!r}, must be one of: "
                           "'amplitude', 'integral', 'frequency'")
        order = np.argsort(key, kind="stable")
        if descending:
            order = order[::-1]
        self.modes = np.ascontiguousarray(self.modes[:, order])
        self.amplitudes = self.amplitudes[order]
        self.eigvals = self.eigvals[order]
        if self.integral_contribution is not None:
            self.integral_contribution = self.integral_contribution[order]

    def mode(self, index: int) -> np.ndarray:
        """
        Return the complex mode shape of a single mode.

        :param index: mode index
        :type index: int
        :return: complex mode shape :math:`\\phi_i` along the spatial grid, shape ``(n_space,)``
        :rtype: np.ndarray
        """
        return self.modes[:, index]

    @property
    def n_modes(self) -> int:
        """
        Number of DMD modes.

        :return: number of modes
        :rtype: int
        """
        return self.modes.shape[1]

    @property
    def n_space(self) -> int:
        """
        Number of spatial grid points of the mode shapes.

        :return: number of spatial points
        :rtype: int
        """
        return self.modes.shape[0]

    @property
    def frequencies(self) -> np.ndarray:
        """
        Temporal frequencies of the modes.

        The frequency is computed from the principal branch of the complex logarithm
        of the eigenvalues, so it lies within the Nyquist limit :math:`|f| \\le 1 / (2\\,dt)`.
        Conjugate pairs appear with opposite signs.

        :return: frequencies :math:`f_i = \\Im(\\log \\lambda_i) / (2\\pi\\,dt)` in hertz, shape ``(n_modes,)``
        :rtype: np.ndarray
        """
        return np.imag(np.log(self.eigvals)) / (2.0 * np.pi * self.dt)

    @property
    def growth_rates(self) -> np.ndarray:
        """
        Growth rates of the modes.

        :return: growth rates :math:`\\sigma_i = \\Re(\\log \\lambda_i) / dt` in 1/s, shape ``(n_modes,)``
        :rtype: np.ndarray
        """
        return np.real(np.log(self.eigvals)) / self.dt

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """
        Indices of complex conjugate mode pairs.

        Real-valued data produce modes in conjugate pairs; each pair is reported once
        as an index tuple ``(i, j)`` with :math:`\\lambda_j = \\bar{\\lambda}_i`.
        Modes that do not pair up, e.g., real eigenvalues such as a constant offset
        mode, are not included.

        .. note::
            The pairs are detected from the current state of the quantities; cache
            the result before modifying eigenvalues or amplitudes if the pair
            structure of the original fit should be preserved.

        :return: list of mutually exclusive index tuples of conjugate pairs
        :rtype: list[tuple[int, int]]
        """
        pairs = []
        assigned = set()
        for i in range(self.n_modes):
            if i in assigned:
                continue
            for j in range(i + 1, self.n_modes):
                if j in assigned:
                    continue
                # report each mode at most once as part of a pair
                if self._is_conjugate_pair(i, j):
                    pairs.append((i, j))
                    assigned.update((i, j))
                    break
        return pairs


if __name__ == "__main__":
    pass
