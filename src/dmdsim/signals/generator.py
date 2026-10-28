"""
    Implements a generator for synthetic modal line fields used as DMD playground data.
"""
import numpy as np


class Mode:
    def __init__(self, wavenumber: float, frequency: float, amplitude: float = 1.0,
                 growth_rate: float = 0.0, phase: float = 0.0) -> None:
        """
        Implement a class for a single modal component of a synthetic line field.

        The mode contributes :math:`A\\,\\exp(\\sigma t)\\,\\cos(\\kappa s - 2\\pi f t + \\phi)`
        to the field, i.e., a traveling wave with wavenumber :math:`\\kappa` and temporal
        frequency :math:`f` that grows or decays with rate :math:`\\sigma`. In the DMD
        picture, the mode corresponds to a conjugate pair of eigenvalues
        :math:`\\lambda = \\exp((\\sigma \\mp 2\\pi \\mathrm{i} f)\\,dt)` with mode shapes
        :math:`\\exp(\\pm \\mathrm{i}\\kappa s)` and amplitudes :math:`A \\exp(\\pm \\mathrm{i}\\phi) / 2`.

        :param wavenumber: spatial wavenumber :math:`\\kappa` in radians per unit length
        :type wavenumber: float
        :param frequency: temporal frequency :math:`f` in hertz
        :type frequency: float
        :param amplitude: amplitude :math:`A` of the mode, defaults to 1.0
        :type amplitude: float, optional
        :param growth_rate: growth rate :math:`\\sigma` in 1/s; positive values grow,
            negative values decay, defaults to 0.0
        :type growth_rate: float, optional
        :param phase: phase shift :math:`\\phi` in radians; shifts the wave pattern
            in space and time simultaneously, defaults to 0.0
        :type phase: float, optional
        :return: None
        :rtype: None
        """
        self.wavenumber = wavenumber
        self.frequency = frequency
        self.amplitude = amplitude
        self.growth_rate = growth_rate
        self.phase = phase
        self._check_parameters()

    def _check_parameters(self) -> None:
        """
        Validate the mode parameters.

        :return: None
        :rtype: None
        """
        assert (np.isfinite(self.wavenumber) and np.isfinite(self.frequency)
                and np.isfinite(self.amplitude) and np.isfinite(self.growth_rate)
                and np.isfinite(self.phase)), (
            f"All mode parameters must be finite, got: {self!r}")
        assert self.amplitude >= 0.0, (
            f"The mode amplitude must be non-negative, got: {self.amplitude}")

    def __repr__(self) -> str:
        """
        Create a debug representation of the mode.

        :return: string representation of all mode parameters
        :rtype: str
        """
        return (f"Mode(wavenumber={self.wavenumber}, frequency={self.frequency}, "
                f"amplitude={self.amplitude}, growth_rate={self.growth_rate}, phase={self.phase})")


class ModalSignal:
    def __init__(self, modes: list[Mode], dt: float, n_space: int, n_time: int, dx: float = 1.0,
                 offset: float = 0.0, noise_std: float = 0.0, seed: int | None = None) -> None:
        """
        Implement a class for a synthetic line field composed of modal traveling waves.

        The field reads :math:`x(s, t) = o + \\sum_i A_i \\exp(\\sigma_i t)
        \\cos(\\kappa_i s - 2\\pi f_i t + \\phi_i) + \\epsilon(s, t)` on the uniform grid
        :math:`s = 0, \\ldots, (n_{space} - 1)\\,dx` and
        :math:`t = 0, \\ldots, (n_{time} - 1)\\,dt`, where :math:`\\epsilon` is optional
        zero-mean Gaussian noise.

        .. note::
            The samples are computed once at construction time; mutating the ``modes``
            afterwards does not change the field.

        :param modes: list of modes to superpose; may be empty
        :type modes: list[Mode]
        :param dt: time step between two consecutive time levels
        :type dt: float
        :param n_space: number of spatial grid points
        :type n_space: int
        :param n_time: number of time levels
        :type n_time: int
        :param dx: spatial grid spacing, defaults to 1.0
        :type dx: float, optional
        :param offset: constant offset :math:`o` added to the field, defaults to 0.0
        :type offset: float, optional
        :param noise_std: standard deviation of the added Gaussian noise; no noise
            is added if set to 0.0, defaults to 0.0
        :type noise_std: float, optional
        :param seed: seed for the noise random number generator; ``None`` uses
            nondeterministic seeding, defaults to None
        :type seed: int | None, optional
        :return: None
        :rtype: None
        """
        self.modes = list(modes)
        self.dt = dt
        self.n_space = n_space
        self.n_time = n_time
        self.dx = dx
        self.offset = offset
        self.noise_std = noise_std
        self.seed = seed
        self._check_parameters()
        self._samples = self._evaluate()

    def _check_parameters(self) -> None:
        """
        Validate the grid and signal parameters.

        :return: None
        :rtype: None
        """
        assert self.n_space >= 2, (
            f"The number of spatial points must be at least 2, got: {self.n_space}")
        assert self.n_time >= 2, (
            f"The number of time levels must be at least 2, got: {self.n_time}")
        assert self.dt > 0.0, f"The time step must be positive, got: {self.dt}"
        assert self.dx > 0.0, f"The spatial grid spacing must be positive, got: {self.dx}"
        assert self.noise_std >= 0.0, (
            f"The noise standard deviation must be non-negative, got: {self.noise_std}")
        assert np.isfinite(self.offset), f"The offset must be finite, got: {self.offset}"

    def _evaluate(self) -> np.ndarray:
        """
        Evaluate the modal superposition on the space-time grid.

        :return: field values with one snapshot per column, shape ``(n_space, n_time)``
        :rtype: np.ndarray
        """
        # broadcast the spatial grid over time and vice versa
        space = self.space[:, None]
        time = self.time[None, :]
        field = np.full((self.n_space, self.n_time), self.offset)
        for mode in self.modes:
            omega = 2.0 * np.pi * mode.frequency
            field += (mode.amplitude * np.exp(mode.growth_rate * time)
                      * np.cos(mode.wavenumber * space - omega * time + mode.phase))
        # add optional zero-mean Gaussian noise
        if self.noise_std > 0.0:
            rng = np.random.default_rng(self.seed)
            field = field + rng.normal(0.0, self.noise_std, field.shape)
        return field

    @property
    def space(self) -> np.ndarray:
        """
        Spatial grid of the field.

        :return: uniform spatial grid from 0 to ``(n_space - 1) * dx``
        :rtype: np.ndarray
        """
        return np.arange(self.n_space) * self.dx

    @property
    def time(self) -> np.ndarray:
        """
        Time grid of the field.

        :return: uniform time grid from 0 to ``(n_time - 1) * dt``
        :rtype: np.ndarray
        """
        return np.arange(self.n_time) * self.dt

    @property
    def samples(self) -> np.ndarray:
        """
        Field values of the synthetic signal.

        :return: real-valued field with one snapshot per column, shape ``(n_space, n_time)``
        :rtype: np.ndarray
        """
        return self._samples


if __name__ == "__main__":
    pass
