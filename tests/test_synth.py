"""
    Implements tests for synthesizing line fields from DMD quantities.
"""
import numpy as np

from dmdsim.core.engine import fit_dmd
from dmdsim.signals.generator import Mode, ModalSignal
from dmdsim.core.synth import mirror_conjugates, synthesize


def _modal_signal() -> ModalSignal:
    """
    Create a deterministic three-mode signal with known modal content.

    The signal contains two traveling waves plus a constant mode, resulting in
    five DMD modes.

    :return: modal signal with known frequencies and growth rates
    :rtype: ModalSignal
    """
    modes = [
        Mode(wavenumber=2.0, frequency=5.0, amplitude=1.5, growth_rate=-0.2, phase=0.3),
        Mode(wavenumber=3.0, frequency=12.0, amplitude=0.8, growth_rate=0.1, phase=-1.0),
        Mode(wavenumber=0.0, frequency=0.0, amplitude=2.0),
    ]
    return ModalSignal(modes, dt=0.01, n_space=64, n_time=300, dx=0.1)


def test_roundtrip() -> None:
    """
    Check that synthesizing from fitted quantities reproduces the original field.

    :return: None
    :rtype: None
    """
    signal = _modal_signal()
    result = fit_dmd(signal.samples, dt=0.01, rank=5)
    rebuilt = synthesize(result.modes, result.amplitudes, result.eigvals, dt=0.01,
                         n_time=signal.n_time)
    assert rebuilt.shape == signal.samples.shape
    assert np.allclose(rebuilt, signal.samples, rtol=1e-6, atol=1e-8)


def test_synthesize_ground_truth() -> None:
    """
    Check the synthesis against the analytic ground truth of a single mode.

    :return: None
    :rtype: None
    """
    mode = Mode(wavenumber=2.0, frequency=5.0, amplitude=1.5, growth_rate=-0.2, phase=0.3)
    signal = ModalSignal([mode], dt=0.01, n_space=64, n_time=200, dx=0.1)
    # each real traveling wave decomposes into one conjugate eigenvalue pair
    shape = np.exp(1j * mode.wavenumber * signal.space)
    eigval = np.exp((mode.growth_rate - 1j * 2.0 * np.pi * mode.frequency) * signal.dt)
    modes = np.stack([shape, np.conj(shape)], axis=1)
    half_amplitude = 0.5 * mode.amplitude * np.exp(1j * mode.phase)
    amplitudes = np.array([half_amplitude, np.conj(half_amplitude)])
    eigvals = np.array([eigval, np.conj(eigval)])
    rebuilt = synthesize(modes, amplitudes, eigvals, dt=signal.dt, n_time=signal.n_time)
    assert np.allclose(rebuilt, signal.samples, rtol=1e-10, atol=1e-12)


def test_modified_eigenvalue() -> None:
    """
    Check that modifying an eigenvalue changes the field as expected.

    :return: None
    :rtype: None
    """
    modes = np.array([[1.0 + 0.0j]])
    amplitudes = np.array([2.0 + 0.0j])
    base = synthesize(modes, amplitudes, np.array([1.0 + 0.0j]), dt=0.01, n_time=100)
    grown = synthesize(modes, amplitudes, np.array([1.5 + 0.0j]), dt=0.01, n_time=100)
    # the neutral eigenvalue keeps the constant field, the enlarged radius grows it
    assert np.allclose(base[:, -1], 2.0)
    assert np.allclose(grown[:, -1], 2.0 * 1.5 ** 99)


def test_mirror_conjugates() -> None:
    """
    Check the conjugate mirroring of modified eigenvalues.

    :return: None
    :rtype: None
    """
    values = np.array([1.0 + 2.0j, 1.0 - 2.0j, 3.0 + 0.0j])
    pairs = [(0, 1)]
    # already conjugate values remain unchanged
    assert np.allclose(mirror_conjugates(values, pairs), values)
    # dragging the first member updates the second one
    modified = values.copy()
    modified[0] = 2.0 + 2.0j
    mirrored = mirror_conjugates(modified, pairs, sources={0})
    assert mirrored[0] == 2.0 + 2.0j
    assert mirrored[1] == np.conj(2.0 + 2.0j)
    # dragging the second member updates the first one
    modified[1] = 0.5 - 1.0j
    mirrored = mirror_conjugates(modified, pairs, sources={1})
    assert mirrored[1] == 0.5 - 1.0j
    assert mirrored[0] == np.conj(0.5 - 1.0j)
    # the input array is not mutated
    assert modified[1] == 0.5 - 1.0j


if __name__ == "__main__":
    pass
