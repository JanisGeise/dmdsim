"""
    Implements tests for the DMD engine wrapper around flowTorch.
"""
import pytest
import numpy as np

from dmdsim.core.engine import fit_dmd
from dmdsim.core.synth import synthesize
from dmdsim.signals.generator import Mode, ModalSignal


def _modal_signal() -> ModalSignal:
    """
    Create a deterministic three-mode signal with known modal content.

    The signal contains two traveling waves with different frequencies and growth
    rates plus a constant mode, resulting in five DMD modes.

    :return: modal signal with known frequencies and growth rates
    :rtype: ModalSignal
    """
    modes = [
        Mode(wavenumber=2.0, frequency=5.0, amplitude=1.5, growth_rate=-0.2, phase=0.3),
        Mode(wavenumber=3.0, frequency=12.0, amplitude=0.8, growth_rate=0.1, phase=-1.0),
        Mode(wavenumber=0.0, frequency=0.0, amplitude=2.0),
    ]
    return ModalSignal(modes, dt=0.01, n_space=64, n_time=300, dx=0.1)


def test_recover_modal_content() -> None:
    """
    Check that the DMD fit recovers the known frequencies and growth rates.

    :return: None
    :rtype: None
    """
    signal = _modal_signal()
    result = fit_dmd(signal.samples, dt=0.01, rank=5)
    assert result.n_modes == 5
    assert result.eigvals.dtype == np.complex128
    # conjugate pairs appear with opposite frequency signs, so compare magnitudes
    expected_frequencies = np.sort(np.array([0.0, 5.0, 5.0, 12.0, 12.0]))
    assert np.allclose(np.sort(np.abs(result.frequencies)), expected_frequencies,
                       rtol=1e-6, atol=1e-8)
    # both members of a pair share the growth rate of their generator mode
    expected_growth = np.sort(np.array([-0.2, -0.2, 0.0, 0.1, 0.1]))
    assert np.allclose(np.sort(result.growth_rates), expected_growth, rtol=1e-6, atol=1e-8)


def test_recover_eigenvalues() -> None:
    """
    Check that the discrete eigenvalues match the analytic ground truth.

    :return: None
    :rtype: None
    """
    signal = _modal_signal()
    result = fit_dmd(signal.samples, dt=0.01, rank=5)
    omega_1 = 2.0 * np.pi * 5.0
    omega_2 = 2.0 * np.pi * 12.0
    expected = np.array([
        np.exp((-0.2 + 1j * omega_1) * 0.01), np.exp((-0.2 - 1j * omega_1) * 0.01),
        np.exp((0.1 + 1j * omega_2) * 0.01), np.exp((0.1 - 1j * omega_2) * 0.01),
        1.0,
    ])
    # match each expected eigenvalue to its closest fitted counterpart
    distances = np.abs(result.eigvals[:, None] - expected[None, :]).min(axis=0)
    assert np.allclose(distances, 0.0, atol=1e-8)


def test_conjugate_pair_detection() -> None:
    """
    Check that the wave modes pair up and the constant mode stays unpaired.

    :return: None
    :rtype: None
    """
    signal = _modal_signal()
    result = fit_dmd(signal.samples, dt=0.01, rank=5)
    pairs = result.pairs
    assert len(pairs) == 2
    paired = {index for pair in pairs for index in pair}
    unpaired = set(range(result.n_modes)) - paired
    assert len(unpaired) == 1
    # the unpaired mode is the constant one with vanishing frequency
    assert np.allclose(np.abs(result.frequencies[list(unpaired)]), 0.0, atol=1e-8)


def test_sort_modes() -> None:
    """
    Check the sorting criteria of the DMD result container.

    :return: None
    :rtype: None
    """
    signal = _modal_signal()
    result = fit_dmd(signal.samples, dt=0.01, rank=5)
    # flowTorch integral contributions (Kou et al., 2017) are available and positive
    assert result.integral_contribution is not None
    assert np.all(result.integral_contribution > 0.0)
    # amplitude sorting: descending |b|
    result.sort(by="amplitude")
    assert np.all(np.diff(np.abs(result.amplitudes)) <= 1e-12)
    # integral sorting: descending flowTorch integral contribution
    result.sort(by="integral")
    assert np.all(np.diff(result.integral_contribution) <= 1e-12)
    # frequency sorting: ascending |f|, the constant mode comes first
    result.sort(by="frequency")
    assert np.all(np.diff(np.abs(result.frequencies)) >= -1e-12)
    assert np.allclose(np.abs(result.frequencies[0]), 0.0, atol=1e-8)
    # sorting must not change the reconstruction (sum over all modes)
    rebuilt = synthesize(result.modes, result.amplitudes, result.eigvals, dt=0.01,
                         n_time=signal.n_time)
    assert np.allclose(rebuilt, signal.samples, rtol=1e-6, atol=1e-8)


def test_invalid_input() -> None:
    """
    Check that invalid fit inputs are rejected.

    :return: None
    :rtype: None
    """
    samples = np.zeros((4, 8))
    with pytest.raises(AssertionError):
        fit_dmd(samples, dt=-1.0)
    with pytest.raises(AssertionError):
        fit_dmd(np.zeros(8), dt=0.01)


if __name__ == "__main__":
    pass
