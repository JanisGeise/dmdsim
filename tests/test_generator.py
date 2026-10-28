"""
    Implements tests for the synthetic modal signal generator.
"""
import pytest
import numpy as np

from dmdsim.signals.generator import Mode, ModalSignal, generate_modes


def test_samples_shape() -> None:
    """
    Check the shapes of the generated field and its grids.

    :return: None
    :rtype: None
    """
    modes = [Mode(wavenumber=2.0, frequency=5.0), Mode(wavenumber=4.0, frequency=11.0)]
    signal = ModalSignal(modes, dt=0.01, n_space=32, n_time=64, dx=0.1)
    assert signal.samples.shape == (32, 64)
    assert signal.space.shape == (32,)
    assert signal.time.shape == (64,)
    assert np.allclose(signal.space[-1], 3.1)
    assert np.allclose(signal.time[-1], 0.63)


def test_offset_only_field() -> None:
    """
    Check that an empty mode list yields the constant offset field.

    :return: None
    :rtype: None
    """
    signal = ModalSignal([], dt=0.1, n_space=8, n_time=16, offset=2.5)
    assert np.allclose(signal.samples, 2.5)


def test_modal_field_values() -> None:
    """
    Check field values of a single mode against the analytic formula.

    :return: None
    :rtype: None
    """
    mode = Mode(wavenumber=2.0, frequency=5.0, amplitude=1.5, growth_rate=-0.2, phase=0.3)
    signal = ModalSignal([mode], dt=0.01, n_space=32, n_time=64, dx=0.1)
    # at s = 0 and t = 0 the field value is the amplitude times the phase cosine
    assert np.allclose(signal.samples[0, 0], 1.5 * np.cos(0.3))
    # along space at t = 0 the field is a phase-shifted cosine
    expected_row = 1.5 * np.cos(2.0 * signal.space + 0.3)
    assert np.allclose(signal.samples[:, 0], expected_row)


def test_noise_determinism() -> None:
    """
    Check that seeded noise is reproducible and different seeds differ.

    :return: None
    :rtype: None
    """
    modes = [Mode(wavenumber=2.0, frequency=5.0)]
    first = ModalSignal(modes, dt=0.01, n_space=16, n_time=32, noise_std=0.1, seed=42)
    second = ModalSignal(modes, dt=0.01, n_space=16, n_time=32, noise_std=0.1, seed=42)
    third = ModalSignal(modes, dt=0.01, n_space=16, n_time=32, noise_std=0.1, seed=43)
    assert np.array_equal(first.samples, second.samples)
    assert not np.array_equal(first.samples, third.samples)


def test_generate_modes() -> None:
    """
    Check the automatic modal composition helper.

    :return: None
    :rtype: None
    """
    first = generate_modes(3, 1.0, 10.0, seed=42)
    second = generate_modes(3, 1.0, 10.0, seed=42)
    third = generate_modes(3, 1.0, 10.0, seed=43)
    # identical seeds reproduce the composition, different seeds do not
    assert [(m.frequency, m.amplitude) for m in first] == [(m.frequency, m.amplitude) for m in second]
    assert first[0].amplitude != third[0].amplitude
    # frequencies are evenly spaced within the bounds
    frequencies = [m.frequency for m in first]
    assert np.allclose(frequencies, np.linspace(1.0, 10.0, 3))
    # wavenumbers follow the dispersion relation with the shared phase speed
    assert np.allclose([m.wavenumber for m in first],
                       2.0 * np.pi * np.array(frequencies) / 10.0)
    # single mode sits at the center of the bounds
    single = generate_modes(1, 2.0, 6.0, seed=1)
    assert np.isclose(single[0].frequency, 4.0)


def test_invalid_parameters() -> None:
    """
    Check that invalid signal and mode parameters are rejected.

    :return: None
    :rtype: None
    """
    with pytest.raises(AssertionError):
        ModalSignal([], dt=-1.0, n_space=8, n_time=16)
    with pytest.raises(AssertionError):
        ModalSignal([], dt=0.1, n_space=1, n_time=16)
    with pytest.raises(AssertionError):
        ModalSignal([], dt=0.1, n_space=8, n_time=16, noise_std=-0.1)
    with pytest.raises(AssertionError):
        Mode(wavenumber=1.0, frequency=1.0, amplitude=-0.5)
    with pytest.raises(AssertionError):
        Mode(wavenumber=float("nan"), frequency=1.0)
    with pytest.raises(AssertionError):
        generate_modes(0, 1.0, 10.0)
    with pytest.raises(AssertionError):
        generate_modes(2, 5.0, 1.0)


if __name__ == "__main__":
    pass
