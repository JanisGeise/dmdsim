# dmdsim — DMD Playground

Interactive playground for building synthetic modal line fields, decomposing them
with Dynamic Mode Decomposition (DMD), and visually exploring all DMD quantities
(eigenvalues, mode shapes, amplitudes, time dynamics, reconstructions).

All DMD computations are handled by
[flowTorch](https://github.com/FlowModelingControl/flowtorch), installed editable
from the local repository `../flowtorch`; the `dmdsim.core.engine` module is the
single wrapper around it. A Bokeh web GUI ("DMD Playground") is planned; currently
the utility is used as a Python library (see *Usage* below).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# pinned dependencies, including the CPU build of torch
.venv/bin/pip install -r requirements.txt
# flowTorch from the local repository, without its full (heavy) requirements
.venv/bin/pip install -e ../flowtorch --no-deps
# the dmdsim package itself
.venv/bin/pip install -e .
```

## Tests

```bash
.venv/bin/pytest
# with coverage
.venv/bin/pytest --cov=dmdsim
```

## Run the GUI

```bash
# starts a local Bokeh server and opens the app in the browser
.venv/bin/dmdsim-app
# equivalent manual start
.venv/bin/bokeh serve src/dmdsim/app
```

The left panel configures the synthetic signal (up to three modes with wavenumber κ,
frequency f, amplitude A, and growth rate σ; grid; offset; optional seeded noise) and
the DMD rank. With *auto-rerun* active, every change refits and updates all views:
eigenvalue plane, DMD spectrum, mode shape viewer (Re/Im/|φ|/arg φ), and the time view
(space-time heatmap with time cursor plus field curves and play animation). Inline
hints under the plots explain each quantity. Deselect modes in the checkbox row to
study their individual contribution to the reconstruction.

The `dmdsim-app` server stops itself automatically about two seconds after the last
browser tab or window is closed. The manual `bokeh serve` variant keeps running until
you stop it with `Ctrl+C`.

## Usage

The core is GUI-independent and can be used from scripts and notebooks.

**Generate a synthetic modal line field** — a superposition of traveling waves
`A·exp(σ·t)·cos(κ·s − 2πf·t + φ)` plus an optional constant offset and noise:

```python
from dmdsim.signals.generator import Mode, ModalSignal

signal = ModalSignal(
    [Mode(wavenumber=2.0, frequency=5.0, amplitude=1.5, growth_rate=-0.2, phase=0.3),
     Mode(wavenumber=3.0, frequency=12.0, amplitude=0.8, growth_rate=0.1, phase=-1.0),
     Mode(wavenumber=0.0, frequency=0.0, amplitude=2.0)],
    dt=0.01, n_space=64, n_time=300, dx=0.1)
samples = signal.samples  # real-valued field, shape (n_space, n_time)
```

**Fit the DMD** (delegates to flowTorch, all quantities returned as numpy arrays):

```python
from dmdsim.core.engine import fit_dmd

result = fit_dmd(samples, dt=0.01, rank=5)
result.eigvals        # discrete eigenvalues λ_i (complex)
result.frequencies    # f_i = Im(log λ_i) / (2π dt)
result.growth_rates   # σ_i = Re(log λ_i) / dt
result.amplitudes     # spectral coefficients b_i (complex)
result.modes          # mode shapes φ_i(s), shape (n_space, n_modes)
result.pairs          # index tuples of complex conjugate pairs
```

**Playground math** — the field is rebuilt as `x(s,t) = Re{Σ φᵢ·bᵢ·λᵢᵗ}`, so
modifying eigenvalues or amplitudes directly changes the reconstruction. The
eigenvalue radius encodes the growth rate, its angle the frequency. Conjugate
partners are mirrored so the field stays real-valued:

```python
import numpy as np
from dmdsim.core.synth import mirror_conjugates, synthesize

pairs = result.pairs  # cache the pair structure before editing!
i, j = pairs[0]
result.eigvals[i] = 0.98 * np.exp(2j * np.pi * 8.0 * 0.01)  # f: -> 8 Hz, decaying
result.eigvals = mirror_conjugates(result.eigvals, pairs, sources={i})
modified = synthesize(result.modes, result.amplitudes, result.eigvals,
                      dt=0.01, n_time=signal.n_time)
```

## Repository

The project is versioned with git; the repository is local only and is never
pushed to any remote.

## Internal documentation

`documentation/` documents the internals for developers and is updated with every
implementation step: `architecture.md` (layers, data flow, design decisions),
`modules.md` (module-by-module reference), and `steps.md` (chronological log).

## Roadmap

See `PLAN.md` for the full roadmap and status. Currently implemented: core
library (M1). Next: Bokeh GUI v1 (M2), interactive eigenvalue dragging (M3),
2D OpenFOAM support via flowTorch (M5, deferred).
