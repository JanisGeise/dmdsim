# dmdsim — DMD Playground

For transparency:

> **Note:** This project is vibe coded — it is developed end-to-end in an
> AI-assisted workflow (human direction and review, AI implementation) using
> [OpenCode](https://opencode.ai) with models provided by
> [ScaDS.AI](https://llm.scads.ai) (GLM-5.3-Flash for coding, alias-vision for
> GUI screenshot review); the ScaDS.AI access is provided via
> [TU Dresden](https://tu-dresden.de), where the project is developed as
> teaching material for the ML-CFD course.

Interactive playground for building synthetic modal line fields, decomposing them
with Dynamic Mode Decomposition (DMD), and visually exploring all DMD quantities
(eigenvalues, mode shapes, amplitudes, time dynamics, reconstructions).

All DMD computations are handled by
[flowTorch](https://github.com/FlowModelingControl/flowtorch), installed editable
from the local repository `../flowtorch`; the `dmdsim.core.engine` module is the
single wrapper around it. The Bokeh web GUI ("DMD Playground") is the main
interface (see *Run the GUI* below); the core library remains usable from
scripts and notebooks (see *Usage* below).

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

The app opens on the **Start** panel: a "DMD in a nutshell" recap of the lecture
concepts (snapshot matrix X → time-shifted X′ → SVD → eigenvalue problem → modes
φ, amplitudes b, eigenvalues λ → reconstruction) and a workflow list with buttons
jumping to the six step panels:

- **1 · Signal** — design the synthetic signal (number of modes Nₘ, frequency
  range fₘᵢₙ–fₘₐₓ, offset, optional seeded noise) and run the DMD fit
  (rank 0 = automatic; auto-rerun refits on every change; "↺ reset parameters"
  restores the defaults). The signal animation plays the created waves on the
  fixed window x ∈ [0, 1].
- **2 · Eigenvalues** — the fitted eigenvalues around the unit circle
  (radius = growth/decay, angle = frequency; shaded half planes mark the
  frequency sign). **Drag them**: the conjugate partner mirrors automatically
  and the reconstruction updates live (|λ| = 1 keeps modes neutral via the
  "lock to unit circle" toggle; the |b| gain spinner adds an extra
  amplification of the selected mode on top of its panel-5 weight). Small
  triangles above each eigenvalue point in the wave's travel direction and
  flip when an eigenvalue crosses the real axis. The λᵗ dynamics plot shows
  the selected mode's spiral animated in time.
- **3 · Spectrum** — each mode's amplitude over frequency (left) and the
  growth-direction plane, growth rate σ over signed phase velocity
  c = −2πf/κ (right): above the zero line the wave travels toward +x, below
  toward −x. Conjugate pairs share their color; dashed lines mark the Nyquist
  limits.
- **4 · Mode shapes** — the quadrature view shows Re(φ) and Im(φ) in one axes
  (with the fitted ghost), so the quarter-wavelength offset between the
  quadratures — their spatial phase relation — is directly visible. The
  "standing ←→ traveling" slider morphs the mode between a pure standing
  pattern and the fitted traveling wave (a standing wave is two equal
  opposite-traveling waves; the readout reports the split). The traveling
  wave is animated next to it with a direction arrow whose length follows
  the net propagation. The phase sub-tab explains the wrapped arg(φ).
- **5 · Reconstruction** — the rebuild vs the original: blend each eigenvalue's
  contribution continuously with a weight slider (0 = silenced, 1 = as
  fitted; conjugate pairs share one slider box and move together) and watch
  the reconstruction and the thin per-mode contribution lines respond.
- **6 · Prediction** — the same model formula advanced beyond the data window:
  the probe-point time series shows the original data ending at the dashed
  boundary while the model curve continues seamlessly from the reconstruction
  into the prediction, compared against the clean continuation of the signal
  (future noise is unpredictable by definition); the reconstruction part is
  shaded and labeled ("reconstruction" / "prediction") so data and forecast
  stay distinguishable, the static frame already peeks a little into the
  prediction, and the time axis starts sliding (peek always ahead of the
  cursor) once the prediction begins. The error curve below (same time axis)
  shows how growing eigenvalues (|λ| > 1) amplify the prediction error while
  decaying ones keep it bounded. The "prediction steps" box sets how many
  time levels the model is advanced (the slider end follows), the probe point
  is adjustable, and the play button sweeps the cursor up to the end time as
  in the other animations.

Every panel carries a **↺ reset button**: panel 1 restores the default signal
parameters (and refits), panels 2–6 restore the fitted DMD values, undoing
all edits made in the playground.

A **light/dark theme** switch sits in the application bar. Inline hints under
the plots explain each quantity; every parameter has a "?" tooltip with meaning
and allowed range. The time sliders and play buttons of panels 1–5 drive the
same shared time cursor; the prediction panel (6) has its own independent
cursor that reaches beyond the data window.

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

Hosted on GitHub: <https://github.com/JanisGeise/dmdsim> (private during
development; intended to be made public later).

## License

`dmdsim` is licensed under the GNU General Public License v3.0 or later
(see `LICENSE`). The DMD computations delegate to
[flowTorch](https://github.com/FlowModelingControl/flowtorch), which is
GPL-3.0 licensed as well.

## Internal documentation

`documentation/` documents the internals for developers and is updated with every
implementation step: `architecture.md` (layers, data flow, design decisions) and
`modules.md` (module-by-module reference). The chronological step log `steps.md`
is kept locally only (untracked).

## Roadmap

Current status: done are the core library (M1), the Bokeh GUI (M2), the
GUI revise phase (M2R), interactive eigenvalue editing (M3),
the UX revision (M4: start panel, light/dark themes, beginner labels), the GUI
polish round (M4b: per-panel resets, growth-direction plane, travel-direction
triangles), the first teaching-concept feature (M4c: the traveling↔standing
morph with the combined quadrature view), and the space–time heatmap (M4d).
Next candidates (teaching concepts):
arg(b) phasor editing, phase-slope (κ) editing, an x–t heatmap, a singular-value
scree plot, guided scenario presets, and a manual signal composition — plus the
deferred backlog (2D OpenFOAM support via flowTorch (M5), dark-mode completion,
DMD variants, session save/load).
