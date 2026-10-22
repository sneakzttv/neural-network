# NumPy Neural Lab

A standalone, offline portfolio demo of a neural network **implemented from
scratch with NumPy**. Train a real 2 → hidden → 1 multilayer perceptron on
generated XOR or circular binary data, inspect measured loss/accuracy curves,
and step through a numeric forward/backward pass in your browser.

## Install and run

Requires Python **3.10+** and NumPy. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:8000**. Press **Train network** to generate all results.
No browser assets, data, or API calls are downloaded. Initial dependency
installation needs internet or a locally available NumPy wheel; subsequent
runs are offline. Stop with Ctrl+C. If the port is busy:

```powershell
.\.venv\Scripts\python.exe app.py --port 8001
```

The server binds only to `127.0.0.1`; there is intentionally no public-host
option. Open the printed URL, not the HTML file directly.

## Interactive tutorial

Seven chapters have **Next**, **Back**, and direct chapter buttons:

1. **Architecture:** tensor shapes, parameter count, nonlinear separation.
2. **Activation:** tanh/ReLU, actual activations, derivative and saturation.
3. **Forward pass:** live weighted sum, hidden features, output sigmoid.
4. **Loss:** target, probability, stable binary cross-entropy.
5. **Backprop:** chain rule with an actual first-weight analytical gradient.
6. **Updates:** SGD and bias-corrected Adam; a clearly labeled hypothetical
   single-example SGD step (not presented as an actual Adam update).
7. **Metrics:** measured train/validation loss and accuracy, overfitting caveats.

The example uses the first training sample **after training**. It is not a
replay of training and does not mutate the final model. Actual updates average
mini-batch gradients. Results arrive when fitting finishes, not as fabricated
live streaming. Both plots have an exact numeric table alternative.

### Suggested experiments

- Start with the defaults: XOR, tanh, Adam, 12 hidden units, 500 epochs,
  learning rate 0.03, seed 7.
- Switch to SGD; try rate 0.3 and 1000 epochs. It generally needs different
  hyperparameters from Adam. Compare the curves rather than expecting parity.
- Switch to the circle dataset, or reduce capacity to two hidden units.
- Keep all settings fixed to reproduce a run; vary just the seed to explore
  sensitivity. Poor results are valid experiment outcomes, not concealed errors.

## What is actually implemented

- Float64 NumPy matrix operations, Xavier-style tanh / He-style ReLU
  initialization, sigmoid binary output.
- Stable logit-space mean BCE (`logaddexp`), manually derived gradients for
  every weight and bias, without autograd.
- Shuffled mini-batch SGD and Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸), including
  Adam bias correction and per-parameter first/second moments.
- Local deterministic uniform samples on [-1, 1]². XOR label is 1 for opposite
  signs; circle label is 1 when x₁²+x₂² > 0.5.
- Deterministic shuffled 75% training / 25% validation split, no validation
  gradients. Initialization and batch order use explicit local RNGs.
- Initial epoch-zero metrics, final metrics, and at most about 151 intervals
  of history; actual training still runs every requested epoch.
- Canvas loss/accuracy curves and 55×55 probability grid with class-colored
  sample points; validation points have white rings.
- Bounded, strictly validated JSON API, visible error messages, request size
  limit, single-training gate, cross-origin/Host protection, no arbitrary file
  serving. This is a local educational server, not hardened production hosting.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Coverage includes central finite-difference checks of **every parameter** for
tanh and ReLU, mean-gradient scaling, numerical stability, exact SGD steps,
two-step Adam state/bias correction, deterministic repeatability, invalid
inputs, and actual loopback HTTP requests/assets/errors. Learning assertions:
default XOR held-out accuracy ≥90%, circle with Adam ≥88%, XOR with tuned SGD
≥88%. These are test expectations, not invented measured benchmark results.
CI runs the suite on Python 3.10, 3.12, and 3.13.

For an additional syntax check:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py network.py tests
```

Browser smoke check: train defaults, confirm metrics and curves appear, navigate
all seven chapters, switch datasets/optimizers, and check a narrow viewport.
No automated browser dependency is required by this project.

## Layout / API

| File | Responsibility |
| --- | --- |
| `network.py` | NumPy model, gradients, optimizers, data, training results |
| `app.py` | Standard-library HTTP server and validated transport |
| `static/` | Offline accessible HTML, CSS, JavaScript tutorial and plots |
| `tests/` | Math, deterministic learning, validation, and HTTP tests |

`GET /api/health` reports readiness. `POST /api/train` accepts JSON settings;
`{}` uses defaults. Settings: `dataset` (`xor`/`circle`), `activation`
(`tanh`/`relu`), `optimizer` (`sgd`/`adam`), `samples` (80–800),
`hidden` (2–32), `epochs` (1–1200), `batch_size` (8–128),
`learning_rate` (>0–1), and `seed` (0–4294967295). Integer controls reject
fractional values and booleans; unknown keys are errors. API errors are JSON
with an `error` field and an appropriate 4xx/5xx status.

## Honest portfolio scope and limitations

This demonstrates the resume theme **“Neural Network From Scratch”**, not an
unverified claim about a previously completed project or production system.
Python and NumPy implement the mathematics; the browser replaces
Matplotlib for plots. **Pandas, Matplotlib, and Jupyter are not used or
required**, and no notebook is claimed. NumPy is the sole runtime dependency.

One hidden layer, two features, binary classification, synthetic noiseless
data, CPU-only synchronous training. No GPU, framework, pretrained model,
cross-validation, independent final test set, persistence, export, or real-world
benchmark. Repeated tuning on validation can overfit that split. Floating-point
results may vary slightly across NumPy/BLAS/platform versions; deterministic
repeatability refers to the same runtime. No model quality is guaranteed for
every allowed hyperparameter choice. The local HTTP server is not suitable
for public deployment or adversarial multi-user traffic.

Source comments explain the backend, frontend, and tests. MIT licensed.
