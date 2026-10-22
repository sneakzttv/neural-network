# Neural Lab — NumPy reference and browser port

**[Live demo: sneakzttv.github.io/neural-network](https://sneakzttv.github.io/neural-network/)**

A standalone, offline portfolio demo of a neural network **implemented from
scratch with NumPy**, with a dependency-free JavaScript port for the browser.
Train a real 2 → hidden → 1 multilayer perceptron on
generated XOR or circular binary data, inspect measured loss/accuracy curves,
and step through a numeric forward/backward pass in your browser.

## Browser and local modes

The UI **always trains in JavaScript**, on GitHub Pages and when served locally.
A dedicated Web Worker keeps CPU-heavy fitting off the UI thread and reports
real epoch progress. Canvas plots and all seven tutorial chapters use that
run's actual final model, not precomputed or simulated results. No `/api` calls,
CDNs, external datasets, analytics, secrets, or npm dependencies are needed.

`network.py` remains the **NumPy reference implementation**; `app.py` still
serves its validated HTTP API as well as the same browser UI. To exercise
NumPy rather than JavaScript, call `POST /api/train` or `network.train({})`.
The local UI is not a NumPy frontend and does not silently switch engines.

Both engines implement Float64 2 → hidden → 1 networks, the same initialization
scaling, dataset rules, 75/25 split, mini-batching, optimizer hyperparameters,
metrics, 55×55 grid, and worked-example schema. **Numbers differ between
engines**: JavaScript uses Mulberry32, Box-Muller normals and Fisher-Yates
shuffles (the batch seed is seed + 1 modulo 2³²); NumPy uses `default_rng` (PCG64). A matching numeric seed therefore
reproduces a run within an engine, not identical samples/weights across engines.
Floating-point functions can also differ slightly between runtimes.

To use just the browser mode locally, serve `static` over HTTP (no NumPy needed):

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory static
```

Open http://127.0.0.1:8000. Do not open `index.html` as a `file://` URL:
Web Workers need HTTP(S). Current browsers with Web Worker and typed-array
support are required. A hosted first visit downloads assets; after downloading
them, training makes no network calls. There is no service worker or guaranteed
offline caching. Inputs and training results stay in memory on your device.

## Install and run the local NumPy reference server

Requires Python **3.10+** and NumPy. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:8000**. Press **Train network** to generate all results.
Browser assets are served from localhost; no external data or API calls are
made. Initial dependency installation needs internet or a locally available NumPy wheel; subsequent
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
mini-batch gradients. Actual epoch progress appears during fitting; results
arrive together when it finishes, not as fabricated metric streaming. Both
plots have an exact numeric table alternative.

### Suggested experiments

- Start with the defaults: XOR, tanh, Adam, 12 hidden units, 500 epochs,
  learning rate 0.03, seed 7.
- Switch to SGD; try rate 0.3 and 1000 epochs. It generally needs different
  hyperparameters from Adam. Compare the curves rather than expecting parity.
- Switch to the circle dataset, or reduce capacity to two hidden units.
- Keep all settings fixed to reproduce a run; vary just the seed to explore
  sensitivity. Poor results are valid experiment outcomes, not concealed errors.

## What is actually implemented

- Float64 NumPy matrices / row-major JavaScript `Float64Array` matrices,
  Xavier-style tanh / He-style ReLU initialization, sigmoid binary output.
- Stable logit-space mean BCE (`logaddexp` in NumPy, softplus in JavaScript),
  manually derived gradients for every weight and bias, without autograd.
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

## GitHub Pages deployment

`.github/workflows/pages.yml` runs dependency-free Node tests on pushes to
`main` or manual dispatch, uploads **only `static/`** with
`actions/upload-pages-artifact@v3`, then deploys with `actions/deploy-pages@v4`.
It grants `contents: read`, `pages: write`, and `id-token: write`, uses the
`github-pages` environment, and serializes deployments with concurrency group
`pages`. No Python server runs on Pages. Relative asset and worker paths work
under the repository subpath `/neural-network/` and at localhost `/`.

A repository administrator must select **Settings → Pages → Build and
deployment → Source: GitHub Actions** once, then merge to `main` (or dispatch
the workflow on the intended branch). The intended URL is the live-demo link
above; a commit alone does not publish it. This change does not alter GitHub
settings or push/deploy anything itself.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test
node --check static/engine.js
node --check static/worker.js
node --check static/app.js
git diff --check
```

Python coverage includes central finite-difference checks of **every parameter** for
tanh and ReLU, mean-gradient scaling, numerical stability, exact SGD steps,
two-step Adam state/bias correction, deterministic repeatability, invalid
inputs, and actual loopback HTTP requests/assets/errors. Learning assertions:
default XOR held-out accuracy ≥90%, circle with Adam ≥88%, XOR with tuned SGD
≥88%. These are test expectations, not invented measured benchmark results.
CI runs the Python suite on 3.10, 3.12, and 3.13, plus a Node 22 job.
Node's built-in `node:test` suite checks every tanh/ReLU parameter gradient,
mean scaling, stable BCE, exact SGD and two-step Adam, deterministic seeds,
dataset rules/splits, validation errors, worker messages, and default XOR
held-out accuracy ≥90%. No package installation is needed (Node 22+).
A local Node 24 run measured **100/100 default XOR validation predictions
correct**, BCE **0.009237**; this describes that seeded synthetic holdout only,
not a real-world benchmark or a guarantee for other settings.

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
| `static/engine.js` | Dependency-free Float64 MLP, datasets and training |
| `static/worker.js` | Off-main-thread training and progress/error messages |
| `static/app.js` | Browser controls, seven-step tutorial and Canvas plots |
| `static/index.html`, `static/style.css` | Accessible UI and local assets |
| `tests/` | Python and Node math, learning, validation, worker and HTTP tests |

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
NumPy implements the reference mathematics; a plain JavaScript port trains
the browser UI, and Canvas replaces Matplotlib for plots. **Pandas, Matplotlib, and Jupyter are not used or
required**, and no notebook is claimed. NumPy is the sole Python runtime dependency; browser training has no dependencies.

One hidden layer, two features, binary classification, synthetic noiseless
data, CPU-only training (synchronous NumPy API; JavaScript in a Web Worker). No GPU, framework, pretrained model,
cross-validation, independent final test set, persistence, export, or real-world
benchmark. Repeated tuning on validation can overfit that split. Floating-point
results may vary across NumPy/BLAS/browser/platform versions; deterministic
repeatability refers to the same engine and runtime. Separate RNG algorithms
mean Python and JavaScript results are not numerically interchangeable.
Training speed depends on the device; progress is sampled at recorded epochs,
not every update. There is no cancel/resume or parallel experiment queue. No model quality is guaranteed for
every allowed hyperparameter choice. The local HTTP server is not suitable
for public deployment or adversarial multi-user traffic.

Source comments explain the backend, frontend, and tests. MIT licensed.
