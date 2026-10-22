"use strict";

// No external chart library: Canvas draws measured values from the browser engine.
const $ = (id) => document.getElementById(id);
const chapters = ["Architecture", "Activation", "Forward pass", "Loss", "Backprop", "Updates", "Metrics"];
let experiment = null;
let chapter = 0;
const f = (value) => Number(value).toFixed(5);
const percent = (value) => `${(100 * value).toFixed(1)}%`;

// Tutorial navigation remains available after each fresh experiment.
chapters.forEach((name, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = `${index + 1}. ${name}`;
  button.addEventListener("click", () => { chapter = index; showLesson(); });
  $("steps").append(button);
});
$("back").addEventListener("click", () => { chapter--; showLesson(); });
$("next").addEventListener("click", () => { chapter++; showLesson(); });

$("settings").addEventListener("submit", async (event) => {
  event.preventDefault();
  // Select fields stay strings; all numeric controls become JSON numbers.
  const settings = Object.fromEntries(new FormData(event.currentTarget));
  for (const key of ["samples", "hidden", "epochs", "batch_size", "learning_rate", "seed"]) {
    settings[key] = Number(settings[key]);
  }
  $("train").disabled = true;
  $("error").hidden = true;
  $("status").textContent = "Training in a browser worker…";
  $("results").setAttribute("aria-busy", "true");
  try {
    const result = await trainInWorker(settings);
    experiment = result;
    chapter = 0;
    render();
    $("status").textContent = `Finished ${result.config.epochs} epochs. The model and tutorial now reflect this run.`;
  } catch (error) {
    // Preserve earlier results but clearly label them as belonging to the prior run.
    $("error").textContent = `Run failed: ${error.message}. Check settings and serve this page over HTTP(S) in a browser with Web Worker support.`;
    $("error").hidden = false;
    $("status").textContent = experiment ? "Previous successful results remain below; this run did not replace them." : "No training result yet.";
  } finally {
    $("train").disabled = false;
    $("results").setAttribute("aria-busy", "false");
  }
});

// Resolve relative to this page: works both at / and /neural-network/.
function trainInWorker(settings) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker.js", document.baseURI));
    const fail = (message) => { worker.terminate(); reject(new Error(message)); };
    worker.onmessage = ({data}) => {
      if (data.type === "progress") {
        $("status").textContent = `Training in your browser: epoch ${data.epoch} / ${data.epochs}.`;
      } else if (data.type === "result") {
        worker.terminate();
        resolve(data.result);
      } else if (data.type === "error") fail(data.error);
    };
    worker.onerror = () => fail("Browser worker failed to load or execute");
    worker.onmessageerror = () => fail("Browser worker returned an unreadable result");
    worker.postMessage(settings);
  });
}

function render() {
  const r = experiment, final = r.history[r.history.length - 1];
  $("empty").hidden = true;
  $("results").hidden = false;
  $("train-accuracy").textContent = percent(final.train.accuracy);
  $("val-accuracy").textContent = percent(final.validation.accuracy);
  $("val-loss").textContent = f(final.validation.loss);
  $("updates").textContent = r.updates;
  $("split-info").textContent = `Seed ${r.config.seed} · ${r.train_count} training / ${r.validation_count} validation samples · ${r.config.optimizer.toUpperCase()} · ${r.config.activation} · 2 → ${r.config.hidden} → 1. History includes epoch zero.`;
  $("data-rule").textContent = r.config.dataset === "xor"
    ? "XOR target: class 1 if x₁ and x₂ have opposite signs. A single straight line cannot separate all four quadrants."
    : "Circle target: class 1 if x₁² + x₂² > 0.5. The ideal boundary is a circle of radius √0.5.";
  drawPlot("loss", "loss");
  drawPlot("accuracy", "accuracy");
  drawBoundary();
  // Build table nodes with textContent rather than injecting untrusted HTML.
  $("metric-rows").replaceChildren();
  r.history.forEach((point) => {
    const row = document.createElement("tr");
    [point.epoch, f(point.train.loss), f(point.validation.loss),
      percent(point.train.accuracy), percent(point.validation.accuracy)].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    $("metric-rows").append(row);
  });
  showLesson();
}

function drawPlot(id, metric) {
  const canvas = $(id), ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, left = 52, right = w - 18, top = 20, bottom = h - 38;
  const history = experiment.history;
  const ymax = metric === "accuracy" ? 1 : Math.max(0.01, ...history.flatMap((p) => [p.train.loss, p.validation.loss])) * 1.08;
  const xmax = experiment.config.epochs;
  ctx.clearRect(0, 0, w, h);
  ctx.font = "12px system-ui";
  // Fixed axis minimum zero avoids visually exaggerating small improvements.
  for (let tick = 0; tick <= 4; tick++) {
    const value = ymax * tick / 4, y = bottom - (bottom - top) * tick / 4;
    ctx.strokeStyle = "#26374d"; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = "#a6b7ce"; ctx.textAlign = "right"; ctx.fillText(value.toFixed(2), left - 8, y + 4);
    const epoch = Math.round(xmax * tick / 4), x = left + (right - left) * tick / 4;
    ctx.textAlign = "center"; ctx.fillText(epoch, x, bottom + 20);
  }
  for (const [split, color] of [["train", "#6cafff"], ["validation", "#ff9e89"]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
    history.forEach((point, index) => {
      const x = left + point.epoch / xmax * (right - left);
      const y = bottom - point[split][metric] / ymax * (bottom - top);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function drawBoundary() {
  const canvas = $("boundary"), ctx = canvas.getContext("2d");
  const margin = 32, width = canvas.width - 2 * margin, height = canvas.height - 2 * margin;
  const {size, probabilities} = experiment.grid;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Invert screen y so mathematical positive y points upward.
  probabilities.forEach((p, i) => {
    const row = Math.floor(i / size), col = i % size;
    const low = [49, 103, 174], high = [198, 99, 83];
    ctx.fillStyle = `rgb(${low.map((value, channel) => Math.round(value * (1 - p) + high[channel] * p)).join(",")})`;
    ctx.fillRect(margin + col * width / size, margin + (size - 1 - row) * height / size, width / size + 1, height / size + 1);
  });
  experiment.points.forEach((point) => {
    const x = margin + (point.x[0] + 1) * width / 2;
    const y = margin + (1 - point.x[1]) * height / 2;
    ctx.beginPath(); ctx.arc(x, y, 3.1, 0, 2 * Math.PI);
    ctx.fillStyle = point.y ? "#ffb39f" : "#8fc6ff"; ctx.fill();
    ctx.strokeStyle = point.split === "validation" ? "#ffffff" : "#172638";
    ctx.lineWidth = 1; ctx.stroke();
  });
  ctx.fillStyle = "#bac9dc"; ctx.font = "12px system-ui"; ctx.textAlign = "center";
  ctx.fillText("−1", margin, canvas.height - 10); ctx.fillText("x₁", canvas.width / 2, canvas.height - 10);
  ctx.fillText("+1", canvas.width - margin, canvas.height - 10);
  ctx.fillText("+1", 15, margin); ctx.fillText("x₂", 15, canvas.height / 2);
  ctx.fillText("−1", 15, canvas.height - margin);
}

function showLesson() {
  if (!experiment) return;
  const {config: c, example: e, history} = experiment;
  const final = history[history.length - 1];
  const first = e.weights.W1[0][0], second = e.weights.W1[1][0], bias = e.weights.b1[0][0];
  const count = 2 * c.hidden + c.hidden + c.hidden + 1;
  const activationText = c.activation === "tanh"
    ? "tanh(z) = (exp(z) − exp(−z)) / (exp(z) + exp(−z)); derivative = 1 − a²."
    : "ReLU(z) = max(0, z); derivative = 1 for z > 0 and 0 otherwise. At exactly zero we choose 0.";
  // Only trusted templates and numeric/validated-enum values enter this markup.
  const lessons = [
    `<h3>Architecture: why a hidden layer?</h3>
     <div class="architecture"><div>2 inputs<br>x₁, x₂</div><div>${c.hidden} hidden units<br>${c.activation}</div><div>1 output<br>sigmoid</div></div>
     <p>A hidden layer bends the decision boundary. Without a nonlinear activation, multiplying two weight matrices is still just one linear map.</p>
     <pre>X: (batch, 2)\nW₁: (2, ${c.hidden}) · b₁: (1, ${c.hidden})\nW₂: (${c.hidden}, 1) · b₂: (1, 1)\nTrainable scalars: 2h + h + h + 1 = ${count}</pre>`,
    `<h3>Activation: turn a weighted sum into a feature</h3>
     <p>${activationText} Every hidden neuron transforms a different weighted sum of the same two inputs.</p>
     <pre>First hidden neuron:\nz₁ = ${f(e.z1[0])}\na₁ = ${c.activation}(z₁) = ${f(e.a1[0])}\nda₁/dz₁ = ${f(e.activation_derivative)}</pre>
     <p>Large tanh inputs saturate and have small gradients. Negative ReLU inputs have zero gradient. Both effects can slow learning.</p>`,
    `<h3>Forward pass: inputs become a probability</h3>
     <p>Matrix multiplication applies every hidden neuron at once. The output sums all hidden features, then sigmoid maps its logit to [0, 1].</p>
     <pre>x = [${e.x.map(f).join(", ")}]\nFirst neuron: z₁ = x₁·W₁[0,0] + x₂·W₁[1,0] + b₁[0]\n= ${f(e.x[0])} × ${f(first)} + ${f(e.x[1])} × ${f(second)} + ${f(bias)}\n= ${f(e.z1[0])}\nAll hidden outputs: [${e.a1.map(f).join(", ")}]\nOutput logit = a₁ @ W₂ + b₂ = ${f(e.logit)}\np = sigmoid(logit) = ${f(e.probability)}</pre>`,
    `<h3>Loss: measure how wrong a probability is</h3>
     <p>Binary cross-entropy penalizes confident wrong predictions strongly. Training minimizes the mean of these losses, not the thresholded accuracy.</p>
     <pre>Target y = ${e.y}\np = ${f(e.probability)}\nBCE = −[y·log(p) + (1−y)·log(1−p)]\nSingle-example BCE = ${f(e.loss)}\nPrediction (p ≥ 0.5): ${e.probability >= 0.5 ? 1 : 0}</pre>
     <p>The browser uses stable softplus(logit) − y·logit (Python: logaddexp(0, logit) − y·logit); this avoids log(0) without clipping gradients.</p>`,
    `<h3>Backprop: reverse the chain rule</h3>
     <p>Sigmoid combined with BCE simplifies the output derivative to p − y. For a batch of n, divide by n before propagating backward.</p>
     <pre>Single example (n = 1):\nδ₂ = p − y = ${f(e.dz2)}\n∂L/∂W₂ = a₁ᵀ · δ₂\nδ₁ = (δ₂ · W₂ᵀ) ⊙ activation′(z₁)\n∂L/∂W₁[0,0] = x₁ × δ₂ × W₂[0,0] × activation′(z₁[0])\n= ${f(e.x[0])} × ${f(e.dz2)} × ${f(e.w2_first)} × ${f(e.activation_derivative)}\n= ${f(e.gradient_w1_first)}</pre>
     <p>⊙ is elementwise multiplication. Bias gradients sum δ across examples. The tests compare every analytical weight and bias gradient with finite differences.</p>`,
    `<h3>Updates: turn a gradient into learning</h3>
     <p>This run used <strong>${c.optimizer.toUpperCase()}</strong>. SGD subtracts ηg. Adam tracks first and second moments of g, with bias correction and ε = 10⁻⁸.</p>
     <pre>SGD: θ ← θ − ηg\nAdam: m ← 0.9m + 0.1g; v ← 0.999v + 0.001g²\nm̂ = m/(1−0.9ᵗ); v̂ = v/(1−0.999ᵗ)\nθ ← θ − ηm̂/(√v̂ + ε)\n\nHypothetical SGD step on this ONE sample (not applied):\n${f(e.weight_before)} − ${c.learning_rate} × ${f(e.gradient_w1_first)}\n= ${f(e.hypothetical_sgd_after)}</pre>
     <p>This illustration is not the final actual update, and it is not an Adam update. Actual training averaged mini-batch gradients and completed ${experiment.updates} updates.</p>`,
    `<h3>Metrics: learning is not just training accuracy</h3>
     <pre>At epoch 0: train loss ${f(history[0].train.loss)}\nAt epoch ${c.epochs}: train loss ${f(final.train.loss)}\nTraining accuracy: ${percent(final.train.accuracy)}\nHeld-out validation accuracy: ${percent(final.validation.accuracy)}\nHeld-out validation BCE: ${f(final.validation.loss)}</pre>
     <p>Accuracy counts correct predictions at threshold 0.5; BCE evaluates confidence. A smaller loss can improve even when accuracy stays unchanged.</p>
     <p>The validation split never contributes gradients. Repeatedly tuning against it can still overfit it: it is not an independent final test set. Small synthetic datasets are educational, not evidence of production performance.</p>
     <p>Next experiment: keep the seed fixed and switch optimizers, then compare curves. Low accuracy is a valid outcome, not a hidden error.</p>`,
  ];
  $("lesson").innerHTML = lessons[chapter];
  $("step-number").textContent = `${chapter + 1} / ${chapters.length}`;
  $("back").disabled = chapter === 0;
  $("next").disabled = chapter === chapters.length - 1;
  [...$("steps").children].forEach((button, index) => {
    if (index === chapter) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
}
