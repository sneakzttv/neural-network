"use strict";

// Shared by the classic Web Worker and dependency-free Node tests. Matrices are
// row-major Float64Arrays: W1=(2,h), b1=(1,h), W2=(h,1), b2=(1,1).
(function (root) {
  const defaults = Object.freeze({dataset: "xor", activation: "tanh", optimizer: "adam",
    samples: 400, hidden: 12, epochs: 500, batch_size: 32, learning_rate: 0.03, seed: 7});

  function parseConfig(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error("Settings must be a JSON object.");
    }
    for (const key of Object.keys(payload)) {
      if (!Object.hasOwn(defaults, key)) throw new Error(`Unknown setting: ${key}.`);
    }
    const config = {...defaults, ...payload};
    for (const [key, choices] of [["dataset", ["xor", "circle"]],
      ["activation", ["tanh", "relu"]], ["optimizer", ["sgd", "adam"]]]) {
      if (!choices.includes(config[key])) throw new Error(`${key} must be one of ${choices.join(", ")}.`);
    }
    for (const [key, low, high] of [["samples", 80, 800], ["hidden", 2, 32],
      ["epochs", 1, 1200], ["batch_size", 8, 128], ["seed", 0, 4294967295]]) {
      if (!Number.isInteger(config[key]) || config[key] < low || config[key] > high) {
        throw new Error(`${key} must be an integer from ${low} to ${high}.`);
      }
    }
    validateRate(config.learning_rate);
    return config;
  }

  function validateRate(rate) {
    if (!Number.isFinite(rate) || rate <= 0 || rate > 1) {
      throw new Error("learning_rate must be finite and in (0, 1].");
    }
  }

  // Mulberry32 gives a reproducible uint32 stream, NOT NumPy's PCG64 stream.
  // Box-Muller converts uniform draws to normals for the same init scaling.
  class Random {
    constructor(seed) { this.state = seed >>> 0; }
    uniform() {
      this.state = (this.state + 0x6D2B79F5) >>> 0;
      let t = this.state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    normal() {
      return Math.sqrt(-2 * Math.log(1 - this.uniform())) * Math.cos(2 * Math.PI * this.uniform());
    }
    permutation(n) {
      const order = Array.from({length: n}, (_, i) => i);
      // Fisher-Yates: each remaining position is equally likely.
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.uniform() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      return order;
    }
  }

  function validateXY(x, y) {
    if (!(x instanceof Float64Array) || !x.length || x.length % 2 || !x.every(Number.isFinite)) {
      throw new Error("X must be a nonempty finite Float64Array of shape (n, 2).");
    }
    if (y !== undefined && (!(y instanceof Float64Array) || y.length !== x.length / 2 ||
      !y.every((v) => v === 0 || v === 1))) {
      throw new Error("y must be a Float64Array of shape (n, 1) with binary labels.");
    }
  }

  function sigmoid(z) {
    const small = Math.exp(-Math.abs(z));
    return z >= 0 ? 1 / (1 + small) : small / (1 + small);
  }
  // Softplus avoids exponential overflow, without clipping away gradients.
  const bce = (z, y) => Math.max(0, z) + Math.log1p(Math.exp(-Math.abs(z))) - y * z;
  const zerosLike = (params) => Object.fromEntries(Object.entries(params).map(([k, v]) => [k, new Float64Array(v.length)]));

  class MLP {
    constructor(hidden = 12, activation = "tanh", seed = 7) {
      parseConfig({hidden, activation, seed});
      this.hidden = hidden;
      this.activation = activation;
      const rng = new Random(seed), scale = Math.sqrt((activation === "relu" ? 2 : 1) / 2);
      this.params = {
        W1: Float64Array.from({length: 2 * hidden}, () => rng.normal() * scale),
        b1: new Float64Array(hidden),
        W2: Float64Array.from({length: hidden}, () => rng.normal() / Math.sqrt(hidden)),
        b2: new Float64Array(1),
      };
    }
    forward(x) {
      validateXY(x);
      const h = this.hidden, n = x.length / 2, p = this.params;
      const z1 = new Float64Array(n * h), a1 = new Float64Array(n * h);
      const logits = new Float64Array(n), probabilities = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let z = p.b2[0];
        for (let j = 0; j < h; j++) {
          const k = i * h + j;
          z1[k] = x[2 * i] * p.W1[j] + x[2 * i + 1] * p.W1[h + j] + p.b1[j];
          a1[k] = this.activation === "tanh" ? Math.tanh(z1[k]) : Math.max(0, z1[k]);
          z += a1[k] * p.W2[j];
        }
        logits[i] = z;
        probabilities[i] = sigmoid(z);
      }
      return {probabilities, z1, a1, logits};
    }
    lossAndGradients(x, y) {
      validateXY(x, y);
      const {probabilities, z1, a1, logits} = this.forward(x);
      const gradients = zerosLike(this.params), h = this.hidden, n = y.length;
      let loss = 0;
      for (let i = 0; i < n; i++) {
        loss += bce(logits[i], y[i]) / n;
        // Mean BCE contributes 1/n once, at the output delta.
        const dz2 = (probabilities[i] - y[i]) / n;
        gradients.b2[0] += dz2;
        for (let j = 0; j < h; j++) {
          const k = i * h + j;
          gradients.W2[j] += a1[k] * dz2;
          const derivative = this.activation === "tanh" ? 1 - a1[k] ** 2 : Number(z1[k] > 0);
          const dz1 = dz2 * this.params.W2[j] * derivative;
          gradients.W1[j] += x[2 * i] * dz1;
          gradients.W1[h + j] += x[2 * i + 1] * dz1;
          gradients.b1[j] += dz1;
        }
      }
      return {loss, gradients};
    }
    metrics(x, y) {
      validateXY(x, y);
      const {probabilities, logits} = this.forward(x);
      let loss = 0, correct = 0;
      for (let i = 0; i < y.length; i++) {
        loss += bce(logits[i], y[i]) / y.length;
        correct += Number(Number(probabilities[i] >= 0.5) === y[i]);
      }
      return {loss, accuracy: correct / y.length};
    }
  }

  class Optimizer {
    constructor(name, learningRate) {
      if (!["sgd", "adam"].includes(name)) throw new Error("optimizer must be sgd or adam.");
      validateRate(learningRate);
      this.name = name; this.learningRate = learningRate; this.t = 0;
      this.m = {}; this.v = {};
    }
    step(params, gradients) {
      // Validate every tensor before mutating any parameter or optimizer state.
      const keys = Object.keys(params);
      if (keys.length !== Object.keys(gradients).length || keys.some((k) =>
        !(params[k] instanceof Float64Array) || !(gradients[k] instanceof Float64Array) ||
        params[k].length !== gradients[k].length || !gradients[k].every(Number.isFinite))) {
        throw new Error("Gradients must be finite and match parameter shapes and keys.");
      }
      this.t++;
      for (const key of keys) {
        const p = params[key], g = gradients[key];
        if (!this.m[key]) { this.m[key] = new Float64Array(p.length); this.v[key] = new Float64Array(p.length); }
        for (let i = 0; i < p.length; i++) {
          if (this.name === "sgd") p[i] -= this.learningRate * g[i];
          else {
            this.m[key][i] = 0.9 * this.m[key][i] + 0.1 * g[i];
            this.v[key][i] = 0.999 * this.v[key][i] + 0.001 * g[i] ** 2;
            const mHat = this.m[key][i] / (1 - 0.9 ** this.t);
            const vHat = this.v[key][i] / (1 - 0.999 ** this.t);
            p[i] -= this.learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
          }
        }
      }
    }
  }

  function subset(x, y, indices) {
    return {x: Float64Array.from(indices.flatMap((i) => [x[2 * i], x[2 * i + 1]])),
      y: Float64Array.from(indices.map((i) => y[i]))};
  }
  function makeData(config) {
    const c = parseConfig(config), rng = new Random(c.seed);
    const x = Float64Array.from({length: c.samples * 2}, () => 2 * rng.uniform() - 1);
    const y = Float64Array.from({length: c.samples}, (_, i) => Number(c.dataset === "xor"
      ? x[2 * i] * x[2 * i + 1] < 0 : x[2 * i] ** 2 + x[2 * i + 1] ** 2 > 0.5));
    const order = rng.permutation(c.samples), split = Math.floor(c.samples * 0.75);
    return {training: subset(x, y, order.slice(0, split)), validation: subset(x, y, order.slice(split))};
  }

  function train(payload = {}, onProgress = () => {}) {
    const config = parseConfig(payload), {training, validation} = makeData(config);
    const {x, y} = training, model = new MLP(config.hidden, config.activation, config.seed);
    const optimizer = new Optimizer(config.optimizer, config.learning_rate);
    // Independent stream: batch order never depends on hidden-layer width.
    const rng = new Random(config.seed + 1), history = [], interval = Math.max(1, Math.ceil(config.epochs / 150));
    function record(epoch) {
      const point = {epoch, train: model.metrics(x, y), validation: model.metrics(validation.x, validation.y)};
      if (![point.train.loss, point.validation.loss].every(Number.isFinite)) {
        throw new Error("Training diverged. Reduce the learning rate.");
      }
      history.push(point);
      onProgress({epoch, epochs: config.epochs});
    }
    record(0);
    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      const order = rng.permutation(y.length);
      for (let start = 0; start < y.length; start += config.batch_size) {
        const batch = subset(x, y, order.slice(start, start + config.batch_size));
        optimizer.step(model.params, model.lossAndGradients(batch.x, batch.y).gradients);
      }
      if (epoch % interval === 0 || epoch === config.epochs) record(epoch);
    }
    if (!Object.values(model.params).every((p) => p.every(Number.isFinite))) {
      throw new Error("Training diverged. Reduce the learning rate.");
    }
    // Worked example uses the FINAL model; the hypothetical update is not applied.
    const oneX = x.slice(0, 2), oneY = y.slice(0, 1), p = model.params;
    const out = model.forward(oneX), sample = model.lossAndGradients(oneX, oneY);
    const rows = (array, width) => Array.from({length: array.length / width}, (_, i) => Array.from(array.slice(i * width, (i + 1) * width)));
    const example = {
      x: Array.from(oneX), y: oneY[0], z1: Array.from(out.z1), a1: Array.from(out.a1),
      logit: out.logits[0], probability: out.probabilities[0], loss: sample.loss,
      dz2: out.probabilities[0] - oneY[0], activation_derivative: config.activation === "tanh"
        ? 1 - out.a1[0] ** 2 : Number(out.z1[0] > 0),
      w2_first: p.W2[0], gradient_w1_first: sample.gradients.W1[0], weight_before: p.W1[0],
      hypothetical_sgd_after: p.W1[0] - config.learning_rate * sample.gradients.W1[0],
      weights: {W1: rows(p.W1, config.hidden), b1: rows(p.b1, config.hidden), W2: rows(p.W2, 1), b2: rows(p.b2, 1)},
    };
    // Same 55x55 row-major grid as Python, first row y=-1, endpoints included.
    const gridX = Float64Array.from({length: 55 * 55 * 2}, (_, i) =>
      -1 + 2 * (i % 2 === 0 ? Math.floor(i / 2) % 55 : Math.floor(i / 110)) / 54);
    const points = [];
    for (const [split, data] of [["train", training], ["validation", validation]]) {
      data.y.forEach((label, i) => points.push({x: Array.from(data.x.slice(i * 2, i * 2 + 2)), y: label, split}));
    }
    return {config, history, example, grid: {size: 55, probabilities: Array.from(model.forward(gridX).probabilities)},
      points, train_count: y.length, validation_count: validation.y.length, updates: optimizer.t};
  }

  const api = {defaults, parseConfig, Random, sigmoid, MLP, Optimizer, makeData, train};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NeuralEngine = api;
})(globalThis);
