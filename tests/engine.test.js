"use strict";
const {test} = require("node:test");
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const vm = require("node:vm");
const {MLP, Optimizer, Random, parseConfig, makeData, sigmoid, train} = require("../static/engine.js");
const tensor = (...values) => new Float64Array(values);
const close = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const x = tensor(-0.8, 0.3, 0.4, -0.7, 0.6, 0.9), y = tensor(1, 1, 0);

for (const activation of ["tanh", "relu"]) {
  test(`${activation}: every analytical gradient matches central finite differences`, () => {
    const model = new MLP(4, activation, 7);
    // Keep ReLU away from its nondifferentiable kink for both perturbations.
    if (activation === "relu") assert.ok(model.forward(x).z1.every((z) => Math.abs(z) > 1e-4));
    const {gradients} = model.lossAndGradients(x, y), epsilon = 1e-6;
    for (const [key, parameter] of Object.entries(model.params)) {
      for (let i = 0; i < parameter.length; i++) {
        const original = parameter[i];
        parameter[i] = original + epsilon;
        const plus = model.lossAndGradients(x, y).loss;
        parameter[i] = original - epsilon;
        const minus = model.lossAndGradients(x, y).loss;
        parameter[i] = original;
        close(gradients[key][i], (plus - minus) / (2 * epsilon), 2e-7);
      }
    }
  });
}

test("mean gradients and loss are invariant to duplicating a batch", () => {
  const model = new MLP(), a = model.lossAndGradients(x, y);
  const b = model.lossAndGradients(Float64Array.from([...x, ...x]), Float64Array.from([...y, ...y]));
  close(a.loss, b.loss);
  for (const key of Object.keys(a.gradients)) a.gradients[key].forEach((g, i) => close(g, b.gradients[key][i]));
});

test("sigmoid and logit BCE remain finite at extreme logits", () => {
  assert.equal(sigmoid(1000), 1); assert.equal(sigmoid(-1000), 0);
  const model = new MLP();
  for (const bias of [-1000, 1000]) {
    model.params.b2[0] = bias;
    const result = model.lossAndGradients(x, y);
    assert.ok(Number.isFinite(result.loss));
    assert.ok(Object.values(result.gradients).every((g) => g.every(Number.isFinite)));
  }
});

test("SGD subtracts the exact learning-rate-scaled gradient", () => {
  const p = {w: tensor(1, -2)}, optimizer = new Optimizer("sgd", 0.1);
  optimizer.step(p, {w: tensor(0.5, -0.25)});
  close(p.w[0], 0.95); close(p.w[1], -1.975); assert.equal(optimizer.t, 1);
});

test("Adam maintains per-tensor moments and two-step bias correction", () => {
  const p = {w: tensor(1, -2), b: tensor(0)}, optimizer = new Optimizer("adam", 0.03);
  const expected = {w: [1, -2], b: [0]}, m = {w: [0, 0], b: [0]}, v = {w: [0, 0], b: [0]};
  for (const [index, g] of [{w: tensor(0.2, -0.5), b: tensor(0.1)},
    {w: tensor(-0.4, 0.1), b: tensor(-0.3)}].entries()) {
    optimizer.step(p, g);
    for (const key of Object.keys(g)) {
      g[key].forEach((value, i) => {
        m[key][i] = 0.9 * m[key][i] + 0.1 * value;
        v[key][i] = 0.999 * v[key][i] + 0.001 * value ** 2;
        expected[key][i] -= 0.03 * (m[key][i] / (1 - 0.9 ** (index + 1))) /
          (Math.sqrt(v[key][i] / (1 - 0.999 ** (index + 1))) + 1e-8);
        close(p[key][i], expected[key][i], 1e-12);
        close(optimizer.m[key][i], m[key][i], 1e-12);
        close(optimizer.v[key][i], v[key][i], 1e-12);
      });
    }
  }
  assert.equal(optimizer.t, 2);
});

test("invalid optimizer gradients fail atomically", () => {
  const optimizer = new Optimizer("adam", 0.03), p = {a: tensor(1), b: tensor(2)};
  for (const g of [{a: tensor(0)}, {a: tensor(1), b: tensor(NaN)}, {a: tensor(1), b: tensor(1, 2)}]) {
    assert.throws(() => optimizer.step(p, g), /Gradients/);
    assert.deepEqual(p, {a: tensor(1), b: tensor(2)});
    assert.equal(optimizer.t, 0); assert.deepEqual(optimizer.m, {});
  }
});

test("settings reject invalid enums, numbers, booleans, objects and unknown keys", () => {
  for (const input of [null, [], 1, {dataset: "other"}, {activation: "sigmoid"}, {optimizer: "other"},
    {epochs: false}, {epochs: 1.5}, {epochs: 1201}, {samples: 79}, {hidden: 33},
    {batch_size: 0}, {seed: -1}, {seed: 4294967296}, {learning_rate: NaN},
    {learning_rate: Infinity}, {learning_rate: "0.1"}, {learning_rate: true},
    {learning_rate: 0}, {learning_rate: 1.1}, {unknown: 1}, JSON.parse('{"__proto__": {}}')]) {
    assert.throws(() => parseConfig(input));
  }
  assert.equal(parseConfig({seed: 4294967295}).seed, 4294967295);
  assert.throws(() => new MLP(1)); assert.throws(() => new MLP(3, "other"));
  assert.throws(() => new Optimizer("other", 0.1));
  assert.throws(() => new Optimizer("adam", true));
});

test("numeric operations reject malformed/nonfinite features and labels", () => {
  const model = new MLP();
  for (const bad of [[], tensor(), tensor(1), tensor(NaN, 0), tensor(Infinity, 0)]) {
    assert.throws(() => model.forward(bad), /X must/);
  }
  for (const bad of [[], tensor(), tensor(0), tensor(0, 1, 2)]) {
    assert.throws(() => model.lossAndGradients(x, bad), /y must/);
  }
});

test("seed deterministically fixes normals, weights, data, split and full training", () => {
  assert.deepEqual(new MLP().params, new MLP().params);
  assert.notDeepEqual(new MLP().params, new MLP(12, "tanh", 8).params);
  const a = new Random(7), b = new Random(7);
  assert.deepEqual(Array.from({length: 20}, () => a.normal()), Array.from({length: 20}, () => b.normal()));
  assert.deepEqual(train({epochs: 4}), train({epochs: 4}));
  assert.deepEqual(train({epochs: 1, seed: 4294967295}), train({epochs: 1, seed: 4294967295}));
});

for (const dataset of ["xor", "circle"]) {
  test(`${dataset}: exact label rule and disjoint 75/25 split`, () => {
    const data = makeData({dataset}), seen = new Set();
    assert.equal(data.training.y.length, 300); assert.equal(data.validation.y.length, 100);
    for (const group of Object.values(data)) group.y.forEach((label, i) => {
      const a = group.x[2 * i], b = group.x[2 * i + 1];
      assert.ok(a >= -1 && a <= 1 && b >= -1 && b <= 1);
      assert.equal(label, Number(dataset === "xor" ? a * b < 0 : a * a + b * b > 0.5));
      const key = `${a},${b}`; assert.ok(!seen.has(key)); seen.add(key);
    });
  });
}

test("default XOR validation accuracy >= 0.9 with measured history and tutorial", (t) => {
  const progress = [], result = train({}, (p) => progress.push(p)), final = result.history.at(-1);
  t.diagnostic(`Default JS XOR validation accuracy: ${final.validation.accuracy}; BCE: ${final.validation.loss}`);
  assert.ok(final.validation.accuracy >= 0.9);
  assert.equal(result.updates, 500 * Math.ceil(300 / 32));
  assert.equal(result.history[0].epoch, 0); assert.equal(final.epoch, 500);
  assert.deepEqual(progress.map((p) => p.epoch), result.history.map((p) => p.epoch));
  assert.equal(result.grid.probabilities.length, 55 * 55);
  assert.ok(result.grid.probabilities.every((p) => p >= 0 && p <= 1));
  assert.equal(result.points.length, 400);
  const e = result.example, h = result.config.hidden;
  close(e.x[0] * e.weights.W1[0][0] + e.x[1] * e.weights.W1[1][0] + e.weights.b1[0][0], e.z1[0]);
  close(e.x[0] * e.dz2 * e.w2_first * e.activation_derivative, e.gradient_w1_first);
  close(e.weight_before - result.config.learning_rate * e.gradient_w1_first, e.hypothetical_sgd_after);
  assert.equal(e.a1.length, h);
  const restored = new MLP(h);
  for (const key of Object.keys(restored.params)) restored.params[key].set(e.weights[key].flat());
  close(restored.forward(tensor(-1, -1)).probabilities[0], result.grid.probabilities[0]);
  close(restored.forward(tensor(1, 1)).probabilities[0], result.grid.probabilities.at(-1));
});

test("worker loads the real engine and sends progress, result and validation errors", () => {
  const messages = [], context = vm.createContext({});
  context.self = {postMessage: (message) => messages.push(message)};
  context.importScripts = (path) => vm.runInContext(readFileSync(join(__dirname, "../static", path), "utf8"), context);
  vm.runInContext(readFileSync(join(__dirname, "../static/worker.js"), "utf8"), context);
  context.self.onmessage({data: {epochs: 2}});
  assert.deepEqual(messages.map((m) => m.type), ["progress", "progress", "progress", "result"]);
  assert.equal(messages.at(-1).result.history.at(-1).epoch, 2);
  context.self.onmessage({data: {epochs: false}});
  assert.equal(messages.at(-1).type, "error");
  assert.match(messages.at(-1).error, /epochs must/);
});
