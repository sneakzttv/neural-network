"""Mathematical and behavioral checks, using only unittest plus NumPy."""

import unittest

import numpy as np

from network import Config, MLP, Optimizer, make_data, sigmoid, train


class GradientTests(unittest.TestCase):
    def test_every_gradient_matches_central_finite_difference(self):
        # Float64 and moderate values make numerical differentiation trustworthy.
        x = np.array([[0.2, -0.8], [0.7, 0.3], [-0.4, 0.6]])
        y = np.array([[1.0], [0.0], [1.0]])
        for activation in ("tanh", "relu"):
            model = MLP(hidden=3, activation=activation, seed=19)
            # Keep the ReLU example away from its nondifferentiable zero kink.
            _, (z1, _, _) = model.forward(x)
            self.assertGreater(np.min(np.abs(z1)), 1e-4)
            _, analytic = model.loss_and_gradients(x, y)
            for name, parameter in model.params.items():
                numeric = np.zeros_like(parameter)
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    epsilon = 1e-6
                    parameter[index] = original + epsilon
                    high, _ = model.loss_and_gradients(x, y)
                    parameter[index] = original - epsilon
                    low, _ = model.loss_and_gradients(x, y)
                    parameter[index] = original
                    numeric[index] = (high - low) / (2 * epsilon)
                with self.subTest(activation=activation, parameter=name):
                    np.testing.assert_allclose(analytic[name], numeric, rtol=2e-5, atol=2e-7)

    def test_mean_reduction_does_not_scale_with_duplicate_examples(self):
        model = MLP(seed=2)
        x, y = np.array([[0.2, -0.6]]), np.array([[1.0]])
        loss, gradients = model.loss_and_gradients(x, y)
        repeated_loss, repeated = model.loss_and_gradients(np.repeat(x, 4, axis=0), np.repeat(y, 4, axis=0))
        self.assertAlmostEqual(loss, repeated_loss)
        for name in gradients:
            np.testing.assert_allclose(gradients[name], repeated[name])

    def test_extreme_logits_remain_finite(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_allclose(sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1])
            model = MLP(hidden=2)
            model.params["W2"].fill(0)
            model.params["b2"].fill(1000)
            loss, gradients = model.loss_and_gradients(np.array([[0.0, 0.0]]), np.array([[0.0]]))
            self.assertAlmostEqual(loss, 1000)
            self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))


class OptimizerTests(unittest.TestCase):
    def test_sgd_applies_exact_gradient_step(self):
        parameters = {"w": np.array([1.0, -2.0])}
        Optimizer("sgd", 0.1).step(parameters, {"w": np.array([2.0, -3.0])})
        np.testing.assert_allclose(parameters["w"], [0.8, -1.7])

    def test_adam_first_step_bias_correction_and_second_step_state(self):
        parameters = {"w": np.array([1.0, -2.0])}
        optimizer = Optimizer("adam", 0.1)
        gradient = np.array([2.0, -3.0])
        optimizer.step(parameters, {"w": gradient})
        expected = np.array([1.0, -2.0]) - 0.1 * gradient / (np.abs(gradient) + 1e-8)
        np.testing.assert_allclose(parameters["w"], expected)
        # A changed gradient distinguishes true accumulated moments from sign-SGD.
        second = np.array([-1.0, 4.0])
        optimizer.step(parameters, {"w": second})
        m = 0.9 * (0.1 * gradient) + 0.1 * second
        v = 0.999 * (0.001 * gradient**2) + 0.001 * second**2
        expected -= 0.1 * (m / (1 - 0.9**2)) / (np.sqrt(v / (1 - 0.999**2)) + 1e-8)
        np.testing.assert_allclose(parameters["w"], expected)
        self.assertEqual(optimizer.t, 2)

    def test_bad_gradient_does_not_mutate_parameters_or_time(self):
        parameters = {"w": np.array([1.0])}
        optimizer = Optimizer("adam", 0.1)
        for gradient in (np.array([np.nan]), np.array([1.0, 2.0])):
            with self.assertRaises(ValueError):
                optimizer.step(parameters, {"w": gradient})
            np.testing.assert_array_equal(parameters["w"], [1])
            self.assertEqual(optimizer.t, 0)


class LearningTests(unittest.TestCase):
    def test_default_xor_is_repeatable_and_learns(self):
        first, second = train({}), train({})
        # Equality includes curves, grid, example values, and deterministic split.
        self.assertEqual(first, second)
        final = first["history"][-1]
        self.assertGreaterEqual(final["validation"]["accuracy"], 0.90)
        self.assertLess(final["train"]["loss"], first["history"][0]["train"]["loss"] * 0.5)
        self.assertEqual(first["updates"], 500 * 10)
        self.assertEqual(first["history"][0]["epoch"], 0)
        self.assertEqual(first["history"][-1]["epoch"], 500)
        self.assertEqual(len(first["grid"]["probabilities"]), 55 * 55)

    def test_circle_and_sgd_can_learn(self):
        # SGD is deliberately allowed more steps than Adam; no claim that both
        # optimizers learn equally fast with identical hyperparameters is made.
        for dataset, optimizer, rate, epochs in (
            ("circle", "adam", 0.02, 500),
            ("xor", "sgd", 0.3, 1000),
        ):
            with self.subTest(dataset=dataset, optimizer=optimizer):
                result = train({"dataset": dataset, "optimizer": optimizer,
                                "learning_rate": rate, "epochs": epochs, "hidden": 16})
                self.assertGreaterEqual(result["history"][-1]["validation"]["accuracy"], 0.88)

    def test_generated_labels_follow_documented_rules(self):
        for dataset in ("xor", "circle"):
            config = Config.parse({"dataset": dataset})
            x, y, vx, vy = make_data(config)
            all_x, all_y = np.vstack([x, vx]), np.vstack([y, vy])
            expected = (all_x[:, 0] * all_x[:, 1] < 0) if dataset == "xor" else (np.sum(all_x**2, axis=1) > 0.5)
            np.testing.assert_array_equal(all_y.ravel(), expected)
            self.assertEqual((len(x), len(vx)), (300, 100))


class ValidationTests(unittest.TestCase):
    def test_invalid_settings_fail_explicitly(self):
        bad = [[], None, {"extra": 1}, {"epochs": True}, {"samples": 79},
               {"hidden": 33}, {"epochs": 1.5}, {"seed": -1},
               {"batch_size": 0}, {"dataset": "moons"}, {"activation": "sigmoid"},
               {"optimizer": "magic"}, {"learning_rate": float("nan")},
               {"learning_rate": float("inf")}, {"learning_rate": 0},
               {"learning_rate": 10**1000}, {"learning_rate": "fast"}]
        for payload in bad:
            with self.subTest(payload=str(payload)[:100]), self.assertRaises(ValueError):
                Config.parse(payload)

    def test_invalid_numeric_inputs(self):
        model = MLP()
        for x in (np.zeros((0, 2)), np.zeros((3, 3)), np.array([[np.inf, 0]]),
                  np.array([["a", "b"]]), [[0, 1]]):
            with self.assertRaises(ValueError):
                model.forward(x)
        for y in (np.array([1.0]), np.array([[2.0]]), np.array([[np.nan]]),
                  np.array([[1 + 0j]])):
            with self.assertRaises(ValueError):
                model.loss_and_gradients(np.array([[0.0, 0.0]]), y)
        for name, rate in (("unknown", 0.1), ("adam", -1), ("sgd", float("nan"))):
            with self.assertRaises(ValueError):
                Optimizer(name, rate)


if __name__ == "__main__":
    unittest.main()
