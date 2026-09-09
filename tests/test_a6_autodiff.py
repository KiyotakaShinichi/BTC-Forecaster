"""Every gradient in the engine, checked against finite differences.

A wrong gradient does not raise. It produces a model that trains, converges,
reports a falling loss curve and learns nothing -- and the only symptom is a
benchmark number that looks like the noise it actually is. Six of the seven deep
architectures in this zoo would sail through that failure.

So every operation the architectures use is checked here against a central
finite-difference approximation, at float64, on random inputs. This file is the
reason the deep results are allowed to be believed at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research.autodiff import (
    Adam,
    Tensor,
    concat,
    numerical_gradient,
    stack,
)

RNG = np.random.default_rng(4242)


def check(build, *shapes, tolerance: float = 1e-6) -> None:
    """Assert the analytic gradient matches central differences for every input.

    ``build`` takes numpy arrays and returns a scalar Tensor, so the same
    callable can be evaluated numerically and differentiated analytically.
    """
    arrays = [RNG.standard_normal(shape) for shape in shapes]

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    loss = build(*tensors)
    loss.backward()
    analytic = [t.grad.copy() for t in tensors]

    for i in range(len(arrays)):
        def scalar(values, index=i):  # noqa: ANN001
            substituted = [
                Tensor(values if j == index else arrays[j].copy()) for j in range(len(arrays))
            ]
            return float(build(*substituted).data)

        numeric = numerical_gradient(scalar, arrays[i].copy())
        assert np.allclose(analytic[i], numeric, atol=tolerance, rtol=1e-4), (
            f"input {i}: max |analytic - numeric| = "
            f"{np.abs(analytic[i] - numeric).max():.3e}"
        )


class TestArithmetic:
    def test_add(self) -> None:
        check(lambda a, b: (a + b).sum(), (3, 4), (3, 4))

    def test_subtract(self) -> None:
        check(lambda a, b: (a - b).sum(), (3, 4), (3, 4))

    def test_multiply(self) -> None:
        check(lambda a, b: (a * b).sum(), (3, 4), (3, 4))

    def test_divide(self) -> None:
        check(lambda a, b: (a / (b * b + 2.0)).sum(), (3, 4), (3, 4))

    def test_power(self) -> None:
        check(lambda a: ((a * a + 1.0) ** 1.5).sum(), (3, 4))

    def test_negation(self) -> None:
        check(lambda a: (-a).sum(), (2, 3))

    def test_scalar_on_the_left(self) -> None:
        check(lambda a: (2.0 - a * 3.0).sum(), (2, 3))


class TestBroadcasting:
    """The single most error-prone part of a hand-written engine.

    Broadcasting is silent going forward and its adjoint is a sum over the
    expanded axes. Get it wrong and every gradient is the right shape and the
    wrong size, which trains to something plausible and useless.
    """

    def test_bias_row_against_a_batch(self) -> None:
        check(lambda x, b: (x + b).sum(), (5, 4), (1, 4))

    def test_column_against_a_matrix(self) -> None:
        check(lambda x, c: (x * c).sum(), (5, 4), (5, 1))

    def test_scalar_tensor_against_a_matrix(self) -> None:
        check(lambda x, s: (x * s).sum(), (5, 4), (1, 1))

    def test_rank_promotion(self) -> None:
        check(lambda x, b: (x + b).sum(), (2, 5, 4), (5, 4))


class TestMatmul:
    def test_two_dimensional(self) -> None:
        check(lambda a, b: (a @ b).sum(), (4, 3), (3, 5))

    def test_batched(self) -> None:
        """Attention needs this: (batch, heads, time, dim) @ transposed."""
        check(lambda a, b: (a @ b).sum(), (2, 3, 4), (2, 4, 5))

    def test_chained_through_a_nonlinearity(self) -> None:
        check(lambda a, b, c: (((a @ b).tanh()) @ c).sum(), (3, 4), (4, 5), (5, 2))


class TestShape:
    def test_reshape(self) -> None:
        check(lambda a: (a.reshape(6, 2) * 2.0).sum(), (3, 4))

    def test_transpose(self) -> None:
        check(lambda a: (a.transpose(1, 0) * 3.0).sum(), (3, 4))

    def test_three_axis_transpose(self) -> None:
        check(lambda a: (a.transpose(0, 2, 1) * 1.5).sum(), (2, 3, 4))

    def test_indexing_a_timestep(self) -> None:
        """How the recurrent cells read one step out of a window."""
        check(lambda a: (a[:, 2, :] ** 2).sum(), (4, 6, 3))

    def test_repeated_indices_accumulate(self) -> None:
        """`np.add.at` rather than `+=`: the same row read twice must receive
        both gradients, not the last one."""
        x = Tensor(RNG.standard_normal((3, 2)), requires_grad=True)
        (x[[0, 0, 1]] ** 2).sum().backward()
        expected = np.zeros((3, 2))
        expected[0] = 2 * x.data[0] * 2
        expected[1] = 2 * x.data[1]
        assert np.allclose(x.grad, expected)


class TestReductions:
    def test_sum_all(self) -> None:
        check(lambda a: (a * a).sum(), (3, 4))

    def test_sum_axis(self) -> None:
        check(lambda a: (a.sum(axis=1) ** 2).sum(), (3, 4))

    def test_sum_axis_keepdims(self) -> None:
        check(lambda a: (a / (a.sum(axis=1, keepdims=True) ** 2 + 3.0)).sum(), (3, 4))

    def test_mean(self) -> None:
        check(lambda a: (a.mean() * 5.0), (3, 4))

    def test_mean_axis(self) -> None:
        check(lambda a: (a.mean(axis=0) ** 2).sum(), (3, 4))

    def test_max_routes_to_the_argmax(self) -> None:
        check(lambda a: a.max(axis=1).sum(), (4, 5))


class TestElementwise:
    def test_exp(self) -> None:
        check(lambda a: (a * 0.5).exp().sum(), (3, 4))

    def test_log(self) -> None:
        check(lambda a: (a * a + 1.0).log().sum(), (3, 4))

    def test_sqrt(self) -> None:
        check(lambda a: (a * a + 1.0).sqrt().sum(), (3, 4))

    def test_tanh(self) -> None:
        check(lambda a: a.tanh().sum(), (3, 4))

    def test_sigmoid(self) -> None:
        check(lambda a: a.sigmoid().sum(), (3, 4))

    def test_sigmoid_does_not_overflow_in_either_tail(self) -> None:
        """A gate saturates the moment a poisoned or extreme input arrives, and
        a RuntimeWarning that fires routinely is one nobody reads when it
        finally means something."""
        import warnings as _warnings

        extreme = Tensor(np.array([[-800.0, -40.0, 0.0, 40.0, 800.0]]))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            value = extreme.sigmoid().data
        assert np.isfinite(value).all()
        assert value[0, 0] == pytest.approx(0.0)
        assert value[0, 2] == pytest.approx(0.5)
        assert value[0, -1] == pytest.approx(1.0)

    def test_relu(self) -> None:
        """Checked away from the kink: the subgradient at exactly zero is a
        convention, and finite differences cannot adjudicate it."""
        x = Tensor(np.array([[-2.0, -0.5, 0.5, 2.0]]), requires_grad=True)
        x.relu().sum().backward()
        assert np.allclose(x.grad, [[0.0, 0.0, 1.0, 1.0]])

    def test_softmax(self) -> None:
        check(lambda a: (a.softmax(axis=-1) * np.arange(4)).sum(), (3, 4))

    def test_softmax_is_stable_at_large_logits(self) -> None:
        """Without the max subtraction these overflow to inf and the row
        becomes NaN -- silently, and only on some batches."""
        big = Tensor(np.array([[300.0, 301.0, 299.0]]))
        probabilities = big.softmax(axis=-1).data
        assert np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0)


class TestJoining:
    def test_concat_splits_the_gradient_back(self) -> None:
        check(lambda a, b: (concat([a, b], axis=1) ** 2).sum(), (3, 2), (3, 4))

    def test_stack_over_time(self) -> None:
        check(lambda a, b, c: (stack([a, b, c], axis=1) ** 2).sum(), (3, 2), (3, 2), (3, 2))


class TestTheTapeItself:
    def test_a_value_used_twice_accumulates_both_paths(self) -> None:
        """d/dx (x*x + x) = 2x + 1. A tape that overwrote instead of accumulating
        would report 2x, and every residual connection in the zoo depends on
        this being right."""
        x = Tensor(np.array([3.0]), requires_grad=True)
        (x * x + x).sum().backward()
        assert x.grad[0] == pytest.approx(7.0)

    def test_deep_chains_do_not_recurse(self) -> None:
        """A 24-step recurrent unrolling is deep enough that naive recursion
        hits Python's stack limit on exactly the architectures this engine is
        for. The traversal is iterative; 500 links proves it."""
        x = Tensor(np.array([0.5]), requires_grad=True)
        value = x
        for _ in range(500):
            value = value * 1.001
        value.sum().backward()
        assert np.isfinite(x.grad).all()
        assert x.grad[0] == pytest.approx(1.001**500, rel=1e-9)

    def test_constants_do_not_require_gradients(self) -> None:
        constant = Tensor(np.ones((2, 2)))
        assert constant.requires_grad is False
        assert constant.grad is None

    def test_requires_grad_propagates_through_operations(self) -> None:
        a = Tensor(np.ones((2, 2)), requires_grad=True)
        b = Tensor(np.ones((2, 2)))
        assert (a + b).requires_grad is True
        assert (b + b).requires_grad is False

    def test_backward_needs_a_scalar(self) -> None:
        with pytest.raises(ValueError, match="scalar"):
            Tensor(np.ones((2, 2)), requires_grad=True).backward()

    def test_zero_grad_clears(self) -> None:
        x = Tensor(np.ones((2, 2)), requires_grad=True)
        (x * 3.0).sum().backward()
        assert x.grad.sum() != 0.0
        x.zero_grad()
        assert x.grad.sum() == 0.0


class TestAdam:
    def test_it_descends_a_quadratic(self) -> None:
        x = Tensor(np.array([5.0, -3.0]), requires_grad=True)
        optimiser = Adam([x], learning_rate=0.1)
        for _ in range(400):
            optimiser.zero_grad()
            (x * x).sum().backward()
            optimiser.step()
        assert np.allclose(x.data, 0.0, atol=1e-3)

    def test_bias_correction_makes_the_first_step_full_sized(self) -> None:
        """Without it the zero-initialised moments make the first steps far too
        small, and with a 60-epoch budget a dozen wasted steps is a fifth of the
        training run."""
        x = Tensor(np.array([1.0]), requires_grad=True)
        optimiser = Adam([x], learning_rate=0.1)
        optimiser.zero_grad()
        (x * 2.0).sum().backward()
        optimiser.step()
        # The corrected first step is almost exactly the learning rate.
        assert abs(1.0 - x.data[0]) == pytest.approx(0.1, rel=1e-3)

    def test_weight_decay_pulls_toward_zero(self) -> None:
        x = Tensor(np.array([1.0]), requires_grad=True)
        optimiser = Adam([x], learning_rate=0.01, weight_decay=0.5)
        for _ in range(200):
            optimiser.zero_grad()
            (x * 0.0).sum().backward()
            optimiser.step()
        assert abs(x.data[0]) < 1.0

    def test_it_ignores_parameters_that_do_not_require_gradients(self) -> None:
        frozen = Tensor(np.array([1.0]))
        optimiser = Adam([frozen], learning_rate=0.1)
        assert optimiser.parameters == []
