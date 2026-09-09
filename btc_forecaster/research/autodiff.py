"""A small reverse-mode autodiff engine over numpy.

Phase 9 asks for seven compact deep architectures and suggests reusing the
existing torch infrastructure. There is none. A3 is a forward *shadow ledger* --
an append-only hash-chained record of pre-committed forecasts -- and contains no
neural code; `torch` appears nowhere in this repository, in any requirement
file, or in `requirements.lock`.

Adding it was considered and rejected on a specific, checkable ground rather
than a preference. `scripts/lock.sh` compiles the lock with `--all-extras`, so
any extra containing torch enters `requirements.lock`, and the `fresh-clone` CI
job installs from that lock with `--require-hashes` on every push. The Linux
wheel pulls a multi-gigabyte CUDA closure. Phase 34 says not to force optional
frameworks into the default install where they make the repository less
reproducible, and that is exactly what it would do.

So the engine is here, in about three hundred lines, and its correctness is not
asserted -- it is *checked*. Every operation is verified against a central
finite-difference approximation in `tests/test_a6_autodiff.py`. That matters
more than the line count: a wrong gradient does not raise. It produces a model
that trains, converges, reports a loss curve and learns nothing, and the only
symptom is a benchmark result that looks like the noise it actually is.

Deliberately not general. Float64 throughout, because a gradient check at
float32 cannot separate a real error from rounding. No GPU, no fused kernels, no
graph optimisation, and no autograd beyond what the seven architectures need.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum a gradient back down to the shape that produced it.

    The single most error-prone part of a hand-written engine. Broadcasting is
    silent in the forward pass -- a `(batch, 1)` bias added to `(batch, units)`
    just works -- and its adjoint is a sum over the axes that were expanded. Get
    it wrong and gradients are the right shape and the wrong size, which trains
    to something plausible and useless.
    """
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """A value on the tape, its gradient, and how to push that gradient back."""

    __slots__ = ("data", "grad", "_backward", "_parents", "requires_grad", "_label")

    # Annotated at class level rather than inline: with __slots__ a bare
    # annotation creates no class attribute, so the slots still work and the
    # type checker no longer has to infer `requires_grad` from an expression
    # that reads `requires_grad`.
    data: np.ndarray
    #: None for constants -- and constants are most of the tape: every input
    #: batch, attention mask and zero-initialised hidden state. Allocating zeros
    #: for all of them would double the memory of a forward pass to hold values
    #: nothing ever reads.
    grad: np.ndarray | None
    requires_grad: bool
    _parents: tuple[Tensor, ...]
    _backward: Callable[[], None]
    _label: str

    def __init__(
        self,
        data: np.ndarray | float | Sequence,
        *,
        requires_grad: bool = False,
        parents: tuple[Tensor, ...] = (),
        backward: Callable[[], None] | None = None,
        label: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._parents = parents
        self._backward = backward or (lambda: None)
        self._label = label

    # -- plumbing --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _accumulate(self, gradient: np.ndarray) -> None:
        if self.requires_grad and self.grad is not None:
            self.grad += _unbroadcast(gradient, self.data.shape)

    @property
    def _gradient(self) -> np.ndarray:
        """The accumulated gradient, for use inside a backward closure.

        A closure runs only for nodes whose `requires_grad` is True, so the
        gradient is never None there. The assertion states that invariant to the
        type checker in one place instead of at nineteen call sites.
        """
        assert self.grad is not None, "a backward closure ran on a constant node"
        return self.grad

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        """Seed this scalar with 1.0 and walk the tape in reverse.

        Topologically ordered rather than recursive: a 24-step recurrent
        unrolling is deep enough that naive recursion hits Python's stack limit
        on exactly the architectures this engine exists for.
        """
        if self.data.size != 1:
            raise ValueError("backward() starts from a scalar loss")

        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in seen:
                    stack.append((parent, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            # Nodes that do not require a gradient have none to push back, and
            # their closures would dereference a `None`. Constants reach the
            # tape constantly -- every input batch, every attention mask, every
            # zero-initialised hidden state -- so this is the common case, not
            # an edge one.
            if node.requires_grad:
                node._backward()

    # -- arithmetic ------------------------------------------------------

    def _binary(self, other: Tensor | float, forward, backward_self, backward_other) -> Tensor:
        rhs = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(forward(self.data, rhs.data), parents=(self, rhs))

        def _back() -> None:
            g = out._gradient
            self._accumulate(backward_self(g, self.data, rhs.data))
            rhs._accumulate(backward_other(g, self.data, rhs.data))

        out._backward = _back
        return out

    def __add__(self, other: Tensor | float) -> Tensor:
        return self._binary(
            other, lambda a, b: a + b, lambda g, a, b: g, lambda g, a, b: g
        )

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self._binary(
            other, lambda a, b: a - b, lambda g, a, b: g, lambda g, a, b: -g
        )

    def __mul__(self, other: Tensor | float) -> Tensor:
        return self._binary(
            other, lambda a, b: a * b, lambda g, a, b: g * b, lambda g, a, b: g * a
        )

    def __truediv__(self, other: Tensor | float) -> Tensor:
        return self._binary(
            other,
            lambda a, b: a / b,
            lambda g, a, b: g / b,
            lambda g, a, b: -g * a / (b * b),
        )

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: float) -> Tensor:
        return Tensor(other) - self

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __pow__(self, power: float) -> Tensor:
        out = Tensor(self.data**power, parents=(self,))

        def _back() -> None:
            self._accumulate(out._gradient * power * self.data ** (power - 1))

        out._backward = _back
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, parents=(self, other))

        def _back() -> None:
            g = out._gradient
            self._accumulate(g @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ g)

        out._backward = _back
        return out

    # -- shape -----------------------------------------------------------

    def reshape(self, *shape: int) -> Tensor:
        out = Tensor(self.data.reshape(shape), parents=(self,))
        original = self.data.shape

        def _back() -> None:
            self._accumulate(out._gradient.reshape(original))

        out._backward = _back
        return out

    def transpose(self, *axes: int) -> Tensor:
        out = Tensor(np.transpose(self.data, axes), parents=(self,))
        inverse = np.argsort(axes)

        def _back() -> None:
            self._accumulate(np.transpose(out._gradient, inverse))

        out._backward = _back
        return out

    def __getitem__(self, key) -> Tensor:  # type: ignore[no-untyped-def]
        out = Tensor(self.data[key], parents=(self,))

        def _back() -> None:
            gradient = np.zeros_like(self.data)
            np.add.at(gradient, key, out._gradient)
            self._accumulate(gradient)

        out._backward = _back
        return out

    # -- reductions ------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=(self,))

        def _back() -> None:
            gradient = out._gradient
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis)
            self._accumulate(np.broadcast_to(gradient, self.data.shape).copy())

        out._backward = _back
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        count = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in np.atleast_1d(axis)]
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    def max(self, axis: int, keepdims: bool = False) -> Tensor:
        """Only used to stabilise softmax; the gradient goes to the argmax."""
        indices = np.argmax(self.data, axis=axis)
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), parents=(self,))

        def _back() -> None:
            gradient = np.zeros_like(self.data)
            expanded = np.expand_dims(indices, axis)
            g = out._gradient if keepdims else np.expand_dims(out._gradient, axis)
            np.put_along_axis(gradient, expanded, g, axis=axis)
            self._accumulate(gradient)

        out._backward = _back
        return out

    # -- elementwise -----------------------------------------------------

    def _unary(self, value: np.ndarray, derivative: Callable[[], np.ndarray]) -> Tensor:
        out = Tensor(value, parents=(self,))

        def _back() -> None:
            self._accumulate(out._gradient * derivative())

        out._backward = _back
        return out

    def exp(self) -> Tensor:
        value = np.exp(self.data)
        return self._unary(value, lambda: value)

    def log(self) -> Tensor:
        return self._unary(np.log(self.data), lambda: 1.0 / self.data)

    def sqrt(self) -> Tensor:
        value = np.sqrt(self.data)
        return self._unary(value, lambda: 0.5 / value)

    def tanh(self) -> Tensor:
        value = np.tanh(self.data)
        return self._unary(value, lambda: 1.0 - value**2)

    def sigmoid(self) -> Tensor:
        """Branch on the sign so neither tail overflows.

        The naive `1/(1+exp(-x))` overflows for large negative x. It still
        returns the right answer -- inf in the denominator gives 0.0 -- but it
        emits a RuntimeWarning per batch, and a warning that fires routinely is
        a warning nobody reads when it finally means something.
        """
        positive = self.data >= 0
        value = np.empty_like(self.data)
        value[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exponentiated = np.exp(self.data[~positive])
        value[~positive] = exponentiated / (1.0 + exponentiated)
        return self._unary(value, lambda: value * (1.0 - value))

    def relu(self) -> Tensor:
        return self._unary(np.maximum(self.data, 0.0), lambda: (self.data > 0.0).astype(float))

    def softmax(self, axis: int = -1) -> Tensor:
        """Numerically stabilised by subtracting the max before exponentiating.

        Without it, attention logits of order 30 overflow to inf and the whole
        row becomes NaN -- silently, and only on some batches.
        """
        shifted = self - self.max(axis=axis, keepdims=True)
        exponentiated = shifted.exp()
        return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    """Join along an axis, splitting the gradient back the way it came."""
    parts = [t.data for t in tensors]
    out = Tensor(np.concatenate(parts, axis=axis), parents=tuple(tensors))
    sizes = [p.shape[axis] for p in parts]

    def _back() -> None:
        offset = 0
        for tensor, size in zip(tensors, sizes, strict=True):
            slicer: list[slice] = [slice(None)] * out.data.ndim
            slicer[axis] = slice(offset, offset + size)
            tensor._accumulate(out._gradient[tuple(slicer)])
            offset += size

    out._backward = _back
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), parents=tuple(tensors))

    def _back() -> None:
        for i, tensor in enumerate(tensors):
            slicer: list = [slice(None)] * out.data.ndim
            slicer[axis] = i
            tensor._accumulate(out._gradient[tuple(slicer)])

    out._backward = _back
    return out


class Adam:
    """Adam, with the bias correction. Written out so the state is inspectable.

    The bias correction is not optional at this scale: the zero-initialised
    moment estimates make the first dozen steps far too small without it, and
    with a 60-epoch budget a dozen wasted steps is a fifth of the training.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self._t += 1
        for i, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            if gradient is None:  # unreachable: the constructor filters these out
                continue
            if self.weight_decay:
                gradient = gradient + self.weight_decay * parameter.data
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * gradient
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * gradient**2
            m_hat = self._m[i] / (1 - self.beta1**self._t)
            v_hat = self._v[i] / (1 - self.beta2**self._t)
            parameter.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def numerical_gradient(
    function: Callable[[np.ndarray], float], values: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Central finite differences. The reference every operation is checked against.

    Central rather than forward: the error is O(h^2) instead of O(h), which is
    the difference between a check that can distinguish a real bug from rounding
    and one that cannot.
    """
    gradient = np.zeros_like(values)
    iterator = np.nditer(values, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = values[index]
        values[index] = original + epsilon
        high = function(values)
        values[index] = original - epsilon
        low = function(values)
        values[index] = original
        gradient[index] = (high - low) / (2 * epsilon)
        iterator.iternext()
    return gradient


__all__ = ["Adam", "Tensor", "concat", "numerical_gradient", "stack"]
