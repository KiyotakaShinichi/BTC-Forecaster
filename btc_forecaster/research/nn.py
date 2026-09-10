"""Layers for the deep challengers, built on the checked autodiff engine.

Everything here is small on purpose. The zoo trains on 1,000 rows, which after a
24-step window is 977 sequences; a model with more parameters than sequences is
not learning a series, it is memorising one. So hidden widths are 16-32, depth
is one or two blocks, and every architecture lands between roughly one and eight
thousand parameters.

Initialisation is Glorot for tanh/sigmoid paths and He for ReLU paths, from one
seeded generator. Not a detail at this size: with 977 sequences and 60 epochs
there is no budget to recover from a bad scale, and an architecture that looks
worse because it was initialised badly would be reported as an architectural
finding.

The recurrent cells are written as explicit loops over the window. That is
slower than a fused kernel and it is the reason the causality is inspectable:
step ``t`` reads ``x[:, t, :]`` and the previous hidden state, and there is
nowhere for a future step to enter.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from .autodiff import Tensor, concat, stack


class Module:
    """Anything with parameters. Registration is by attribute scan.

    Deliberately simple: a module's parameters are its own `Tensor` attributes
    plus those of any sub-module it holds. There is no explicit registration
    call to forget, which is the usual way a layer ends up silently frozen.
    """

    def parameters(self) -> list[Tensor]:
        found: list[Tensor] = []
        seen: set[int] = set()

        def walk(value: object) -> None:
            if isinstance(value, Tensor):
                if value.requires_grad and id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                for item in value.__dict__.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(self)
        return found

    def parameter_count(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        return self.forward(*args, **kwargs)  # type: ignore[no-any-return]

    def forward(self, *args: object, **kwargs: object) -> Tensor:  # pragma: no cover
        raise NotImplementedError


def _init(rng: np.random.Generator, shape: tuple[int, ...], *, relu: bool) -> Tensor:
    """He for ReLU paths, Glorot otherwise.

    With 977 sequences and a 60-epoch budget there is no room to recover from a
    bad scale, and a model that looked worse because of its initialisation would
    be written down as an architectural result.
    """
    fan_in = shape[0]
    fan_out = shape[-1]
    gain = 2.0 if relu else 1.0
    scale = np.sqrt(gain / (fan_in if relu else 0.5 * (fan_in + fan_out)))
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=True)


class Linear(Module):
    def __init__(
        self, rng: np.random.Generator, in_features: int, out_features: int, *, relu: bool = False
    ) -> None:
        self.weight = _init(rng, (in_features, out_features), relu=relu)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x @ self.weight + self.bias


class LayerNorm(Module):
    """Normalise over the feature axis.

    Load-bearing for the transformer at this depth: without it the residual
    stream drifts in scale across the block and the attention logits go with it.
    """

    def __init__(self, features: int, epsilon: float = 1e-5) -> None:
        self.gain = Tensor(np.ones((1, features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, features)), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        variance = (centred * centred).mean(axis=-1, keepdims=True)
        return centred / (variance + self.epsilon).sqrt() * self.gain + self.bias


class CausalConv1d(Module):
    """Dilated causal convolution over time.

    Causality is enforced by construction rather than by masking: the output at
    step ``t`` is built from inputs at ``t - d*(k-1) ... t``, and steps before
    the start are zero-padded. There is no path by which ``t+1`` reaches ``t``,
    which is the property a mask can silently get wrong by one.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = _init(rng, (kernel_size * in_channels, out_channels), relu=True)
        self.bias = Tensor(np.zeros((1, out_channels)), requires_grad=True)
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """(batch, time, channels) -> (batch, time, out_channels)."""
        batch, time, _ = x.shape
        pad = self.dilation * (self.kernel_size - 1)
        padded = concat(
            [Tensor(np.zeros((batch, pad, self.in_channels))), x], axis=1
        )
        taps = [
            padded[:, i * self.dilation : i * self.dilation + time, :]
            for i in range(self.kernel_size)
        ]
        joined = concat(taps, axis=-1)
        flat = joined.reshape(batch * time, self.kernel_size * self.in_channels)
        return (flat @ self.weight + self.bias).reshape(batch, time, -1)


class GruCell(Module):
    """A GRU step. Two gates, written out so the recursion is readable."""

    def __init__(self, rng: np.random.Generator, in_features: int, hidden: int) -> None:
        self.hidden = hidden
        self.update = Linear(rng, in_features + hidden, hidden)
        self.reset = Linear(rng, in_features + hidden, hidden)
        self.candidate = Linear(rng, in_features + hidden, hidden)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:  # type: ignore[override]
        joined = concat([x, h], axis=-1)
        z = self.update(joined).sigmoid()
        r = self.reset(joined).sigmoid()
        n = self.candidate(concat([x, r * h], axis=-1)).tanh()
        return (1.0 - z) * n + z * h


class LstmCell(Module):
    """An LSTM step. Three gates and a cell state.

    The forget-gate bias is initialised to 1.0, which is the standard fix for
    short-sequence training: at zero the gate starts near 0.5 and the cell state
    decays by half per step, so a 24-step window arrives with almost nothing
    from its start.
    """

    def __init__(self, rng: np.random.Generator, in_features: int, hidden: int) -> None:
        self.hidden = hidden
        self.forget = Linear(rng, in_features + hidden, hidden)
        self.forget.bias = Tensor(np.ones((1, hidden)), requires_grad=True)
        self.input = Linear(rng, in_features + hidden, hidden)
        self.output = Linear(rng, in_features + hidden, hidden)
        self.candidate = Linear(rng, in_features + hidden, hidden)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> Tensor:  # type: ignore[override]
        h, c = state
        joined = concat([x, h], axis=-1)
        f = self.forget(joined).sigmoid()
        i = self.input(joined).sigmoid()
        o = self.output(joined).sigmoid()
        g = self.candidate(joined).tanh()
        new_c = f * c + i * g
        new_h = o * new_c.tanh()
        return stack([new_h, new_c], axis=0)


class MultiHeadAttention(Module):
    """Scaled dot-product attention with a causal mask.

    The mask is additive and applied before the softmax, which is the only
    formulation that survives the numerical stabilisation: a multiplicative mask
    after the softmax renormalises the surviving weights and quietly lets a
    future position influence the total.
    """

    def __init__(self, rng: np.random.Generator, d_model: int, heads: int) -> None:
        if d_model % heads:
            raise ValueError("d_model must divide evenly across heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model
        self.query = Linear(rng, d_model, d_model)
        self.key = Linear(rng, d_model, d_model)
        self.value = Linear(rng, d_model, d_model)
        self.project = Linear(rng, d_model, d_model)

    def _split(self, x: Tensor, batch: int, time: int) -> Tensor:
        return x.reshape(batch, time, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        batch, time, _ = x.shape
        flat = x.reshape(batch * time, self.d_model)
        q = self._split(self.query(flat).reshape(batch, time, self.d_model), batch, time)
        k = self._split(self.key(flat).reshape(batch, time, self.d_model), batch, time)
        v = self._split(self.value(flat).reshape(batch, time, self.d_model), batch, time)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        # Additive -inf above the diagonal, before the softmax.
        mask = np.triu(np.full((time, time), -1e9), k=1)
        weights = (scores + Tensor(mask.reshape(1, 1, time, time))).softmax(axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch * time, self.d_model)
        return self.project(attended).reshape(batch, time, self.d_model)


def positional_encoding(time: int, d_model: int) -> Tensor:
    """Fixed sinusoidal positions. Not learned: at 977 sequences a learned
    table of 24 x d_model would spend parameters on memorising an ordering that
    is already known."""
    position = np.arange(time)[:, None]
    index = np.arange(d_model)[None, :]
    angle = position / np.power(10_000.0, (2 * (index // 2)) / d_model)
    encoding = np.where(index % 2 == 0, np.sin(angle), np.cos(angle))
    return Tensor(encoding.reshape(1, time, d_model))


def mse(prediction: Tensor, target: np.ndarray) -> Tensor:
    difference = prediction - Tensor(target.reshape(prediction.shape))
    return (difference * difference).mean()


def iterate_minibatches(
    n: int, batch_size: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Shuffled mini-batches.

    Shuffling *sequences* is not shuffling *time*: each sequence already
    contains its own ordered window, so the order in which complete windows are
    presented to the optimiser carries no temporal information and randomising
    it only decorrelates the gradient steps.
    """
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def parameters_of(modules: Sequence[Module]) -> list[Tensor]:
    found: list[Tensor] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                found.append(parameter)
    return found


__all__ = [
    "CausalConv1d",
    "GruCell",
    "LayerNorm",
    "Linear",
    "LstmCell",
    "Module",
    "MultiHeadAttention",
    "iterate_minibatches",
    "mse",
    "parameters_of",
    "positional_encoding",
]
