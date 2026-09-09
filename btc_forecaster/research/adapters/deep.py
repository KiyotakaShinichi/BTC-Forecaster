"""Seven compact deep temporal challengers, on the checked numpy engine.

Each consumes a genuine temporal window -- 24 causal feature rows -- and emits
one next-bar log return. That constraint is what makes them a family rather than
a collection: a "LSTM" fed one scalar row at a time is not a sequence model, and
Phase 9 rules it out explicitly.

The seven are architecturally distinct, not seven parameterisations of one idea:

``mlp``          flattens the window; no temporal structure at all. The control.
``cnn_1d``       two causal convolutions; local patterns, fixed receptive field.
``tcn``          dilated residual convolutions; exponential receptive field.
``gru``          gated recurrence, two gates.
``lstm``         gated recurrence with a separate cell state, three gates.
``transformer``  causal self-attention; every step can read every earlier one.
``nbeats``       residual backcast/forecast stacks; no recurrence, no attention.

**Capacity is the binding constraint, not compute.** After a 24-step window the
1,000 training rows leave 977 sequences, so widths are 16-32 and six of the
seven land between roughly 1.3k and 9k parameters.

N-BEATS is the exception and it is left as one. Its backcast head projects back
to the full flattened window, which makes it structurally parameter-heavy: even
at hidden 16 it carries roughly 18k parameters, eighteen per training sequence.
Shrinking it until it fit a sentence would have hidden the interesting part --
the parameters-to-sequences ratio is exactly the axis Phase 24 asks about, and
N-BEATS is where this zoo can actually see it.

**Every choice that could be tuned is frozen instead.** One learning rate, one
batch size, one epoch ceiling, one seed, for all seven. Early stopping watches
DEV and never HOLDOUT. The intent is to compare architectures under an identical
budget; a per-model tuning loop would compare how long each was tuned for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..autodiff import Adam, Tensor
from ..contracts import (
    Capability,
    EvaluationContext,
    Family,
    ModelFitError,
    Preprocessing,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from ..nn import (
    CausalConv1d,
    GruCell,
    LayerNorm,
    Linear,
    LstmCell,
    Module,
    MultiHeadAttention,
    iterate_minibatches,
    mse,
    positional_encoding,
)
from ..preprocessing import Scaler, fit_scaler
from ..registry import ZooRegistration, register
from ..windows import WindowSpec, build_sequences, evaluation_windows

#: One training budget for all seven, so the comparison is architectural.
SEED = 20260909
LOOKBACK = 24
MAX_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
#: Stop when DEV has not improved for this many epochs. Watching DEV, never
#: HOLDOUT -- an early-stop epoch chosen on the evaluation block is a
#: hyperparameter fitted to the thing being reported.
PATIENCE = 8
#: Wall-clock ceiling per model. A model that exceeds it is recorded as
#: RESOURCE_LIMIT with the epochs it managed, not left running.
TRAIN_SECONDS_BUDGET = 300.0


@dataclass
class TrainingTrace:
    """What the training loop did, for the model card and the resource registry."""

    epochs_run: int
    best_epoch: int
    best_dev_loss: float
    train_loss: float
    seconds: float
    stopped_early: bool
    hit_time_budget: bool

    def as_dict(self) -> dict:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_dev_loss": self.best_dev_loss,
            "final_train_loss": self.train_loss,
            "training_seconds": round(self.seconds, 3),
            "stopped_early": self.stopped_early,
            "hit_time_budget": self.hit_time_budget,
        }


class DeepModel(ZooModel):
    """Shared training loop, scaling and window handling for the seven.

    Subclasses build a network and define its forward pass. Everything that
    could differ between them by accident -- the seed, the optimiser, the
    stopping rule, the target scaling -- is here so that it cannot.
    """

    family = Family.DEEP
    preprocessing = Preprocessing.STANDARDIZED
    capabilities = frozenset({Capability.POINT, Capability.SERIALIZE})
    resource_class = ResourceClass.MODERATE
    needs_calibration = True

    hidden = 24

    def __init__(self) -> None:
        super().__init__()
        self._modules: list[Module] = []
        self._scaler: Scaler | None = None
        self._feature_names: tuple[str, ...] = ()
        self._target_mean = 0.0
        self._target_std = 1.0
        self._trace: TrainingTrace | None = None
        self._window = WindowSpec(lookback=LOOKBACK)
        self._train_sequences: np.ndarray | None = None
        self._train_targets: np.ndarray | None = None
        self._rng = np.random.default_rng(SEED)

    # -- subclass surface ------------------------------------------------

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        raise NotImplementedError

    def _forward(self, x: Tensor) -> Tensor:
        """(batch, lookback, features) -> (batch, 1)."""
        raise NotImplementedError

    # -- lifecycle -------------------------------------------------------

    def _fit(self, train: TrainingSet) -> None:
        self._scaler = fit_scaler(train.X, self.preprocessing)
        self._feature_names = tuple(train.X.columns)
        scaled = self._scaler.transform(train.X)

        targets = train.y.to_numpy(dtype=float)
        # The target is standardised too. Daily log returns are order 1e-2, and
        # a squared loss at that scale is order 1e-4 -- small enough that one
        # learning rate cannot serve both the output layer and the input layer.
        self._target_mean = float(targets.mean())
        self._target_std = float(targets.std()) or 1.0

        sequences = build_sequences(scaled, train.y, train.feature_bar, self._window)
        self._train_sequences = sequences.X
        self._train_targets = (sequences.y - self._target_mean) / self._target_std

        self._rng = np.random.default_rng(SEED)
        self._modules = self._build(self._rng, sequences.n_features)

    def _calibrate(self, context: EvaluationContext) -> None:
        """Train, with DEV deciding when to stop.

        Fitting happens here rather than in `_fit` because the stopping rule
        needs a block that estimation never touched, and the contract hands that
        block to `calibrate`. A model stopped on its own training loss would
        stop when it had finished memorising.
        """
        if self._train_sequences is None or self._train_targets is None or self._scaler is None:
            raise ModelFitError(f"{self.model_id} was not prepared before calibration")
        train_sequences, train_targets = self._train_sequences, self._train_targets

        dev_X = self._scaler.transform(context.X)
        dev_sequences = build_sequences(dev_X, context.y, context.feature_bar, self._window)
        dev_targets = (dev_sequences.y - self._target_mean) / self._target_std

        parameters = [p for module in self._modules for p in module.parameters()]
        optimiser = Adam(
            parameters, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        best_loss = np.inf
        best_state = [p.data.copy() for p in parameters]
        best_epoch = 0
        since_improvement = 0
        started = time.perf_counter()
        epoch = 0
        train_loss = np.inf
        hit_budget = False

        for epoch in range(1, MAX_EPOCHS + 1):
            batch_losses = []
            for indices in iterate_minibatches(len(train_targets), BATCH_SIZE, self._rng):
                optimiser.zero_grad()
                prediction = self._forward(Tensor(train_sequences[indices]))
                loss = mse(prediction, train_targets[indices])
                loss.backward()
                optimiser.step()
                batch_losses.append(float(loss.data))
            train_loss = float(np.mean(batch_losses))

            dev_prediction = self._forward(Tensor(dev_sequences.X))
            dev_loss = float(mse(dev_prediction, dev_targets).data)

            if dev_loss < best_loss - 1e-9:
                best_loss, best_epoch = dev_loss, epoch
                best_state = [p.data.copy() for p in parameters]
                since_improvement = 0
            else:
                since_improvement += 1

            if since_improvement >= PATIENCE:
                break
            if time.perf_counter() - started > TRAIN_SECONDS_BUDGET:
                hit_budget = True
                break

        # Restore the parameters DEV liked best, not the ones the last epoch
        # happened to leave behind.
        for parameter, saved in zip(parameters, best_state, strict=True):
            parameter.data = saved

        self._trace = TrainingTrace(
            epochs_run=epoch,
            best_epoch=best_epoch,
            best_dev_loss=best_loss,
            train_loss=train_loss,
            seconds=time.perf_counter() - started,
            stopped_early=since_improvement >= PATIENCE,
            hit_time_budget=hit_budget,
        )

    def _predict_point(self, context: EvaluationContext) -> np.ndarray:
        if self._scaler is None:
            raise ModelFitError(f"{self.model_id} has no fitted scaler")
        if self._trace is None:
            raise ModelFitError(
                f"{self.model_id} was never trained; deep models train during "
                "calibration so that DEV decides when to stop"
            )
        # The full design matrix, so the first evaluation window can reach back
        # into realised history rather than starting from nothing.
        full = self._scaler.transform(
            _full_design(context, self._feature_names)
        )
        windows = evaluation_windows(full, context.target_bars, self._window)
        outputs = []
        # Batched so a 705-origin evaluation does not build one enormous tape.
        for start in range(0, len(windows), 256):
            block = self._forward(Tensor(windows[start : start + 256]))
            outputs.append(np.asarray(block.data, dtype=float).ravel())
        standardised = np.concatenate(outputs)
        return standardised * self._target_std + self._target_mean

    # -- reporting -------------------------------------------------------

    def parameter_count(self) -> int | None:
        if not self._modules:
            return None
        seen: set[int] = set()
        total = 0
        for module in self._modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    total += int(parameter.data.size)
        return total

    def hyperparameters(self) -> dict:
        configuration = {
            "lookback": LOOKBACK,
            "hidden": self.hidden,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seed": SEED,
            "early_stopping_block": "DEV",
            "target_scaling": "standardised on train",
        }
        if self._trace is not None:
            configuration["training"] = self._trace.as_dict()
        return configuration


def _full_design(context: EvaluationContext, names: tuple[str, ...]):  # type: ignore[no-untyped-def]
    """The design matrix a window may draw from, checked for column agreement.

    Falls back to the evaluation rows alone when no full matrix was supplied --
    which costs the first `lookback` origins and never reaches forward.
    """
    design = context.design if context.design is not None else context.X
    if tuple(design.columns) != names:
        raise ModelFitError(
            f"the design matrix has a different feature set: expected {list(names)}"
        )
    return design


# --------------------------------------------------------------------------
# The seven architectures.
# --------------------------------------------------------------------------


class Mlp(DeepModel):
    """Flattens the window. No temporal structure whatsoever.

    The control the other six are measured against: if none of them beats a
    model that cannot tell the order of its own inputs, the conclusion is about
    the series rather than about architectures.
    """

    model_id = "mlp"
    hidden = 32

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        flat = LOOKBACK * n_features
        self.l1 = Linear(rng, flat, self.hidden, relu=True)
        self.l2 = Linear(rng, self.hidden, self.hidden // 2, relu=True)
        self.out = Linear(rng, self.hidden // 2, 1)
        return [self.l1, self.l2, self.out]

    def _forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, x.shape[1] * x.shape[2])
        return self.out(self.l2(self.l1(flat).relu()).relu())


class TemporalCnn(DeepModel):
    """Two causal convolutions and a mean over time.

    A fixed receptive field of five bars: local pattern detection, and nothing
    that can reach the start of a 24-step window.
    """

    model_id = "cnn_1d"
    hidden = 16

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        self.c1 = CausalConv1d(rng, n_features, self.hidden, kernel_size=3)
        self.c2 = CausalConv1d(rng, self.hidden, self.hidden, kernel_size=3)
        self.out = Linear(rng, self.hidden, 1)
        return [self.c1, self.c2, self.out]

    def _forward(self, x: Tensor) -> Tensor:
        h = self.c1(x).relu()
        h = self.c2(h).relu()
        return self.out(h.mean(axis=1))


class Tcn(DeepModel):
    """Dilated residual convolutions: receptive field grows exponentially.

    Dilations 1, 2, 4 with kernel 3 reach 15 bars against the CNN's five, at
    the same depth. That is the architectural claim being tested, and it is the
    only difference from `cnn_1d` worth having both.
    """

    model_id = "tcn"
    hidden = 16

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        self.project = Linear(rng, n_features, self.hidden, relu=True)
        self.blocks = [
            CausalConv1d(rng, self.hidden, self.hidden, kernel_size=3, dilation=d)
            for d in (1, 2, 4)
        ]
        self.out = Linear(rng, self.hidden, 1)
        return [self.project, *self.blocks, self.out]

    def _forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        h = self.project(x.reshape(batch * time, x.shape[2])).relu().reshape(
            batch, time, self.hidden
        )
        for block in self.blocks:
            # Residual: the block learns a correction, which is what lets three
            # of them stack without the signal degrading.
            h = h + block(h).relu()
        return self.out(h[:, -1, :])


class Gru(DeepModel):
    """Gated recurrence over the window; the last hidden state is the summary."""

    model_id = "gru"
    hidden = 24

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        self.cell = GruCell(rng, n_features, self.hidden)
        self.out = Linear(rng, self.hidden, 1)
        return [self.cell, self.out]

    def _forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        h = Tensor(np.zeros((batch, self.hidden)))
        for t in range(time):
            h = self.cell(x[:, t, :], h)
        return self.out(h)


class Lstm(DeepModel):
    """Three gates and a separate cell state.

    Registered beside the GRU rather than instead of it because the extra gate
    and the persistent cell are the hypothesis: at 977 sequences the additional
    parameters may cost more than the memory buys.
    """

    model_id = "lstm"
    hidden = 24

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        self.cell = LstmCell(rng, n_features, self.hidden)
        self.out = Linear(rng, self.hidden, 1)
        return [self.cell, self.out]

    def _forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        h = Tensor(np.zeros((batch, self.hidden)))
        c = Tensor(np.zeros((batch, self.hidden)))
        for t in range(time):
            state = self.cell(x[:, t, :], (h, c))
            h, c = state[0], state[1]
        return self.out(h)


class TransformerEncoder(DeepModel):
    """One causal self-attention block with a feed-forward projection.

    Every step can attend to every earlier step directly, which is the
    architectural difference from recurrence: no information bottleneck at the
    hidden state. One block and two heads, because at 977 sequences a stack
    would be memorising.
    """

    model_id = "transformer"
    hidden = 16
    heads = 2

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        self.embed = Linear(rng, n_features, self.hidden)
        self.attention = MultiHeadAttention(rng, self.hidden, self.heads)
        self.norm1 = LayerNorm(self.hidden)
        self.ff1 = Linear(rng, self.hidden, self.hidden * 2, relu=True)
        self.ff2 = Linear(rng, self.hidden * 2, self.hidden)
        self.norm2 = LayerNorm(self.hidden)
        self.out = Linear(rng, self.hidden, 1)
        return [
            self.embed,
            self.attention,
            self.norm1,
            self.ff1,
            self.ff2,
            self.norm2,
            self.out,
        ]

    def _forward(self, x: Tensor) -> Tensor:
        batch, time, features = x.shape
        h = self.embed(x.reshape(batch * time, features)).reshape(batch, time, self.hidden)
        h = h + positional_encoding(time, self.hidden)

        attended = self.attention(h)
        h = (h + attended).reshape(batch * time, self.hidden)
        h = self.norm1(h)
        h = h + self.ff2(self.ff1(h).relu())
        h = self.norm2(h).reshape(batch, time, self.hidden)
        return self.out(h[:, -1, :])


class NBeats(DeepModel):
    """Residual backcast/forecast stacks. No recurrence, no attention.

    Each block reads the residual left by the ones before it, emits a backcast
    it subtracts and a forecast it adds. The interesting property at this sample
    size is that it is a pure feed-forward decomposition -- so if it matches the
    recurrent models, recurrence was not what mattered.

    Generic basis rather than trend/seasonality: the seasonality basis would
    impose a weekly period this series does not have.
    """

    model_id = "nbeats"
    hidden = 16
    blocks = 2

    def _build(self, rng: np.random.Generator, n_features: int) -> list[Module]:
        flat = LOOKBACK * n_features
        self.stack = []
        for _ in range(self.blocks):
            block = {
                "h1": Linear(rng, flat, self.hidden, relu=True),
                "h2": Linear(rng, self.hidden, self.hidden, relu=True),
                "backcast": Linear(rng, self.hidden, flat),
                "forecast": Linear(rng, self.hidden, 1),
            }
            self.stack.append(block)
        return [module for block in self.stack for module in block.values()]

    def _forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        residual = x.reshape(batch, x.shape[1] * x.shape[2])
        forecast: Tensor | None = None
        for block in self.stack:
            hidden = block["h2"](block["h1"](residual).relu()).relu()
            residual = residual - block["backcast"](hidden)
            partial = block["forecast"](hidden)
            forecast = partial if forecast is None else forecast + partial
        assert forecast is not None
        return forecast


_DEEP = (
    (Mlp, "Flattened-window MLP: two hidden layers, no temporal structure.",
     ("The control. If nothing beats a model that cannot tell the order of its "
      "own inputs, the finding is about the series, not about architectures.",)),
    (TemporalCnn, "Two causal convolutions (kernel 3) and a mean over time.",
     ("Fixed five-bar receptive field.",)),
    (Tcn, "Dilated residual causal convolutions at dilations 1, 2, 4.",
     ("Reaches 15 bars against cnn_1d's 5 at the same depth; that is the whole "
      "reason both are registered.",)),
    (Gru, "GRU over the 24-step window; last hidden state projected to a scalar.", ()),
    (Lstm, "LSTM with forget-gate bias initialised to 1.0.",
     ("Registered beside the GRU because the extra gate is the hypothesis: at "
      "977 sequences the parameters may cost more than the memory buys.",)),
    (TransformerEncoder, "One causal self-attention block, two heads, sinusoidal positions.",
     ("Additive -inf mask before the softmax, so no future position can "
      "influence the normalisation.",)),
    (NBeats, "Two generic residual backcast/forecast blocks.",
     ("Generic basis rather than trend/seasonality: a seasonal basis would "
      "impose a weekly period this series does not have.",
      "The parameter-heavy outlier by construction: its backcast head projects "
      "back to the full flattened window, giving roughly 18 parameters per "
      "training sequence against 1-9 for the other six. Left that way on "
      "purpose -- the ratio is the finding.")),
)

for _cls, _description, _notes in _DEEP:
    register(
        ZooRegistration(
            model_id=_cls.model_id,
            factory=_cls,
            family=Family.DEEP,
            resource_class=ResourceClass.MODERATE,
            description=_description,
            requires=(),
            notes=(
                *_notes,
                f"Trained on the numpy autodiff engine in this repository, whose "
                f"gradients are checked against central finite differences. torch "
                f"is not a dependency: the lock is compiled with --all-extras and "
                f"installed by the fresh-clone CI job, so a torch extra would put "
                f"a multi-gigabyte CUDA closure in every clean install.",
                f"Budget frozen and shared across all seven: lookback {LOOKBACK}, "
                f"max {MAX_EPOCHS} epochs, batch {BATCH_SIZE}, lr {LEARNING_RATE}, "
                f"patience {PATIENCE}, seed {SEED}. Early stopping watches DEV.",
            ),
        )
    )


__all__ = [
    "BATCH_SIZE",
    "LEARNING_RATE",
    "LOOKBACK",
    "MAX_EPOCHS",
    "PATIENCE",
    "SEED",
    "TRAIN_SECONDS_BUDGET",
    "DeepModel",
    "Gru",
    "Lstm",
    "Mlp",
    "NBeats",
    "Tcn",
    "TemporalCnn",
    "TrainingTrace",
    "TransformerEncoder",
]
