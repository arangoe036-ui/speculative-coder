"""Early-exit cascade verification: read half the target's weights when it is sure.

A decode forward is bound by streaming weights, and the target reads all 28 of its
layers to score every token. If the residual stream already "knows" the answer
halfway up, the remaining layers are wasted traffic. This projects an intermediate
hidden state to the vocabulary and, when that intermediate distribution is confident
enough, accepts the drafted token without finishing the forward.

Two things about this are unlike everything else in the project, and both are
load-bearing.

This engine is lossy
--------------------
Every other strategy here is distribution-preserving: drafts can be arbitrarily bad
because the *full* target decides each token. Accepting on layer 14's confidence
replaces that decision with an approximation of it. At temperature 0 the true ``p`` is
one-hot at the full model's argmax; if the intermediate projection is confident about
a different token, an early accept emits something the target would never have
produced. So the number that matters is not the layer saving but the **agreement
rate**: when the gate fires, how often does it agree with the full model? That is
measured, not assumed -- see :func:`measure_cascade`.

The skipped layers still owe their KV entries
---------------------------------------------
Exiting at layer 14 means layers 14-27 never see that position, so their caches have
no entry for it. Accepted tokens become context, and any later full pass needs those
entries, so the skipped work has to be repaid. Early exit therefore does not delete
compute -- it *defers* it, and only pays off if the backlog can be repaid in a batch
(a wide forward costs barely more than a narrow one, so several deferred positions can
be caught up at once). :class:`CascadeBudget` accounts for that honestly rather than
counting the deferral as a saving.

The projection detail that matters
----------------------------------
``lm_head`` expects states that have passed the model's final norm, so the
intermediate state is normalised with that same norm before projection. Skipping it
makes the mid-layer distribution far worse and would understate the idea; this is the
standard logit-lens construction.
"""

from __future__ import annotations

import torch


def decoder_layers(model):
    """The target's decoder layer list, whatever the wrapper looks like."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError("could not locate decoder layers on this model")
    return layers


def final_norm(model):
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise AttributeError("could not locate the final norm on this model")
    return norm


def project_intermediate(model, hidden: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits from an intermediate hidden state.

    The final norm is applied first: ``lm_head`` was fitted to normalised states, and
    projecting a raw residual-stream vector through it gives a much weaker
    distribution. Omitting the norm would make early exit look worse than it is.
    """
    head = model.get_output_embeddings()
    normed = final_norm(model)(hidden)
    return head(normed.to(head.weight.dtype))


class CascadeBudget:
    """Weight-traffic accounting for early exit, including deferred work.

    ``layers_run`` counts layer executions actually performed. ``layers_owed`` counts
    executions deferred by an early exit that a later full pass will have to repay.
    Reporting only the first would credit the cascade with savings it has merely
    postponed.
    """

    def __init__(self, total_layers: int, early_layer_idx: int):
        self.total_layers = total_layers
        self.early_layer_idx = early_layer_idx
        self.layers_run = 0
        self.layers_owed = 0
        self.tokens = 0
        self.early_exits = 0
        self.full_passes = 0

    def record_early_exit(self, positions: int = 1) -> None:
        self.layers_run += self.early_layer_idx * positions
        self.layers_owed += (self.total_layers - self.early_layer_idx) * positions
        self.early_exits += positions
        self.tokens += positions

    def record_full_pass(self, positions: int = 1, repaid: int = 0) -> None:
        self.layers_run += self.total_layers * positions
        # A full pass over `repaid` deferred positions clears their debt at the same
        # time, which is the only way early exit can actually pay.
        cleared = min(repaid, self.layers_owed // max(1, self.total_layers - self.early_layer_idx))
        self.layers_owed -= cleared * (self.total_layers - self.early_layer_idx)
        self.layers_run += (self.total_layers - self.early_layer_idx) * repaid
        self.full_passes += positions
        self.tokens += positions

    @property
    def layers_per_token(self) -> float:
        return self.layers_run / self.tokens if self.tokens else 0.0

    @property
    def layers_per_token_with_debt(self) -> float:
        """Layers per token counting work still owed -- the honest figure."""
        if not self.tokens:
            return 0.0
        return (self.layers_run + self.layers_owed) / self.tokens

    @property
    def early_exit_rate(self) -> float:
        return self.early_exits / self.tokens if self.tokens else 0.0

    def summary(self) -> str:
        return (
            f"{self.tokens} tokens | early exits {self.early_exit_rate:.1%} | "
            f"layers/token {self.layers_per_token:.1f} run, "
            f"{self.layers_per_token_with_debt:.1f} including deferred | "
            f"baseline {self.total_layers}"
        )


@torch.no_grad()
def measure_cascade(
    model,
    input_ids: torch.Tensor,
    early_layer_idx: int,
    threshold: float,
) -> dict:
    """Would an early-exit gate at this layer be safe, and how often would it fire?

    One teacher-forced forward gives both the intermediate states and the true final
    logits, so the gate can be evaluated against ground truth at every position at
    once. Reports:

    * ``fire_rate`` -- how often the intermediate distribution clears ``threshold``.
      A gate that never fires saves nothing; one that always fires is not a gate.
    * ``agreement`` -- among fired positions, how often the intermediate argmax equals
      the *full model's* argmax. This is the losslessness cost, and it is the number
      the whole idea rests on.
    * ``agreement_all`` -- the same over every position, for context on how good the
      intermediate projection is in general.
    """
    outputs = model(
        input_ids=input_ids, output_hidden_states=True, use_cache=False
    )
    # hidden_states[0] is the embedding output, so index i is the input to layer i.
    intermediate = outputs.hidden_states[early_layer_idx]
    early_logits = project_intermediate(model, intermediate[0]).float()
    early_probs = torch.softmax(early_logits, dim=-1)

    early_top = early_probs.max(dim=-1)
    early_argmax = early_top.indices
    confidence = early_top.values
    true_argmax = outputs.logits[0].float().argmax(dim=-1)

    fired = confidence > threshold
    agree = early_argmax == true_argmax

    n = int(true_argmax.numel())
    n_fired = int(fired.sum())
    return {
        "positions": n,
        "fire_rate": n_fired / max(1, n),
        "agreement": float((agree & fired).sum()) / max(1, n_fired),
        "agreement_all": float(agree.sum()) / max(1, n),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(confidence.median()),
        "fired": n_fired,
        "wrong_accepts": int((fired & ~agree).sum()),
    }
