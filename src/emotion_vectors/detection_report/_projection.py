"""Section 6: the paper's neutral-projection confound-removal step.

The source paper finds the directions that vary most while the model reads
emotionally neutral text and subtracts them out of the probes; what remains
should be more purely emotional. This module applies that step to the
full-corpus probe set and formats the before-and-after table. The registered
pass rule stays in the notebook cell.

Scores here center on the 12 probes themselves (see :mod:`._data`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float

from emotion_vectors.analysis import project_out_neutral

from ._data import IT_LABEL, READOUTS, ModelSweep, battery_counts
from ._scale import ScaleCorpus

# Layers and readouts the table walks: the geometry peak, the layer the
# centered-readout resolution later passed at, and the deepest collected
# layer, under the two readouts the campaign reported.
HIGHLIGHT_LAYERS = (33, 39, 57)
HIGHLIGHT_READOUTS = ("last", "mean_content")
# A row is shown when its two batteries together clear this, so the table
# stays short; the deepest layer's content-mean row is always shown as the
# fixed reference point the campaign record quotes.
TABLE_INTEREST_TOTAL = 8


def projection_probes(
    scale: ScaleCorpus,
) -> tuple[
    Float[np.ndarray, "battery_emotions layers d_model"],
    Float[np.ndarray, "battery_emotions layers d_model"],
    dict[int, int],
]:
    """Apply the neutral projection to the full-corpus probes, layer by layer.

    Input: the :class:`~._scale.ScaleCorpus`. Returns ``(unprojected,
    projected, components_removed)``, the last mapping each layer to how many
    neutral principal components were subtracted (enough to cover half the
    neutral variance, the paper's threshold).
    """
    unprojected = scale.means[-1]
    projected = np.zeros_like(unprojected)
    components_removed: dict[int, int] = {}
    for layer_pos, layer in enumerate(scale.layers):
        projected[:, layer_pos, :], n_removed = project_out_neutral(
            unprojected[:, layer_pos, :], scale.neutral[:, layer_pos, :]
        )
        components_removed[int(layer)] = int(n_removed)
    return unprojected, projected, components_removed


def projection_table(
    model: ModelSweep,
    probe_sets: dict[str, Float[np.ndarray, "battery_emotions layers d_model"]],
    layers: list[int],
    components_removed: dict[int, int],
) -> dict[str, Any]:
    """Score the battery before and after the projection at the shown layers.

    Inputs: the instruct model's sweep, {"unprojected": ..., "projected":
    ...}, the shared layer grid, and ``projection_probes``'s
    components-removed map. Returns ``stats`` with ``lines`` (the printed
    record) and ``counts`` {(probe set, format, layer, readout): (paper,
    held-out)} for every cell scored.
    """
    missing = [readout for readout in HIGHLIGHT_READOUTS if readout not in READOUTS]
    if missing:
        raise ValueError(f"table asks for unswept readouts {missing}")
    lines = [
        f"model: {IT_LABEL}",
        f"neutral PCs removed per layer (50% of the neutral variance): {components_removed}",
        "",
        f"{'probes':12s} {'fmt':6s} {'layer':5s} {'readout':13s} {'paper':>6s} {'heldout':>8s}",
    ]
    counts: dict[tuple[str, str, int, str], tuple[int, int]] = {}
    for name, probes in probe_sets.items():
        for fmt in model.formats:
            for layer in HIGHLIGHT_LAYERS:
                layer_pos = layers.index(layer)
                for readout in HIGHLIGHT_READOUTS:
                    paper, held_out = battery_counts(
                        model, probes[:, layer_pos, :], fmt, readout, layer_pos
                    )
                    counts[name, fmt, layer, readout] = (paper, held_out)
                    is_reference_row = layer == HIGHLIGHT_LAYERS[-1] and readout == "mean_content"
                    if paper + held_out >= TABLE_INTEREST_TOTAL or is_reference_row:
                        lines.append(
                            f"{name:12s} {fmt:6s} {layer:5d} {readout:13s}"
                            f" {paper:4d}/12 {held_out:6d}/12"
                        )
    return {"lines": lines, "counts": counts}
