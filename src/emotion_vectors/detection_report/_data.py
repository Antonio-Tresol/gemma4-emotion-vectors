"""Data resolution and shared vocabulary for the notebook-03 exhibit library.

Notebook 03 reports the detection campaign: seven pre-registered attempts to
make the emotion probes work as detectors of an implied emotion. This module
resolves every input through ``emotion_vectors.artifacts.fetch`` (local
``results/`` first, then the published Hugging Face datasets), so the notebook
runs unchanged on any clone, and pins the vocabulary every figure shares: the
constants of the registered pass rule, one fixed color per tracked probe, and
labels that name the ROLE of each arm rather than its filename.

TWO CENTERING CONVENTIONS COEXIST IN THIS NOTEBOOK, deliberately, and must not
be unified. A cosine is only defined once a baseline has been subtracted from
the probes, and the campaign used two baselines:

* Sections 1 and 2 score the corpus probe sets, which were extracted for all
  171 emotion words. Those sections center on the FULL extraction (the "sweep
  convention": ``center_pool`` is all 171 mean vectors).
* Sections 3 to 6 score probe sets that only ever extracted the 12 scenario
  emotions (self-generated stories, dialogues, the scale corpus). Those
  sections center on the 12 probes themselves, because no 171-word pool
  exists for them.

Both are registered; mixing them silently would move every number. Every
scoring entry point below therefore takes ``center_pool`` explicitly.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jaxtyping import Float

from emotion_vectors.artifacts import fetch  # local results/ first, HF otherwise
from emotion_vectors.probe_prompts import SCENARIOS
from emotion_vectors.scoring import score_battery

# Plotly never wraps a title, it just runs off the canvas, so every title in
# this package is wrapped by ``figure_title`` before it is set. These are
# characters that fit per pixel of figure width, measured from rendered PNGs
# at title_font_size 13: the headline renders at full size, the subtitle
# lines inside <sup> render smaller and fit more.
HEADLINE_CHARS_PER_PIXEL = 0.125
SUBTITLE_CHARS_PER_PIXEL = 0.17

# Constants of the registered detection read (TREE.md Q1.H2.E2/E4, registered
# before any scoring).
N_EMOTIONS = 12  # candidate emotions ranked for each scenario
N_SCENARIOS = 12  # scenario prompts in each of the two sets
TOP_K = 3  # a scenario is correct if the target emotion ranks in the top 3
PASS_BAR = 8  # the registered bar: >= 8 of 12 correct on BOTH scenario sets
# top-3 of 12 emotions by luck: 3/12 chance per scenario, so 3 of 12 expected
CHANCE_CORRECT = N_SCENARIOS * TOP_K / N_EMOTIONS
REGISTERED_LAYER = 33  # the layer the registered single-layer reads are defined at

IT_LABEL = "gemma-4-31b-it"  # the instruction-tuned model under study
BASE_LABEL = "gemma-4-31b (base)"  # the untuned comparison model

# Ways of pooling a prompt's token activations into one vector. READOUTS and
# READOUT_LABELS keep their names because notebook 03 imports READOUTS, and the
# entries are also part of the stored activation-array keys. The label names
# what each pooling DOES, so a figure legend never shows a bare key.
READOUTS = ["last", "mean_all", "mean_content"]
READOUT_LABELS = {
    "last": "last token",
    "mean_all": "mean of all tokens",
    "mean_content": "mean of content tokens",
}
# The two scenario sets, and the ROLE each one plays in the registered rule.
# The keys "scenario" and "heldout" are the ``kind`` values in the stored
# prompt records, so they stay as they are.
SCENARIO_SET_ROLES = {
    "scenario": "the paper's twelve scenarios: pick the best cell",
    "heldout": "twelve fresh held-out scenarios: confirm it",
}
SCENARIO_SET_SHORT = {"scenario": "paper", "heldout": "held-out"}

# label -> (mean-activation bundle, sweep subtree, prompt format that model
# was swept in for the single-format reads of section 2). The base model has
# no chat template, so its sweep holds the plain format only.
MODEL_SOURCES = {
    IT_LABEL: ("emotion_vectors_it_means.npz", "probe_sweep_it", "chat"),
    BASE_LABEL: ("emotion_vectors/emotion_means.npz", "probe_sweep", "plain"),
}


def figure_title(headline: str, subtitles: list[str], width_px: int) -> str:
    """Wrap a question-form headline plus subtitle lines to the figure width.

    Inputs: ``headline`` (the question and its answer, headline values already
    formatted in), ``subtitles`` (each becomes one smaller line, in order:
    what one mark is, then the anchors and the evidence file), and
    ``width_px`` (the figure's ``width``, which decides how many characters
    fit on a line). Returns the ``<br>``-joined title string.
    """
    if width_px <= 0:
        raise ValueError(f"figure width must be positive, got {width_px}")
    lines = textwrap.wrap(headline, width=int(width_px * HEADLINE_CHARS_PER_PIXEL))
    for subtitle in subtitles:
        wrapped = textwrap.wrap(subtitle, width=int(width_px * SUBTITLE_CHARS_PER_PIXEL))
        lines += [f"<sup>{line}</sup>" for line in wrapped]
    return "<br>".join(lines)


@dataclass
class ModelSweep:
    """One model's probe bundle plus its swept scenario and template prompts.

    Fields: ``label`` (the model, as every figure names it), ``means``
    [extracted_emotions layers d_model] (the mean activation per emotion word,
    171 words for the corpus extraction), ``emotions`` (row order of
    ``means``), ``layers`` (the collected layer numbers), ``formats`` (prompt
    formats this model was swept in), ``probe_index`` (rows of ``means`` for
    the 12 scenario emotions, in scenario order), ``scenario_set_rows``
    (prompt row indices per scenario set), ``prompts`` (the swept prompt
    records).
    """

    label: str
    means: Float[np.ndarray, "extracted_emotions layers d_model"]
    emotions: list[str]
    layers: list[int]
    formats: list[str]
    probe_index: list[int]
    scenario_set_rows: dict[str, list[int]]
    prompts: list[dict[str, Any]]
    _activations: Any = None  # the open npz; sliced lazily, cached per pooling
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    def activations(self, fmt: str, pooling: str) -> Float[np.ndarray, "prompts layers d_model"]:
        """Every swept prompt's activation under one (format, pooling), float32.

        The npz stores float16; every score in this notebook is computed after
        the same upcast, so the cast happens here once instead of per call.
        """
        # the stored arrays are keyed "<format>_<pooling>", e.g. "chat_last"
        key = f"{fmt}_{pooling}"
        if key not in self._cache:
            if key not in self._activations:
                raise KeyError(f"{self.label}: sweep has no '{key}' array")
            self._cache[key] = self._activations[key].astype(np.float32)
        return self._cache[key]

    def scenario_activations(
        self, fmt: str, pooling: str, kind: str, layer_pos: int
    ) -> Float[np.ndarray, "scenarios d_model"]:
        """One set's 12 scenario activations at one layer."""
        acts = self.activations(fmt, pooling)
        return np.stack([acts[row, layer_pos] for row in self.scenario_set_rows[kind]])

    def probe_means(self, layer_pos: int) -> Float[np.ndarray, "scenario_emotions d_model"]:
        """The 12 scenario probes at one layer, in scenario order."""
        return self.means[self.probe_index, layer_pos, :]

    def emotion_means(
        self, names: list[str], layer_pos: int
    ) -> Float[np.ndarray, "named_emotions d_model"]:
        """Named emotions' probes at one layer, in the order given."""
        missing = [name for name in names if name not in self.emotions]
        if missing:
            raise KeyError(f"{self.label}: no extracted probe for {missing}")
        return self.means[[self.emotions.index(name) for name in names], layer_pos, :]

    def template_rows(self) -> dict[str, list[tuple[int, int]]]:
        """{template name: [(number in the sentence, prompt row)]}, ascending.

        The numerical-intensity templates repeat one sentence with only a
        number changed; this groups the swept prompts back into those series.
        """
        series: dict[str, list[tuple[int, int]]] = {}
        for row, prompt in enumerate(self.prompts):
            if prompt["kind"] == "template":
                series.setdefault(prompt["name"], []).append((int(prompt["x"]), row))
        if not series:
            raise ValueError(f"{self.label}: sweep holds no numerical-intensity templates")
        return {name: sorted(points) for name, points in series.items()}

    def center_pool(self, layer_pos: int) -> Float[np.ndarray, "extracted_emotions d_model"]:
        """The full extraction at one layer: the sweep convention's baseline."""
        return self.means[:, layer_pos, :]


@dataclass(frozen=True)
class SweepContext:
    """Both models' sweeps, sharing one layer grid and one scenario order.

    ``models[label]`` is that model's :class:`ModelSweep`; ``layers`` is the
    shared layer grid; ``probe_order`` is the 12 target emotions in scenario
    order, which is the row order of every probe matrix here.
    """

    models: dict[str, ModelSweep]
    layers: list[int]
    probe_order: list[str]

    def model(self, label: str) -> ModelSweep:
        """One model's sweep, by label, failing loudly on an unknown label."""
        if label not in self.models:
            raise KeyError(f"no sweep loaded for {label}; have {list(self.models)}")
        return self.models[label]


def _load_model_sweep(
    label: str, means_path: str, sweep_dir: str, probe_order: list[str]
) -> ModelSweep:
    """Resolve and assemble one model's sweep; every missing input raises."""
    bundle = np.load(fetch(means_path), allow_pickle=True)
    emotions = [str(emotion) for emotion in bundle["emotions"]]
    layers = [int(layer) for layer in bundle["layers"]]
    sweep = np.load(fetch(f"{sweep_dir}/activations.npz"), allow_pickle=True)
    if [int(layer) for layer in sweep["layers"]] != layers:
        raise ValueError(f"{label}: sweep layers differ from the probe bundle's layers")
    prompts = [
        json.loads(line) for line in fetch(f"{sweep_dir}/prompts.jsonl").read_text().splitlines()
    ]
    # "scenario" and "heldout" are the ``kind`` values in the stored prompt records
    scenario_set_rows = {
        kind: [pos for pos, prompt in enumerate(prompts) if prompt["kind"] == kind]
        for kind in ("scenario", "heldout")
    }
    for kind, rows in scenario_set_rows.items():
        if len(rows) != N_SCENARIOS:
            raise ValueError(
                f"{label}: the {kind} set has {len(rows)} prompts, expected {N_SCENARIOS}"
            )
    return ModelSweep(
        label=label,
        means=bundle["means"].astype(np.float32),
        emotions=emotions,
        layers=layers,
        formats=[str(fmt) for fmt in sweep["formats"]],
        probe_index=[emotions.index(target) for target in probe_order],
        scenario_set_rows=scenario_set_rows,
        prompts=prompts,
        _activations=sweep,
    )


def load_sweep_context() -> SweepContext:
    """Load both models' probe bundles and swept activations.

    Returns the :class:`SweepContext` every section reads from. Raises if the
    two models disagree on the layer grid (the figures overlay them) or if the
    base model turns out to carry a chat format (it has no chat template, and
    section 1's base panel is labelled on that assumption).
    """
    probe_order = [target for _, target, _ in SCENARIOS]
    models = {
        label: _load_model_sweep(label, means_path, sweep_dir, probe_order)
        for label, (means_path, sweep_dir, _fmt) in MODEL_SOURCES.items()
    }
    layer_grids = {label: tuple(model.layers) for label, model in models.items()}
    if len(set(layer_grids.values())) != 1:
        raise ValueError(f"models disagree on the layer grid: {layer_grids}")
    if models[BASE_LABEL].formats != ["plain"]:
        raise ValueError(
            f"base model has no chat template, so plain format only; got {models[BASE_LABEL].formats}"
        )
    return SweepContext(
        models=models,
        layers=models[IT_LABEL].layers,
        probe_order=probe_order,
    )


def battery_counts(
    model: ModelSweep,
    probe_means: Float[np.ndarray, "scenario_emotions d_model"],
    fmt: str,
    pooling: str,
    layer_pos: int,
    center_pool: Float[np.ndarray, "pool d_model"] | None = None,
) -> tuple[int, int]:
    """(paper, held-out) scenarios of 12 whose target emotion ranks top-3.

    The registered statistic, evaluated on both scenario sets at once.
    ``center_pool`` picks the centering convention (see the module docstring):
    pass the full 171-word extraction for sections 1 and 2, leave it None to
    center on the 12 probes themselves for sections 3 to 6. This function keeps
    its name because notebook 03 imports it.
    """
    counts = [
        score_battery(
            probe_means,
            model.scenario_activations(fmt, pooling, kind, layer_pos),
            center_pool=center_pool,
        )[0]
        for kind in ("scenario", "heldout")
    ]
    return counts[0], counts[1]
