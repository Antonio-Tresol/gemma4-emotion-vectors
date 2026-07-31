"""Data resolution and shared vocabulary for the notebook-08 exhibit library.

Notebook 08 reports the Q3 first tranche: while the instruction-tuned model
reads a multi-phase story, can the current emotion be read out of its
activations (the identity read), and does the incoming emotion start rising
before the phase boundary (the anticipation read, registry name "R1")? This
module resolves every input through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the
published Hugging Face datasets), so the notebook runs unchanged on any clone.

Nothing here re-scores. Every function slices the frozen evidence JSONs written
by ``scripts/score_q3_gate_r1.py``.

TWO FACTORS ARE CROSSED IN THIS NOTEBOOK AND ARE EASY TO CONFUSE, so the
vocabulary below keeps them apart by construction:

* a PROBE SET is a set of emotion directions, named by whose stories the
  probes were built from (``PROBE_SET_LABEL``);
* a STORY SET is the set of stories the model was reading while its
  activations were recorded, named by who wrote those stories
  (``StorySet.label``).

"DeepSeek" names one of each, so no label in this package is ever the bare word
"deepseek": every figure legend, axis tick, and printed line spells out which
of the two roles it means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from emotion_vectors.artifacts import fetch  # local results/ first, HF otherwise

# figure_title and its two wrapping constants live in emotion_vectors.figure_text.
# Three byte-identical copies used to sit in the three report packages; they are
# re-exported here so this package's own modules keep importing them from _data.
from emotion_vectors.figure_text import (
    HEADLINE_CHARS_PER_PIXEL,
    SUBTITLE_CHARS_PER_PIXEL,
    figure_title,
)

PROBED_MODEL = "google/gemma-4-31b-it"  # the model whose reading was recorded

# The six extracted layers and the registered primary cell (TREE.md Q3.H1.E1,
# registered before any scoring ran).
LAYERS = [6, 15, 24, 33, 42, 51]
PRIMARY_LAYER = 33

# Grading anchors, all registered. The identity bar is the top decile of the
# probe set (rank 17 of 171, rank 1 of 12), which as a fraction of how many
# probes the set holds is 0.099 and 0.083; the figures draw one line at 0.1 for
# both. Chance is where the wrong-emotion shuffle (N2) lands: half the set.
IDENTITY_BAR_FRACTION = 0.1
CHANCE_FRACTION = 0.5
# A dot within this much of chance is called chance-like by the section-3 zones.
CHANCE_LIKE_FRACTION = 0.48
# The anticipation bar: N2 p < 0.001 AND a mean lead of at least half the
# calibrated per-token noise sd of that layer.
R1_BAR_IN_NOISE_SD = 0.5

# Probe sets, in the order every figure shows them, with labels that name the
# ROLE (whose stories the probes were built from) rather than the bare key.
PROBE_SET_ORDER = ["corpus", "selfgen", "deepseek"]
PROBE_SET_SIZE = {"corpus": 171, "selfgen": 12, "deepseek": 12}
PROBE_SET_LABEL = {
    "corpus": "corpus probes (171, external stories)",
    "selfgen": "self-gen probes (12, stories the probed model wrote)",
    "deepseek": "DeepSeek probes (12, DeepSeek-written stories)",
}
# Short two-line versions for axis ticks, where the full role sentence would
# collide with its neighbours. The role still reads off the tick; the full
# sentence stays in the legend, the subtitles, and the printed record.
PROBE_SET_TICK_LABEL = {
    "corpus": "corpus probes<br>(171, external)",
    "selfgen": "self-gen probes<br>(12, probed model)",
    "deepseek": "DeepSeek probes<br>(12, DeepSeek)",
}
# Shortest form that still says "these are probes, and this is how many",
# for the y ticks of section 3, where the tick pairs a probe set with a story set.
PROBE_SET_SHORT = {
    "corpus": "corpus probes (171)",
    "selfgen": "self-gen probes (12)",
    "deepseek": "DeepSeek probes (12)",
}
PROBE_SET_COLORS = {"corpus": "#7f7f7f", "selfgen": "#1f77b4", "deepseek": "#d62728"}
# One marker shape per story set, so a reader can tell which story set a
# dot belongs to even in a monochrome print.
STORY_SET_SYMBOLS = ["circle", "square", "diamond", "triangle-up"]

# The null probe set: 24 fixed-seed random directions (registry name N1). It is
# carried in every arm's shards but carries no per-emotion correspondence, so
# the identity and anticipation reads (both defined on "the tagged emotion's
# own probe") score zero phases against it. ``random_null_lines`` reports that
# emptiness at runtime rather than letting the notebook assume a null band.
# The value is a key inside the evidence JSONs, so it stays byte-identical.
NULL_PROBE_SET = "random"

# (label, evidence file key, arm key inside that file), in the order every
# figure shows the story sets. The v1 file carries the constant-emotion
# control arm alongside its true arm.
STORY_SET_SOURCES = [
    ("Gemma-written stories (v1 probes)", "it", "true_arm"),
    ("DeepSeek-written stories", "deepseek", "true_arm"),
    ("constant-emotion control stories", "it", "control_arm"),
    ("Gemma-written stories (v2 post-fix probes)", "it_v2", "true_arm"),
]
STORY_SET_TICK_LABEL = {
    "Gemma-written stories (v1 probes)": "Gemma-written stories<br>(v1 probe shards)",
    "DeepSeek-written stories": "DeepSeek-written<br>stories",
    "constant-emotion control stories": "constant-emotion<br>control stories",
    "Gemma-written stories (v2 post-fix probes)": "Gemma-written stories<br>(v2 post-fix probes)",
}
# Evidence files, by the key ``STORY_SET_SOURCES`` refers to them by. Only the
# v2 file is optional: it was scored after the first tranche, and a clone
# without it still renders every figure with one column fewer.
EVIDENCE_FILES = {
    "it": ("q3_gate_r1_it.json", True),
    "deepseek": ("q3_gate_r1_deepseek.json", True),
    "it_v2": ("q3_gate_r1_it_v2.json", False),
}


def title_top(height_px: int, pad_px: int = 16) -> float:
    """Normalized y for a top-anchored title that clears the canvas edge.

    Plotly's ``title_y`` is a fraction of figure height measured from the
    bottom, so a value copied between figures of different heights lands a
    different number of pixels from the top and clips the first line. Inputs:
    the figure's ``height``, and how many pixels of clearance the first line
    needs. Returns the ``title_y`` to pair with ``title_yanchor="top"``.
    """
    if height_px <= pad_px:
        raise ValueError(f"figure height {height_px} leaves no room for a title")
    return 1.0 - pad_px / height_px


@dataclass(frozen=True)
class StorySet:
    """One scored story set: what the model was reading, and how it scored.

    Fields: ``label`` (the story set, as every figure and printed line names
    it), ``tick_label`` (the same label pre-wrapped for an axis tick), and
    ``arm`` (that arm's parsed block of the evidence JSON, holding
    ``n_stories``, the identity read under ``gate_G`` and the anticipation
    read under ``r1_anticipation``, each keyed by probe set). The evidence
    files use the older name ``gate_G`` for the identity read.
    """

    label: str
    tick_label: str
    arm: dict[str, Any]

    @property
    def n_stories(self) -> int:
        """How many stories of this set were scored."""
        return int(self.arm["n_stories"])

    def banks(self, read: str) -> list[str]:
        """Probe sets this arm scored on ``read``, in canonical order.

        The public name is pinned because notebook cells call it.

        ``read`` is ``"gate_G"`` (the identity read, under the older name the
        evidence files use) or ``"r1_anticipation"``. A probe set appears only
        if it carries a per-layer grid: the null set (N1) and any set absent
        from the arm's shards are excluded, which is what makes every figure's
        layout follow the evidence instead of a typed shape.
        """
        scored = self.arm[read]
        return [
            probe_set for probe_set in PROBE_SET_ORDER if scored.get(probe_set, {}).get("per_layer")
        ]

    def identity_cell(self, probe_set: str, layer: int) -> dict[str, Any] | None:
        """The identity read at one (probe set, layer), or None if not scored.

        The returned cell holds ``median_rank`` (the tagged emotion's median
        rank within its probe set), ``n2_p`` (the wrong-emotion shuffle p
        value), ``bar_rank`` and ``passes``.
        """
        return self.arm["gate_G"].get(probe_set, {}).get("per_layer", {}).get(str(layer))

    def r1_cell(self, probe_set: str, layer: int) -> dict[str, Any] | None:
        """The anticipation read at one (probe set, layer), or None if not scored.

        The returned cell holds ``mean_lead`` (mean centered-cosine rise in the
        16 tokens before the boundary over the 16 before those), ``noise_sd``
        (that layer's calibrated per-token noise), ``n2_p``, ``bar_lead`` and
        ``passes``.
        """
        return self.arm["r1_anticipation"].get(probe_set, {}).get("per_layer", {}).get(str(layer))

    def lead_in_noise_sd(self, probe_set: str, layer: int) -> float | None:
        """The anticipation lead in units of that layer's noise sd, or None.

        Dividing by the layer's calibrated noise is what lets the six layers
        share one scale: a lead of 1.0 means "one per-token noise sd".
        """
        cell = self.r1_cell(probe_set, layer)
        if cell is None or not cell.get("noise_sd"):
            return None
        return float(cell["mean_lead"]) / float(cell["noise_sd"])

    def null_probe_set_counts(self) -> tuple[int, int]:
        """(phases, transitions) the random-direction null set (N1) scored."""
        # "gate_G" is the identity read's key in the evidence files, kept as-is
        identity_read = self.arm["gate_G"].get(NULL_PROBE_SET, {})
        r1_read = self.arm["r1_anticipation"].get(NULL_PROBE_SET, {})
        return int(identity_read.get("n_phases", 0)), int(r1_read.get("n_transitions", 0))


@dataclass(frozen=True)
class TransitionEvidence:
    """Every scored story set, sharing one layer grid.

    Fields: ``story_sets`` ({label: :class:`StorySet`}, in the order the
    figures show them), ``layers`` (the six extracted layers), and
    ``degradations`` (one printed sentence per optional evidence file that did
    not resolve, so a missing input is announced rather than silently dropping
    a column).
    """

    story_sets: dict[str, StorySet]
    layers: list[int]
    degradations: list[str]

    def story_set(self, label: str) -> StorySet:
        """One story set by label, failing loudly on an unknown label."""
        if label not in self.story_sets:
            raise KeyError(f"no story set loaded for {label!r}; have {list(self.story_sets)}")
        return self.story_sets[label]

    def scored_cells(self, read: str) -> list[tuple[StorySet, str]]:
        """Every (story set, probe set) pair scored on ``read``.

        This is the list every figure's layout is derived from, so a probe set
        that an arm never carried can never become a typed row or column.
        """
        return [
            (story_set, probe_set)
            for story_set in self.story_sets.values()
            for probe_set in story_set.banks(read)
        ]


def _check_layer_grid(label: str, arm: dict[str, Any]) -> None:
    """Raise unless every scored probe set of ``arm`` covers exactly ``LAYERS``."""
    expected = [str(layer) for layer in LAYERS]
    # "gate_G" is the identity read's key in the evidence files, kept as-is
    for read in ("gate_G", "r1_anticipation"):
        for probe_set, probe_set_result in arm[read].items():
            grid = probe_set_result.get("per_layer")
            if grid and list(grid) != expected:
                raise ValueError(
                    f"{label}: {read} probe set {probe_set!r} covers layers"
                    f" {list(grid)}, expected {expected}"
                )


def load_transition_evidence() -> TransitionEvidence:
    """Resolve the identity / anticipation evidence files and assemble the arms.

    Returns the :class:`TransitionEvidence` every section reads from. The two
    required files raise if absent; the v2 file is optional and its absence is
    recorded in ``degradations`` for the notebook cell to print. Raises if any
    scored probe set covers a layer grid other than the six extracted layers,
    since every figure indexes its cells by those layers.
    """
    parsed: dict[str, dict[str, Any]] = {}
    degradations: list[str] = []
    for key, (filename, required) in EVIDENCE_FILES.items():
        try:
            parsed[key] = json.loads(fetch(filename).read_text())
        except FileNotFoundError:
            if required:
                raise
            degradations.append(
                f"{filename} did not resolve: every figure below is drawn with one story"
                " set fewer, and the design grid marks its column absent"
            )
    story_sets: dict[str, StorySet] = {}
    for label, file_key, arm_key in STORY_SET_SOURCES:
        if file_key not in parsed:
            continue
        arm = parsed[file_key][arm_key]
        _check_layer_grid(label, arm)
        story_sets[label] = StorySet(label, STORY_SET_TICK_LABEL[label], arm)
    if not story_sets:
        raise ValueError("no story set resolved; the notebook has nothing to report")
    return TransitionEvidence(story_sets=story_sets, layers=list(LAYERS), degradations=degradations)


def random_null_lines(evidence: TransitionEvidence) -> list[str]:
    """What the random-direction null (N1) actually scored, per story set.

    The registered plan ran every statistic on 24 random directions as the
    meaninglessness floor. Both reads are defined on "the rank / the lead of
    the phase's OWN tagged emotion", and a random direction has no emotion
    attached, so the scorer kept zero phases for that set. These lines print
    the counts so the notebook states the honest null instead of assuming a
    null band was measured.
    """
    lines = []
    for story_set in evidence.story_sets.values():
        n_phases, n_transitions = story_set.null_probe_set_counts()
        lines.append(
            f"{story_set.label:44s} random-direction null (N1):"
            f" {n_phases} phases and {n_transitions} transitions scored"
        )
    total = sum(sum(each.null_probe_set_counts()) for each in evidence.story_sets.values())
    lines.append(
        f"N1 total across all {len(evidence.story_sets)} story sets: {total} scored units."
        " The meaninglessness floor in this tranche therefore rests on the wrong-emotion"
        " shuffle (N2) alone, which IS measured at every cell."
    )
    return lines
