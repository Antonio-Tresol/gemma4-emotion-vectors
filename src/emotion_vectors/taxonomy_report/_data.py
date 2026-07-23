"""Loaders, shared constants, and the core data helpers for the notebook-11
exhibits: record-dump resolution, bank slicing, the gate and R1 reads, the
cluster bootstrap, and the house layer-slider pattern."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int, Num

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.artifacts import fetch

# The six extracted layers; PRIMARY_LAYER is the registered primary cell.
LAYERS = [6, 15, 24, 33, 42, 51]
PRIMARY_LAYER = 33
# Bootstrap convention shared by every CI in the notebook: one generator
# seeded with SEED, N_BOOT cluster resamples over triple_id per interval.
SEED = 20260723
N_BOOT = 2000
# The five designed triple families (S2's x axis and every per-family split).
FAMILIES = [
    "A_superposition",
    "B_conflict",
    "D_timescale",
    "E_arousal_mismatch",
    "F_valence_spread",
]
# S7 phase classes: first-third vs last-third correctness of the tagged probe.
S7_CATEGORIES = ["always right", "converges", "loses it", "never right"]

# |delta valence| bin edges for the S3 binned read (NRC valence scale).
S3_VALENCE_GAP_EDGES = np.array([0.0, 0.4, 0.8, 1.2, 2.0])

# One arm's record dump: score arrays plus the phase/transition metadata rows.
Arm = dict[str, object]
Arms = dict[str, Arm]
# One metadata row (phase or transition) as dumped: parsed JSON, honestly untyped.
Row = dict[str, object]
Vad = dict[str, tuple[float, float]]
ConfidenceInterval = tuple[float, float]
# Per-layer stats dict: {layer: {stat name: value}}.
LayerStats = dict[int, dict[str, object]]
# The gate read for one (arm, bank): tagged-emotion ranks, winner probe
# positions (both [kept_phases, layers]), and the kept phase metadata rows.
GateRead = tuple[
    Int[np.ndarray, "kept_phases layers"],
    Int[np.ndarray, "kept_phases layers"],
    list[Row],
]
# The R1 read for one (arm, bank): incoming-emotion leads and the kept
# transition metadata rows.
LeadRead = tuple[Float[np.ndarray, "kept_trans layers"], list[Row]]


def repo_root() -> Path:
    """Checkout root, found by walking up from the current directory.

    Repo-relative inputs (the VAD lexicon, the designed triples file) resolve
    the same way from the repo root and from notebooks/, with no machine
    absolute paths anywhere.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {here}")


def load_arms(arm_names: list[str]) -> Arms:
    """Per-record dumps for the given arms, via the artifact resolver.

    Each entry carries the arrays exactly as ``scripts/dump_q3_records.py``
    wrote them: ``phase_scores`` [phase_records, layers, probes],
    ``phase_thirds`` [phase_records, 3, layers, probes], ``trans_lead``
    [transition_records, layers, probes], plus the metadata row lists
    (``phases``, ``transitions``) and the ``labels`` probe-name list
    ("bank:emotion" strings, one per probe column).
    """
    arms: Arms = {}
    for arm in arm_names:
        score_arrays = np.load(fetch(f"q3_records_{arm}.npz"))
        meta = json.loads(fetch(f"q3_records_{arm}_meta.json").read_text())
        arms[arm] = {
            "phase_scores": score_arrays["phase_scores"],
            "phase_thirds": score_arrays["phase_third_scores"],
            "trans_lead": score_arrays["trans_lead"],
            "phases": meta["phase_records"],
            "transitions": meta["transition_records"],
            "labels": meta["probe_labels"],
        }
    return arms


def load_vad() -> Vad:
    """NRC VAD lexicon: ``{term: (valence, arousal)}``, both in [-1, 1].

    Third-party license, so it is deliberately NOT fetchable (see
    ``emotion_vectors.artifacts``): data/lexicons is populated manually.
    """
    lexicon = repo_root() / "data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
    return load_nrc_vad(lexicon)


def load_has_nonaffect() -> dict[int, bool]:
    """``triple_id -> has_nonaffect``, joined from the designed triples file."""
    triples_file = repo_root() / "scripts/combined_story_gen/emotions_triples_v1.json"
    triples = json.loads(triples_file.read_text())
    return {triple_id: triple["has_nonaffect"] for triple_id, triple in enumerate(triples)}


def bank_columns(arms: Arms, arm: str, bank: str) -> tuple[list[int], list[str]]:
    """Probe columns belonging to one bank, plus their emotion names."""
    labels = arms[arm]["labels"]
    columns = [pos for pos, label in enumerate(labels) if label.startswith(bank + ":")]
    names = [labels[pos].split(":", 1)[1] for pos in columns]
    return columns, names


def gate_ranks(arms: Arms, arm: str, bank: str) -> GateRead:
    """The gate read: rank of the tagged emotion within its bank, per phase.

    Keeps the phases whose tagged emotion has a probe in the bank. Returns
    (ranks [kept_phases, layers] with 1 = the tagged probe scored highest,
    winner probe position [kept_phases, layers], kept phase metadata rows).
    """
    columns, names = bank_columns(arms, arm, bank)
    phase_records = arms[arm]["phases"]
    kept = [pos for pos, row in enumerate(phase_records) if row["emotion"] in names]
    # scores: [kept_phases, layers, bank_probes] phase-mean centered cosines
    scores = arms[arm]["phase_scores"][kept][:, :, columns]
    target = np.array([names.index(phase_records[pos]["emotion"]) for pos in kept])
    # rank = 1 + how many bank probes out-score the tagged probe
    target_score = scores[np.arange(len(kept)), :, target]
    ranks = (scores > target_score[:, :, None]).sum(axis=2) + 1
    winner = scores.argmax(axis=2)
    kept_rows = [phase_records[pos] for pos in kept]
    return ranks, winner, kept_rows


def true_leads(arms: Arms, arm: str, bank: str) -> LeadRead:
    """The R1 read: anticipation lead of the incoming emotion, per transition.

    Keeps the transitions whose incoming emotion has a probe in the bank.
    Returns (leads [kept_transitions, layers] in centered-cosine units,
    kept transition metadata rows).
    """
    columns, names = bank_columns(arms, arm, bank)
    transition_records = arms[arm]["transitions"]
    kept = [pos for pos, row in enumerate(transition_records) if row["to_emotion"] in names]
    leads = arms[arm]["trans_lead"][kept][:, :, columns]
    target = np.array([names.index(transition_records[pos]["to_emotion"]) for pos in kept])
    kept_rows = [transition_records[pos] for pos in kept]
    return leads[np.arange(len(kept)), :, target], kept_rows


def cluster_boot_ci(
    values: Num[np.ndarray, " values"],
    triple_ids: list[int],
    rng: np.random.Generator,
    stat: Callable[[Num[np.ndarray, " resample"]], float] = np.mean,
) -> ConfidenceInterval:
    """95% CI for stat(values) from a cluster bootstrap over triple_id.

    Whole triples are resampled (with replacement) rather than individual
    values, because stories from one designed triple share a text lineage.
    Consumes exactly one ``rng.integers`` draw per resample, so CI values
    depend on the generator's position; the notebook threads one generator
    through the sections in a fixed order.
    """
    by_triple: dict[int, list[int]] = {}
    for value_pos, triple_id in enumerate(triple_ids):
        by_triple.setdefault(triple_id, []).append(value_pos)
    clusters = list(by_triple.values())
    resample_stats = []
    for _ in range(N_BOOT):
        picked_clusters = rng.integers(0, len(clusters), size=len(clusters))
        resample_idx = np.concatenate([clusters[pick] for pick in picked_clusters])
        resample_stats.append(stat(values[resample_idx]))
    return float(np.percentile(resample_stats, 2.5)), float(np.percentile(resample_stats, 97.5))


def layer_slider(fig: go.Figure, traces_per_layer: int, prefix: str = "layer ") -> go.Figure:
    """Show one layer's trace block at a time via a slider (house pattern).

    Assumes traces were added layer-major: ``traces_per_layer`` consecutive
    traces per layer, in LAYERS order. Defaults the slider to PRIMARY_LAYER.
    """
    n_traces = len(fig.data)
    assert n_traces == traces_per_layer * len(LAYERS), (n_traces, traces_per_layer)
    for trace_pos, trace in enumerate(fig.data):
        trace.visible = trace_pos // traces_per_layer == LAYERS.index(PRIMARY_LAYER)
    steps = []
    for layer_pos, layer in enumerate(LAYERS):
        visible = [pos // traces_per_layer == layer_pos for pos in range(n_traces)]
        steps.append(dict(method="update", args=[{"visible": visible}], label=str(layer)))
    fig.update_layout(
        sliders=[
            dict(
                active=LAYERS.index(PRIMARY_LAYER),
                steps=steps,
                currentvalue={"prefix": prefix},
                pad={"t": 40},
            )
        ]
    )
    return fig
