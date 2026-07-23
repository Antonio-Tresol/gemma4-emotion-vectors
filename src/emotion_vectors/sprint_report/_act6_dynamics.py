"""Act VI exhibits: the Q3 factorial heatmap pair and the E2 addendum lines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots

from ..analysis import load_nrc_vad

Stats = dict[str, Any]
RowSources = dict[str, tuple[Mapping, str]]

# the six layers the Q3 trajectory arms were extracted at
LAYERS_Q3 = [6, 15, 24, 33, 42, 51]
# the registered primary cell of the Q3 factorial
PRIMARY_LAYER_Q3 = 33
BANKS = ["corpus", "selfgen", "deepseek"]
BANK_SIZE = {"corpus": 171, "selfgen": 12, "deepseek": 12}
BANK_LABEL = {
    "corpus": "corpus-171 probes",
    "selfgen": "self-gen probes (12)",
    "deepseek": "DeepSeek probes (12)",
}


def _factorial_cell(source: Mapping, arm: str, read: str, bank: str, layer: int) -> Mapping | None:
    """One (arm, read, bank, layer) cell of a Q3 gate/R1 evidence file, or None."""
    return source[arm][read].get(bank, {}).get("per_layer", {}).get(str(layer))


def _act6_lines(row_sources: RowSources, base_arm: Mapping, calibration: Mapping) -> list[str]:
    """The Act VI printout: the layer-33 grid, the base-reader per-layer summary
    (backing the Act VI markdown's numbers), and the noise floor."""
    lines = []
    for label, (source, arm) in row_sources.items():
        parts = []
        for bank in BANKS:
            gate_cell = _factorial_cell(source, arm, "gate_G", bank, PRIMARY_LAYER_Q3)
            lead_cell = _factorial_cell(source, arm, "r1_anticipation", bank, PRIMARY_LAYER_Q3)
            if gate_cell:
                lead_sd = (
                    (lead_cell["mean_lead"] / lead_cell["noise_sd"])
                    if lead_cell and lead_cell.get("noise_sd")
                    else float("nan")
                )
                parts.append(f"{bank}: rank {gate_cell['median_rank']:g}, lead {lead_sd:+.1f} sd")
        lines.append(f"{label.replace('<br>', ' '):50s} " + " | ".join(parts))
    base_gate = base_arm["true_arm"]["gate_G"]["corpus"]["per_layer"]
    base_lead = base_arm["true_arm"]["r1_anticipation"]["corpus"]["per_layer"]
    lines.append(
        "base reader, corpus-171 bank, median rank by layer: "
        + ", ".join(f"L{layer} {base_gate[str(layer)]['median_rank']:g}" for layer in LAYERS_Q3)
    )
    base_lead_sds = [
        base_lead[str(layer)]["mean_lead"] / base_lead[str(layer)]["noise_sd"]
        for layer in LAYERS_Q3
    ]
    base_worst_p = max(base_lead[str(layer)]["n2_p"] for layer in LAYERS_Q3)
    lines += [
        f"base reader anticipation lead: {min(base_lead_sds):.1f}-{max(base_lead_sds):.1f}x "
        f"noise sd, passes at all {len(LAYERS_Q3)} layers (worst shuffle p = {base_worst_p:.4f})",
        f"noise floor at layer 33 (instruct-arm calibration): "
        f"{calibration['random_probe']['33']['cos_sd']:.4f} cosine units",
    ]
    return lines


def _act6_grids(row_sources: RowSources, layer: int) -> tuple[list, list, list, list]:
    """Identity z, anticipation z, and their star-marked cell labels at one layer."""
    identity_grid, anticipation_grid, identity_labels, anticipation_labels = [], [], [], []
    for _, (source, arm) in row_sources.items():
        identity_row, anticipation_row, identity_text, anticipation_text = [], [], [], []
        for bank in BANKS:
            gate_cell = _factorial_cell(source, arm, "gate_G", bank, layer)
            lead_cell = _factorial_cell(source, arm, "r1_anticipation", bank, layer)
            if gate_cell:
                star = " *" if gate_cell.get("passes") else ""
                identity_row.append(gate_cell["median_rank"] / BANK_SIZE[bank])
                identity_text.append(f"{gate_cell['median_rank']:g}/{BANK_SIZE[bank]}{star}")
            else:
                identity_row.append(None)
                identity_text.append("n/a")
            if lead_cell and lead_cell.get("noise_sd"):
                lead_sd = lead_cell["mean_lead"] / lead_cell["noise_sd"]
                star = " *" if lead_cell.get("passes") else ""
                anticipation_row.append(lead_sd)
                anticipation_text.append(f"{lead_sd:+.1f}{star}")
            else:
                anticipation_row.append(None)
                anticipation_text.append("n/a")
        identity_grid.append(identity_row)
        anticipation_grid.append(anticipation_row)
        identity_labels.append(identity_text)
        anticipation_labels.append(anticipation_text)
    return identity_grid, anticipation_grid, identity_labels, anticipation_labels


def _act6_title(
    row_sources: RowSources, gemma_v1: Mapping, deepseek_arm: Mapping, base_arm: Mapping, layer: int
) -> str:
    """The figure title, recomputed for the slider's layer so verdicts never go stale."""
    small_bank_ranks, it_corpus_ranks = [], []
    for label, (source, arm) in row_sources.items():
        if "BASE" in label:
            continue
        for bank in ("selfgen", "deepseek"):
            gate_cell = _factorial_cell(source, arm, "gate_G", bank, layer)
            if gate_cell:
                small_bank_ranks.append(gate_cell["median_rank"])
        gate_cell = _factorial_cell(source, arm, "gate_G", "corpus", layer)
        if gate_cell:
            it_corpus_ranks.append(gate_cell["median_rank"])
    base_rank = base_arm["true_arm"]["gate_G"]["corpus"]["per_layer"][str(layer)]["median_rank"]
    gemma_lead_cell = _factorial_cell(gemma_v1, "true_arm", "r1_anticipation", "selfgen", layer)
    gemma_lead = gemma_lead_cell["mean_lead"] / gemma_lead_cell["noise_sd"]
    deepseek_best = max(
        cell["mean_lead"] / cell["noise_sd"]
        for bank in BANKS
        if (cell := _factorial_cell(deepseek_arm, "true_arm", "r1_anticipation", bank, layer))
        and cell.get("noise_sd")
    )
    return (
        "Does the model track and anticipate what it reads? Identity follows the "
        "probe bank; anticipation only on Gemma-written stories<br>"
        f"<sup>layer {layer}: 12-probe banks rank "
        f"{min(small_bank_ranks):g}-{max(small_bank_ranks):g} "
        f"of 12 vs corpus-171 rank {min(it_corpus_ranks):g}-{max(it_corpus_ranks):g} of 171 "
        f"(instruct reader) and {base_rank:g} of 171 (base reader); self-gen lead "
        f"{gemma_lead:+.1f}x noise on Gemma-written vs best {deepseek_best:+.1f}x on "
        "DeepSeek-written</sup><br>"
        "<sup>one cell = one (story set + reader, probe bank); every row is the instruct "
        "reader except the marked base-reader row; * = passes its registered bar | evidence: "
        "q3_gate_r1_(it, deepseek, it_v2, base).json</sup>"
    )


def _act6_add_heatmaps(fig: go.Figure, row_sources: RowSources) -> None:
    """One (identity, anticipation) heatmap pair per layer; only layer 33 starts
    visible, the slider toggles the rest. Both encodings get titled colorbars."""
    for layer in LAYERS_Q3:
        identity_grid, anticipation_grid, identity_labels, anticipation_labels = _act6_grids(
            row_sources, layer
        )
        visible = layer == PRIMARY_LAYER_Q3
        fig.add_trace(
            go.Heatmap(
                z=identity_grid,
                x=[BANK_LABEL[bank] for bank in BANKS],
                y=list(row_sources),
                text=identity_labels,
                texttemplate="%{text}",
                colorscale="Blues_r",
                zmin=0,
                zmax=0.6,
                colorbar=dict(
                    title=dict(text="rank / bank size", side="right"),
                    x=0.42,
                    len=0.75,
                    thickness=12,
                ),
                visible=visible,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=anticipation_grid,
                x=[BANK_LABEL[bank] for bank in BANKS],
                y=list(row_sources),
                text=anticipation_labels,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(
                    title=dict(text="lead (noise-sd)", side="right"),
                    x=1.0,
                    len=0.75,
                    thickness=12,
                ),
                visible=visible,
            ),
            row=1,
            col=2,
        )


def act6_factorial_figure(
    gemma_v1: Mapping,
    deepseek_arm: Mapping,
    gemma_v2: Mapping,
    base_arm: Mapping,
    calibration: Mapping,
) -> tuple[go.Figure, Stats]:
    """Act VI: the factorial heatmap pair (identity / anticipation) with layer slider.

    Inputs are the four Q3 evidence files (``q3_gate_r1_it.json`` supplies both
    the Gemma-written v1 row and the control row, ``q3_gate_r1_deepseek.json``,
    ``q3_gate_r1_it_v2.json``, ``q3_gate_r1_base.json``) plus the instruct-arm
    instrument calibration. Every slider step retitles the figure with that
    layer's own verdict numbers. Returns the figure and stats with the printed
    ``lines`` (the layer-33 grid, the base-reader per-layer summary, and the
    noise floor).
    """
    row_sources: RowSources = {
        "Gemma-written (v1 probes)": (gemma_v1, "true_arm"),
        "DeepSeek-written stories": (deepseek_arm, "true_arm"),
        "constant-emotion control": (gemma_v1, "control_arm"),
        "Gemma-written (v2 post-fix probes)": (gemma_v2, "true_arm"),
        "Gemma-written, BASE reader<br>(base-lineage corpus probes)": (base_arm, "true_arm"),
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.16,
        subplot_titles=(
            "identity (gate G): rank / bank size<br>lower + darker = better; * = passes bar",
            "anticipation (R1): pre-boundary lead, noise-sd units<br>"
            "blue positive = rising early; * = passes bar",
        ),
    )
    _act6_add_heatmaps(fig, row_sources)
    steps = []
    for layer_pos, layer in enumerate(LAYERS_Q3):
        visibility = [pos == layer_pos for pos in range(len(LAYERS_Q3)) for _ in (0, 1)]
        title = _act6_title(row_sources, gemma_v1, deepseek_arm, base_arm, layer)
        steps.append(
            dict(
                method="update",
                label=f"layer {layer}",
                args=[{"visible": visibility}, {"title.text": title}],
            )
        )
    fig.update_yaxes(autorange="reversed")  # first row_sources row on top
    for annotation in fig.layout.annotations:  # only the panel titles exist at this point
        annotation.font = dict(size=12)
    fig.update_layout(
        sliders=[
            dict(
                active=LAYERS_Q3.index(PRIMARY_LAYER_Q3),
                steps=steps,
                y=-0.30,
                currentvalue=dict(prefix="showing: "),
            )
        ],
        title=_act6_title(row_sources, gemma_v1, deepseek_arm, base_arm, PRIMARY_LAYER_Q3),
        title_font_size=14,
        width=1250,
        height=540,
        margin=dict(t=130, b=140),
    )
    base_gate = base_arm["true_arm"]["gate_G"]["corpus"]["per_layer"]
    base_lead = base_arm["true_arm"]["r1_anticipation"]["corpus"]["per_layer"]
    stats = {
        "lines": _act6_lines(row_sources, base_arm, calibration),
        "base_gate_ranks": {layer: base_gate[str(layer)]["median_rank"] for layer in LAYERS_Q3},
        "base_lead_sds": {
            layer: base_lead[str(layer)]["mean_lead"] / base_lead[str(layer)]["noise_sd"]
            for layer in LAYERS_Q3
        },
    }
    return fig, stats


# --------------------------------------------------------------------------- E2 addendum


def _vad_confusion_line(
    phase_scores: Float[np.ndarray, "phase selfgen_probe"],
    tagged: list[str],
    selfgen_emotions: list[str],
    vad_lexicon_path: Path,
) -> str:
    """The confusion-structure read: on wrongly-won phases, is the winning probe
    closer to the target in valence-arousal (VAD) space than a random wrong probe?

    ``phase_scores`` axes: (phase, selfgen_probe), mean centered cosines at the
    primary layer. Random draws use the fixed seed 20260723 so the printed
    numbers reproduce exactly. Raises FileNotFoundError when the third-party
    NRC VAD lexicon is absent (the caller prints the degradation instead).
    """
    vad = load_nrc_vad(vad_lexicon_path)
    rng = np.random.default_rng(20260723)
    dist_winner, dist_random = [], []
    for emotion_pos, emotion in enumerate(selfgen_emotions):
        for row in (phase_row for phase_row, tag in enumerate(tagged) if tag == emotion):
            winner_pos = int(np.argmax(phase_scores[row]))
            if winner_pos == emotion_pos:
                continue
            random_pos = rng.choice(
                [pos for pos in range(len(selfgen_emotions)) if pos != emotion_pos]
            )
            target_valence, target_arousal = vad[emotion]
            for store, pos in ((dist_winner, winner_pos), (dist_random, random_pos)):
                valence, arousal = vad[selfgen_emotions[pos]]
                store.append(float(np.hypot(valence - target_valence, arousal - target_arousal)))
    return (
        f"wrong-winner VAD distance to target {np.mean(dist_winner):.2f} "
        f"vs random wrong probe {np.mean(dist_random):.2f} "
        f"(n={len(dist_winner)} wrong phases)"
    )


def act6_addendum_lines(records: Mapping, records_meta: Mapping, vad_lexicon_path: Path) -> Stats:
    """Act VI addendum (E2): per-emotion top-1 rates, family leads, confusion structure.

    ``records`` is the loaded ``q3_records_it_v2.npz`` per-record score dump
    (``phase_scores`` axes: (phase, layer, probe) mean centered cosines;
    ``trans_lead`` axes: (transition, layer, probe) anticipation leads) and
    ``records_meta`` its sidecar. The confusion-structure read needs the NRC
    VAD lexicon at ``vad_lexicon_path``; when the file is absent the returned
    lines end with the printed degradation notice instead (explicit, never a
    quiet skip).
    """
    layer_pos_33 = records_meta["layers"].index(33)
    probe_labels = records_meta["probe_labels"]
    selfgen_cols = [pos for pos, label in enumerate(probe_labels) if label.startswith("selfgen:")]
    selfgen_emotions = [probe_labels[pos].split(":", 1)[1] for pos in selfgen_cols]

    # per-emotion top-1 rate over phases tagged with a battery emotion; slice
    # (phase, layer, probe) down to (phase, selfgen_probe) at the primary layer
    phase_scores = records["phase_scores"][:, layer_pos_33, :][:, selfgen_cols]
    tagged = [record["emotion"] for record in records_meta["phase_records"]]
    top1 = {}
    for emotion_pos, emotion in enumerate(selfgen_emotions):
        rows = [row for row, tag in enumerate(tagged) if tag == emotion]
        wins = sum(int(np.argmax(phase_scores[row]) == emotion_pos) for row in rows)
        top1[emotion] = wins / len(rows)
    ranked = sorted(top1, key=lambda emotion: -top1[emotion])
    lines = [
        "top-1 rate, layer 33: best "
        + ", ".join(f"{emotion} {top1[emotion]:.2f}" for emotion in ranked[:3])
        + " | worst "
        + ", ".join(f"{emotion} {top1[emotion]:.2f}" for emotion in ranked[-3:])
    ]

    # family mean anticipation lead, transitions into battery emotions;
    # trans_lead axes: (transition, layer, probe), sliced at the primary layer
    lead33 = records["trans_lead"][:, layer_pos_33, :]
    family_leads: dict[str, list[float]] = {}
    for row, transition in enumerate(records_meta["transition_records"]):
        if transition["to_emotion"] in selfgen_emotions:
            col = selfgen_cols[selfgen_emotions.index(transition["to_emotion"])]
            family_leads.setdefault(transition["category"], []).append(lead33[row, col])
    for family in sorted(family_leads, key=lambda fam: -np.mean(family_leads[fam])):
        values = family_leads[family]
        lines.append(f"  {family:20s} mean lead {np.mean(values):+.4f} (n={len(values)})")

    # confusion structure in valence-arousal space, guarded on the third-party lexicon
    try:
        lines.append(_vad_confusion_line(phase_scores, tagged, selfgen_emotions, vad_lexicon_path))
    except FileNotFoundError:
        lines.append("NRC VAD lexicon not present; confusion-structure numbers are in notebook 11")
    return {"lines": lines, "top1": top1, "family_leads": family_leads}
