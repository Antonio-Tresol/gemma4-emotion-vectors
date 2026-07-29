"""E11 exhibits: corpus samples, lexical diversity, detection, and geometry.

One builder per notebook-07 section. Each figure builder returns
``(figure, stats)`` where ``stats["lines"]`` holds every line the notebook
prints, so no number appears in markdown without being computed beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float

from ._data import (
    CHANCE_CORRECT,
    GEOMETRY_LAYER,
    N_EMOTIONS,
    PASS_BAR,
    PROBED_MODEL,
    REQUESTED_PER_EMOTION,
    STORY_SOURCE_COLORS,
    STORY_SOURCE_LABELS,
    STORY_SOURCE_ORDER,
    STORY_SOURCE_TICK_LABELS,
    TOP_K,
    StorySourceEvidence,
    figure_title,
    per_layer_counts,
)

# story-source key -> (evidence attribute, name of its row inside that R1
# grid); the fixed-DeepSeek row is called "strong_external" in e11_lineage.json
R1_SOURCES = {
    "selfgen": ("e11", "selfgen"),
    "weak_external": ("e11", "weak_external"),
    "fixed_deepseek": ("e11", "strong_external"),
    "diverse_deepseek": ("full_grid", "diverse_deepseek_n1024"),
}
SAMPLE_EMOTION = "sad"
SAMPLE_CHARS = 500
HOUSE_PROTAGONIST = "Elias"  # the self-generated corpus's recurring character


def corpus_overview(corpora: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Section 1 stats (no figure): sizes, one sample story each, Elias share.

    Input: ``load_corpora()`` output. Returns ``stats`` with:
      lines               -- the printed record: requested against kept
                             stories per corpus (with the per-emotion
                             minimum, which is where the dose-response
                             curves stop), the first "sad" story of each
                             corpus truncated to 500 characters, and the
                             share of self-generated stories naming a
                             character Elias
      story_counts        -- corpus label -> kept stories in total
      per_emotion_minimum -- corpus label -> smallest per-emotion count
      elias_share         -- fraction of self-generated stories naming Elias
    """
    story_counts = {
        label: sum(len(stories) for stories in corpus.values()) for label, corpus in corpora.items()
    }
    # requested against kept: the filters drop a handful of stories, so the
    # smallest per-emotion count (not the requested round number) is the
    # largest corpus size the dose-response subsampling can reach
    per_emotion_minimum = {
        label: min(len(stories) for stories in corpus.values()) for label, corpus in corpora.items()
    }
    lines = ["stories per corpus (requested by the recipe -> kept after filtering):"]
    for label, corpus in corpora.items():
        if len(corpus) != N_EMOTIONS:
            raise ValueError(f"{label}: {len(corpus)} emotions, expected the {N_EMOTIONS} emotions")
        requested = REQUESTED_PER_EMOTION[label]
        lines.append(
            f"  {label}: {requested * len(corpus)} requested"
            f" ({requested} x {len(corpus)} emotions) -> {story_counts[label]} kept;"
            f" smallest per-emotion count {per_emotion_minimum[label]}"
        )
    for label, corpus in corpora.items():
        if SAMPLE_EMOTION not in corpus:
            raise KeyError(f"{label}: no '{SAMPLE_EMOTION}' stories to sample")
        lines.append("")
        lines.append(
            f"=== {label} | assigned emotion: {SAMPLE_EMOTION} (first story, unpicked) ==="
        )
        lines.append(corpus[SAMPLE_EMOTION][0][:SAMPLE_CHARS])
    selfgen_label = next(label for label in corpora if label.startswith("self-generated"))
    selfgen_stories = [story for stories in corpora[selfgen_label].values() for story in stories]
    elias_share = sum(HOUSE_PROTAGONIST in story for story in selfgen_stories) / len(
        selfgen_stories
    )
    lines.append("")
    lines.append(
        f"self-generated stories naming a character {HOUSE_PROTAGONIST}: "
        f"{elias_share:.1%} of {len(selfgen_stories)}"
    )
    return {
        "lines": lines,
        "story_counts": story_counts,
        "per_emotion_minimum": per_emotion_minimum,
        "elias_share": elias_share,
    }


def _add_diversity_anchors(
    fig: go.Figure,
    fixed_overlap: float,
    values: list[float | None],
    unmeasured_label: str,
) -> None:
    """Add the grading scale to the phrase-repetition bars.

    0 (the bottom axis) is maximal phrase diversity and the fixed-DeepSeek bar
    is the measured diversity floor. Both labels go in the right margin,
    because everything near the bottom of the plot is already occupied by the
    near-zero bar and its value label. The unmeasured arm is named in place so
    an absent bar cannot be misread as zero overlap.
    """
    fig.add_hline(y=fixed_overlap, line_dash="dot")
    for anchor_y, anchor_text in (
        (
            fixed_overlap,
            f"measured diversity floor:<br>fixed DeepSeek, {fixed_overlap:.4f}",
        ),
        (
            max(value for value in values if value is not None),
            "higher = the corpus keeps<br>reusing the same phrasing",
        ),
    ):
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=anchor_y,
            text=anchor_text,
            showarrow=False,
            align="left",
            xanchor="left",
            font=dict(size=11, color="#444444"),
        )
    fig.add_annotation(
        x=unmeasured_label,
        y=0,
        yshift=60,
        text="not measured<br>(R4 covered the E11 arms only)",
        showarrow=False,
        font=dict(size=11, color="#666666"),
    )


def diversity_figure(evidence: StorySourceEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 2 exhibit: is phrase repetition what makes probes work or fail?

    Bars the exploratory R4 read (mean pairwise 5-gram Jaccard overlap
    within each measured corpus) from e11_lineage.json. Returns
    ``(figure, stats)``; ``stats["lines"]`` prints the overlap ratio and
    ``stats["overlap"]`` maps story source -> overlap (None where unmeasured).
    """
    r4 = evidence.e11["r4_diversity_EXPLORATORY"]
    measured = ["selfgen", "weak_external", "strong_external"]
    display = {  # e11 slot name -> canonical story-source key
        "selfgen": "selfgen",
        "weak_external": "weak_external",
        "strong_external": "fixed_deepseek",
    }
    overlap: dict[str, float | None] = {}
    for slot in measured:
        slot_stats = r4.get(slot)
        overlap[display[slot]] = slot_stats["mean_pairwise_jaccard"] if slot_stats else None
    selfgen_overlap = overlap["selfgen"]
    fixed_overlap = overlap["fixed_deepseek"]
    if selfgen_overlap is None or fixed_overlap is None:
        raise ValueError("R4 must be measured on selfgen and strong_external to plot the ratio")
    ratio = selfgen_overlap / fixed_overlap

    keys = ["selfgen", "weak_external", "fixed_deepseek"]
    labels = [STORY_SOURCE_TICK_LABELS[key] for key in keys]
    values = [overlap[key] for key in keys]
    width_px = 900
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[STORY_SOURCE_COLORS[key] for key in keys],
            # the fixed-DeepSeek bar is ~53x shorter, so its own value has to
            # be written next to it or it reads as a missing bar
            text=[f"{value:.4f}" if value is not None else "" for value in values],
            textposition="outside",
            showlegend=False,
        )
    )
    _add_diversity_anchors(fig, fixed_overlap, values, labels[keys.index("weak_external")])
    fig.update_layout(
        title=figure_title(
            "Is phrase repetition what separates working probes from failing ones? No: the"
            f" self-generated corpus repeats {ratio:.0f}x more 5-word phrases than fixed DeepSeek"
            f" ({selfgen_overlap:.4f} vs {fixed_overlap:.4f}), yet both story sets yield working"
            " probes",
            [
                "one bar = mean pairwise 5-gram Jaccard overlap between stories within one"
                " corpus; 0 (the bottom axis) = no two stories share any 5-word phrase, which"
                " is maximal diversity, and higher means more repetitive",
                "exploratory R4 read, at most 30 stories per emotion sampled; probes read from"
                f" {PROBED_MODEL}; evidence: e11_lineage.json (the diverse arm was not"
                " measured)",
            ],
            width_px,
        ),
        title_font_size=13,
        yaxis_title="mean pairwise 5-gram Jaccard overlap (0 = fully diverse)",
        xaxis_title="story source (who wrote the stories)",
        width=width_px,
        height=560,
        # right margin holds the two grading-anchor labels
        margin=dict(t=190, r=220),
    )
    lines = [
        f"self-generated overlap {selfgen_overlap:.4f} is {ratio:.0f}x fixed DeepSeek's"
        f" {fixed_overlap:.4f}; weak external not measured"
    ]
    return fig, {"lines": lines, "overlap": overlap, "ratio": ratio}


DETECTION_WIDTH_PX = 1150


def _detection_title(pass_counts: dict[str, int], n_layers: int, view_label: str) -> str:
    """Per-slider-step title: the (view-independent) verdict plus which set of
    scenarios the color currently shows."""
    return figure_title(
        "Whose stories give probes that detect at the most layers? Fixed DeepSeek:"
        f" {pass_counts['fixed_deepseek']} of {n_layers} layers pass, vs diverse"
        f" {pass_counts['diverse_deepseek']}, self-generated {pass_counts['selfgen']}, weak"
        f" external {pass_counts['weak_external']}",
        [
            f"one cell = scenarios placed correctly, of 12; the color shows the {view_label},"
            " the cell text shows both counts as paper/held-out, and * marks a cell clearing"
            f" the mark we fixed before scoring (at least {PASS_BAR} correct on BOTH sets of"
            " scenarios)",
            f"failure anchor: ranking the target in the top {TOP_K} of 12 emotions by luck"
            f" gives {CHANCE_CORRECT:.0f} of 12 correct; strength anchor: the {PASS_BAR}-of-12"
            f" mark fixed in advance; probes read from {PROBED_MODEL}; evidence:"
            " e11_lineage.json and e12_diverse_fullcorpus_grid.json",
        ],
        DETECTION_WIDTH_PX,
    )


def _add_detection_heatmaps(
    fig: go.Figure,
    views: list[tuple[str, Any]],
    counts: dict[str, dict[int, tuple[int, int]]],
    layers: list[int],
    text: list[list[str]],
    row_labels: dict[str, str],
) -> None:
    """Add one heatmap trace per scenario-set view (only the first is visible).

    All three carry the same cell text, so switching views recolors the grid
    without changing the counts a reader is checking.
    """
    for view_pos, (_view_label, reduce_pair) in enumerate(views):
        z: Float[np.ndarray, "story_sources layers"] = np.array(
            [[reduce_pair(*counts[key][layer]) for layer in layers] for key in STORY_SOURCE_ORDER],
            dtype=float,
        )
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[str(layer) for layer in layers],
                y=[row_labels[key] for key in STORY_SOURCE_ORDER],
                text=text,
                texttemplate="%{text}",
                visible=(view_pos == 0),
                colorscale="Blues",
                zmin=0,
                zmax=12,
                colorbar=dict(
                    title=f"scenarios correct of 12<br>(chance {CHANCE_CORRECT:.0f}, the mark"
                    f" fixed<br>in advance {PASS_BAR})"
                ),
            )
        )


def detection_figure(evidence: StorySourceEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 3 exhibit: the registered R1 detection read, per layer.

    Heatmap of correct-scenario counts per (story source, layer) with a
    slider over three color views: the worse of the two scenario sets
    (default), the paper's scenarios, the held-out scenarios. The verdict
    (passing layers) never depends on the view; the title restates which set
    the color shows. Returns ``(figure, stats)``; ``stats["lines"]`` prints
    each story source's passing layers and best layer, ``stats["n_passing"]``
    maps story source -> count.
    """
    arms = {
        # "r1_dual_battery" is the frozen key in the evidence JSONs: the
        # detection read scored on both sets of scenarios
        key: getattr(evidence, attribute)["r1_dual_battery"][name]
        for key, (attribute, name) in R1_SOURCES.items()
    }
    counts = {key: per_layer_counts(arm) for key, arm in arms.items()}
    layers = sorted(counts["selfgen"])
    for key, arm_counts in counts.items():
        if sorted(arm_counts) != layers:
            raise ValueError(f"{key}: layer grid differs from selfgen's")

    row_labels = {  # row label restates n and the generator role
        "selfgen": "self-generated n=256<br>(stories by the probed model)",
        "weak_external": "weak external<br>(stories by gemma-4-4B)",
        "fixed_deepseek": "fixed DeepSeek n=256<br>(stories by deepseek-v4-pro)",
        "diverse_deepseek": "diverse DeepSeek n=1024<br>(stories by deepseek-v4-pro)",
    }
    views = [  # (slider label, per-(paper, held-out) reduction shown as color)
        ("worse of the two sets of scenarios", lambda paper, held_out: min(paper, held_out)),
        ("the paper's twelve scenarios", lambda paper, held_out: paper),
        ("the twelve held-out scenarios", lambda paper, held_out: held_out),
    ]
    # cell text is view-independent: both counts, starred when passing
    text = [
        [
            f"{paper}/{held_out}" + (" *" if paper >= PASS_BAR and held_out >= PASS_BAR else "")
            for layer in layers
            for paper, held_out in [counts[key][layer]]
        ]
        for key in STORY_SOURCE_ORDER
    ]
    pass_counts = {key: len(arms[key]["passing_layers"]) for key in STORY_SOURCE_ORDER}

    fig = go.Figure()
    _add_detection_heatmaps(fig, views, counts, layers, text, row_labels)
    steps = [  # one step per scenario-set view: flip visibility, restate it
        dict(
            method="update",
            label=view_label,
            args=[
                {"visible": [pos == view_pos for pos in range(len(views))]},
                {"title.text": _detection_title(pass_counts, len(layers), view_label)},
            ],
        )
        for view_pos, (view_label, _) in enumerate(views)
    ]
    fig.update_layout(
        title=_detection_title(pass_counts, len(layers), views[0][0]),
        title_font_size=13,
        xaxis_title=f"layer of {PROBED_MODEL}",
        # rows read top-down in the canonical story-source order (plotly
        # stacks categorical y from the bottom, so the axis is reversed)
        yaxis=dict(autorange="reversed", title="story source (who wrote the stories)"),
        width=DETECTION_WIDTH_PX,
        height=560,
        margin=dict(t=170, b=140),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "color shows: "},
                # push the slider clear of the x-axis title below the heatmap
                pad={"t": 70},
                steps=steps,
            )
        ],
    )
    lines = [
        f"{STORY_SOURCE_LABELS[key]}: {pass_counts[key]} passing layers"
        f" {arms[key]['passing_layers']}, best layer {arms[key]['best']['layer']}"
        f" ({arms[key]['best']['counts'][0]}/{arms[key]['best']['counts'][1]})"
        for key in STORY_SOURCE_ORDER
    ]
    return fig, {"lines": lines, "n_passing": pass_counts, "layers": layers}


def _geometry_matrices(
    evidence: StorySourceEvidence,
) -> tuple[
    Float[np.ndarray, "story_sources story_sources"],
    Float[np.ndarray, "story_sources story_sources"],
    dict[tuple[str, str], tuple[float, float]],
]:
    """Assemble the symmetric cosine and RSA matrices from both evidence files.

    Returns ``(cosine, rsa, pairs)``. Unmeasured pairs stay NaN so the figure
    can label them rather than draw them as zero; the diagonal is 1 by
    definition (a probe set against itself).
    """
    # merge the two evidence files' pair dicts, mapping their slot names to
    # the canonical story-source keys
    slot_to_key = {
        "selfgen": "selfgen",
        "selfgen_postfix": "selfgen",
        "weak_external": "weak_external",
        "strong_external": "fixed_deepseek",
        "fixed_deepseek_n256": "fixed_deepseek",
        "diverse_deepseek_n1024": "diverse_deepseek",
    }
    # "r2_cross_lineage_layer33" is the frozen key the scorers wrote: the
    # pairwise geometry read between probe sets at layer 33
    raw_pairs = {
        **evidence.e11["r2_cross_lineage_layer33"],
        **evidence.full_grid["r2_cross_lineage_layer33"],
    }
    n = len(STORY_SOURCE_ORDER)
    cosine: Float[np.ndarray, "story_sources story_sources"] = np.full((n, n), np.nan)
    rsa: Float[np.ndarray, "story_sources story_sources"] = np.full((n, n), np.nan)
    # a probe set agrees with itself perfectly: the diagonal is the strength anchor
    np.fill_diagonal(cosine, 1.0)
    np.fill_diagonal(rsa, 1.0)
    pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for pair_name, pair_stats in raw_pairs.items():
        slot_a, slot_b = pair_name.split("_vs_")
        key_a, key_b = slot_to_key[slot_a], slot_to_key[slot_b]
        row, col = STORY_SOURCE_ORDER.index(key_a), STORY_SOURCE_ORDER.index(key_b)
        cosine[row, col] = cosine[col, row] = pair_stats["mean_contrast_cos"]
        rsa[row, col] = rsa[col, row] = pair_stats["rsa_12x12"]
        pairs[key_a, key_b] = (pair_stats["mean_contrast_cos"], pair_stats["rsa_12x12"])
    return cosine, rsa, pairs


def _geometry_cell_text(
    cosine: Float[np.ndarray, "story_sources story_sources"],
    rsa: Float[np.ndarray, "story_sources story_sources"],
) -> list[list[str]]:
    """Cell labels for the geometry matrix: both numbers per measured pair,
    an explicit note on the diagonal, and an explicit note where a pair was
    never scored (a blank cell would read as zero agreement)."""
    n = len(STORY_SOURCE_ORDER)
    text = []
    for row in range(n):
        row_text = []
        for col in range(n):
            if row == col:
                row_text.append("same probe set<br>(1 by definition)")
            elif np.isnan(cosine[row, col]):
                # an unmeasured pair must say so, or a blank cell reads as zero
                row_text.append("pair not<br>measured")
            else:
                row_text.append(f"cos {cosine[row, col]:.2f}<br>RSA {rsa[row, col]:.2f}")
        text.append(row_text)
    return text


def geometry_figure(evidence: StorySourceEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 5 exhibit: the registered R2 geometry read at layer 33.

    Symmetric source-by-source heatmap: color and top number = mean
    per-emotion contrast cosine (direction agreement), bottom number = RSA
    over the 12x12 emotion-similarity matrices (shape agreement). The
    diagonal is 1 by definition and serves as the strength anchor; 0 =
    unrelated directions is the failure anchor. Returns ``(figure, stats)``;
    ``stats["lines"]`` prints every measured pair and ``stats["pairs"]``
    maps (a, b) -> (cosine, rsa).
    """
    cosine, rsa, pairs = _geometry_matrices(evidence)

    def cos_to(key: str) -> float:
        """Contrast cosine between one source's probes and the self-generated set."""
        row = STORY_SOURCE_ORDER.index("selfgen")
        return float(cosine[row, STORY_SOURCE_ORDER.index(key)])

    text = _geometry_cell_text(cosine, rsa)
    axis_labels = [STORY_SOURCE_TICK_LABELS[key] for key in STORY_SOURCE_ORDER]
    width_px = 1000
    fig = go.Figure(
        go.Heatmap(
            z=cosine,
            x=axis_labels,
            y=axis_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="mean contrast cosine<br>(1 = same directions,<br>0 = unrelated,"
                "<br>-1 = opposed)"
            ),
        )
    )
    diverse_rsa = pairs["selfgen", "diverse_deepseek"][1]
    fig.update_layout(
        title=figure_title(
            "Does a stronger generator recover the probed model's own probe directions? Partly:"
            f" cosine to the self-generated set is {cos_to('fixed_deepseek'):.2f} (fixed"
            f" DeepSeek) against {cos_to('weak_external'):.2f} (weak external), and the diverse"
            f" arm keeps the shape (RSA {diverse_rsa:.2f}) more than the directions (cos"
            f" {cos_to('diverse_deepseek'):.2f})",
            [
                f"one cell = one pair of probe sets compared at layer {GEOMETRY_LAYER}, the"
                " geometry peak, which is the only layer the registered R2 read is defined at:"
                " top number and cell color = mean per-emotion contrast cosine (do the"
                " directions agree?), bottom number = RSA of the 12x12 emotion-similarity"
                " matrices (does the shape of emotion space agree?)",
                "anchors: the diagonal is 1 by definition (a probe set against itself, perfect"
                " agreement), and 0 is the failure anchor (unrelated directions); probes read"
                f" from {PROBED_MODEL}; evidence: e11_lineage.json and"
                " e12_diverse_fullcorpus_grid.json",
            ],
            width_px,
        ),
        title_font_size=13,
        xaxis_title="probes built from stories by",
        # rows read top-down in the canonical story-source order, matching
        # the detection heatmap
        yaxis=dict(autorange="reversed", title="probes built from stories by"),
        width=width_px,
        height=740,
        # the wrapped title runs to seven lines above the matrix
        margin=dict(t=240),
    )
    lines = [
        f"{STORY_SOURCE_LABELS[key_a]} vs {STORY_SOURCE_LABELS[key_b]}:"
        f" contrast cos {cos_value:.3f}, RSA {rsa_value:.3f}"
        for (key_a, key_b), (cos_value, rsa_value) in pairs.items()
    ]
    return fig, {"lines": lines, "pairs": pairs}
