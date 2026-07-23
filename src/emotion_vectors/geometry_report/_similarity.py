"""Sections 3, 4, and 6: the similarity-structure exhibits.

Clustered centered-cosine heatmaps (section 3), the k-means/t-SNE family
view, paper Figure 6 with t-SNE substituted for UMAP (section 4), and the
across-depth RSA (section 6). Each builder returns ``(figure, stats)`` with
``stats["lines"]`` holding every line the notebook prints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Int
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from ._context import LATE_BAND, GeometryContext

CLUSTER_SEED = 20260722  # fixed seed shared by k-means and t-SNE: reruns reproduce exactly
N_FAMILIES = 10  # k-means k=10, as the paper
FAMILY_COLORS = qualitative.Plotly  # 10 colors, one per k-means cluster

# Every per-panel-per-layer store below is keyed (model label, layer).
ModelLayerKey = tuple[str, int]
SimilarityByKey = dict[ModelLayerKey, Float[np.ndarray, "emotions emotions"]]
TickLabelsByKey = dict[ModelLayerKey, list[str]]
LeafOrderByKey = dict[ModelLayerKey, Int[np.ndarray, " emotions"]]
# one (t-SNE embedding, k-means family labels) pair per key
EmbeddingByKey = dict[
    ModelLayerKey, tuple[Float[np.ndarray, "emotions 2"], Int[np.ndarray, " emotions"]]
]
RsaByLabel = dict[str, Float[np.ndarray, "layers layers"]]


def centered_cosine(
    layer_means: Float[np.ndarray, "emotions d_model"],
) -> Float[np.ndarray, "emotions emotions"]:
    """Pairwise cosine similarity after removing the layer's mean vector."""
    centered = layer_means - layer_means.mean(axis=0, keepdims=True)
    normed = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    return normed @ normed.T


def _clustered_similarities(
    ctx: GeometryContext,
) -> tuple[SimilarityByKey, SimilarityByKey, TickLabelsByKey, LeafOrderByKey]:
    """Per (label, layer): the full centered-cosine matrix, its clustered-order
    heatmap z (float32 at 2 dp keeps the saved notebook light), the every-10th
    tick labels in clustered order, and the hierarchy leaf order itself."""
    tick_positions = list(range(0, len(ctx.emotions), 10))
    similarity_full: SimilarityByKey = {}
    heatmap_z: SimilarityByKey = {}
    tick_labels: TickLabelsByKey = {}
    leaf_order: LeafOrderByKey = {}
    for label in ctx.labels:
        for layer_pos, layer in enumerate(ctx.layers):
            similarity = centered_cosine(ctx.models[label][:, layer_pos, :])
            similarity_full[label, layer] = similarity
            order = hierarchy.leaves_list(hierarchy.linkage(1 - similarity, method="average"))
            heatmap_z[label, layer] = np.round(similarity[np.ix_(order, order)], 2).astype(
                np.float32
            )
            tick_labels[label, layer] = [ctx.emotions[order[t]] for t in tick_positions]
            leaf_order[label, layer] = order
    return similarity_full, heatmap_z, tick_labels, leaf_order


def _s3_structure_agreement(
    ctx: GeometryContext, similarity_full: SimilarityByKey
) -> dict[int, float]:
    """Per layer: Pearson r between the two readers' upper-triangle
    similarities — do they agree which emotions resemble each other?"""
    pair_upper = np.triu_indices(len(ctx.emotions), k=1)  # each unordered emotion pair once
    it, base = ctx.labels
    return {
        layer: float(
            np.corrcoef(
                similarity_full[it, layer][pair_upper], similarity_full[base, layer][pair_upper]
            )[0, 1]
        )
        for layer in ctx.layers
    }


def _s3_title(ctx: GeometryContext, layer: int, agreement: dict[int, float]) -> str:
    """Per-layer verdict: how much the two readers' structures agree."""
    return (
        f"Do the synonym-cluster blocks survive instruction tuning at layer {layer}? "
        f"instruct-vs-base structure agreement r = {agreement[layer]:.2f} (1 = identical, 0 ="
        " unrelated)"
        "<br><sup>one cell = centered-cosine similarity of two emotions' vectors; rows/columns"
        " ordered by clustering, independently per panel and per layer; red diagonal blocks ="
        " near-synonym families, blue = opposed emotions</sup>"
        "<br><sup>failure would be a structureless field; activations read by each panel's model"
        " from the same gemma-4-4B-written stories | maps to: Anthropic Figure 5 + open"
        " replication fig1_cosine_similarity</sup>"
    )


def _s3_slider_steps(
    ctx: GeometryContext,
    tick_labels: TickLabelsByKey,
    agreement: dict[int, float],
    trace_keys: list[ModelLayerKey],
) -> list[dict]:
    """One step per layer: visibility mask, re-labeled clustered ticks, and the
    per-layer verdict title."""
    it, base = ctx.labels
    return [
        dict(
            method="update",
            label=str(layer),
            args=[
                {"visible": [key[1] == layer for key in trace_keys]},
                {
                    "xaxis.ticktext": tick_labels[it, layer],
                    "yaxis.ticktext": tick_labels[it, layer],
                    "xaxis2.ticktext": tick_labels[base, layer],
                    "yaxis2.ticktext": tick_labels[base, layer],
                    "title.text": _s3_title(ctx, layer, agreement),
                },
            ],
        )
        for layer in ctx.layers
    ]


def _s3_figure(
    ctx: GeometryContext,
    heatmap_z: SimilarityByKey,
    tick_labels: TickLabelsByKey,
    agreement: dict[int, float],
) -> go.Figure:
    """Paired clustered heatmaps, one hidden trace per (panel, layer); the
    slider flips visibility and swaps the clustered ticks and verdict title."""
    tick_positions = list(range(0, len(ctx.emotions), 10))
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(ctx.labels), horizontal_spacing=0.14)
    trace_keys: list[ModelLayerKey] = []
    for col, label in enumerate(ctx.labels, start=1):
        for layer in ctx.layers:
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_z[label, layer],
                    visible=(layer == ctx.default_layer),
                    colorscale="RdBu",
                    reversescale=True,
                    zmin=-1,
                    zmax=1,
                    showscale=(col == 2),
                    colorbar=dict(
                        title="centered cosine<br>(+1 = near-synonym<br>vectors, -1 = opposed)"
                    ),
                ),
                row=1,
                col=col,
            )
            trace_keys.append((label, layer))
        fig.update_xaxes(
            tickvals=tick_positions,
            ticktext=tick_labels[label, ctx.default_layer],
            tickangle=90,
            tickfont=dict(size=6),
            title_text="emotion (clustered order, every 10th labeled)",
            row=1,
            col=col,
        )
        fig.update_yaxes(
            tickvals=tick_positions,
            ticktext=tick_labels[label, ctx.default_layer],
            tickfont=dict(size=6),
            autorange="reversed",
            row=1,
            col=col,
        )
    fig.update_layout(
        title=_s3_title(ctx, ctx.default_layer, agreement),
        title_font_size=14,
        height=760,
        width=1200,
        margin=dict(t=150),
        sliders=[
            dict(
                active=ctx.layers.index(ctx.default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 85},
                steps=_s3_slider_steps(ctx, tick_labels, agreement, trace_keys),
            )
        ],
    )
    return fig


def s3_similarity_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 3 exhibit: do the synonym-cluster blocks survive tuning?

    Clusters each model's centered-cosine matrix per layer, paired heatmaps
    with a layer slider; the per-layer title reports how much the two readers'
    structures agree. Returns ``(figure, stats)``; ``stats["lines"]`` is the
    printed record (each reader's first cluster block at the default layer)
    and ``stats["agreement"]`` maps layer -> structure-agreement r.
    """
    similarity_full, heatmap_z, tick_labels, leaf_order = _clustered_similarities(ctx)
    lines = [
        f"{label} first cluster block at layer {ctx.default_layer}: "
        f"{[ctx.emotions[pos] for pos in leaf_order[label, ctx.default_layer][:8]]}"
        for label in ctx.labels
    ]
    agreement = _s3_structure_agreement(ctx, similarity_full)
    fig = _s3_figure(ctx, heatmap_z, tick_labels, agreement)
    return fig, {"lines": lines, "agreement": agreement}


def _cluster_embeddings(ctx: GeometryContext) -> EmbeddingByKey:
    """Per (label, layer): the t-SNE embedding [emotions 2] (float32 at 2 dp)
    and the k-means family labels [emotions], both fit on the centered vectors
    with fixed seeds."""
    embeddings: EmbeddingByKey = {}
    for label in ctx.labels:
        for layer_pos, layer in enumerate(ctx.layers):
            centered = ctx.models[label][:, layer_pos, :] - ctx.models[label][:, layer_pos, :].mean(
                axis=0
            )
            kmeans = KMeans(n_clusters=N_FAMILIES, n_init=10, random_state=CLUSTER_SEED).fit(
                centered
            )
            embedding = TSNE(
                n_components=2, perplexity=20, random_state=CLUSTER_SEED
            ).fit_transform(centered)
            embeddings[label, layer] = (np.round(embedding, 2).astype(np.float32), kmeans.labels_)
    return embeddings


def _s4_title(ctx: GeometryContext, layer: int, ari: dict[int, float]) -> str:
    """Per-layer verdict: adjusted Rand index between the readers' partitions."""
    return (
        f"Do the two readers group emotions into the same families at layer {layer}? "
        f"cluster agreement ARI = {ari[layer]:.2f} (1 = identical partitions, 0 = chance overlap)"
        "<br><sup>one point = one emotion at its t-SNE position; color = k-means cluster (k=10,"
        " as the paper), clustered independently per panel and per layer, so colors do not track"
        " across panels or layers</sup>"
        "<br><sup>success: the paper's joy/hope, calm/content, grief families visible in both"
        " panels; failure: no such families | activations read by each panel's model | maps to:"
        " Anthropic Figure 6 (t-SNE substituted for the paper's UMAP)</sup>"
    )


def _s4_figure(
    ctx: GeometryContext, embeddings: EmbeddingByKey, ari: dict[int, float]
) -> go.Figure:
    """Paired t-SNE scatters, one hidden trace per (panel, layer, cluster); the
    slider flips visibility and swaps the per-layer verdict title."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(ctx.labels), horizontal_spacing=0.06)
    trace_layers = []
    for col, label in enumerate(ctx.labels, start=1):
        for layer in ctx.layers:
            embedding, family = embeddings[label, layer]
            for cluster_id in range(N_FAMILIES):
                member_idx = np.where(family == cluster_id)[0]
                fig.add_trace(
                    go.Scatter(
                        x=embedding[member_idx, 0],
                        y=embedding[member_idx, 1],
                        visible=(layer == ctx.default_layer),
                        mode="markers+text",
                        text=[ctx.emotions[pos] for pos in member_idx],
                        textposition="top center",
                        textfont=dict(size=6),
                        marker=dict(size=6, color=FAMILY_COLORS[cluster_id]),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
                trace_layers.append(layer)
        fig.update_xaxes(title_text="t-SNE dim 1", row=1, col=col)
        fig.update_yaxes(title_text="t-SNE dim 2", row=1, col=col)
    steps = [  # visibility mask plus the per-layer verdict title
        dict(
            method="update",
            label=str(layer),
            args=[
                {"visible": [t == layer for t in trace_layers]},
                {"title.text": _s4_title(ctx, layer, ari)},
            ],
        )
        for layer in ctx.layers
    ]
    fig.update_layout(
        title=_s4_title(ctx, ctx.default_layer, ari),
        title_font_size=14,
        height=700,
        width=1250,
        margin=dict(t=140),
        sliders=[
            dict(
                active=ctx.layers.index(ctx.default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 40},
                steps=steps,
            )
        ],
    )
    return fig


def s4_families_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 4 exhibit: do the paper's emotion families appear on both
    readers? (paper Figure 6, t-SNE substituted for UMAP).

    Returns ``(figure, stats)``. ``stats["lines"]`` is the printed record:
    each reader's k-means memberships at the default layer (first 8 members
    per cluster) and the cross-reader ARI range; ``stats["ari"]`` maps
    layer -> adjusted Rand index between the two readers' partitions.
    """
    embeddings = _cluster_embeddings(ctx)
    it, base = ctx.labels
    ari = {
        layer: float(adjusted_rand_score(embeddings[it, layer][1], embeddings[base, layer][1]))
        for layer in ctx.layers
    }
    lines = []
    for label in ctx.labels:
        lines.append(f"{label} (k-means k={N_FAMILIES}, layer {ctx.default_layer}):")
        family = embeddings[label, ctx.default_layer][1]
        for cluster_id in range(N_FAMILIES):
            members = [ctx.emotions[pos] for pos in np.where(family == cluster_id)[0]][:8]
            lines.append(f"  cluster {cluster_id}: {members}")
    lines.append(
        f"cross-reader cluster agreement (ARI): {min(ari.values()):.2f} to"
        f" {max(ari.values()):.2f} "
        f"across layers; {ari[ctx.default_layer]:.2f} at layer {ctx.default_layer}"
    )
    return _s4_figure(ctx, embeddings, ari), {"lines": lines, "ari": ari}


def _s6_figure(
    ctx: GeometryContext, rsa_by_label: RsaByLabel, late_stability: dict[str, float]
) -> go.Figure:
    """Paired RSA heatmaps; the title carries the verdict computed upstream."""
    it, base = ctx.labels
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{label}" for label in ctx.labels],
        horizontal_spacing=0.14,
    )
    for col, label in enumerate(ctx.labels, start=1):
        fig.add_trace(
            go.Heatmap(
                z=rsa_by_label[label],
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                showscale=(col == 2),
                colorbar=dict(
                    title="agreement r between layers'<br>similarity structures<br>(1 = same view"
                    " of emotion<br>similarity, 0 = unrelated)"
                ),
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            tickvals=list(range(len(ctx.layers))),
            ticktext=[str(layer) for layer in ctx.layers],
            title_text="layer",
            tickfont=dict(size=9),
            row=1,
            col=col,
        )
        fig.update_yaxes(
            tickvals=list(range(len(ctx.layers))),
            ticktext=[str(layer) for layer in ctx.layers],
            title_text="layer",
            tickfont=dict(size=9),
            autorange="reversed",
            row=1,
            col=col,
        )
    fig.update_layout(
        title=(
            "Is each reader's emotion geometry stable across depth? Base yes, instruct fragments: "
            f"mean late-band ({LATE_BAND[0]}-{LATE_BAND[1]}) agreement {late_stability[base]:.2f}"
            f" base vs {late_stability[it]:.2f} instruct"
            "<br><sup>one cell (i, j) = Pearson r between layer i's and layer j's 171x171"
            " emotion-similarity structures; a bright mid-to-late block = the geometry"
            " consolidates early and stays stable</sup>"
            "<br><sup>activations read by each panel's model from gemma-4-4B-written stories | "
            "maps to: Anthropic Figure 9 (RSA substitute for the open replication's fig_cka)</sup>"
        ),
        title_font_size=14,
        height=600,
        width=1150,
        margin=dict(t=140),
    )
    return fig


def s6_rsa_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 6 exhibit: is the geometry stable across depth, or fragmenting?

    Correlates similarity structures across layers (RSA) per model, paired
    matrices; the stability numbers are computed first so the title states
    the verdict. Returns ``(figure, stats)``; ``stats["lines"]`` is the
    printed record and ``stats["late_stability"]`` the mean within-late-band
    RSA per model.
    """
    pair_upper = np.triu_indices(len(ctx.emotions), k=1)  # each unordered emotion pair once
    late_pos = [
        pos for pos, layer in enumerate(ctx.layers) if LATE_BAND[0] <= layer <= LATE_BAND[1]
    ]
    rsa_by_label: RsaByLabel = {}
    late_stability: dict[str, float] = {}
    lines = []
    for label in ctx.labels:
        means = ctx.models[label]
        similarity_flat = np.stack(
            [
                centered_cosine(means[:, layer_pos, :])[pair_upper]
                for layer_pos in range(len(ctx.layers))
            ]
        )
        rsa_by_label[label] = np.corrcoef(similarity_flat)
        late_stability[label] = rsa_by_label[label][np.ix_(late_pos, late_pos)][
            np.triu_indices(len(late_pos), k=1)
        ].mean()
        lines.append(
            f"{label}: mean RSA within layers {LATE_BAND[0]}-{LATE_BAND[1]} "
            f"{late_stability[label]:.3f}, "
            f"layer 0 vs layers {LATE_BAND[0]}-{LATE_BAND[1]}"
            f" {rsa_by_label[label][0, late_pos].mean():.3f}"
        )
    fig = _s6_figure(ctx, rsa_by_label, late_stability)
    return fig, {"lines": lines, "rsa": rsa_by_label, "late_stability": late_stability}
