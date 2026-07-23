"""Act I exhibits: the per-layer PC1-valence curve and the E3/E4 addendum lines."""

from __future__ import annotations

import re
from typing import Any, Mapping

import plotly.graph_objects as go

Stats = dict[str, Any]


def _act1_stats(
    geometry: Mapping,
    geometry_it: Mapping,
    geometry_demotion: Mapping,
    extraction_audit: Mapping,
) -> Stats:
    """Peak, demotion, robustness, late-band, and paper-reference numbers for Act I.

    The paper's reported correlation is parsed from the evidence file's own
    ``literature_reference`` note rather than typed by hand.
    """
    peak_layer, peak_row = max(
        geometry["per_layer"].items(), key=lambda item: abs(item[1]["pc1_valence"]["pearson_r"])
    )
    peak_abs_r = abs(peak_row["pc1_valence"]["pearson_r"])
    it_pc1_33 = geometry_it["per_layer"]["33"]["pc1_valence"]["pearson_r"]
    audit_geo = extraction_audit["lineages"]["corpus_base"]["geometry"]
    late_band = [
        layer for layer in sorted(int(k) for k in geometry["per_layer"]) if 33 <= layer <= 57
    ]
    late_vals = [
        abs(geometry["per_layer"][str(layer)]["pc1_valence"]["pearson_r"]) for layer in late_band
    ]
    paper_r = float(
        re.search(r"PC1-valence r=([0-9.]+)", geometry["literature_reference"]).group(1)
    )
    lines = [
        f"base model  | peak PC1-valence |r| = {peak_abs_r:.3f} at layer {peak_layer}",
        f"instruct    | PC1-valence |r| at layer 33 = {abs(it_pc1_33):.3f} (the demotion); "
        f"valence resurfaces as PC{geometry_demotion['it']['best_pc']} "
        f"at |r| = {geometry_demotion['it']['best_r']:.3f}",
        f"post-fix robustness check (E4b, base geometry): peak |r| "
        f"{audit_geo['peak_abs_r_before']:.4f} -> {audit_geo['peak_abs_r_after']:.4f} "
        f"(delta {audit_geo['peak_abs_r_delta']:.4f})",
        f"base late band (layers {late_band[0]}-{late_band[-1]}): |r| holds "
        f"{min(late_vals):.2f}-{max(late_vals):.2f}",
    ]
    return {
        "lines": lines,
        "peak_abs_r": peak_abs_r,
        "peak_layer": int(peak_layer),
        "it_pc1_33_abs": abs(it_pc1_33),
        "late_band_range": (min(late_vals), max(late_vals)),
        "paper_r": paper_r,
    }


def _act1_anchors(fig: go.Figure, paper_r: float) -> None:
    """Grading anchors: the paper's reported strength, the zero (no structure)
    floor, and the late band the claim quotes."""
    fig.add_hline(
        y=paper_r,
        line_dash="dashdot",
        annotation_text=f"paper's reported PC1-valence r = {paper_r:.2f}",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        annotation_text="0 = no valence structure on PC1",
        annotation_position="bottom right",
    )
    fig.add_vrect(
        x0=33,
        x1=57,
        fillcolor="#888888",
        opacity=0.08,
        line_width=0,
        annotation_text="late band (layers 33-57)",
        annotation_position="top left",
    )


def act1_geometry_figure(
    geometry: Mapping,
    geometry_it: Mapping,
    geometry_demotion: Mapping,
    extraction_audit: Mapping,
    model_base: str,
    model_it: str,
) -> tuple[go.Figure, Stats]:
    """Act I: the per-layer PC1-valence correlation curve, base vs instruct reader.

    Inputs are the parsed evidence files ``emotion_geometry_correlations.json``
    (base), ``emotion_geometry_correlations_it.json`` (instruct),
    ``geometry_pc_demotion.json``, and ``e4b_extraction_impact.json``. Returns
    the anchored curve figure and stats with the printed ``lines``, the base
    peak, the late-band range, and the paper's reported correlation.
    """
    stats = _act1_stats(geometry, geometry_it, geometry_demotion, extraction_audit)
    base_curve = {
        int(k): abs(v["pc1_valence"]["pearson_r"]) for k, v in geometry["per_layer"].items()
    }
    it_curve = {
        int(k): abs(v["pc1_valence"]["pearson_r"]) for k, v in geometry_it["per_layer"].items()
    }
    fig = go.Figure()
    fig.add_scatter(
        x=sorted(base_curve),
        y=[base_curve[layer] for layer in sorted(base_curve)],
        mode="lines+markers",
        name=f"{model_base} (base reader)",
    )
    fig.add_scatter(
        x=sorted(it_curve),
        y=[it_curve[layer] for layer in sorted(it_curve)],
        mode="lines+markers",
        name=f"{model_it} (instruct reader)",
    )
    _act1_anchors(fig, stats["paper_r"])
    late_low, late_high = stats["late_band_range"]
    fig.update_layout(
        title=(
            "Does the first principal component track valence? Base reader: yes, "
            "across the whole late band; instruct reader: demoted<br>"
            f"<sup>base |r| holds {late_low:.2f}-{late_high:.2f} over layers 33-57 "
            f"(peak {stats['peak_abs_r']:.3f} at layer {stats['peak_layer']}); instruct |r| at "
            f"layer 33 = {stats['it_pc1_33_abs']:.2f}, valence resurfacing as "
            f"PC{geometry_demotion['it']['best_pc']} at |r| = "
            f"{geometry_demotion['it']['best_r']:.2f}</sup><br>"
            "<sup>one dot = one (reader model, layer): |Pearson r| between PC1 of the 171 "
            "emotion vectors and human valence norms | evidence: "
            "emotion_geometry_correlations(_it).json</sup>"
        ),
        title_font_size=15,
        xaxis_title="layer",
        yaxis_title="|Pearson r| between PC1 and valence norms",
        width=1050,
        height=480,
        margin=dict(t=120),
    )
    return fig, stats


def act1_addendum_lines(pc_structure: Mapping, rsa_frag: Mapping) -> Stats:
    """Act I addendum (E3/E4): what displaced valence, and the RSA ablation read.

    Formats ``it_pc_structure.json`` (per-PC identities of the instruct top
    components) and ``rsa_fragmentation.json`` (top-component ablation and
    cross-model agreement). Returns stats whose ``lines`` reproduce the
    notebook printout (the empty string is the printed blank line).
    """
    it_pcs = {row["pc"]: row for row in pc_structure["it"]["per_pc"]}
    score_corr = pc_structure["alignment"]["score_corr_it_by_base"]
    angles = pc_structure["alignment"]["principal_angles_deg_top3"]
    length_r = pc_structure["story_length_read"]["it"]["r_mean_story_len_per_pc"]
    it_ablation = rsa_frag["it"]["ablation"]
    base_ablation = rsa_frag["base"]["ablation"]
    cross = rsa_frag["cross_model"]
    cross_post = rsa_frag["cross_model_ablated_post_hoc"]
    pc1_best_axis_r = max(
        abs(it_pcs[1]["r_valence"]), abs(it_pcs[1]["r_arousal"]), abs(it_pcs[1]["r_dominance"])
    )
    lines = [
        f"it-PC1: evr {it_pcs[1]['evr']:.2f}, best |r| vs valence/arousal/dominance "
        f"{pc1_best_axis_r:.2f}, "
        f"best |score corr| vs base top-5 {max(abs(c) for c in score_corr[0]):.2f}, "
        f"first principal angle {angles[0]:.1f} deg -> inserted structure",
        f"it-PC2: |score corr| vs base-PC2 {abs(score_corr[1][1]):.2f} (arousal survivor) but "
        f"r(story length) {length_r[1]:+.2f} vs r(arousal) {it_pcs[2]['r_arousal']:+.2f} "
        "-> length confound",
        f"it-PC3: |score corr| vs base-PC1 {abs(score_corr[2][0]):.2f} "
        f"(valence {it_pcs[3]['r_valence']:+.2f}, dominance {it_pcs[3]['r_dominance']:+.2f}) "
        "-> the bundle survives",
        "",
        f"RSA ablation: instruct mid-to-late coherence {it_ablation['0']['mid_to_late_mean']:.2f} "
        f"-> {it_ablation['1']['mid_to_late_mean']:.2f} with the top component removed; "
        f"base {base_ablation['0']['mid_to_late_mean']:.2f} -> "
        f"{base_ablation['1']['mid_to_late_mean']:.2f} (only hurts)",
        f"cross-model late-band agreement: {cross['it_late_x_base_late_mean']:.2f} unablated -> "
        f"{cross_post['it_k2_late_x_base_late_mean']:.2f} with instruct top-2 removed "
        "(post-hoc read): base-like geometry intact underneath",
    ]
    return {"lines": lines}
