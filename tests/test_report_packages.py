"""Contract tests for the notebook exhibit packages under ``src/emotion_vectors``.

Extracting notebook code into importable functions is only worth the trouble if
something then calls those functions without a notebook. This module is that
caller. It does not re-derive the science (the notebooks print the numbers and
``results/`` holds the evidence); it enforces the house contract every figure
builder is supposed to satisfy, so a regression is caught by ``check.sh``
rather than by a reader scrolling a rendered notebook.

Each contract below exists because it was violated at least once during the
report-notebook passes:

* a figure whose title froze while its layer slider moved the data underneath
  (notebook 08, caught only by rasterizing a non-default slider step),
* a slider whose per-step value array no longer matched the trace count
  (notebook 11, caught by an assert that a later edit would have broken),
* a machine-local absolute path reaching a committed artifact, which
  ``AGENTS.md`` forbids outright.

Data availability: every builder reads frozen evidence through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the project
datasets on Hugging Face). A clone without that data cannot run these tests, so
each fixture skips with an explicit reason naming what was missing. That is a
printed degradation, not a silent one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from emotion_vectors import (
    corpora_report,
    detection_report,
    geometry_report,
    lineage_report,
    taxonomy_report,
)

# Absolute paths that must never appear in a committed figure or printed record.
MACHINE_LOCAL_PATH = re.compile(r"(/Users/|/home/|/workspace/|[A-Za-z]:\\\\)")

# Builders are addressed as (package attribute name, callable factory) pairs so
# a failure names the exhibit a reader would recognise.
FigureResult = tuple[go.Figure, dict[str, Any]]


def _load_or_skip(loader: Callable[[], Any], description: str) -> Any:
    """Call an evidence loader, skipping the test with a named reason if the
    evidence is not reachable from this clone.

    Inputs: ``loader`` (zero-argument callable), ``description`` (what it
    loads, used in the skip message). Returns whatever the loader returns.
    """
    try:
        return loader()
    except Exception as error:  # noqa: BLE001 - the reason is reported, not swallowed
        pytest.skip(f"{description} unavailable in this clone: {type(error).__name__}: {error}")


def figure_texts(figure: go.Figure) -> list[str]:
    """Every human-readable string a figure puts on screen: title, axis and
    subplot titles, legend entries, annotations, colorbar titles, slider
    labels. Used by the no-local-path and labelling contracts."""
    texts: list[str] = []
    layout = figure.layout
    if layout.title and layout.title.text:
        texts.append(layout.title.text)
    for annotation in layout.annotations or ():
        if annotation.text:
            texts.append(annotation.text)
    for axis_name in dir(layout):
        if not (axis_name.startswith("xaxis") or axis_name.startswith("yaxis")):
            continue
        axis = getattr(layout, axis_name, None)
        if axis is not None and getattr(axis, "title", None) and axis.title.text:
            texts.append(axis.title.text)
    for slider in layout.sliders or ():
        texts.extend(step.label for step in slider.steps if step.label)
    for trace in figure.data:
        if getattr(trace, "name", None):
            texts.append(trace.name)
        colorbar = getattr(getattr(trace, "marker", None), "colorbar", None) or getattr(
            trace, "colorbar", None
        )
        if colorbar is not None and getattr(colorbar, "title", None) and colorbar.title.text:
            texts.append(colorbar.title.text)
    return texts


# Exceptions to the printed-record contract, named so they stay reviewable.
#
# taxonomy_report's early sections hand their record to `cross_arm_lines`,
# which combines several sections into one printout, so a per-figure record
# would print the same numbers twice.

# geometry_report.s5_loadings_figure is not deliberate: it is the one exhibit
# with no printed record at all, so its section states no number a reader can
# check. Fixing it changes notebook 02's stored output.
BUILDERS_WITHOUT_A_PRINTED_RECORD = {
    "geometry_report.s5_loadings_figure",
    "taxonomy_report.s1_top1_figure",
    "taxonomy_report.s2_family_figure",
    "taxonomy_report.s3_affective_distance_figure",
    "taxonomy_report.s4_confusion_figure",
    "taxonomy_report.s5_position_nonaffect_figure",
}


def assert_figure_contract(result: FigureResult, name: str) -> None:
    """Assert the house contract on one ``(figure, stats)`` return value.

    Checks: the return shape; a non-empty printed record of strings (except for
    the named exceptions above); a title that asks the question it answers; and
    no machine-local path anywhere in the figure's visible text or its printed
    record.
    """
    assert isinstance(result, tuple) and len(result) == 2, (
        f"{name} must return (figure, stats), got {type(result)!r}"
    )
    figure, stats = result
    assert isinstance(figure, go.Figure), f"{name} first return value must be a plotly Figure"
    assert isinstance(stats, dict), f"{name} second return value must be a stats dict"
    assert stats, f"{name} returned an empty stats mapping"

    lines = stats.get("lines")
    if name in BUILDERS_WITHOUT_A_PRINTED_RECORD:
        assert lines is None, (
            f"{name} now returns a printed record; remove it from"
            " BUILDERS_WITHOUT_A_PRINTED_RECORD so the contract is enforced"
        )
    else:
        assert isinstance(lines, list) and lines, (
            f"{name} stats must carry a non-empty 'lines' record"
        )
        assert all(isinstance(line, str) for line in lines), (
            f"{name} stats['lines'] must be strings (they are printed verbatim)"
        )
        # blank entries are allowed: a multi-section record separates its
        # sections with them. What is not allowed is a record that says nothing.
        assert any(line.strip() for line in lines), f"{name} stats['lines'] holds no non-blank line"

    title = figure.layout.title.text if figure.layout.title else None
    assert title, f"{name} has no title"
    assert "?" in title, (
        f"{name} title must be question-form (the house rule: state the question, then its own"
        f" answer). Got: {title[:120]!r}"
    )

    for text in [*figure_texts(figure), *(lines or ())]:
        assert not MACHINE_LOCAL_PATH.search(text), (
            f"{name} exposes a machine-local absolute path: {text[:160]!r}"
        )


# FIXED, and the empty set is kept so a regression has a name. The defect was:
# every taxonomy_report exhibit drove its slider through the shared
# `layer_slider`, which toggled trace visibility without restating the title,
# so the title kept the primary layer's verdict while the slider moved the data
# under it. `layer_slider` now takes a per-layer title callback and all six
# notebook-11 exhibits pass one, so no figure is exempt.

# The assertion is inverted for anything listed here, so an exemption cannot
# outlive its defect: fixing a figure makes this test fail until it leaves the
# list.
FIGURES_WITH_FROZEN_SLIDER_TITLES: set[str] = set()


def assert_slider_contract(figure: go.Figure, name: str) -> None:
    """Assert that a layer/view slider cannot leave a stale verdict on screen.

    Every step must restate the title (a frozen title over moving data is the
    defect this catches), and any per-trace value array a step supplies must
    match the number of traces it addresses.
    """
    known_frozen = name in FIGURES_WITH_FROZEN_SLIDER_TITLES
    for slider_pos, slider in enumerate(figure.layout.sliders or ()):
        assert slider.steps, f"{name} slider {slider_pos} has no steps"
        for step in slider.steps:
            arguments = step.args or ()
            updates: dict[str, Any] = {}
            for argument in arguments:
                if isinstance(argument, dict):
                    updates.update(argument)
            restates_title = any("title" in key for key in updates)
            if known_frozen:
                assert not restates_title, (
                    f"{name} now restates its title per slider step; remove it from"
                    " FIGURES_WITH_FROZEN_SLIDER_TITLES so the contract is enforced"
                )
                continue
            assert restates_title, (
                f"{name} slider step {step.label!r} does not restate the title; a frozen title"
                " over moving data is a defect"
            )
            # a restyle array addresses traces positionally: it may not be longer
            # than the traces on the figure
            for key, value in updates.items():
                if isinstance(value, (list, tuple)) and key in {"y", "x", "z", "visible", "text"}:
                    assert len(value) <= len(figure.data), (
                        f"{name} slider step {step.label!r} supplies {len(value)} values for"
                        f" '{key}' but the figure has {len(figure.data)} traces"
                    )


# --------------------------------------------------------------------------
# corpora_report (notebook 01)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpora_context():
    return _load_or_skip(corpora_report.load_corpora_context, "the story corpora")


def test_corpora_figures_meet_the_contract(corpora_context) -> None:
    for name, builder in (
        ("s1_catalog_figure", lambda: corpora_report.s1_catalog_figure(corpora_context)),
        ("s2_leakage_figure", lambda: corpora_report.s2_leakage_figure(corpora_context)),
        ("s3_bundles_figure", corpora_report.s3_bundles_figure),
    ):
        result = builder()
        assert_figure_contract(result, f"corpora_report.{name}")
        assert_slider_contract(result[0], f"corpora_report.{name}")


def test_corpora_catalog_counts_match_the_grouped_files(corpora_context) -> None:
    """The catalog is a census: every row's text count must equal what its own
    grouped file holds, so the front page cannot drift from the data."""
    for entry in corpora_context.catalog:
        if entry.grouped_path is None:
            continue
        rows = [json.loads(line) for line in open(entry.grouped_path)]
        counted = sum(
            len(row.get("texts", row.get("stories", row.get("dialogues", [])))) for row in rows
        )
        assert counted == entry.n_texts, (
            f"catalog row {entry.key!r} claims {entry.n_texts} texts, file holds {counted}"
        )


# --------------------------------------------------------------------------
# lineage_report (notebook 07)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lineage_evidence():
    return _load_or_skip(lineage_report.load_evidence, "the E11/E12 lineage evidence")


def test_lineage_figures_meet_the_contract(lineage_evidence) -> None:
    builders = (
        "diversity_figure",
        "detection_figure",
        "dose_response_figure",
        "geometry_figure",
        "preference_figure",
        "extraction_format_figure",
    )
    for name in builders:
        result = getattr(lineage_report, name)(lineage_evidence)
        assert_figure_contract(result, f"lineage_report.{name}")
        assert_slider_contract(result[0], f"lineage_report.{name}")


def test_lineage_geometry_pair_agrees_across_both_evidence_files(lineage_evidence) -> None:
    """The self-generated vs fixed-DeepSeek pair is stored in two evidence
    files. If they ever diverge, one figure would quietly contradict another."""
    from_e11 = lineage_evidence.e11["r2_cross_lineage_layer33"]["selfgen_vs_strong_external"]
    grid = lineage_evidence.full_grid["r2_cross_lineage_layer33"]
    shared = [value for key, value in grid.items() if "selfgen" in key and "fixed" in key]
    for other in shared:
        assert other["mean_contrast_cos"] == pytest.approx(from_e11["mean_contrast_cos"])
        assert other["rsa_12x12"] == pytest.approx(from_e11["rsa_12x12"])


# --------------------------------------------------------------------------
# detection_report (notebook 03)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep_context():
    return _load_or_skip(detection_report.load_sweep_context, "the probe sweep activations")


def test_detection_sweep_and_intensity_figures_meet_the_contract(sweep_context) -> None:
    instruct = next(iter(sweep_context.models.values()))
    grids = detection_report.sweep_grids(instruct)
    result = detection_report.sweep_figure(instruct, grids, "test source note")
    assert_figure_contract(result, "detection_report.sweep_figure")
    assert_slider_contract(result[0], "detection_report.sweep_figure")

    curves = detection_report.intensity_curves(sweep_context)
    result = detection_report.intensity_figure(curves, sweep_context.layers)
    assert_figure_contract(result, "detection_report.intensity_figure")
    assert_slider_contract(result[0], "detection_report.intensity_figure")


def test_detection_registered_sign_read_stays_at_layer_33(sweep_context) -> None:
    """The numerical-intensity verdict is registered at layer 33: the instruct
    arm gets 11 of 11 directions and the base arm reproduces the archived 7 of
    11 (TREE Q1.H2.E1, prediction P2)."""
    curves = detection_report.intensity_curves(sweep_context)
    layer_pos = sweep_context.layers.index(33)
    scores = detection_report.direction_scores(curves, layer_pos)
    correct = {label: score["correct"] for label, score in scores.items()}
    assert 11 in correct.values(), f"no arm reached 11 of 11 at layer 33: {correct}"
    assert 7 in correct.values(), f"the base arm must reproduce 7 of 11 at layer 33: {correct}"


# --------------------------------------------------------------------------
# geometry_report (notebook 02)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def geometry_context():
    return _load_or_skip(geometry_report.load_geometry_context, "the emotion-vector bundles")


def test_geometry_figures_meet_the_contract(geometry_context) -> None:
    builders = (
        "s1_replication_figure",
        "s2_circumplex_figure",
        "s3_similarity_figure",
        "s4_families_figure",
        "s5_loadings_figure",
        "s6_rsa_figure",
        "s9_fragmentation_figure",
    )
    for name in builders:
        result = getattr(geometry_report, name)(geometry_context)
        assert_figure_contract(result, f"geometry_report.{name}")
        assert_slider_contract(result[0], f"geometry_report.{name}")


def test_geometry_centered_cosine_is_symmetric_with_unit_diagonal(geometry_context) -> None:
    """The similarity matrix every geometry figure rests on must be a proper
    similarity: symmetric, and 1 where an emotion meets itself."""
    layer_pos = geometry_context.layers.index(geometry_context.default_layer)
    for label, means in geometry_context.models.items():
        similarity = geometry_report.centered_cosine(means[:, layer_pos, :])
        assert np.allclose(similarity, similarity.T, atol=1e-6), f"{label}: not symmetric"
        assert np.allclose(np.diag(similarity), 1.0, atol=1e-6), f"{label}: diagonal is not 1"


def geometry_report_centered(layer_means):
    """Thin indirection so the import stays inside the test module's scope."""
    return geometry_report.centered_cosine(layer_means)


# --------------------------------------------------------------------------
# taxonomy_report (notebook 11)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def taxonomy_arms():
    return _load_or_skip(
        # "base" is needed by s6_named_confusions, which reads the base arm
        lambda: taxonomy_report.load_arms(["it_v2", "deepseek", "base"]),
        "the Q3 tracking arms",
    )


def test_taxonomy_figures_meet_the_contract(taxonomy_arms) -> None:
    vad = _load_or_skip(taxonomy_report.load_vad, "the NRC VAD lexicon")
    rng = np.random.default_rng(0)
    results = {
        "s1_top1_figure": taxonomy_report.s1_top1_figure(taxonomy_arms),
        "s2_family_figure": taxonomy_report.s2_family_figure(taxonomy_arms, rng),
        "s3_affective_distance_figure": taxonomy_report.s3_affective_distance_figure(
            taxonomy_arms, vad, rng
        ),
        "s4_confusion_figure": taxonomy_report.s4_confusion_figure(taxonomy_arms, vad, rng),
        "s5_position_nonaffect_figure": taxonomy_report.s5_position_nonaffect_figure(
            taxonomy_arms, taxonomy_report.load_has_nonaffect(), rng
        ),
        "s6_named_confusions": taxonomy_report.s6_named_confusions(taxonomy_arms),
        "s7_thirds_figure": taxonomy_report.s7_thirds_figure(taxonomy_arms),
        "s8_geometry_figure": taxonomy_report.s8_geometry_figure(taxonomy_arms, vad),
    }
    for name, result in results.items():
        assert_figure_contract(result, f"taxonomy_report.{name}")
        assert_slider_contract(result[0], f"taxonomy_report.{name}")


def test_every_taxonomy_section_builder_is_covered() -> None:
    """The contract tests enumerate builders by name, so a builder that is
    never listed is never enforced. This pins the enumeration: if a new
    sNN_ figure builder appears in the package, it must be added above."""
    covered = {
        "s1_top1_figure",
        "s2_family_figure",
        "s3_affective_distance_figure",
        "s4_confusion_figure",
        "s5_position_nonaffect_figure",
        "s6_named_confusions",
        "s7_thirds_figure",
        "s8_geometry_figure",
    }
    exported = {
        name
        for name in dir(taxonomy_report)
        if re.fullmatch(r"s\d+_\w+", name) and callable(getattr(taxonomy_report, name))
    }
    assert exported == covered, (
        "taxonomy_report's section builders and the tested set have diverged;"
        f" untested: {sorted(exported - covered)}, stale entries: {sorted(covered - exported)}"
    )
