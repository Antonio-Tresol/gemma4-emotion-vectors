"""Publish experiment-level activations, prompts, and scores to Hugging Face.

Open-science goal: a replicator should be able to check every number in the
report notebooks WITHOUT re-running inference. This publishes the activation
tensors, the exact prompts they came from, and the scored outputs, in one
dataset repo with a per-file schema card. Corpora and extraction vectors are
published separately (scripts/publish_generated_corpora.py, the extraction
pipeline's --publish flow).

Not uploaded: the NRC VAD lexicon (third-party license — the card points to
its source) and anything reproducible from the repo alone.

    uv run python scripts/publish_experiment_artifacts.py [--public]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# (path under results/, one-line schema note for the card)
ARTIFACTS: list[tuple[str, str]] = [
    (
        "probe_sweep/",
        "BASE model battery+template activations: activations.npz keys plain_{last,mean_all,"
        "mean_content} [prompts, 20 layers, 5376] fp16, layers=range(0,60,3); prompts.jsonl "
        "index-aligned (kind: scenario/heldout/template)",
    ),
    (
        "probe_sweep_it/",
        "INSTRUCT model battery+template activations: same schema, plus chat_* keys "
        "(chat-template formatted, thinking-OFF tokenizer default)",
    ),
    (
        "probe_validation/",
        "E1 battery activations, base model, layers 33/39 (acts key), plain format",
    ),
    (
        "template_thinking_on/",
        "E8 thinking-ON arm: the 37 numerical-template prompts re-collected with "
        "enable_thinking=True, chat_* keys, same layer grid",
    ),
    (
        "preferences_it/",
        "Q1.H3.E1 (plain format) A/B preference logits + activity-token activations. CAVEAT: "
        "collected BEFORE the left-padding fix (TREE Q1.H3.E4) — logits read mid-prompt for "
        "non-longest rows; kept for the record, superseded by preferences_it_fixed",
    ),
    (
        "preferences_it_chat/",
        "Q1.H3.E3 (chat format) preference collection. Same pre-fix CAVEAT as preferences_it; "
        "superseded by preferences_it_chat_fixed",
    ),
    (
        "steering_it/",
        "Q1.H3.E2 alpha=2 steered A/B logits per battery emotion. Same pre-fix CAVEAT; "
        "superseded by steering_it_fixed",
    ),
    (
        "steering_it_a8/",
        "Q1.H3.E2 alpha=8 arm. Same pre-fix CAVEAT; superseded by steering_it_a8_fixed",
    ),
    (
        "preferences_it_fixed/",
        "E1 rerun with the padding fix (padding-agnostic last-token indexing) — the "
        "evidence-bearing version",
    ),
    (
        "preferences_it_chat_fixed/",
        "E3 rerun with the padding fix — the evidence-bearing version",
    ),
    (
        "steering_it_fixed/",
        "E2 alpha=2 rerun with the padding fix — the evidence-bearing version",
    ),
    (
        "steering_it_a8_fixed/",
        "E2 alpha=8 rerun with the padding fix — the evidence-bearing version",
    ),
    (
        "emotion_vectors_it_means.npz",
        "instruct-model emotion means [171, 20, 5376] fp16 — LINEAGE: extracted from the EXTERNAL gemma-4-4B story corpus (snae/emotion_stories_gemma_4_4B)",
    ),
    (
        "e6_scale_means.npz",
        "E6 scale-test probe means at n in {16,64,128,256} stories/emotion — LINEAGE: SELF-GENERATED stories (written by the probed model, abotresol/emotion-stories-gemma-4-31b-it)",
    ),
    (
        "e7_neutral_bundle.npz",
        "E7 neutral-transcript activation vectors [128, 20, 5376] — LINEAGE: neutral transcripts written by the probed model",
    ),
    ("e8_template_diagnostic.json", "E8 confound diagnostic, instruct last-token readout"),
    ("e8_template_diagnostic_mean_all.json", "E8, mean-all readout"),
    ("e8_template_diagnostic_mean_content.json", "E8, mean-content readout"),
    ("e8_template_diagnostic_base.json", "E8, base model"),
    ("e8_template_diagnostic_mean_all_base.json", "E8, base, mean-all"),
    ("e8_template_diagnostic_mean_content_base.json", "E8, base, mean-content"),
    ("e8_template_diagnostic_thinking_on.json", "E8, thinking-ON arm"),
    ("e9_centered_readout.json", "E9 centered-cosine readout grid + falsify reads, both models"),
    (
        "e5_cross_lineage_n256.json",
        "cross-lineage comparison at converged scale: self-gen vs corpus directions cos 0.219",
    ),
    (
        "e10_lineage_headtohead.json",
        "lineage head-to-head under working readouts: own-story probes 11/12+9/12 at layer 33, corpus probes pass nowhere (matched conventions)",
    ),
    ("logit_lens_base_L33.json", "logit lens tables, base layer 33 (partial positive)"),
    ("logit_lens_base_L57.json", "logit lens, base layer 57"),
    ("logit_lens_it_L33_normed.json", "logit lens, instruct layer 33 (negative)"),
    ("logit_lens_it_L57.json", "logit lens, instruct layer 57 (negative)"),
    ("falsify_c1_scorecard.json", "falsify gate scorecard, C1 geometry (survived)"),
    ("falsify_c2_scorecard.json", "falsify gate scorecard, C2 noise ceiling (failed)"),
    ("falsify_c3_scorecard.json", "falsify gate scorecard, C3 probe null (weakened)"),
    ("falsify_h3_scorecard.json", "H3 preference scorecard (pre-fix arms; superseded)"),
    ("falsify_h3_steering_scorecard.json", "H3 steering scorecard (pre-fix arms; superseded)"),
    ("emotion_geometry_correlations.json", "per-layer PC-valence/arousal correlations, base"),
    ("emotion_geometry_correlations_it.json", "same, instruct"),
    ("geometry_pc_demotion.json", "instruction-tuning valence-demotion diagnostic"),
    ("probe_stability.json", "bootstrap probe-direction stability at corpus scale"),
    ("vllm_activation_bench.json", "vLLM hook feasibility + numerics parity bench"),
    ("activation_engine_bench.json", "three-engine activation-extraction throughput bench"),
]

REPO_SUFFIX = "emotion-vectors-experiment-artifacts"


def card(commit: str, rows: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"| `{p}` | {d} |" for p, d in rows)
    return f"""# Emotion-vectors replication on Gemma-4-31B: experiment artifacts

Every activation tensor, prompt set, and scored output behind the report
notebooks of [cbai-cambria-project](https://github.com/Antonio-Tresol/cbai-cambria-project)
(commit `{commit}`), published so replication does NOT require re-running
inference. The research record (hypotheses, pre-registered predictions,
verdicts) is the repo's TREE.md; the daily log is RESEARCH_LOG.md.

Models: google/gemma-4-31b (base) and google/gemma-4-31b-it (instruct),
bf16. Layers: range(0, 60, 3) unless stated. d_model = 5376. Activations
fp16 in npz unless stated; prompts index-aligned in the sibling
prompts.jsonl of each directory.

KNOWN INSTRUMENT BUG, labeled per-directory below: collections made before
2026-07-22 with batched right-padding assumptions carry mid-prompt logit
reads (TREE Q1.H3.E4). The *_fixed directories are the evidence-bearing
versions; the originals are kept for the record.

Not included: the NRC VAD lexicon v2.1 (third-party; obtain from
saifmohammad.com), the story corpora and extraction vectors (published as
separate datasets under this account), and anything derivable from the
code alone.

| file / directory | contents |
|---|---|
{lines}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    api = HfApi()
    repo = f"{api.whoami()['name']}/{REPO_SUFFIX}"
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    present = [(p, d) for p, d in ARTIFACTS if (args.results / p.rstrip("/")).exists()]
    missing = [p for p, _ in ARTIFACTS if not (args.results / p.rstrip("/")).exists()]
    api.upload_file(
        path_or_fileobj=card(commit, present).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    for path, _ in present:
        src = args.results / path.rstrip("/")
        if src.is_dir():
            api.upload_folder(
                folder_path=str(src),
                repo_id=repo,
                repo_type="dataset",
                path_in_repo=path.rstrip("/"),
            )
        else:
            api.upload_file(
                path_or_fileobj=src.read_bytes(),
                path_in_repo=path,
                repo_id=repo,
                repo_type="dataset",
            )
        print(f"uploaded {path}")
    if missing:
        print(f"not yet present (re-run after they land): {missing}")
    print(f"-> https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
