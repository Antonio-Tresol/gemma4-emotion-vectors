# Data index: every dataset, where it lives, and how to load it

One page for humans and agents: what data exists, which Hugging Face (HF)
dataset holds it, and how code should load it. The machine-readable version
of this mapping is `ROUTES` in `src/emotion_vectors/artifacts.py`; if this
page and that table disagree, the table wins and this page has a bug.

## How loading works (use this, never hardcode paths)

```python
from emotion_vectors.artifacts import fetch
path = fetch("q3_records_it_v2.npz")          # flat file
path = fetch("combined_trajectories/manifest.jsonl")  # routed subtree
```

`fetch()` resolves a `results/`-relative path: the local `results/` tree
first, then the HF dataset that owns that subtree (`ROUTES`), and for any
path with no route, the catch-all artifacts repo. Private repos need an
`HF_TOKEN=` line in the repo-root `.env` (gitignored; each machine keeps its
own). Publishing goes through the `scripts/publish_*.py` scripts, each of
which writes its dataset card; new evidence files are added to
`scripts/publish_experiment_artifacts.py`'s `ARTIFACTS` list.

## The catch-all: experiment artifacts

| dataset | contents |
|---|---|
| [`abotresol/emotion-vectors-experiment-artifacts`](https://huggingface.co/datasets/abotresol/emotion-vectors-experiment-artifacts) | Every scored output, scorecard, calibration, and per-record substrate the report notebooks cite: detection sweeps, preference/steering runs (pre- and post-fix), falsify scorecards, geometry diagnostics (`it_pc_structure.json`, `rsa_fragmentation.json`), Q3 gate/anticipation scores (`q3_gate_r1_*.json`), and the five per-record dumps `q3_records_{it,it_v2,deepseek,deepseek_constant,base}.npz` + `_meta.json`. Its README lists a one-line schema per file. |

## Story corpora (text the models read or wrote)

| dataset | contents |
|---|---|
| external corpus `snae/emotion_stories_gemma_4_4B` | The open replication's 171-emotion story corpus written by gemma-4-4B; extraction source for all corpus-lineage vectors. Not ours. |
| `abotresol/emotion-stories-gemma-4-31b-it` | Self-generated stories (the probed instruct model writing about each battery emotion). |
| `abotresol/emotion-dialogues-gemma-4-31b` / `-it` | Dialogue-format probe corpora, base and instruct generators. |
| `abotresol/neutral-transcripts-gemma-4-31b-it` | Neutral (no-emotion) transcripts, the neutral-projection control. |
| `abotresol/emotion-stories-deepseek-v4-pro` | Fixed-prompt DeepSeek-written battery stories (strong-generator arm). |
| `abotresol/emotion-stories-deepseek-v4-pro-diverse` | Prompt-diversified DeepSeek corpus (the scale-and-diversity dose-response arm). |
| `abotresol/emotion-combined-stories-gemma-4-31b-it` | The Q3 three-emotion transition stories, Gemma-written (primary substrate text). |
| `abotresol/emotion-combined-stories-deepseek-v4-pro` | Same 173-triple recipe, DeepSeek-written. |
| `abotresol/emotion-combined-stories-deepseek-v4-pro-constant-control` | Constant-emotion control stories (scene changes, no emotion change). |

## Vector / activation sets (extraction outputs)

Pre-fix sets are kept for the padding-bug before/after audit; the `-postfix`
sets are the blessed instrument for all work from 2026-07-22 on.

| dataset | contents | public? |
|---|---|---|
| `abotresol/emotion-vectors-gemma-4-31b` / `-it` | Corpus-lineage emotion means, pre-fix, base/instruct | yes |
| `abotresol/emotion-vectors-gemma-4-31b-postfix` / `-it-postfix` | Post-fix re-extractions (the blessed sets) | private until sprint end |
| `abotresol/emotion-selfstory-vectors-gemma-4-31b-it` (+ `-postfix`) | Self-generated-lineage vectors | pre-fix yes / postfix private |
| `abotresol/emotion-dialogue-vectors-gemma-4-31b` / `-it` | Dialogue-lineage vectors | yes |
| `abotresol/neutral-vectors-gemma-4-31b-it` (+ `-postfix`) | Neutral-transcript activation vectors | pre-fix yes / postfix private |
| `abotresol/emotion-deepseek-vectors-gemma-4-31b-it` | Fixed-DeepSeek battery contrasts (best detection probes; also the Q3 "deepseek" probe bank) | yes |
| `abotresol/emotion-deepseek-diverse-vectors-gemma-4-31b-it` | Diverse-DeepSeek per-story vectors (dose-response subsampling needs no GPU) | yes |
| `abotresol/emotion-vectors-gemma-4-31b-smoke` | Smoke-test artifact, ignore | yes |

## Per-token trajectory substrates (the Q3 shards)

Each holds `manifest.jsonl`, `probe_labels.json`, `run_config.json`, and
`shards/<story_id>.npz` with per-token probe dots and norms.

| dataset | reader model | stories |
|---|---|---|
| `abotresol/emotion-combined-trajectories-gemma-4-31b-it` | instruct | Gemma-written (v1 probe bank) |
| `abotresol/emotion-combined-trajectories-gemma-4-31b-it-v2` | instruct | Gemma-written (post-fix corpus + selfgen + deepseek banks; primary) |
| `abotresol/emotion-combined-trajectories-gemma-4-31b` | base | Gemma-written (corpus base-lineage bank) |
| `abotresol/emotion-combined-trajectories-deepseek-stories-gemma-4-31b-it` | instruct | DeepSeek-written |
| `abotresol/emotion-combined-trajectories-constant-control-gemma-4-31b-it` | instruct | constant-emotion control |

## Deliberately NOT published

- The NRC VAD v2.1 lexicon (third-party license): download from
  saifmohammad.com into `data/lexicons/NRC-VAD-Lexicon-v2.1/`.
- Anything derivable from the code alone.

Unrelated to this project: `abotresol/gemma3-refusal-axis-data` (prior
thesis work).
