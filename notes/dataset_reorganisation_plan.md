# Reorganising the published datasets, 2026-08-01

Twenty-nine dataset repositories, 55,861 files. Nothing breaks a Hugging Face
hard limit, but almost nothing is reusable: the dataset viewer is off
everywhere, no card declares a `configs:` block, and reading anything requires
this project's own `fetch()`.

Measured against the Hub's stated recommendations
(<https://huggingface.co/docs/hub/repositories-recommendations>):

| recommendation | limit | our worst | verdict |
|---|---|---|---|
| files per repo | <100k | 5,894 | fine |
| entries per folder | <10k | 5,888 (trajectory `shards/`) | fine, at 59% of the cap |
| file size | <200GB | 66MB | fine |
| formats integrated with the ecosystem | Parquet / WebDataset | `.npy` / `.npz` | **fails: no viewer** |
| avoid custom loading code | — | `emotion_vectors.artifacts.fetch` | **the reuse risk the Hub names** |
| dataset card | required | 29 of 29 | fine |

The Hub also documents the mechanism we are not using: "It is also possible to
define multiple subsets (also called *configurations*) for the same dataset
(e.g. if the dataset has various independent files)"
(<https://huggingface.co/docs/hub/datasets-manual-configuration>).

## Phase 1: the story corpora (the change worth making first)

Nine repositories, about 82MB of JSONL, become **one** dataset with two
subsets. This is the highest-value change because the corpora are the most
reusable thing the project produced: the headline finding is that *who writes
the stories* decides whether any of this works, and right now nobody can
compare those corpora without cloning the repo and calling our loader.

**`abotresol/emotion-story-corpora`**, Parquet, viewer on.

Each subset is a union of its corpora with a `source` column, so the
story-source comparison is a filter in the viewer rather than nine downloads.

### Subset `single_emotion` — 18,944 rows

| `source` | rows | replaces |
|---|---|---|
| `gemma-4-31b-it` | 3,072 | `emotion-stories-gemma-4-31b-it` |
| `deepseek-v4-pro` | 3,072 | `emotion-stories-deepseek-v4-pro` |
| `deepseek-v4-pro-diverse` | 12,288 | `emotion-stories-deepseek-v4-pro-diverse` |
| `dialogues-gemma-4-31b` | 192 | `emotion-dialogues-gemma-4-31b` |
| `dialogues-gemma-4-31b-it` | 192 | `emotion-dialogues-gemma-4-31b-it` |
| `neutral-gemma-4-31b-it` | 128 | `neutral-transcripts-gemma-4-31b-it` |

Columns: `source`, `emotion`, `text`, and the generation fields that only some
corpora carry, left null elsewhere: `model`, `leaked`, `tokens_out`, `slot`,
`error`.

### Subset `three_emotion` — 11,882 rows

| `source` | rows | replaces |
|---|---|---|
| `gemma-4-31b-it` | 5,888 | `emotion-combined-stories-gemma-4-31b-it` |
| `deepseek-v4-pro` | 5,785 | `emotion-combined-stories-deepseek-v4-pro` |
| `deepseek-v4-pro-constant-control` | 209 | `...-constant-control` |

Columns: `source`, `triple_id`, `emotions`, `permutation`, `perm_idx`, `mode`,
`category`, `has_nonaffect`, `text`, `tags`, `tags_match_emotions`,
`leaked_words`, `n_tags`, `n_chars`, `seed`, `had_thinking_trace`,
`generator_model`, `tokens_out`, `error`.

Note the naming defect this consolidation removes: `emotion-stories-gemma-4-31b-it`
currently ships its stories in files called `dialogues_raw.jsonl` and
`dialogues_grouped.jsonl`, which is simply wrong.

## Phase 2: the vector sets — deliberately NOT merged

Tempting and wrong. The pre-fix and post-fix sets differ by a padding bug that
reversed two published null results, and each `-postfix` repo carries a
`LINEAGE.md` saying so. Folding them into one repo with subset names would put
the buggy and corrected vectors one dropdown apart. The separation is the
provenance guarantee; keep it.

What to do instead: keep the repos, and make each card state in its first line
whether it is the corrected set or its superseded predecessor.

## Phase 3: the trajectory shards — the one structural issue

Four repositories hold 5,888 `.npz` files in a single `shards/` folder. That is
under the Hub's 10k-per-folder ceiling but is the closest thing here to a real
violation, and it is why those repos have no viewer. The Hub's advice is
explicit: "merge data into fewer files", and use WebDataset or Parquet.

Deferred, not dismissed: converting them means re-verifying every trajectory
number, which is a bigger job than the corpora and touches published results.

## What does NOT change

`emotion_vectors.artifacts.fetch` keeps working against the existing repos.
Nothing in this plan deletes or renames a published dataset, so every URL in
every card, notebook and log still resolves. The consolidated corpora dataset
is additive; the old corpora repos get a card line pointing at it.
