"""Bootstrap stability of probe directions from per-story shards — E5 prerequisite.

Quantifies the corpus-scale gap (ours ~9 stories/emotion vs the paper's ~1,000):
resample each emotion's stories with replacement, recompute the token-weighted
mean and the centered contrast direction, and measure cosine to the full-sample
direction. Run where the shards live (pod). Writes results/probe_stability.json.
"""

import json
from pathlib import Path

import numpy as np

from emotion_vectors.story_store import load_manifest

rng = np.random.default_rng(20260721)
out = Path("results/emotion_vectors")
manifest = [r for r in load_manifest(out / "manifest.jsonl").values() if r.get("error") is None]
by_emotion: dict[str, list[dict]] = {}
for r in manifest:
    by_emotion.setdefault(r["emotion"], []).append(r)
layers = json.loads((out / "run_config.json").read_text())["layers"]
LP = layers.index(33)


def load_vec(row):
    return np.load(out / "shards" / (row["story_id"] + ".npy"))[LP].astype(np.float64)


def wmean(rows, vecs):
    w = np.array([r["n_tokens"] for r in rows], dtype=np.float64)
    return (vecs * w[:, None]).sum(0) / w.sum()


emos = list(by_emotion)
all_vecs = {e: np.stack([load_vec(r) for r in by_emotion[e]]) for e in emos}
full_means = {e: wmean(by_emotion[e], all_vecs[e]) for e in emos}

# raw mean-vector stability per emotion
raw_stats = []
for e in emos:
    rows, vecs, full = by_emotion[e], all_vecs[e], full_means[e]
    cos = []
    for _ in range(200):
        idx = rng.integers(0, len(rows), len(rows))
        b = wmean([rows[i] for i in idx], vecs[idx])
        cos.append(float(full @ b / (np.linalg.norm(full) * np.linalg.norm(b))))
    raw_stats.append((np.mean(cos), np.percentile(cos, 5)))
raw = np.array(raw_stats)
print(
    f"raw mean-vector bootstrap cosine (layer 33): mean {raw[:, 0].mean():.4f}, "
    f"worst-emotion p5 {raw[:, 1].min():.4f}"
)

# stability of the centered CONTRAST direction (the probe actually used)
grand = np.mean(list(full_means.values()), axis=0)
cos_c = {e: [] for e in emos}
for _ in range(50):
    boot_means = {}
    for e in emos:
        idx = rng.integers(0, len(by_emotion[e]), len(by_emotion[e]))
        boot_means[e] = wmean([by_emotion[e][i] for i in idx], all_vecs[e][idx])
    g = np.mean(list(boot_means.values()), axis=0)
    for e in emos:
        a, b = full_means[e] - grand, boot_means[e] - g
        cos_c[e].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
cvals = np.array([np.mean(v) for v in cos_c.values()])
print(
    f"centered CONTRAST direction bootstrap cosine: mean {cvals.mean():.4f}, min {cvals.min():.4f}"
)
worst = sorted(cos_c, key=lambda e: float(np.mean(cos_c[e])))[:5]
print("least stable emotions:", [(e, round(float(np.mean(cos_c[e])), 3)) for e in worst])

Path("results/probe_stability.json").write_text(
    json.dumps(
        {
            "layer": 33,
            "n_bootstrap_raw": 200,
            "n_bootstrap_contrast": 50,
            "seed": 20260721,
            "raw_mean_cosine": round(float(raw[:, 0].mean()), 4),
            "raw_worst_emotion_p5": round(float(raw[:, 1].min()), 4),
            "contrast_mean_cosine": round(float(cvals.mean()), 4),
            "contrast_min_cosine": round(float(cvals.min()), 4),
            "contrast_per_emotion": {e: round(float(np.mean(v)), 4) for e, v in cos_c.items()},
            "note": "contrast = centered probe direction, the quantity the battery uses; "
            "~9 stories/emotion vs the paper's ~1000",
        },
        indent=2,
    )
    + "\n"
)
print("written -> results/probe_stability.json")
