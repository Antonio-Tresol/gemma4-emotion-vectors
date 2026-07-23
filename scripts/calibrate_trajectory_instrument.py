"""Q3 trajectory-instrument calibration — NO true-probe outcome reads.

Everything here uses (a) per-token norms, (b) the 24 fixed-seed RANDOM probe
columns, and (c) simulation with matched noise. The 171 corpus + 12 selfgen
probe columns are never read, so the Q3.H1.E1 scoring registration stays
clean: this script characterizes the instrument, not the effect.

What it measures (all layers, fixed-seed 300-story SEQUENTIAL sample):
  1. norm drift across story position (E8 lesson: drift fakes trends);
  2. random-cosine scale, position-drift |rho|, within-phase noise sd, AR(1);
  3. null bands for the registered reads: anticipation lead (W=16) and
     sustained-crossover existence (K=5) under random-pair assignment;
  4. step-blur decidability: false-ramp rate on simulated TRUE STEPS with
     boundary jitter, and detection power on simulated TRUE RAMPS, at the
     measured layer-33 noise. This sets which amplitudes/widths are decidable
     per-transition and motivates the hierarchical width read.

Shards are fetched from the published HF substrate into the HF cache (or use
--shard-dir to reuse an existing local sample).

    uv run python scripts/calibrate_trajectory_instrument.py            # -it arm
    uv run python scripts/calibrate_trajectory_instrument.py --arm base
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LAYERS = [6, 15, 24, 33, 42, 51]
SAMPLE_SEED = 20260722
REPOS = {
    "it": "abotresol/emotion-combined-trajectories-gemma-4-31b-it",
    "base": "abotresol/emotion-combined-trajectories-gemma-4-31b",
}
MANIFESTS = {
    "it": "results/combined_trajectories/manifest.jsonl",
    "base": "results/combined_trajectories_base/manifest.jsonl",
}
W = 16  # anticipation window (tokens); phase p5 length is 54, so 2W fits
K = 5   # sustained-run length for the crossover-existence null
SIM_WINDOW = 48        # fit window [-24, +24) around the boundary
DELTA_BIC = 2          # ramp must beat step by this much
RAMP_WIDTHS = [2, 4, 6, 8, 12, 16, 24]  # 10-90% widths on the fit grid


def sample_ids(manifest: str, n: int) -> list[str]:
    rows = [json.loads(line) for line in open(manifest)]
    seq = [
        r for r in rows
        if r["mode"] == "SEQUENTIAL" and not r["error"]
        and len(r["phase_token_starts"]) == 3
    ]
    random.Random(SAMPLE_SEED).shuffle(seq)
    return [r["story_id"] for r in seq[:n]]


def load_stories(repo: str, ids: list[str], rand_idx: list[int],
                 shard_dir: str | None) -> list[dict]:
    if shard_dir:
        paths = [Path(shard_dir) / f"{sid}.npz" for sid in ids]
    else:
        from huggingface_hub import hf_hub_download
        with ThreadPoolExecutor(16) as ex:
            paths = list(ex.map(
                lambda sid: hf_hub_download(
                    repo, f"shards/{sid}.npz", repo_type="dataset"), ids))
    stories = []
    for p in paths:
        d = np.load(p)
        stories.append(dict(
            cos=(d["dots"][:, :, rand_idx].astype(np.float32)
                 / d["norms"].astype(np.float32)[:, :, None]),  # (T, 6, 24)
            norms=d["norms"].astype(np.float32),
            starts=d["phase_token_starts"].tolist(),
            T=int(d["norms"].shape[0])))
    return stories


def norm_drift(stories: list[dict]) -> dict:
    out = {}
    for li, layer in enumerate(LAYERS):
        r = [s["norms"][int(.75 * s["T"]):, li].mean()
             / s["norms"][:int(.25 * s["T"]), li].mean() for s in stories]
        out[str(layer)] = round(float(np.mean(r)), 4)
    return out


def random_probe_stats(stories: list[dict]) -> dict:
    stats = {}
    for li, layer in enumerate(LAYERS):
        sd_all, rho, ar1, phase_sd = [], [], [], []
        for s in stories:
            c = s["cos"][:, li, :]
            sd_all.append(c.std(axis=0))
            tr = np.argsort(np.argsort(np.arange(s["T"])))
            for j in range(0, 24, 3):
                cr = np.argsort(np.argsort(c[:, j]))
                rho.append(abs(np.corrcoef(tr, cr)[0, 1]))
            bounds = s["starts"] + [s["T"]]
            for p in range(len(bounds) - 1):
                seg = c[bounds[p]:bounds[p + 1], :]
                if len(seg) < 12:
                    continue
                x = seg - np.linspace(seg[0], seg[-1], len(seg))
                phase_sd.append(x.std(axis=0).mean())
                x0 = x[:-1] - x[:-1].mean(0)
                x1 = x[1:] - x[1:].mean(0)
                ar1.append((x0 * x1).sum()
                           / np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum()))
        stats[str(layer)] = dict(
            cos_sd=float(np.mean(sd_all)),
            drift_absrho_med=float(np.median(rho)),
            drift_absrho_p90=float(np.quantile(rho, .9)),
            phase_noise_sd=float(np.mean(phase_sd)),
            ar1=float(np.mean(ar1)))
    return stats


def null_reads(stories: list[dict]) -> dict:
    nulls = {}
    pair_rng = np.random.default_rng(1)
    for li, layer in enumerate(LAYERS):
        anticip, crossed, total = [], 0, 0
        for s in stories:
            c = s["cos"][:, li, :]
            for b in s["starts"][1:]:
                if b < 2 * W or s["T"] - b < W:
                    continue
                jin, jout = pair_rng.choice(24, 2, replace=False)
                anticip.append(float(
                    c[b - W:b, jin].mean() - c[b - 2 * W:b - W, jin].mean()))
                lo, hi = max(0, b - W), min(s["T"], b + 2 * W)
                dcurve = c[lo:hi, jin] - c[lo:hi, jout]
                run, hit = 0, False
                for v in np.sign(dcurve - dcurve[0]):
                    run = run + 1 if v > 0 else 0
                    if run >= K:
                        hit = True
                        break
                crossed += hit
                total += 1
        a = np.array(anticip)
        nulls[str(layer)] = dict(
            anticip=dict(mean=float(a.mean()), sd=float(a.std()),
                         p95=float(np.quantile(a, .95)), n=len(a)),
            cross_frac=crossed / total)
    return nulls


def _fit_step_vs_ramp(y: np.ndarray, tgrid: np.ndarray) -> tuple[float, float, float]:
    best = {}
    n = len(y)
    for name, widths in [("step", [1e-6]), ("ramp", RAMP_WIDTHS)]:
        sse_best, w_best = np.inf, None
        for c in range(-8, 9, 2):
            for w in widths:
                f = 1 / (1 + np.exp(np.clip(-(tgrid - c) / max(w / 4.6, 1e-6),
                                            -60, 60)))
                X = np.stack([np.ones(n), f], 1)
                beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = res[0] if len(res) else float(((y - X @ beta) ** 2).sum())
                if sse < sse_best:
                    sse_best, w_best = sse, w
        k = 3 if name == "step" else 4
        best[name] = (n * np.log(sse_best / n) + k * np.log(n), w_best)
    return best["step"][0], best["ramp"][0], best["ramp"][1]


def _gen_ar1(n: int, length: int, sigma: float, phi: float,
             rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal((n, length)) * sigma * np.sqrt(1 - phi ** 2)
    x = np.zeros((n, length))
    x[:, 0] = e[:, 0] / np.sqrt(1 - phi ** 2)
    for t in range(1, length):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def decidability_sims(sigma: float, phi: float, nsim: int) -> dict:
    tgrid = np.arange(SIM_WINDOW) - SIM_WINDOW / 2
    rng = np.random.default_rng(7)
    out = {"true_step_false_ramp": {}, "true_ramp_power": {}}
    for amp_sd in [1, 2, 4]:
        for jit in [0, 2, 4]:
            x = _gen_ar1(nsim, SIM_WINDOW, sigma, phi, rng)
            jitters = np.round(rng.standard_normal(nsim) * jit)
            for i in range(nsim):
                x[i] += amp_sd * sigma * (tgrid >= jitters[i])
            wins, widths = 0, []
            for y in x:
                bs, br, w = _fit_step_vs_ramp(y, tgrid)
                if br < bs - DELTA_BIC:
                    wins += 1
                    widths.append(w)
            out["true_step_false_ramp"][f"amp{amp_sd}sd_jit{jit}"] = dict(
                ramp_wins_frac=wins / nsim,
                median_spurious_width=float(np.median(widths)) if widths else None)
    for amp_sd in [1, 2, 4]:
        for width in [8, 16, 24]:
            x = _gen_ar1(nsim, SIM_WINDOW, sigma, phi, rng)
            jitters = np.round(rng.standard_normal(nsim) * 2)
            for i in range(nsim):
                x[i] += amp_sd * sigma / (1 + np.exp(np.clip(
                    -(tgrid - jitters[i]) / (width / 4.6), -60, 60)))
            wins, widths = 0, []
            for y in x:
                bs, br, w = _fit_step_vs_ramp(y, tgrid)
                if br < bs - DELTA_BIC:
                    wins += 1
                    widths.append(w)
            out["true_ramp_power"][f"amp{amp_sd}sd_w{width}"] = dict(
                ramp_wins_frac=wins / nsim,
                median_fitted_width=float(np.median(widths)) if widths else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["it", "base"], default="it")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nsim", type=int, default=300)
    ap.add_argument("--shard-dir", default=None,
                    help="reuse local shard dir instead of downloading")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels_path = Path(MANIFESTS[args.arm]).parent / "probe_labels.json"
    labels = json.load(open(labels_path))
    rand_idx = [i for i, l in enumerate(labels) if l.startswith("random")]
    assert len(rand_idx) == 24, f"expected 24 random probes, got {len(rand_idx)}"

    ids = sample_ids(MANIFESTS[args.arm], args.n)
    stories = load_stories(REPOS[args.arm], ids, rand_idx, args.shard_dir)
    print(f"loaded {len(stories)} stories ({args.arm} arm)")

    out = {
        "arm": args.arm, "n_stories": len(stories), "layers": LAYERS,
        "sample_seed": SAMPLE_SEED, "read_params": dict(W=W, K=K),
        "norm_lastq_over_firstq": norm_drift(stories),
        "random_probe": random_probe_stats(stories),
        "null_reads": null_reads(stories),
    }
    p33 = out["random_probe"]["33"]
    out["sims"] = decidability_sims(p33["phase_noise_sd"], p33["ar1"], args.nsim)
    out["sim_params"] = dict(sigma=p33["phase_noise_sd"], phi=p33["ar1"],
                             nsim=args.nsim, delta_bic=DELTA_BIC,
                             window=SIM_WINDOW, ramp_jitter_sd=2)

    out_path = args.out or f"results/trajectory_instrument_calibration_{args.arm}.json"
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
