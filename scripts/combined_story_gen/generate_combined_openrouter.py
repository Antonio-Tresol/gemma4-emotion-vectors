"""Q3 prep — the combined-story battery written by a strong external generator.

Same recipe as generate_combined_stories.py in every respect except the
generator: SYSTEM_PROMPT, the 173-triple x 12-combo grid, the seeded
setting/name picks, the QC gates (length, leakage, tag-match), and the row
schema are all imported from that script; only the backend swaps from local
vLLM (Gemma writing its own stories) to OpenRouter. Motivated by E11/E12:
generator QUALITY dominates probe function, so Q3's trajectory substrate
gets a strong-generator arm. Gemma still does the reading — trajectories
are the probed model's per-token states over the text, whoever wrote it.

Per the same Q3 gate note as the parent script: generation is legitimate
preparation, but no run from this script counts as a logged Q3 experiment
result until the orchestrator confirms the gate and registers predictions.

OpenRouter cannot seed sampling, so the combo seed drives only the
setting/name pick (recorded per row); story-level sampling is not
bit-reproducible, the corpus file itself is the frozen artifact.

    .venv/bin/python scripts/combined_story_gen/generate_combined_openrouter.py --smoke
    .venv/bin/python scripts/combined_story_gen/generate_combined_openrouter.py --per-triple 36
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from generate_combined_stories import (
    N_COMBOS,
    SYSTEM_PROMPT,
    assemble_grouped,
    build_user_prompt,
    combos_for,
    emotion_tags,
    existing_counts,
    git_commit,
    leaked_words,
    load_triples,
    row_key,
    strip_thinking,
    tags_match_emotions,
)

API = "https://openrouter.ai/api/v1/chat/completions"


def one_request(job: dict, args: argparse.Namespace, key: str) -> dict:
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(job["perm"], job["mode"], job["seed"]),
                },
            ],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": 1500,
            "reasoning": {"enabled": False},
        }
    ).encode()
    req = Request(
        API,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read())
            raw_text = response["choices"][0]["message"]["content"].strip()
            return {
                **job,
                "raw_text": raw_text,
                "model": response.get("model", args.model),
                "tokens_out": response.get("usage", {}).get("completion_tokens"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — API flakiness: retry, then record
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {**job, "raw_text": None, "model": args.model, "error": last_error}


def qc_row(res: dict) -> tuple[dict | None, str]:
    """QC gates in the parent script's order; returns (row-or-None, verdict)."""
    text, had_think = strip_thinking(res["raw_text"])
    emotions = res["triple"]["emotions"]
    leaks = leaked_words(text, emotions)
    tags = emotion_tags(text)
    tags_ok = tags_match_emotions(tags, emotions)
    if len(text) < 100:
        return None, "short"
    if leaks:
        return None, "leak"
    if not tags_ok:
        return None, "tag_mismatch"
    return {
        "triple_id": res["triple_id"],
        "emotions": emotions,
        "category": res["triple"].get("category"),
        "has_nonaffect": res["triple"].get("has_nonaffect"),
        "mode": res["mode"],
        "permutation": list(res["perm"]),
        "perm_idx": res["perm_idx"],
        "text": text,
        "had_thinking_trace": had_think,
        "leaked_words": leaks,
        "n_tags": len(tags),
        "tags": tags,
        "tags_match_emotions": tags_ok,
        "n_chars": len(text),
        "seed": res["seed"],
        "generator_model": res["model"],
        "tokens_out": res.get("tokens_out"),
        "error": None,
    }, "kept"


def pending_jobs(
    triples: list[dict], counts: dict[str, int], samples_per_combo: int, base_seed: int
) -> list[dict]:
    jobs = []
    for triple_id, triple in enumerate(triples):
        combos = combos_for(triple["emotions"])
        for combo_idx, (mode, perm, perm_idx) in enumerate(combos):
            seed = base_seed + triple_id * 100 + combo_idx  # parent script's seed scheme
            need = samples_per_combo - counts.get(row_key(triple_id, mode, perm_idx), 0)
            for _ in range(max(0, need)):
                jobs.append(
                    {
                        "triple_id": triple_id,
                        "triple": triple,
                        "mode": mode,
                        "perm": perm,
                        "perm_idx": perm_idx,
                        "seed": seed,
                    }
                )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument(
        "--triples-file", type=Path, default=Path(__file__).parent / "emotions_triples_v1.json"
    )
    parser.add_argument("--per-triple", type=int, default=36)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--smoke", action="store_true", help="2 triples, 1 sample/combo")
    parser.add_argument(
        "--constant-battery",
        action="store_true",
        help="CONTROL arm: one (e, e, e) pseudo-triple per battery emotion — same "
        "SEQUENTIAL scaffold, transition events, and tag structure with NO emotion "
        "change; any transition-locked signal here is scene-change artifact",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined_stories_deepseek"))
    args = parser.parse_args()
    load_dotenv()
    key = os.environ["OPENROUTER_KEY"]
    if args.smoke:
        args.per_triple = N_COMBOS
    if args.per_triple % N_COMBOS != 0:
        raise SystemExit(f"--per-triple must be a multiple of {N_COMBOS}, got {args.per_triple}")
    samples_per_combo = args.per_triple // N_COMBOS
    if args.constant_battery:
        battery = (
            "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised"
        )
        triples = [
            {
                "emotions": [emotion, emotion, emotion],
                "category": "CONTROL_CONSTANT",
                "has_nonaffect": False,
            }
            for emotion in battery.split()
        ]
    else:
        triples = load_triples(args.triples_file)
    if args.smoke:
        triples = triples[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "stories_raw.jsonl"
    counts = existing_counts(raw_path)
    jobs = pending_jobs(triples, counts, samples_per_combo, args.seed)
    # CLI args verbatim, with Path values stringified for JSON
    args_jsonable = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(args).items()
    }
    config = {
        **args_jsonable,
        "n_combos": N_COMBOS,
        "samples_per_combo": samples_per_combo,
        "n_requests": len(jobs),
        "git_commit": git_commit(),
        "note": "OpenRouter sampling is unseeded; seeds drive setting/name picks only",
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"{len(jobs)} requests pending (resuming from {sum(counts.values())} kept) "
        f"with {args.model} ({args.workers} workers)"
    )
    start_time = time.monotonic()
    stats = {"kept": 0, "short": 0, "leak": 0, "tag_mismatch": 0, "api_error": 0}
    total_tokens_out = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, raw_path.open("a") as sink:
        futures = [pool.submit(one_request, job, args, key) for job in jobs]
        for n_done, future in enumerate(as_completed(futures), 1):
            res = future.result()
            total_tokens_out += res.get("tokens_out") or 0
            if res["error"] is not None:
                stats["api_error"] += 1
            else:
                row, verdict = qc_row(res)
                stats[verdict] += 1
                if row is not None:
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
            if n_done % 100 == 0 or n_done == len(jobs):
                rate = n_done / max(time.monotonic() - start_time, 1)
                eta_minutes = (len(jobs) - n_done) / max(rate, 0.01) / 60
                print(
                    f"{n_done}/{len(jobs)} | {stats} | {total_tokens_out} tokens out | "
                    f"{rate:.1f} req/s | ETA {eta_minutes:.1f} min",
                    flush=True,
                )
    n_grouped = assemble_grouped(raw_path, args.out_dir / "stories_grouped.jsonl")
    print(f"DONE: {stats} | {n_grouped} triples grouped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
