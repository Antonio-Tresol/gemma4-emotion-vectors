"""Q1.H2.E11 generation: the strong-external story corpus via OpenRouter.

Third-lineage test: same instruction as the self-generated corpus, verbatim,
but written by a strong external model (default deepseek/deepseek-chat) —
dissociating story quality from generator-distribution match. 12 battery
emotions x 256 stories, matching E6's converged scale. Leakage is FLAGGED
per row (emotion word appearing in its own story), mirroring the corpus
convention: rows are kept, the extraction-side filter decides.

    .venv/bin/python scripts/generate_openrouter_stories.py  # laptop, needs
    OPENROUTER_KEY in .env; ~3,072 requests, resumable by row count.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

BATTERY = (
    "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised".split()
)
INSTRUCTION = (
    "Write a short third-person story, around 150 words, about a person experiencing "
    "{emotion}. Do not name the emotion anywhere in the text."
)
# E12 diversity arm: same instruction core, but the protagonist and setting are
# pinned per slot so the corpus can't collapse onto one scaffold (Gemma's
# "Elias" failure mode). Slot j -> personas[j % 8] x settings[(j // 8) % 8],
# a deterministic 64-cell grid; at 1024 stories/emotion each cell gets 16.
DIVERSE_INSTRUCTION = (
    "Write a short third-person story, around 150 words, about {persona} {setting} "
    "experiencing {emotion}. Do not name the emotion anywhere in the text."
)
PERSONAS = [
    "a teenage girl",
    "an elderly fisherman",
    "a young nurse",
    "a middle-aged bus driver",
    "a retired schoolteacher",
    "a first-year college student",
    "a single father",
    "a street musician",
]
SETTINGS = [
    "in a crowded city market",
    "in a remote coastal village",
    "during a long winter night",
    "in a hospital waiting room",
    "on a cross-country train",
    "in a small mountain town",
    "during a summer festival",
    "in a quiet suburban kitchen",
]
API = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(emotion: str, slot: int, diverse: bool) -> str:
    if not diverse:
        return INSTRUCTION.format(emotion=emotion)
    persona = PERSONAS[slot % len(PERSONAS)]
    setting = SETTINGS[(slot // len(PERSONAS)) % len(SETTINGS)]
    return DIVERSE_INSTRUCTION.format(persona=persona, setting=setting, emotion=emotion)


def one_story(
    model: str, emotion: str, slot: int, temperature: float, key: str, diverse: bool
) -> dict:
    prompt = build_prompt(emotion, slot, diverse)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            # v4-pro spends completion tokens on hidden reasoning; a tight cap
            # truncates stories mid-sentence (caught 2026-07-22, ~2% of rows)
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
            text = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            leaked = bool(re.search(rf"\b{re.escape(emotion)}\w*", text, re.IGNORECASE))
            return {
                "emotion": emotion,
                "slot": slot,
                "text": text,
                "leaked": leaked,
                "model": response.get("model", model),
                "tokens_out": usage.get("completion_tokens"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — API flakiness: retry, then record
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {
        "emotion": emotion,
        "slot": slot,
        "text": None,
        "leaked": None,
        "model": model,
        "error": last_error,
    }


def resume_slots(out_path: Path) -> dict[str, set[int]]:
    """Completed (emotion, slot) pairs so each grid cell fills exactly once;
    legacy rows without a slot field fall back to arrival-order numbering."""
    done: dict[str, set[int]] = {emotion: set() for emotion in BATTERY}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if row["error"] is None:
                slots = done[row["emotion"]]
                if row.get("slot") is not None:
                    slots.add(row["slot"])
                else:
                    # legacy row: assign the next arrival-order slot number
                    slots.add(len(slots))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--per-emotion", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("results/openrouter_stories"))
    parser.add_argument("--smoke", action="store_true", help="2 stories per emotion")
    parser.add_argument(
        "--diverse", action="store_true", help="pin persona x setting per slot (E12 diversity arm)"
    )
    args = parser.parse_args()
    load_dotenv()
    key = os.environ["OPENROUTER_KEY"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "stories_raw.jsonl"
    stories_per_emotion = 2 if args.smoke else args.per_emotion
    done_slots = resume_slots(out_path)
    # every (emotion, slot) grid cell not already filled by a previous run
    todo = [
        (emotion, slot)
        for emotion in BATTERY
        for slot in range(stories_per_emotion)
        if slot not in done_slots[emotion]
    ]
    config = {
        "model": args.model,
        "instruction": DIVERSE_INSTRUCTION if args.diverse else INSTRUCTION,
        "diverse": args.diverse,
        "personas": PERSONAS if args.diverse else None,
        "settings": SETTINGS if args.diverse else None,
        "per_emotion": stories_per_emotion,
        "temperature": args.temperature,
        "resumed_counts": {emotion: len(slots) for emotion, slots in done_slots.items()},
        "n_requests": len(todo),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"generating {len(todo)} stories with {args.model} ({args.workers} workers)")
    start_time = time.monotonic()
    n_ok = n_err = total_tokens_out = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a") as sink:
        futures = [
            pool.submit(one_story, args.model, emotion, slot, args.temperature, key, args.diverse)
            for emotion, slot in todo
        ]
        for n_done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if row["error"] is None:
                n_ok += 1
            else:
                n_err += 1
            total_tokens_out += row.get("tokens_out") or 0
            if n_done % 100 == 0 or n_done == len(todo):
                rate = n_done / max(time.monotonic() - start_time, 1)
                eta_minutes = (len(todo) - n_done) / max(rate, 0.01) / 60
                print(
                    f"{n_done}/{len(todo)} | ok {n_ok} err {n_err} | "
                    f"{total_tokens_out} tokens out | "
                    f"{rate:.1f} req/s | ETA {eta_minutes:.1f} min",
                    flush=True,
                )
    print(f"DONE: {n_ok} stories, {n_err} errors, {total_tokens_out} completion tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
