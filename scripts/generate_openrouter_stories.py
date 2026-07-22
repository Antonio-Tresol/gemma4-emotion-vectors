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

BATTERY = "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised".split()
INSTRUCTION = (
    "Write a short third-person story, around 150 words, about a person experiencing "
    "{emotion}. Do not name the emotion anywhere in the text."
)
API = "https://openrouter.ai/api/v1/chat/completions"


def one_story(model: str, emotion: str, temperature: float, key: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": INSTRUCTION.format(emotion=emotion)}],
            "temperature": temperature,
            "max_tokens": 400,
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
                out = json.loads(resp.read())
            text = out["choices"][0]["message"]["content"].strip()
            usage = out.get("usage", {})
            leaked = bool(re.search(rf"\b{re.escape(emotion)}\w*", text, re.IGNORECASE))
            return {
                "emotion": emotion,
                "text": text,
                "leaked": leaked,
                "model": out.get("model", model),
                "tokens_out": usage.get("completion_tokens"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — API flakiness: retry, then record
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {"emotion": emotion, "text": None, "leaked": None, "model": model, "error": last}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--per-emotion", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("results/openrouter_stories"))
    parser.add_argument("--smoke", action="store_true", help="2 stories per emotion")
    args = parser.parse_args()
    load_dotenv()
    key = os.environ["OPENROUTER_KEY"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "stories_raw.jsonl"
    per = 2 if args.smoke else args.per_emotion
    done: dict[str, int] = {e: 0 for e in BATTERY}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if row["error"] is None:
                done[row["emotion"]] = done.get(row["emotion"], 0) + 1
    todo = [e for e in BATTERY for _ in range(max(0, per - done[e]))]
    config = {
        "model": args.model,
        "instruction": INSTRUCTION,
        "per_emotion": per,
        "temperature": args.temperature,
        "resumed_counts": done,
        "n_requests": len(todo),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"generating {len(todo)} stories with {args.model} ({args.workers} workers)")
    t0 = time.monotonic()
    n_ok = n_err = tok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a") as sink:
        futures = [pool.submit(one_story, args.model, e, args.temperature, key) for e in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            n_ok += row["error"] is None
            n_err += row["error"] is not None
            tok += row.get("tokens_out") or 0
            if i % 100 == 0 or i == len(todo):
                rate = i / max(time.monotonic() - t0, 1)
                print(
                    f"{i}/{len(todo)} | ok {n_ok} err {n_err} | {tok} tokens out | "
                    f"{rate:.1f} req/s | ETA {(len(todo) - i) / max(rate, 0.01) / 60:.1f} min",
                    flush=True,
                )
    print(f"DONE: {n_ok} stories, {n_err} errors, {tok} completion tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
