"""Q3 LLM-judge passes over the SEQUENTIAL story battery (OpenRouter).

Two tasks, both blind to activations AND to the tagged phase boundaries —
judges see only clean story text plus a target emotion, so cue positions
cannot anchor on the boundary they are meant to test:

  cue      For each transition, mark the EARLIEST verbatim phrase where the
           incoming emotion is signaled (read R1's cue-referenced twin,
           TREE.md Q3.H1.E1). Output rows carry the clean-text char offset;
           token mapping happens scoring-side via phase_token_starts.
  display  Construct-validity QC: rate how clearly a phase passage conveys
           its tagged emotion (clear/partial/absent). Separates "probes
           don't track" from "stories don't display" if the gate read fails.

Resumable: rows append to the output JSONL keyed (story_id, unit, model);
existing keys are skipped on rerun. Judge-instrument QC: run the same subset
under two models (--model), then compare with --agreement A.jsonl B.jsonl.

    uv run python scripts/judge_transition_cues.py --task cue --limit 30
    uv run python scripts/judge_transition_cues.py --task cue --limit 30 \
        --model google/gemini-2.5-flash
    uv run python scripts/judge_transition_cues.py --agreement \
        results/judge_cue_deepseek.jsonl results/judge_cue_gemini.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_vectors.trajectories import kept_rows, parse_story, story_id  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
STORIES = Path("results/combined_stories/stories_raw.jsonl")
SAMPLE_SEED = 20260722  # same fixed-seed ordering as the calibration sample

CUE_SYSTEM = (
    "You annotate short stories for an emotion-timing research project. "
    "Given a story and a target emotion, find the EARLIEST point in the story "
    "where that emotion begins to be conveyed, hinted at, or foreshadowed — "
    "through word choice, imagery, physical behavior, dialogue, or events. "
    "The emotion is never named explicitly; judge from how the text reads. "
    "Respond with ONLY a JSON object, no other text: "
    '{"quote": "<the EXACT verbatim contiguous substring of 4-12 words where '
    'the emotion is first signaled — copy it character-for-character>", '
    '"none": false} — or {"quote": "", "none": true} if the emotion is never '
    "conveyed anywhere in the story."
)

DISPLAY_SYSTEM = (
    "You rate short passages for an emotion-portrayal research project. Given "
    "a passage and a target emotion, rate how clearly the passage conveys that "
    "emotion (which is never named explicitly): 'clear' = unmistakably the "
    "dominant emotional tone; 'partial' = present but weak, mixed, or "
    "ambiguous; 'absent' = not conveyed. Respond with ONLY a JSON object: "
    '{"rating": "clear"|"partial"|"absent"}'
)


def call_judge(model: str, system: str, user: str, key: str, retries: int = 3) -> dict[str, object]:
    for attempt in range(retries):
        try:
            r = requests.post(
                API,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(m.group(0))
        except Exception as exc:  # noqa: BLE001 — retry then surface
            if attempt == retries - 1:
                return {"error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2**attempt)
    return {"error": "unreachable"}


def locate(quote: str, text: str) -> int:
    """Char offset of the judge's quote in clean text; -1 if not found."""
    if not quote:
        return -1
    i = text.find(quote)
    if i < 0:
        i = text.casefold().find(quote.casefold())
    if i < 0:  # tolerate whitespace-normalization drift
        norm = re.sub(r"\s+", " ", text)
        j = norm.casefold().find(re.sub(r"\s+", " ", quote).casefold())
        if j >= 0:
            running, k = 0, 0
            for k, ch in enumerate(text):
                if running == j:
                    break
                running += 0 if (ch.isspace() and k and text[k - 1].isspace()) else 1
            i = k
    return i


def build_units(task: str, limit: int | None) -> list[dict]:
    rows = [r for r in kept_rows(STORIES) if r["mode"] == "SEQUENTIAL"]
    parsed = []
    for r in rows:
        p = parse_story(r["text"], r["mode"])
        if len(p.phase_char_starts) != 3:
            continue
        parsed.append((story_id(r), r, p))
    random.Random(SAMPLE_SEED).shuffle(parsed)
    units = []
    for sid, r, p in parsed:
        bounds = list(p.phase_char_starts) + [len(p.clean_text)]
        if task == "cue":
            for ti in (1, 2):  # transitions INTO phases 1 and 2
                units.append(
                    dict(
                        story_id=sid,
                        unit=f"tr{ti}",
                        emotion=p.phase_emotions[ti],
                        text=p.clean_text,
                        boundary_char=p.phase_char_starts[ti],
                        category=r["category"],
                    )
                )
        else:  # display: every phase is a unit
            for pi in range(3):
                units.append(
                    dict(
                        story_id=sid,
                        unit=f"ph{pi}",
                        emotion=p.phase_emotions[pi],
                        text=p.clean_text[bounds[pi] : bounds[pi + 1]],
                        category=r["category"],
                    )
                )
        if limit and len(units) >= limit:
            break
    return units[:limit] if limit else units


def run_task(args: argparse.Namespace, key: str) -> None:
    units = build_units(args.task, args.limit)
    out = Path(
        args.out or f"results/judge_{args.task}_{args.model.split('/')[-1].split(':')[0]}.jsonl"
    )
    done = set()
    if out.exists():
        done = {(json.loads(l)["story_id"], json.loads(l)["unit"]) for l in open(out)}
    todo = [u for u in units if (u["story_id"], u["unit"]) not in done]
    print(f"{args.task}: {len(units)} units, {len(done)} done, {len(todo)} to run -> {out}")
    system = CUE_SYSTEM if args.task == "cue" else DISPLAY_SYSTEM

    def work(u: dict[str, object]) -> dict[str, object]:
        if args.task == "cue":
            user = f"Target emotion: {u['emotion']}\n\nStory:\n{u['text']}"
        else:
            user = f"Target emotion: {u['emotion']}\n\nPassage:\n{u['text']}"
        res = call_judge(args.model, system, user, key)
        row = {k: u[k] for k in ("story_id", "unit", "emotion", "category")}
        row["model"] = args.model
        if "error" in res:
            row["error"] = res["error"]
        elif args.task == "cue":
            row["none"] = bool(res.get("none"))
            row["quote"] = res.get("quote", "")
            row["cue_char"] = -1 if row["none"] else locate(row["quote"], u["text"])
            row["boundary_char"] = u["boundary_char"]
            row["cue_minus_boundary"] = (
                None if row["cue_char"] < 0 else row["cue_char"] - u["boundary_char"]
            )
        else:
            row["rating"] = res.get("rating")
        return row

    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as fh:
        n_err = 0
        for i, row in enumerate(ex.map(work, todo)):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_err += "error" in row
            if i and i % 50 == 0:
                print(f"  {i}/{len(todo)} ({n_err} errors)")
            if n_err > max(10, 0.2 * (i + 1)):
                raise SystemExit(
                    f"error rate too high ({n_err}/{i + 1}) — check model id / key / rate limits"
                )
    print(f"done: {len(todo)} rows, {n_err} errors")


def agreement(file_a: str, file_b: str) -> None:
    def load(f: str) -> dict[tuple[str, object], dict[str, object]]:
        return {(r["story_id"], r["unit"]): r for r in map(json.loads, open(f)) if "error" not in r}

    a, b = load(file_a), load(file_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no overlapping units")
    if "rating" in a[keys[0]]:
        same = sum(a[k]["rating"] == b[k]["rating"] for k in keys)
        print(f"display agreement: {same}/{len(keys)} exact ({same / len(keys):.2f})")
        return
    diffs, none_disagree = [], 0
    for k in keys:
        ra, rb = a[k], b[k]
        if ra["none"] != rb["none"]:
            none_disagree += 1
        elif not ra["none"] and ra["cue_char"] >= 0 and rb["cue_char"] >= 0:
            diffs.append(abs(ra["cue_char"] - rb["cue_char"]))
    print(
        f"cue agreement over {len(keys)} units: none-flag disagreements "
        f"{none_disagree}; |char-offset delta| median "
        f"{statistics.median(diffs) if diffs else float('nan'):.0f}, "
        f"within 80 chars (~16 tokens): "
        f"{sum(d <= 80 for d in diffs)}/{len(diffs)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cue", "display"], default="cue")
    ap.add_argument("--model", default="deepseek/deepseek-v4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agreement", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.agreement:
        agreement(*args.agreement)
        return
    key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key and Path(".env").exists():
        for line in open(".env"):
            if line.startswith(("OPENROUTER_KEY", "OPENROUTER_API_KEY")):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("no OPENROUTER_KEY in env or .env")
    run_task(args, key)


if __name__ == "__main__":
    main()
