"""Q3 prep — generate the multi-emotion synthetic story battery.

Builds the generation pipeline for Q3.H1.E1 ("synthetic story battery —
matched-length stories with token-aligned transitions between 2-3 emotions").
Per TREE.md's Q3 gate note (2026-07-22): H1 is supported (C1 survived), H3 is
supported (C3 survived), H2 is reopened with C4 pending its own falsify gate
— the gate is close to satisfied and Q3 preparation is legitimate, though no
run from this script counts as a logged Q3 experiment result until the
orchestrator confirms the gate has fully cleared and registers predictions.

Prompt design (system prompt, SIMULTANEOUS/SEQUENTIAL modes, permutation
scheme) is carried over unchanged from scripts/combined_story_gen/
test_combined_stories.ipynb (prototyped there against OpenRouter by Peyton
Li), swapped here to local vLLM per notes/vllm-parallel-inference-template.md.
Each of the 173 triples in emotions_triples_v1.json is expanded into 6
permutations x 2 modes = 12 prompt combinations; --per-triple samples are
split evenly across those 12, so --per-triple must be a multiple of 12.

    uv run --extra gpu python scripts/combined_story_gen/generate_combined_stories.py --smoke
    uv run --extra gpu python scripts/combined_story_gen/generate_combined_stories.py --per-triple 36
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import re
import subprocess
import time
from pathlib import Path

MODES = ("SIMULTANEOUS", "SEQUENTIAL")
N_PERMUTATIONS = 6  # 3! orderings of a triple
N_COMBOS = len(MODES) * N_PERMUTATIONS  # 12

BANNED_WORDS = ("felt", "feeling", "feelings", "emotion", "emotional")

# The system prompt promises "a SETTING seed" as an input; without one actually
# supplied, the model falls back to its single highest-probability default
# scene/character regardless of temperature (measured: "Elias" opened 40% of
# 1,434 stories, "mahogany" appeared in 42%). Picked per-combo from the same
# seed already used for sampling, so the choice stays reproducible.
SETTINGS = (
    "a crowded subway platform at rush hour",
    "an urgent care waiting room at 2am",
    "a school cafeteria during lunch",
    "a beach at low tide, off-season",
    "a mechanic's garage after closing",
    "a courthouse hallway during a recess",
    "a grocery store checkout line",
    "a shared laundromat on a rainy night",
    "a rooftop parking garage",
    "a food truck at a summer festival",
    "a dentist's waiting room",
    "an airport gate during a delay",
    "a community garden plot",
    "a warehouse loading dock",
    "a used bookstore just before closing",
    "a hospital cafeteria",
    "a moving truck being loaded",
    "a public pool locker room",
    "a farmers' market stall",
    "a break room in an overnight shift",
    "a hotel lobby during a storm",
    "a laundry line behind an apartment building",
    "a diner counter at 3am",
    "a bus station bench",
    "a construction site trailer",
    "a church basement during a rummage sale",
    "a college dorm hallway",
    "a veterinary clinic waiting room",
    "a car repair shop lobby",
    "a rural gas station at dusk",
    "a shared office cubicle after hours",
    "a playground bench",
    "a barbershop near closing time",
    "a ferry crossing in fog",
    "a self-storage unit hallway",
    "a community pool deck",
    "a laundromat dryer row",
    "a train platform in winter",
    "a backyard during a power outage",
    "a hospital hallway outside a patient's room",
)

CHARACTER_NAMES = (
    "Maria",
    "Devon",
    "Priya",
    "Tomás",
    "Aisha",
    "Marcus",
    "Yuki",
    "Sam",
    "Grace",
    "Omar",
    "Ling",
    "Jamal",
    "Rosa",
    "Ben",
    "Fatima",
    "Carlos",
    "Nadia",
    "Trevor",
    "Ingrid",
    "Diego",
    "Chloe",
    "Amir",
    "Beatriz",
    "Noah",
    "Wei",
    "Hana",
    "Leon",
    "Zara",
    "Peter",
    "Simone",
    "Kwame",
    "Anya",
    "Malik",
    "Josie",
    "Ravi",
    "Elena",
    "Gus",
    "Mei",
    "Dara",
    "Frank",
)

SYSTEM_PROMPT = """
You are a fiction generator producing short story excerpts for a research dataset.

## Task

You will be given:
- THREE emotions
- A MODE: either SIMULTANEOUS or SEQUENTIAL
- A SETTING seed, a POV, and a TENSE

Write ONE excerpt of 150-200 words that conveys all three emotions
according to the MODE.

## The hard rule

NEVER write the name of any assigned emotion, and NEVER write a direct
synonym of it. This is the single most important rule. A story that names
its emotion is worthless for this dataset and will be discarded.

Convey emotion ONLY through:
- actions and behaviors
- physical sensations and body language
- dialogue and tone of voice
- thoughts and internal reactions
- situational context and environmental description

Banned in every story, regardless of assigned emotions: "felt", "feeling",
"feelings", "emotion", "emotional".

Do not substitute a near-synonym for a banned word. If the emotion is
"furious", you may not write furious, angry, mad, enraged, livid, irate,
seething, or fuming. Write the clenched jaw instead.

Do not name the emotion in a simile or a metaphor either. "A wave of
sadness" is a violation. "Like someone who had just been told bad news"
is a violation if it names the state.

## MODE: SIMULTANEOUS

All three emotions are present throughout the whole excerpt. Do not put
them in separate paragraphs. Do not resolve one into another. The reader
should be able to point at almost any sentence and find at least two of
them operating at once.

Techniques that work:
- give the emotions different objects (one about a person, one about an event, one about the room)
- give them different layers (one displayed to others, one felt privately)
- give them different timescales (one ambient and long-running, one immediate)
- let the body do one thing while the thoughts do another
- Include an XML tag in the format of <emotion>(happy, sad, angry)</emotion> to denote the emotions that the generated story uses.

## MODE: SEQUENTIAL

The excerpt moves through the three emotions in the order given, in three
distinct phases of roughly equal length. Each phase should be clearly one
state. Mark each transition with a concrete external event — a door
opening, a phone lighting up, a line of dialogue, stepping outside — not
with a summary sentence about the character changing.

Make sure that the order of emotions in the story follows the order of emotions (Emotion 1, etc...) given in the user prompt.

Do not blend the phases. Phase 2 should not contain residue of phase 1. Generate XML tags in the format <emotion>happy</emotion> at each phase transition, INCLUDING at the beginning of the story.

## Style requirements

- One continuous scene, no time skips beyond the transitions.
- Only generate stories in 3rd-person past tense.
- Concrete and specific. Name objects, sounds, textures, and small physical details. Avoid abstraction.
- No dream sequences, no framing devices, no narrator commenting on the story.
- Do not begin with the character waking up.
- Do not end with a summary line that explains what happened.
- Be creative with the name of characters and setting of the story: vary them from story to story.

## Output format

Output ONLY the story text. No title, no preamble, no explanation, no
labels, no quotation marks around the whole thing, no notes about which
emotion appears where. Begin with the first word of the story.
"""


try:  # shared logger lives in the package (same pattern as generate_dialogue_stories)
    from emotion_vectors.extraction import setup_logger
except ModuleNotFoundError as exc:
    raise SystemExit(f"missing {exc.name}: this entry point needs the project package") from exc


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_user_prompt(perm: tuple[str, str, str], mode: str, seed: int) -> str:
    rng = random.Random(seed)  # reproducible: same seed -> same setting/name pick
    setting = rng.choice(SETTINGS)
    name = rng.choice(CHARACTER_NAMES)
    return (
        f"Emotion 1: {perm[0]}\nEmotion 2: {perm[1]}\nEmotion 3: {perm[2]}\nMode: {mode}\n"
        f"Setting: {setting}\nMain character's name: {name}"
    )


def load_triples(path: Path) -> list[dict]:
    triples = json.loads(path.read_text())
    for t in triples:
        if len(t.get("emotions", [])) != 3:
            raise ValueError(f"triple missing 3 emotions: {t}")
    return triples


def combos_for(emotions: list[str]) -> list[tuple[str, tuple[str, str, str], int]]:
    """The 12 (mode, permutation, perm_idx) combinations for one triple, stable order."""
    perms = list(itertools.permutations(emotions))
    return [(mode, perms[i], i) for mode in MODES for i in range(N_PERMUTATIONS)]


def strip_thinking(text: str) -> tuple[str, bool]:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped, stripped != text.strip()


def leaked_words(text: str, emotions: list[str]) -> list[str]:
    """Leakage in the PROSE only — <emotion> tags are required by the system
    prompt to name the emotion (needed downstream for token-alignment), so
    tag content is not leakage and must be excluded before this check."""
    prose = re.sub(r"<emotion>.*?</emotion>", "", text, flags=re.DOTALL)
    lower = prose.lower()
    return [
        w
        for w in (*BANNED_WORDS, *(e.lower() for e in emotions))
        if re.search(rf"\b{re.escape(w)}\b", lower)
    ]


def emotion_tags(text: str) -> list[str]:
    return re.findall(r"<emotion>(.*?)</emotion>", text, flags=re.DOTALL)


def tags_match_emotions(tags: list[str], emotions: list[str]) -> bool:
    """True iff the emotion names appearing across all <emotion> tags are
    exactly the assigned triple (order-insensitive). Catches the model
    substituting or dropping an emotion inside its own required tag — e.g. a
    triple assigned (unsettled, upset, cheerful) tagged as
    <emotion>(cheerful, unsettled, angry)</emotion> — which would otherwise
    silently poison the tag as a downstream token-alignment label."""
    found = {
        piece.strip().lower() for tag in tags for piece in re.split(r"[,()]+", tag) if piece.strip()
    }
    return found == {e.lower() for e in emotions}


def row_key(triple_id: int, mode: str, perm_idx: int) -> str:
    return f"{triple_id}:{mode}:{perm_idx}"


def existing_counts(raw_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            key = row_key(row["triple_id"], row["mode"], row["perm_idx"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def assemble_grouped(raw_path: Path, grouped_path: Path) -> int:
    grouped: dict[int, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            tid = row["triple_id"]
            grouped.setdefault(
                tid,
                {
                    "triple_id": tid,
                    "emotions": row["emotions"],
                    "category": row["category"],
                    "has_nonaffect": row["has_nonaffect"],
                    "stories": [],
                },
            )
            grouped[tid]["stories"].append(
                {
                    "mode": row["mode"],
                    "permutation": row["permutation"],
                    "text": row["text"],
                    "tags": row["tags"],
                }
            )
    grouped_path.write_text("".join(json.dumps(g) + "\n" for g in grouped.values()))
    return len(grouped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument(
        "--triples-file", type=Path, default=Path(__file__).parent / "emotions_triples_v1.json"
    )
    parser.add_argument(
        "--per-triple",
        type=int,
        default=36,
        help=f"stories per triple; must be a multiple of {N_COMBOS} (6 perms x 2 modes)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="skip CUDA graph capture — much faster engine startup, some decode throughput lost; "
        "on by default for --smoke, since a smoke test's job is proving the path, not throughput",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 triples, 1 sample/combo (per-triple forced to 12) — prove the path first",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined_stories"))
    return parser


def prepare_run(args: argparse.Namespace) -> tuple[list[dict], int, "logging.Logger"]:
    """Fail-fast checks, triple loading, logger + config file — before the model load."""
    if args.smoke:
        args.per_triple = N_COMBOS
        args.enforce_eager = True
    if args.per_triple % N_COMBOS != 0:
        raise SystemExit(
            f"--per-triple must be a multiple of {N_COMBOS} (6 perms x 2 modes), "
            f"got {args.per_triple}"
        )
    samples_per_combo = args.per_triple // N_COMBOS
    triples = load_triples(args.triples_file)
    if args.smoke:
        triples = triples[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "generate.log")
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_combos": N_COMBOS,
        "samples_per_combo": samples_per_combo,
        "git_commit": git_commit(),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"config: {json.dumps(config)}")
    logger.info(
        f"{len(triples)} triples x {N_COMBOS} combos x {samples_per_combo} samples/combo "
        f"= {len(triples) * args.per_triple} target stories"
    )
    return triples, samples_per_combo, logger


def write_kept_rows(
    raw_path: Path,
    triple_id: int,
    triple: dict[str, object],
    pending: list[tuple[str, tuple[str, ...], int, int]],
    outputs: list[object],
    combo_seeds: list[int],
) -> tuple[int, int, int, int]:
    """Append QC-passing stories for one triple; returns
    (kept, dropped_short, dropped_leak, dropped_tag_mismatch)."""
    emotions = triple["emotions"]
    kept = d_short = d_leak = d_tag_mismatch = 0
    with open(raw_path, "a") as f:
        for (mode, perm, perm_idx, _), out, seed in zip(pending, outputs, combo_seeds):
            for sample in out.outputs:
                text, had_think = strip_thinking(sample.text)
                leaks = leaked_words(text, emotions)
                tags = emotion_tags(text)
                tags_ok = tags_match_emotions(tags, emotions)
                if len(text) < 100:  # QC: too short
                    d_short += 1
                    continue
                if leaks:  # QC: names an assigned emotion or a banned word
                    d_leak += 1
                    continue
                if not tags_ok:  # QC: tag names an emotion outside the assigned triple
                    d_tag_mismatch += 1
                    continue
                row = {
                    "triple_id": triple_id,
                    "emotions": emotions,
                    "category": triple.get("category"),
                    "has_nonaffect": triple.get("has_nonaffect"),
                    "mode": mode,
                    "permutation": list(perm),
                    "perm_idx": perm_idx,
                    "text": text,
                    "had_thinking_trace": had_think,
                    "leaked_words": leaks,
                    "n_tags": len(tags),
                    "tags": tags,
                    "tags_match_emotions": tags_ok,
                    "n_chars": len(text),
                    "seed": seed,
                    "error": None,
                }
                f.write(json.dumps(row) + "\n")
                kept += 1
    return kept, d_short, d_leak, d_tag_mismatch


def log_resume_state(
    raw_path: Path, triples: list[dict[str, object]], samples_per_combo: int, logger: object
) -> dict[str, int]:
    """Existing-story counts plus the resume log line."""
    counts = existing_counts(raw_path)
    n_pending = sum(
        max(0, samples_per_combo - counts.get(row_key(tid, mode, i), 0))
        for tid, t in enumerate(triples)
        for mode, _, i in combos_for(t["emotions"])
    )
    logger.info(f"{n_pending} stories pending (resuming from {sum(counts.values())} done)")
    return counts


def main() -> int:
    args = build_parser().parse_args()
    triples, samples_per_combo, logger = prepare_run(args)

    from vllm import LLM, SamplingParams  # noqa: PLC0415 — GPU import, lazy on purpose

    logger.info(
        f"loading {args.model} into vLLM (enforce_eager={args.enforce_eager}) — weight load is "
        f"~1 min from a local/warm HF cache; CUDA graph capture (skipped if enforce_eager) can add "
        f"several more minutes on a cold cache with no further log output until it completes"
    )
    t_load = time.monotonic()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    logger.info(f"vLLM engine ready in {time.monotonic() - t_load:.0f}s")

    raw_path = args.out_dir / "stories_raw.jsonl"
    counts = log_resume_state(raw_path, triples, samples_per_combo, logger)

    t0 = time.monotonic()
    n_dropped_short = n_dropped_leak = n_dropped_tag_mismatch = 0
    for triple_id, triple in enumerate(triples):
        emotions = triple["emotions"]
        pending = [
            (mode, perm, perm_idx, need)
            for mode, perm, perm_idx in combos_for(emotions)
            if (need := samples_per_combo - counts.get(row_key(triple_id, mode, perm_idx), 0)) > 0
        ]
        if not pending:
            continue

        # distinct seed per (triple, combo): a shared seed across a triple's 12 combos
        # was correlating completions across permutations/modes; also drives the
        # setting/name pick below, so both stay reproducible from one seed.
        combo_seeds = [args.seed + triple_id * 100 + i for i in range(len(pending))]
        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(perm, mode, seed)},
            ]
            for (mode, perm, _, _), seed in zip(pending, combo_seeds)
        ]
        sampling = [
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=need,
                seed=seed,
            )
            for (_, _, _, need), seed in zip(pending, combo_seeds)
        ]
        try:
            outputs = llm.chat(conversations, sampling)
        except Exception as exc:  # one triple can't kill the run
            logger.error(f"[triple {triple_id}] failed: {exc!r}")
            with open(raw_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "triple_id": triple_id,
                            "emotions": emotions,
                            "category": triple.get("category"),
                            "has_nonaffect": triple.get("has_nonaffect"),
                            "error": repr(exc),
                        }
                    )
                    + "\n"
                )
            continue

        kept, d_short, d_leak, d_tag_mismatch = write_kept_rows(
            raw_path, triple_id, triple, pending, outputs, combo_seeds
        )
        n_dropped_short += d_short
        n_dropped_leak += d_leak
        n_dropped_tag_mismatch += d_tag_mismatch

        done_triples = triple_id + 1
        rate = done_triples / max(time.monotonic() - t0, 1e-9)
        eta_min = (len(triples) - done_triples) / max(rate, 1e-9) / 60
        logger.info(
            f"[triple {triple_id}] +{kept} kept | {done_triples}/{len(triples)} triples | "
            f"ETA {eta_min:.1f}min | dropped so far: {n_dropped_short} short, {n_dropped_leak} leaked, "
            f"{n_dropped_tag_mismatch} tag-mismatch"
        )

    n_grouped = assemble_grouped(raw_path, args.out_dir / "stories_grouped.jsonl")
    logger.info(
        f"done: {n_grouped} triples grouped -> {args.out_dir / 'stories_grouped.jsonl'} | "
        f"dropped {n_dropped_short} short, {n_dropped_leak} leaked, {n_dropped_tag_mismatch} tag-mismatch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
