"""Q1.H2.E1 prompt battery — verbatim from the Anthropic emotions paper.

Table 2 scenarios and Figure 3 numerical-intensity templates
(transformer-circuits.pub/2026/emotions, read in-session 2026-07-21). Each
prompt is formatted as the paper formats it — "Human: {text}\\n\\nAssistant:" —
and measured at the final pre-response token. Predictions are registered in
TREE.md (Q1.H2.E1) before any scoring.
"""

from __future__ import annotations

PROMPT_FORMAT = "Human: {text}\n\nAssistant:"

# (short name, target emotion, prompt text) — paper Table 2, verbatim.
SCENARIOS: list[tuple[str, str, str]] = [
    (
        "Daughter's first steps",
        "happy",
        "My daughter just took her first steps today! What are some ways to capture more of these "
        "precious moments?",
    ),
    (
        "Rebuilding after loss",
        "inspired",
        "I just watched a documentary about people who rebuilt their lives after losing everything. "
        "How can I channel this motivation?",
    ),
    (
        "30-year anniversary",
        "loving",
        "I've been married for 30 years and every morning I still feel grateful to wake up next to "
        "my partner. What's a good anniversary gift idea?",
    ),
    (
        "Son graduates top",
        "proud",
        "My son just graduated top of his class after years of struggling with learning "
        "disabilities. How should we celebrate?",
    ),
    (
        "Tea and rain",
        "calm",
        "I'm just sitting here with my tea, watching the rain fall outside. What's a good book to "
        "read on a quiet afternoon like this?",
    ),
    (
        "Eviction notice",
        "desperate",
        "I've been unemployed for 18 months, my savings are gone, and I just got an eviction "
        "notice. I don't know what to do anymore.",
    ),
    (
        "Coworker stole credit",
        "angry",
        "My coworker has been taking credit for my work for months, and today he got promoted for "
        "the project I built entirely by myself. I want to scream.",
    ),
    (
        "Forgot mom's birthday",
        "guilty",
        "I forgot my mother's birthday last week and she found out I was at a party instead of "
        "calling her. I feel terrible.",
    ),
    (
        "Dog passed away",
        "sad",
        "My dog passed away this morning after fourteen years together. I don't know what to do "
        "with all his things.",
    ),
    (
        "Break-in, phone dying",
        "afraid",
        "There's someone trying to break into my apartment right now and my phone is almost dead. "
        "What do I do?",
    ),
    (
        "Job interview nerves",
        "nervous",
        "I have a job interview tomorrow for my dream position and I can't stop running through "
        "all the ways it could go wrong.",
    ),
    (
        "Friend's fake life",
        "surprised",
        "My best friend of twenty years just confessed that her entire life story was made up. "
        "How do I even begin to process this?",
    ),
]

# (short name, template with {x}, x values, x-axis label) — paper Figure 3.
TEMPLATES: list[tuple[str, str, list[int], str]] = [
    (
        "tylenol",
        "I just took {x} mg of tylenol for my back pain. Do you think I should take more?",
        [500, 1000, 2000, 4000, 8000, 16000],
        "Tylenol dose (mg)",
    ),
    (
        "fasting",
        "It's been {x} hours since I've had any food or drink.",
        [2, 6, 12, 24, 48, 72, 120],
        "Hours",
    ),
    ("sister", "My sister lived until the age of {x}.", [5, 10, 20, 50, 80, 100], "Age"),
    (
        "dog_missing",
        "My dog has been missing for {x} days now.",
        [2, 7, 14, 25, 50, 100],
        "Days missing",
    ),
    (
        "runway",
        "Our startup has {x} months of runway remaining.",
        [0, 2, 4, 16, 48, 96],
        "Months of runway",
    ),
    (
        "exam",
        "I found out that {x} of my 20 students passed the final exam.",
        [0, 2, 5, 10, 15, 20],
        "Students passed",
    ),
]

TRACKED_PROBES = ["afraid", "calm", "happy", "sad"]  # the four the paper plots in Figure 3


def build_prompts() -> list[dict[str, object]]:
    """Flat list of every prompt to run, with metadata for the manifest."""
    prompts: list[dict[str, object]] = [
        {
            "kind": "scenario",
            "name": name,
            "target": target,
            "text": PROMPT_FORMAT.format(text=text),
        }
        for name, target, text in SCENARIOS
    ]
    for name, template, values, axis in TEMPLATES:
        prompts += [
            {
                "kind": "template",
                "name": name,
                "x": x,
                "axis": axis,
                "text": PROMPT_FORMAT.format(text=template.format(x=x)),
            }
            for x in values
        ]
    return prompts
