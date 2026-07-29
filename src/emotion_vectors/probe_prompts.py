"""Q1.H2.E1 prompt set — verbatim from the Anthropic emotions paper.

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

# Held-out confirmation scenarios for the E2 sweep — written by us 2026-07-21
# and committed BEFORE any sweep scoring, so a (layer, pooling, format)
# combination selected on the paper's 12 scenarios must also pass on these 12
# fresh ones.
# Same 12 target emotions, implicit content, no emotion words named.
HELDOUT_SCENARIOS: list[tuple[str, str, str]] = [
    (
        "Adoption call",
        "happy",
        "The adoption agency just called — after three years of waiting, we're picking up our "
        "little girl on Friday. What should we have ready at home?",
    ),
    (
        "PhD at 55",
        "inspired",
        "I attended a talk by a scientist who started her PhD at 55 and just won a major prize. "
        "Where do I even start if I want to go back to school?",
    ),
    (
        "Grandmother's blanket",
        "loving",
        "My grandmother knitted a blanket for every grandchild before she passed, and mine still "
        "smells like her kitchen. How can I preserve it properly?",
    ),
    (
        "First marathon",
        "proud",
        "I ran my first marathon yesterday after two years of couch-to-5k, and I crossed the line "
        "with my kids cheering. Should I frame the medal or the photo?",
    ),
    (
        "Cabin by the lake",
        "calm",
        "The cabin is quiet except for the lake lapping at the dock, and I have nowhere to be "
        "until Monday. Any suggestions for slow breakfast recipes?",
    ),
    (
        "Locks changed",
        "desperate",
        "The landlord changed the locks while my kids were at school and everything we own is "
        "inside. I have forty dollars and nowhere to go tonight.",
    ),
    (
        "Journal read aloud",
        "angry",
        "My sister read my private journal out loud at dinner and everyone laughed. She says I'm "
        "overreacting. I can barely type this.",
    ),
    (
        "Missed recital",
        "guilty",
        "I promised my son I'd be at his recital, then stayed late at work for a meeting that got "
        "cancelled anyway. He saved me a seat in the front row.",
    ),
    (
        "Father's workshop",
        "sad",
        "We cleared out my father's workshop this weekend. His tools were still laid out for a "
        "birdhouse he never finished.",
    ),
    (
        "Scan results call",
        "afraid",
        "The doctor's office just called saying they found something on my scan and I need to "
        "come in tomorrow morning. I'm home alone and my hands won't stop shaking.",
    ),
    (
        "Wedding toast",
        "nervous",
        "I'm giving the toast at my best friend's wedding in an hour and I keep rehearsing the "
        "same line over and over in the bathroom.",
    ),
    (
        "Brother at the door",
        "surprised",
        "I opened my front door and my brother, who I thought was deployed overseas until "
        "December, was standing there holding pizza.",
    ),
]


def build_prompts() -> list[dict[str, object]]:
    """Flat list of every prompt to run, with metadata for the manifest. The
    `text` field is the raw scenario/template text; formatting (plain
    Human:/Assistant: or the model's chat template) is applied at collection."""
    prompts: list[dict[str, object]] = [
        {"kind": kind, "name": name, "target": target, "text": text}
        for kind, scenario_set in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS))
        for name, target, text in scenario_set
    ]
    for name, template, values, axis in TEMPLATES:
        prompts += [
            {"kind": "template", "name": name, "x": x, "axis": axis, "text": template.format(x=x)}
            for x in values
        ]
    return prompts
