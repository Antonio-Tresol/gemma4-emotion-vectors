"""Build the CAMBRIA capstone write-up as one HTML file: docs/index.html.

Run it as `uv run python build.py` from this directory. It is stdlib-only, so
any interpreter works, but a stock macOS has no bare `python`.

Every number rendered in the deck is passed in from DATA below, and every entry
in DATA was copied from a notebook's printed record or an evidence file in the
repository (the source is named in the comment above each block). Nothing here
is recomputed and nothing is typed from memory.

Usage: python build.py   (writes presentation.html next to this script)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).parent
# The extracted figure data. Defaults to a sibling `data/` directory so the
# build runs on any clone; set EMOTION_DECK_DATA to point elsewhere.
SCRATCH = Path(os.environ.get("EMOTION_DECK_DATA", HERE / "data"))
STORY_JSON = SCRATCH / "story_data.json"
STORY_TEXT_JSON = SCRATCH / "story_text.json"  # the story itself, from results/combined_stories
EMO_BY_LAYER_JSON = SCRATCH / "emo_by_layer.json"  # per-layer confusion, all six layers
# three stories with the SAME three emotions but very different measured tracking
# quality, chosen by mean gate rank at layer 33 (best / median / worst of the
# stories whose phases all fall inside the 12-probe bank). Curves rebuilt with
# the notebook-05 recipe and checked against its committed figure.
THREE_STORIES_JSON = SCRATCH / "three_stories.json"
# the three 20x20 layer-by-layer RSA matrices from notebook 02 section 9
RSA_JSON = SCRATCH / "rsa_matrices.json"
# per-layer detection scores behind the lineage figure, so the chart can show
# the measurement instead of asserting a count derived from it
LINEAGE_LAYERS_JSON = SCRATCH / "lineage_layers.json"
# preference and steering evidence, written by scripts/export_deck_preferences.py;
# extracted rather than transcribed because the arrays are per-emotion and the
# source file has a near-twin whose numbers are the pre-fix ones
PREFERENCES_JSON = SCRATCH / "preferences.json"

# ---------------------------------------------------------------------------
# All figures below come from notebooks/02 and notebooks/11 printed records and
# notebooks/08. Source notebook named per block.
# ---------------------------------------------------------------------------

# notebook 02, "PC1: evr ... |r| valence ... arousal ... dominance ... story-length"
PCS = {
    "base": [
        {
            "pc": 1,
            "evr": 0.151,
            "valence": 0.83,
            "arousal": 0.02,
            "dominance": 0.66,
            "length": 0.15,
        },
        {
            "pc": 2,
            "evr": 0.123,
            "valence": 0.09,
            "arousal": 0.55,
            "dominance": 0.08,
            "length": 0.41,
        },
        {
            "pc": 3,
            "evr": 0.061,
            "valence": 0.05,
            "arousal": 0.08,
            "dominance": 0.05,
            "length": 0.18,
        },
        {
            "pc": 4,
            "evr": 0.045,
            "valence": 0.09,
            "arousal": 0.44,
            "dominance": 0.22,
            "length": 0.14,
        },
        {
            "pc": 5,
            "evr": 0.042,
            "valence": 0.15,
            "arousal": 0.01,
            "dominance": 0.06,
            "length": 0.21,
        },
    ],
    "instruct": [
        {
            "pc": 1,
            "evr": 0.279,
            "valence": 0.11,
            "arousal": 0.04,
            "dominance": 0.17,
            "length": 0.39,
        },
        {
            "pc": 2,
            "evr": 0.097,
            "valence": 0.13,
            "arousal": 0.35,
            "dominance": 0.23,
            "length": 0.66,
        },
        {
            "pc": 3,
            "evr": 0.063,
            "valence": 0.72,
            "arousal": 0.21,
            "dominance": 0.52,
            "length": 0.10,
        },
        {
            "pc": 4,
            "evr": 0.055,
            "valence": 0.36,
            "arousal": 0.06,
            "dominance": 0.25,
            "length": 0.16,
        },
        {
            "pc": 5,
            "evr": 0.027,
            "valence": 0.08,
            "arousal": 0.13,
            "dominance": 0.10,
            "length": 0.12,
        },
    ],
}

# notebook 02, "|score correlation|, instruct PCs (rows) x base PCs (columns), layer 33"
SCORE_GRID = [
    [0.14, 0.02, 0.09, 0.09, 0.05],
    [0.24, 0.57, 0.06, 0.11, 0.22],
    [0.83, 0.37, 0.02, 0.02, 0.00],
    [0.34, 0.06, 0.01, 0.21, 0.17],
    [0.00, 0.06, 0.60, 0.36, 0.04],
]
PRINCIPAL_ANGLES = [86.1, 56.0, 46.1]  # notebook 02, top-3 subspaces

# notebook 02, "it-PC1 extremes"
IT_PC1_LOW = [
    "awestruck",
    "jubilant",
    "miserable",
    "skeptical",
    "hostile",
    "fulfilled",
    "smug",
    "lazy",
]
IT_PC1_HIGH = [
    "irritated",
    "on edge",
    "uneasy",
    "impatient",
    "frustrated",
    "bored",
    "disturbed",
    "scared",
]

# NOTE: this file once hardcoded a per-emotion table transcribed from the
# notebook's printed TOP-3 confusion tables. That was wrong — an emotion missing
# from its own top-3 was recorded as 0 ("never wins") when the true rate was just
# small (angry 0.151, calm 0.094, nervous 0.058 at layer 33). Every rate now comes
# from emo_by_layer.json over all six layers, so the top-3 display limit cannot
# leak into the numbers again.

# notebook 11, "L6: top-1 0.58; wrong-winner VAD dist 0.78 vs shuffle 1.12 ..."
# and "pearson r(lead, |dV|) ... per layer"
BY_LAYER = [
    {"layer": 6, "top1": 0.58, "vad": 0.78, "shuffle": 1.12, "r_dval": 0.027},
    {"layer": 15, "top1": 0.33, "vad": 0.96, "shuffle": 1.12, "r_dval": 0.061},
    {"layer": 24, "top1": 0.57, "vad": 0.89, "shuffle": 1.07, "r_dval": 0.143},
    {"layer": 33, "top1": 0.27, "vad": 1.08, "shuffle": 1.22, "r_dval": 0.001},
    {"layer": 42, "top1": 0.41, "vad": 0.92, "shuffle": 1.15, "r_dval": 0.186},
    {"layer": 51, "top1": 0.30, "vad": 0.99, "shuffle": 1.24, "r_dval": 0.255},
]

# notebook 07 printed record: R1 passing layers per probe lineage (bar = 8 of 12
# scenarios correct on BOTH batteries), phrase repetition, and the preference read.
LINEAGE = [
    {
        "key": "weak",
        "label": "gemma-4-E4B",
        "sub": "a smaller external model",
        "layers": 1,
        "pref": 0.593,
        "overlap": None,
        "n": 1539,
    },
    {
        "key": "self",
        "label": "the probed model itself",
        "sub": "self-generated stories",
        "layers": 5,
        "pref": 0.616,
        "overlap": 0.0386,
        "n": 3072,
    },
    {
        "key": "diverse",
        "label": "deepseek-v4-pro, varied prompts",
        "sub": "persona x setting grid",
        "layers": 7,
        "pref": 0.772,
        "overlap": None,
        "n": 12262,
    },
    {
        "key": "fixed",
        "label": "deepseek-v4-pro, one prompt",
        "sub": "a stronger external writer",
        "layers": 9,
        "pref": 0.706,
        "overlap": 0.0007,
        "n": 3070,
    },
]
LINEAGE_DOSE = {  # stories per emotion -> mean passing layers, fixed-prompt DeepSeek arm
    "8": 3.8,
    "16": 5.0,
    "32": 6.2,
    "64": 8.6,
    "128": 7.8,
    "255": 9.0,
}
ELIAS_SHARE = 0.982  # share of self-generated stories naming a character Elias
LINEAGE_BAR = 8  # registered pass bar, of 12 scenarios, on both batteries

FACTS = {
    # notebook 02
    "base_pc1_valence": 0.828,
    "it_pc1_valence": 0.113,
    "it_pc3_valence": 0.718,
    "base_pc1_evr": 0.151,
    "it_pc1_evr": 0.279,
    "rsa_it_late": 0.793,
    "rsa_base_late": 0.941,
    "cross_model_unablated": 0.286,
    "cross_model_top1_removed": 0.604,
    "logit_lens_base": "5 of 12",
    "ari_at_33": 0.06,
    # notebook 11 / 08
    "gate_rank_it": 3.0,
    "gate_rank_deepseek": 5.0,
    "gate_rank_control": 4.5,
    "r1_lead_it": 0.0117,
    "selfgen_rank_range": "1 to 3",
    "corpus_rank_range": "38 to 76",
    "n_phases": 8938,
    "n_transitions": 5934,
}

HERE = Path(__file__).resolve().parent
CONTENT = HERE / "content"

# Prose lives in docs/content/, one file per section, named in reading order;
# styling in docs/style.css; charts in docs/charts.js. Edit a paragraph there
# and re-run this script. All three are inlined into the output, because the
# published page must stay one self-contained file that works from any clone.
SKELETON = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<!-- This page carried `robots: noindex, nofollow, noarchive` while the work was
     private and shared by link. It was removed on 2026-07-27 when the repository
     was opened up: the page is now meant to be found. Anything that belongs in
     <head> belongs HERE, in the template — a hand-edit to index.html survives
     exactly until the next `python build.py`, which is how the original meta tag
     was lost once already. -->
<title>How are emotions represented in large language models? A study with Gemma 4 31B</title>
<meta name="description" content="Replicating Anthropic's emotion-vector result on Gemma 4 31B: the base model reproduces the circumplex, the instruction-tuned model buries it under an axis we cannot explain, and emotion tracking through a story is real but small.">
<meta property="og:title" content="How are emotions represented in large language models?">
<meta property="og:description" content="A short research sprint on emotion vectors in Gemma 4 31B: replication, what instruction tuning displaces, and reading emotion token by token through a story.">
<meta property="og:type" content="article">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>__CSS__</style></head>
<body>
__BODY__
<script>
__JS__
</script>
</body></html>"""


def read_content() -> str:
    """Every content file, in filename order, which is reading order."""
    files = sorted(CONTENT.glob("*.html"))
    if not files:
        raise FileNotFoundError(f"no content files in {CONTENT}")
    return "\n\n".join(f.read_text(encoding="utf-8").rstrip("\n") for f in files)


def build() -> str:
    story = json.loads(STORY_JSON.read_text(encoding="utf-8"))
    payload = {
        "story": story,
        "storyText": json.loads(STORY_TEXT_JSON.read_text(encoding="utf-8")),
        "emoByLayer": json.loads(EMO_BY_LAYER_JSON.read_text(encoding="utf-8")),
        "three": json.loads(THREE_STORIES_JSON.read_text(encoding="utf-8")),
        "rsa": json.loads(RSA_JSON.read_text(encoding="utf-8")),
        "lineageLayers": json.loads(LINEAGE_LAYERS_JSON.read_text(encoding="utf-8")),
        "prefs": json.loads(PREFERENCES_JSON.read_text(encoding="utf-8")),
        "pcs": PCS,
        "grid": SCORE_GRID,
        "angles": PRINCIPAL_ANGLES,
        "itpc1": {"low": IT_PC1_LOW, "high": IT_PC1_HIGH},
        "byLayer": BY_LAYER,
        "lineage": LINEAGE,
        "dose": LINEAGE_DOSE,
        "elias": ELIAS_SHARE,
        "bar": LINEAGE_BAR,
        "facts": FACTS,
    }
    html = (
        SKELETON.replace("__BODY__", read_content())
        .replace("__JS__", (HERE / "charts.js").read_text(encoding="utf-8").rstrip("\n"))
        .replace("__CSS__", (HERE / "style.css").read_text(encoding="utf-8").rstrip("\n"))
        .replace("__DATA__", json.dumps(payload))
    )
    return link_notebooks(mark_external_links(html))


# The arrow the course site uses: a diagonal stroke with a corner, drawn in the
# accent colour by the .link-plain rule above. Sized and stroked in CSS, not
# here, because a presentation attribute cannot read a CSS custom property.
EXTERNAL_ARROW = (
    '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M1 11L11 1M11 1H4M11 1V8"/></svg>'
)

# Only absolute http(s) anchors that carry no attributes beyond href. In-page
# anchors (href="#...") and the script-built navigation links are left alone,
# which is what keeps this from touching the tick bar and the section index.
REPO_BLOB = "https://github.com/Antonio-Tresol/gemma4-emotion-vectors/blob/main/"
_SRC_BLOCK = re.compile(r'(<div class="src"[^>]*>)(.*?)(</div>)', re.DOTALL)
_NOTEBOOK = re.compile(r"(notebooks/\d\d_[A-Za-z0-9_]+\.ipynb)")


def link_notebooks(html: str) -> str:
    """Point every notebook named in a source line at the notebook itself.

    A figure that names its notebook and then makes you go and find it is
    only half a citation. Restricted to the source lines, which is where these
    names appear, and skipped where one already carries a link. The anchors
    get a class, which is also what keeps mark_external_links from giving them
    the arrow treatment: these should stay quiet mono text, not become another
    bold link in the middle of a caption.
    """

    def link_one(match: re.Match[str]) -> str:
        name = match.group(1)
        return (
            f'<a class="link-src" href="{REPO_BLOB}{name}" '
            f'target="_blank" rel="noopener noreferrer">{name}</a>'
        )

    def rewrite_block(match: re.Match[str]) -> str:
        head, body, tail = match.groups()
        if "<a " in body:
            return match.group(0)
        return head + _NOTEBOOK.sub(link_one, body) + tail

    return _SRC_BLOCK.sub(rewrite_block, html)


_BARE_EXTERNAL_ANCHOR = re.compile(r'<a href="((?:https?://|mailto:)[^"]+)">(.*?)</a>', re.DOTALL)


def mark_external_links(html: str) -> str:
    """Give every outbound link the course site's treatment.

    Three things happen to each one: the .link-plain class, a new-tab target
    with `rel="noopener noreferrer"`, and a trailing arrow. Done here rather
    than by hand at each of the eleven call sites so a link added later cannot
    forget, and so the rule lives in one readable place.

    The arrow is glued to the final word inside a `.nw` span. Without that it
    can wrap onto a line of its own and read as a stray mark. The link text is
    also whitespace-collapsed, since these anchors span source lines and the
    split would otherwise treat a newline as part of the last word.
    """

    def rewrite(match: re.Match[str]) -> str:
        href, text = match.group(1), " ".join(match.group(2).split())
        # An address is not a page: no arrow, because the arrow means "this
        # opens somewhere else on the web", and no new tab, because a mail
        # client is not a tab. It still needs the class, or it falls through to
        # browser-default blue, which is how the correspondence line ended up
        # the only 1994-looking link on the page.
        if href.startswith("mailto:"):
            return f'<a class="link-plain" href="{href}">{text}</a>'
        head, _, last_word = text.rpartition(" ")
        # Any closing tags on the last word stay OUTSIDE the .nw span. Without
        # this, link text wrapped in <em> produced <em>...<span>Model</em></span>:
        # the em opened outside the span and closed inside it.
        word, closers = re.match(r"(.*?)((?:</[a-zA-Z]+>)*)$", last_word, re.S).groups()
        tail = f'<span class="nw">{word}{EXTERNAL_ARROW}</span>{closers}'
        label = f"{head} {tail}" if head else tail
        return (
            f'<a class="link-plain" href="{href}" '
            f'target="_blank" rel="noopener noreferrer">{label}</a>'
        )

    return _BARE_EXTERNAL_ANCHOR.sub(rewrite, html)


# Em dashes were cut from 60 to 1 on 2026-07-26 and then crept back three
# separate times in later edits, because nothing checked. A dash count is a poor
# proxy for good prose, but it is an excellent proxy for THIS regression, which
# is the one that keeps happening.
EM_DASH = "\u2014"
EM_DASH_BUDGET = 4


def visible_prose(html: str) -> str:
    """The page as a reader sees it: no script, no style, no tags."""
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", body).replace("&mdash;", EM_DASH)


def check_em_dashes(html: str) -> None:
    """Fail the build rather than let the dash habit creep back unnoticed."""
    used = visible_prose(html).count(EM_DASH)
    if used > EM_DASH_BUDGET:
        raise SystemExit(
            f"em-dash budget exceeded: {used} in the rendered prose, "
            f"budget {EM_DASH_BUDGET}.\n"
            "Use a comma, a colon, or a full stop. If a dash is genuinely "
            "right, raise the budget deliberately rather than by accident."
        )
    print(f"em dashes in prose: {used} (budget {EM_DASH_BUDGET})")


# MPLS scientific-writing rules with a mechanical signature. The rest of that
# guidance (topic sentences, information order, stress position) needs a reader.
MAX_SENTENCE_WORDS = 30
SMOTHERED_VERB = re.compile(
    r"\b(make|makes|made|perform|performs|performed|provide|provides|provided|"
    r"conduct|conducts|conducted|undertake|give|gives|gave|reach|reaches|"
    # plain nouns that merely end in -ion are not smothered verbs. Without this
    # guard the pattern fires on "makes emotion", on a page about emotions.
    r"reached)\s+(a|an|the)?\s*(?!emotions?\b|questions?\b|sections?\b|versions?\b"
    r"|directions?\b|dimensions?\b|fractions?\b)\w+(ion|ance|ence|ment)\b",
    re.I,
)


# Prose the reader sees but that lives inside a <script> block: the method and
# glossary panels are built as JS strings and injected at runtime. They must be
# checked, or the longest sentences on the page hide from their own gate.
JS_PROSE_SPAN = re.compile(r'<span class="(?:gl|where|t-sub|note)"[^>]*>(.*?)</span>', re.S)

BLOCK_TAG = re.compile(r"</?(?:p|h1|h2|h3|div|li|details|summary|section|br|td|th)\b[^>]*>", re.I)
# Label spans (a callout's heading, a takeaway's tag) sit inline but read as
# their own line. Without a break here a label glues onto the sentence after it.
LABEL_SPAN = re.compile(
    r"</span>\s*<span\b"  # adjacent spans: legend entries
    r"|</span>(?=\s*<b|\s*[A-Z])"  # a label running into its sentence
    r'|<span class="(?:k|lbl|kicker|secno)"[^>]*>'  # callout and takeaway labels
)


def _sentences_in(fragment: str) -> list[str]:
    """Split one already-tagless block of text into sentences.

    Whitespace is collapsed FIRST so that a sentence hard-wrapped across several
    source lines is measured as the one sentence a reader meets, not as several
    short ones.
    """
    block = re.sub(r"\s+", " ", fragment).strip()
    if len(block) < 30:
        return []
    # a sentence may open with a number ("1.0 means the two axes..."), so the
    # lookahead admits digits too, or the pair is measured as one long sentence
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“])", block)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def prose_sentences(html: str) -> list[str]:
    """Reader-visible sentences, one per returned string.

    Block boundaries come from HTML tags only. An earlier version split on "\\n",
    but this file hard-wraps its markup, so one 46-word sentence arrived as five
    short fragments and the gate reported a pass it had never checked.
    """
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)  # comments are not prose
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S | re.I)
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    # the reader-visible prose inside those scripts, recovered by its own markup.
    # Strip the JS string-concatenation seams so "...next `+ `emotion" reads as
    # the one phrase the reader sees rather than gaining two stray tokens.
    for js in scripts:
        for match in JS_PROSE_SPAN.findall(js):
            fragment = re.sub(r"`\s*\+\s*`", "", match)
            # a ternary offers the reader one branch at a time, so its branches
            # are separate sentences, not one run-on
            fragment = re.sub(r"[\"`]\s*[?:]\s*[\"`]", "</p><p>", fragment)
            body += "<p>" + fragment + "</p>"

    body = LABEL_SPAN.sub("\x00", body)

    body = BLOCK_TAG.sub("\x00", body)
    text = re.sub(r"<[^>]+>", " ", body)
    for entity, plain in (
        ("&mdash;", "-"),
        ("&times;", "x"),
        ("&amp;", "&"),
        ("&nbsp;", " "),
        ("&middot;", "-"),
        ("&quot;", '"'),
        ("&mu;", "u"),
        ("&phi;", "p"),
        ("&ell;", "l"),
        ("&rarr;", "->"),
    ):
        text = text.replace(entity, plain)
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)

    return [s for block in text.split("\x00") for s in _sentences_in(block)]


def check_prose(html: str) -> None:
    """Fail the build on the two rules a script can actually judge."""
    sentences = prose_sentences(html)
    counts = [len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)]) for s in sentences]
    long_ones = [(n, s) for n, s in zip(counts, sentences) if n > MAX_SENTENCE_WORDS]
    smothered = sorted({m.group(0) for s in sentences for m in SMOTHERED_VERB.finditer(s)})
    problems = []
    if long_ones:
        problems.append(
            f"{len(long_ones)} sentence(s) over {MAX_SENTENCE_WORDS} words:\n"
            + "\n".join(f"  [{n}] {s[:120]}" for n, s in sorted(long_ones, reverse=True))
        )
    if smothered:
        problems.append("smothered verbs (use the plain verb): " + ", ".join(smothered))
    if problems:
        raise SystemExit("prose check failed\n" + "\n".join(problems))
    print(
        f"prose: {len(sentences)} sentences, mean {sum(counts) / len(counts):.1f} words, "
        f"longest {max(counts)} (limit {MAX_SENTENCE_WORDS})"
    )


# Words a reader has already had to ask about. Each was fixed only after being
# queried, so the list exists to stop them returning rather than to be clever.
# A term earns removal from this list by being defined on the page at first use.
BANNED_WORDS = {
    "stack": "say \u201cthe model\u2019s layers\u201d; the verb \u201cstacked\u201d is fine",
    "battery": "say what is in it: \u201cthe paper\u2019s twelve scenarios\u201d",
    "batteries": "say \u201cboth sets of scenarios\u201d",
    "arm": "say \u201cmodel\u201d, \u201cversion\u201d or \u201ccondition\u201d",
    "arms": "say \u201cthe two models we compare\u201d",
    # "probe bank" alone was the hole: the page carried the bare noun ten times,
    # in pseudo-code and a glossary, while this check reported it clean.
    "probe bank": "say \u201cthe twelve emotion vectors\u201d",
    "bank": "say \u201cthe twelve emotion vectors\u201d, or name the source",
    "banks": "say \u201cthe vector sets\u201d",
    "readout": "say what the code actually does",
    "unablated": "say \u201cas measured\u201d",
    "attractor": "say \u201cthe wrong answers pile onto X\u201d",
    "attractors": "say \u201cthe wrong answers pile onto X\u201d",
    "lineage": "say \u201cstory source\u201d or \u201ccorpus\u201d",
    "substrate": "say \u201cthe stories\u201d or \u201cthe corpus\u201d",
    "free parameter": "say that nothing in the method fixes it",
    "ablation": "say what was removed and what happened",
    "the read": "say \u201cTakeaway\u201d",
    "registered bar": "say \u201cthe mark we fixed in advance\u201d",
}


# One figure, one card, four supports. The rule is written out in README.md;
# this checks the two halves of it a script can judge. It exists because the
# expand control reads the how-to and the source line OUT of the figure's card,
# so a figure whose reading lives in a sibling card expands with nothing to
# read, and an independent audit found four figures in that state.
FIGURE_HOST = re.compile(r"<div[^>]*\bdata-figtitle=\"([^\"]+)\"[^>]*>", re.I)


def check_figures(html: str) -> None:
    """Every result figure carries its how-to block and its source line."""
    problems = []
    for match in FIGURE_HOST.finditer(html):
        title = match.group(1)
        tag = match.group(0)
        # the card a figure sits in ends at the next card or the section end,
        # whichever comes first; that span is what the expand control clones
        rest = html[match.end() :]
        end = min(
            (i for i in (rest.find('<div class="card"'), rest.find("</section>")) if i != -1),
            default=len(rest),
        )
        card = rest[:end]
        if "data-schematic" in tag:
            continue  # carries no measurement, so there is nothing to grade or cite
        if 'details class="howto"' not in card:
            problems.append(f"  no how-to block: {title}")
        if 'class="src"' not in card:
            problems.append(f"  no source line: {title}")
    if problems:
        raise SystemExit("figure check failed\n" + "\n".join(problems))
    total = len(FIGURE_HOST.findall(html))
    print(f"figures: {total}, each with a how-to block and a source line")


FIGURE_REF = re.compile(r"\bFigures?\s+(\d+)\s*(<!--\s*fig:([A-Za-z0-9_]+)\s*-->)?")
MARKED_NUMBER = re.compile(r"(\d+)\s*<!--\s*fig:([A-Za-z0-9_]+)\s*-->")


def check_figure_references(html: str) -> None:
    """Every "Figure N" in prose must name its chart id in an adjacent comment,
    and N must be that chart's position in document order.

    Figure numbers are assigned at runtime by numberFigures(), in document
    order, so inserting a chart renumbers everything after it. The glossary
    once said "Figure 1" about a chart that had become Figure 6; this check
    turns that silent drift into a build failure.
    """
    body = re.sub(r"(?is)<script\b.*?</script>", " ", html)
    hosts = []
    for match in FIGURE_HOST.finditer(body):
        id_match = re.search(r'\bid="([A-Za-z0-9_-]+)"', match.group(0))
        hosts.append(id_match.group(1) if id_match else None)
    problems = []
    for number, marker, _chart_id in FIGURE_REF.findall(body):
        if not marker:
            problems.append(f'  "Figure {number}" carries no <!-- fig:chartId --> marker')
    for number, chart_id in MARKED_NUMBER.findall(body):
        position = int(number)
        if not (1 <= position <= len(hosts)) or hosts[position - 1] != chart_id:
            actual = hosts.index(chart_id) + 1 if chart_id in hosts else "absent"
            problems.append(
                f"  a mention says figure {position} is {chart_id}, "
                f"but that chart is figure {actual} in document order"
            )
    if problems:
        raise SystemExit("figure reference check failed\n" + "\n".join(problems))
    marked = len(MARKED_NUMBER.findall(body))
    print(f"figure references: {marked} marked, each pointing at the chart it names")


def check_jargon(html: str) -> None:
    """Fail the build on terms a reader has already had to ask about."""
    prose = " ".join(prose_sentences(html)).lower()
    hits = [
        f"  {word!r}: {fix}"
        for word, fix in BANNED_WORDS.items()
        if re.search(r"\b" + re.escape(word) + r"\b", prose)
    ]
    if hits:
        raise SystemExit(
            "jargon check failed; these were each queried by a reader before:\n" + "\n".join(hits)
        )
    print(f"jargon: none of the {len(BANNED_WORDS)} flagged terms present")


if __name__ == "__main__":
    out = HERE / "index.html"
    html = build()
    check_em_dashes(html)
    check_prose(html)
    check_figures(html)
    check_figure_references(html)
    check_jargon(html)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
