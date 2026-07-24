"""Build the CAMBRIA capstone presentation as one self-contained HTML file.

Every number rendered in the deck is passed in from DATA below, and every entry
in DATA was copied from a notebook's printed record or an evidence file in the
repository (the source is named in the comment above each block). Nothing here
is recomputed and nothing is typed from memory.

Usage: python build.py   (writes presentation.html next to this script)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).parent
# The extracted figure data. Defaults to a sibling `data/` directory so the
# build runs on any clone; set EMOTION_DECK_DATA to point elsewhere.
SCRATCH = Path(os.environ.get("EMOTION_DECK_DATA", HERE / "data"))
STORY_JSON = SCRATCH / "story_data.json"
STORY_TEXT_JSON = SCRATCH / "story_text.json"   # the story itself, from results/combined_stories
EMO_BY_LAYER_JSON = SCRATCH / "emo_by_layer.json"  # per-layer confusion, all six layers
# three stories with the SAME three emotions but very different measured tracking
# quality, chosen by mean gate rank at layer 33 (best / median / worst of the
# stories whose phases all fall inside the 12-probe bank). Curves rebuilt with
# the notebook-05 recipe and checked against its committed figure.
THREE_STORIES_JSON = SCRATCH / "three_stories.json"
# the three 20x20 layer-by-layer RSA matrices from notebook 02 section 9
RSA_JSON = SCRATCH / "rsa_matrices.json"

# ---------------------------------------------------------------------------
# All figures below come from notebooks/02 and notebooks/11 printed records and
# notebooks/08. Source notebook named per block.
# ---------------------------------------------------------------------------

# notebook 02, "PC1: evr ... |r| valence ... arousal ... dominance ... story-length"
PCS = {
    "base": [
        {"pc": 1, "evr": 0.151, "valence": 0.83, "arousal": 0.02, "dominance": 0.66, "length": 0.15},
        {"pc": 2, "evr": 0.123, "valence": 0.09, "arousal": 0.55, "dominance": 0.08, "length": 0.41},
        {"pc": 3, "evr": 0.061, "valence": 0.05, "arousal": 0.08, "dominance": 0.05, "length": 0.18},
        {"pc": 4, "evr": 0.045, "valence": 0.09, "arousal": 0.44, "dominance": 0.22, "length": 0.14},
        {"pc": 5, "evr": 0.042, "valence": 0.15, "arousal": 0.01, "dominance": 0.06, "length": 0.21},
    ],
    "instruct": [
        {"pc": 1, "evr": 0.279, "valence": 0.11, "arousal": 0.04, "dominance": 0.17, "length": 0.39},
        {"pc": 2, "evr": 0.097, "valence": 0.13, "arousal": 0.35, "dominance": 0.23, "length": 0.66},
        {"pc": 3, "evr": 0.063, "valence": 0.72, "arousal": 0.21, "dominance": 0.52, "length": 0.10},
        {"pc": 4, "evr": 0.055, "valence": 0.36, "arousal": 0.06, "dominance": 0.25, "length": 0.16},
        {"pc": 5, "evr": 0.027, "valence": 0.08, "arousal": 0.13, "dominance": 0.10, "length": 0.12},
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
IT_PC1_LOW = ["awestruck", "jubilant", "miserable", "skeptical", "hostile", "fulfilled", "smug", "lazy"]
IT_PC1_HIGH = ["irritated", "on edge", "uneasy", "impatient", "frustrated", "bored", "disturbed", "scared"]

# NOTE: an earlier version of this file hardcoded a per-emotion table transcribed
# from the notebook's printed TOP-3 confusion tables. That was wrong: when an
# emotion's own probe did not appear in its own top-3 list, the transcription
# recorded 0 ("never wins") when the true rate was simply small (angry 0.151,
# calm 0.094, nervous 0.058 at layer 33). The deck now reads every rate from
# emo_by_layer.json, computed directly from the taxonomy package over all six
# layers, so the top-3 display limit cannot leak into the numbers again.

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
    {"key": "weak", "label": "gemma-4-4B", "sub": "a smaller external model",
     "layers": 1, "pref": 0.593, "overlap": None, "n": 1539},
    {"key": "self", "label": "the probed model itself", "sub": "self-generated stories",
     "layers": 5, "pref": 0.616, "overlap": 0.0386, "n": 3072},
    {"key": "diverse", "label": "deepseek-v4-pro, varied prompts", "sub": "persona x setting grid",
     "layers": 7, "pref": 0.772, "overlap": None, "n": 12262},
    {"key": "fixed", "label": "deepseek-v4-pro, one prompt", "sub": "a stronger external writer",
     "layers": 9, "pref": 0.706, "overlap": 0.0007, "n": 3070},
]
LINEAGE_DOSE = {  # stories per emotion -> mean passing layers, fixed-prompt DeepSeek arm
    "8": 3.8, "16": 5.0, "32": 6.2, "64": 8.6, "128": 7.8, "255": 9.0,
}
ELIAS_SHARE = 0.982  # share of self-generated stories naming a character Elias
LINEAGE_BAR = 8      # registered pass bar, of 12 scenarios, on both batteries

FACTS = {
    # notebook 02
    "base_pc1_valence": 0.828, "it_pc1_valence": 0.113, "it_pc3_valence": 0.718,
    "base_pc1_evr": 0.151, "it_pc1_evr": 0.279,
    "rsa_it_late": 0.793, "rsa_base_late": 0.941,
    "cross_model_unablated": 0.286, "cross_model_top1_removed": 0.604,
    "logit_lens_base": "5 of 12", "ari_at_33": 0.06,
    # notebook 11 / 08
    "gate_rank_it": 3.0, "gate_rank_deepseek": 5.0, "gate_rank_control": 4.5,
    "r1_lead_it": 0.0117, "selfgen_rank_range": "1 to 3", "corpus_rank_range": "38 to 76",
    "n_phases": 8938, "n_transitions": 5934,
}

CSS = """
:root{--bg:#F4F4F4;--surface:#FFFFFF;--surface-alt:#FAFAFA;--border:#E5E5E5;
--text:#0A0A0A;--body:#262626;--muted:#737373;--orange:#CC785C;--navy:#1d3557;
--red:#e63946;--teal:#457b9d;--green:#009E73;--amber:#E69F00;
/* neutrals and the single alert colour, so no chart invents its own */
--grey-soft:#D4D4D4;--grey-mid:#BDBDBD;--alert:#b3202c;
--display:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
--serif:'Source Serif 4',Georgia,serif;--mono:'JetBrains Mono','SF Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--body);font-family:var(--serif);font-size:16.5px;line-height:1.65;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
.narrow{max-width:820px}
nav{position:sticky;top:0;z-index:50;background:rgba(244,244,244,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--border)}
nav .wrap{display:flex;gap:22px;align-items:center;height:56px;overflow-x:auto}
nav a{font-family:var(--display);font-size:13px;font-weight:500;color:var(--muted);text-decoration:none;
white-space:nowrap;padding:4px 0;border-bottom:2px solid transparent}
nav a:hover{color:var(--text)}
nav a.on{color:var(--text);border-bottom-color:var(--orange)}
nav a.done{color:var(--body)}
nav .pos{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:auto;
padding-left:16px;white-space:nowrap}
nav .brand{font-weight:700;color:var(--text);margin-right:6px;letter-spacing:-.01em;white-space:nowrap}
header{padding:78px 0 54px;border-bottom:1px solid var(--border);background:var(--surface)}
h1{font-family:var(--display);font-size:clamp(34px,5vw,52px);font-weight:700;color:var(--text);
letter-spacing:-.025em;line-height:1.08;margin-bottom:18px}
h2{font-family:var(--display);font-size:30px;font-weight:650;color:var(--text);letter-spacing:-.02em;
margin-bottom:8px;line-height:1.2}
h3{font-family:var(--display);font-size:19px;font-weight:600;color:var(--text);margin-bottom:8px;
letter-spacing:-.01em}
.lede{font-size:20px;line-height:1.55;color:var(--body)}
.kicker{font-family:var(--mono);font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;
color:var(--orange);font-weight:600;margin-bottom:14px}
section{padding:62px 0;border-bottom:1px solid var(--border);scroll-margin-top:56px}
section:nth-of-type(even){background:var(--surface)}
p{margin-bottom:14px;max-width:76ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:24px 26px}
section:nth-of-type(even) .card{background:var(--surface-alt)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:860px){.grid2,.grid3{grid-template-columns:1fr}}
.stat{font-family:var(--display);font-size:38px;font-weight:700;letter-spacing:-.02em;line-height:1}
.stat.small{font-size:27px}
.stat-label{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;
letter-spacing:.08em;margin-top:8px}
.muted{color:var(--muted)}
.mono{font-family:var(--mono);font-size:13px}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:4px;
border:1px solid var(--border);background:var(--surface);color:var(--muted);margin-right:6px}
.tag.ok{color:#0a7a53;border-color:#0a7a5333;background:#0a7a530f}
.tag.no{color:#b3202c;border-color:#b3202c33;background:#b3202c0f}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
button.seg{font-family:var(--display);font-size:13px;font-weight:500;padding:7px 15px;border-radius:7px;
border:1px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer}
button.seg.on{background:var(--text);color:#fff;border-color:var(--text)}
button.seg:hover:not(.on){border-color:var(--muted);color:var(--text)}
input[type=range]{-webkit-appearance:none;height:4px;background:var(--border);border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;border-radius:50%;
background:var(--orange);cursor:pointer;border:2px solid #fff;box-shadow:0 1px 3px #0003}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:11.5px;color:var(--muted);
margin-top:10px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.howto{margin-top:14px;border-top:1px dashed var(--border);padding-top:12px}
.howto summary{font-family:var(--display);font-size:13px;font-weight:600;color:var(--muted);cursor:pointer}
.howto[open] summary{color:var(--text);margin-bottom:8px}
.howto p{font-size:14.5px;color:var(--body)}
.callout{border-left:3px solid var(--orange);padding:14px 0 14px 18px;background:transparent;margin:18px 0}
.callout .k{font-family:var(--display);font-weight:650;color:var(--text);display:block;margin-bottom:4px}
svg text{font-family:var(--mono)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
td{padding:8px 10px;border-bottom:1px solid var(--border);font-family:var(--serif)}
td.num{font-family:var(--mono);font-size:13px;text-align:right}
.ct{border:1px solid var(--border);border-radius:9px;background:var(--surface);margin:14px 0;
overflow:hidden}
.ct-tabs{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface-alt)}
.ct-tabs button{font-family:var(--display);font-size:12.5px;font-weight:500;padding:9px 16px;border:0;
white-space:nowrap;
background:transparent;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent}
.ct-tabs button.on{color:var(--text);border-bottom-color:var(--orange);background:var(--surface)}
.ct-body{padding:16px 18px;overflow-x:auto}
.ct-label{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;padding:9px 14px;margin-left:auto;align-self:center;text-align:right;
line-height:1.35}
pre.code{font-family:var(--mono);font-size:12.5px;line-height:1.65;white-space:pre;margin:0}
.k-kw{color:var(--orange);font-weight:600}.k-str{color:#1d3557}.k-num{color:#457b9d}
.k-cm{color:var(--muted);font-style:italic}.k-fn{color:#0A0A0A;font-weight:600}
.mathblock{font-family:'Source Serif 4',Georgia,serif;font-size:16.5px;color:var(--text)}
.mathblock .eq{display:block;margin:22px 0 6px;padding-left:2px;line-height:2.5;
white-space:normal}
.mathblock i{font-style:italic;font-family:'Source Serif 4',Georgia,serif}
.mathblock .op{color:var(--muted);padding:0 .3em;font-style:normal}
.mathblock .gl{display:block;font-family:var(--display);font-size:13.5px;color:var(--body);
line-height:1.62;margin:8px 0 20px 2px;padding-left:12px;border-left:2px solid var(--border)}
.mathblock .gl:last-child{margin-bottom:2px}
/* a real fraction: numerator over a rule over denominator */
.frac{display:inline-flex;flex-direction:column;align-items:center;vertical-align:middle;
margin:0 .28em;line-height:1.22}
.frac>.num{padding:0 .4em .12em}
.frac>.den{padding:.12em .4em 0;border-top:1.3px solid currentColor}
/* a real big operator: limits above and below the glyph */
.bigop{display:inline-flex;flex-direction:column;align-items:center;vertical-align:middle;
margin:0 .22em;line-height:1}
.bigop>.above{font-size:.62em;line-height:1.15;white-space:nowrap}
.bigop>.glyph{font-size:1.95em;line-height:.92;font-family:'Source Serif 4',Georgia,serif}
.bigop>.below{font-size:.62em;line-height:1.15;white-space:nowrap}
/* norms, subscripts and a couple of decorations */
.mathblock .norm{padding:0 .06em}
.mathblock sub{font-size:.7em}
.mathblock sup{font-size:.7em}
.mathblock .bar{text-decoration:overline;text-underline-offset:.1em;padding:0 .04em}
.mathblock .paren{font-size:1.5em;line-height:0;vertical-align:-.18em;color:var(--muted)}
.symtab{width:100%;border-collapse:collapse;margin:6px 0 2px}
.symtab td{padding:5px 8px;border-bottom:1px solid var(--border);vertical-align:top}
.symtab td:first-child{font-family:'Source Serif 4',Georgia,serif;font-size:15.5px;white-space:nowrap;
width:1%;padding-right:16px;color:var(--text)}
.symtab td:last-child{font-family:var(--display);font-size:13px;color:var(--body);line-height:1.5}
.symtab tr:last-child td{border-bottom:0}
.symkey{margin-top:14px;border-top:1px dashed var(--border);padding-top:12px}
.symkey>summary{font-family:var(--display);font-size:12.5px;font-weight:600;color:var(--muted);
cursor:pointer;list-style:none}
.symkey>summary::-webkit-details-marker{display:none}
.symkey>summary:before{content:"▸ ";color:var(--orange)}
.symkey[open]>summary:before{content:"▾ "}
.symkey[open]>summary{color:var(--text);margin-bottom:6px}
.mathblock .where{font-family:var(--display);font-size:12.5px;color:var(--muted);
margin:2px 0 0 2px;display:block}
#tip{position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .1s;
background:#0A0A0A;color:#fff;border-radius:7px;padding:9px 12px;font-family:var(--display);
font-size:13px;line-height:1.45;max-width:290px;box-shadow:0 4px 14px #0003}
#tip b{font-weight:650}
#tip .t-sub{color:#BDBDBD;font-size:12px;display:block;margin-top:3px}
.storybox{font-family:var(--serif);font-size:15.5px;line-height:1.72;background:var(--surface-alt);
border:1px solid var(--border);border-radius:9px;padding:18px 20px;max-height:330px;overflow-y:auto}
.storybox .ph{transition:opacity .18s}
.storybox .ph.dim{opacity:.32}
.storybox .phtag{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
display:block;margin:10px 0 4px;font-weight:600}
.storybox .ph:first-child .phtag{margin-top:0}
h2 .secno{font-family:var(--mono);font-size:.62em;font-weight:600;color:var(--orange);
margin-right:.5em;vertical-align:.12em;letter-spacing:0}
.takeaway{margin-top:16px;border-left:3px solid var(--orange);padding:12px 0 12px 16px;
font-family:var(--display);font-size:15px;line-height:1.55;color:var(--text)}
.takeaway b{font-weight:650}
.takeaway .lbl{display:block;font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
letter-spacing:.1em;color:var(--orange);margin-bottom:4px}
.index{display:grid;grid-template-columns:repeat(2,1fr);gap:0 26px;margin-top:22px}
@media(max-width:760px){.index{grid-template-columns:1fr}}
.index a{display:flex;gap:12px;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--border);
text-decoration:none;color:var(--body);font-family:var(--display);font-size:14.5px}
.index a:hover{color:var(--text)}
.index a .n{font-family:var(--mono);font-size:12px;color:var(--orange);font-weight:600;min-width:20px}
.index a .t{flex:1}
.kbd{display:inline-block;font-family:var(--mono);font-size:11px;border:1px solid var(--border);
border-bottom-width:2px;border-radius:4px;padding:1px 6px;background:var(--surface);color:var(--muted);
margin:0 2px}
#bar{position:fixed;top:55px;left:0;height:2px;background:var(--orange);z-index:60;width:0;
opacity:.75;transition:width .12s}
.figwrap{position:relative}
.expand{position:absolute;top:6px;right:6px;z-index:2;font-family:var(--display);font-size:11.5px;
padding:4px 10px;border:1px solid var(--border);border-radius:6px;background:var(--surface);
color:var(--muted);cursor:pointer;opacity:.55;transition:opacity .15s}
.figwrap:hover .expand{opacity:1}
.expand:hover{color:var(--text);border-color:var(--muted)}
#modal{position:fixed;inset:0;z-index:100;background:rgba(10,10,10,.72);display:none;
align-items:center;justify-content:center;padding:26px}
#modal.on{display:flex}
#modalInner{background:var(--surface);border-radius:12px;padding:20px 24px;max-width:96vw;
max-height:94vh;overflow:auto;width:1180px}
#modalInner svg{width:100%;height:auto}
#modalClose{float:right;font-family:var(--display);font-size:13px;border:1px solid var(--border);
border-radius:6px;padding:5px 12px;background:var(--surface);cursor:pointer;color:var(--muted)}
footer{padding:44px 0 64px;color:var(--muted);font-size:14px}
.src{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:10px}
"""


def build() -> str:
    story = json.loads(STORY_JSON.read_text())
    payload = {
        "story": story,
        "storyText": json.loads(STORY_TEXT_JSON.read_text()),
        "emoByLayer": json.loads(EMO_BY_LAYER_JSON.read_text()),
        "three": json.loads(THREE_STORIES_JSON.read_text()),
        "rsa": json.loads(RSA_JSON.read_text()),
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
    html = TEMPLATE.replace("__CSS__", CSS).replace("__DATA__", json.dumps(payload))
    return html


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Do language models track emotions in stories?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>__CSS__</style></head>
<body>
<div id="tip"></div><div id="bar"></div>
<div id="modal"><div id="modalInner"><button id="modalClose">close &times;</button>
<div id="modalBody"></div></div></div>
<nav><div class="wrap">
  <span class="brand">Emotion vectors in Gemma&nbsp;4</span>
  <a href="#question">Question</a><a href="#replication">Replication</a>
  <a href="#displaced">What moved</a><a href="#story">One story</a>
  <a href="#emotions">Per emotion</a><a href="#layers">Layers</a><a href="#probes">Probe quality</a><a href="#next">Next</a><a href="#methods">Methods</a><a href="#appendix">Appendix</a>
</div></nav>

<header><div class="wrap">
  <div class="kicker">CAMBRIA capstone &middot; Hannah Kim &middot; Peyton Li &middot; Antonio Badilla Olivas</div>
  <h1>Does a language model keep track of<br>how a story <em>feels</em>?</h1>
  <p class="lede narrow">We replicated Anthropic's emotion-vector work on Gemma&nbsp;4&nbsp;31B, then pushed it
  somewhere the paper did not go: stories where the emotion <em>changes</em>. The base model reproduces the
  emotion geometry almost exactly. The instruction-tuned model does not, and what replaced it is the most
  interesting thing we found.</p>
  <div class="controls" style="margin-top:24px">
    <span class="tag ok">base model: circumplex replicates</span>
    <span class="tag no">instruct: valence demoted to PC3</span>
    <span class="tag">emotions tracked in stories, unevenly</span>
  </div>
  <div class="index" id="coverIndex"></div>
  <div class="muted" style="font-family:var(--display);font-size:13px;margin-top:18px">
    Press <span class="kbd">&rarr;</span> or <span class="kbd">space</span> for the next section,
    <span class="kbd">&larr;</span> to go back, <span class="kbd">1</span>&ndash;<span class="kbd">9</span>
    or <span class="kbd">0</span> to jump straight to a numbered section.
  </div>
</div></header>

<section id="question"><div class="wrap">
  <div class="kicker">What we were trying to find out</div>
  <h2><span class="secno">1</span>Three questions, in order</h2>
  <div class="grid3" style="margin-top:26px">
    <div class="card"><h3>1. Does it replicate?</h3><p style="font-size:15px">Anthropic reports that
    emotion vectors, one direction per emotion, lay out in a valence-by-arousal plane: the
    <em>circumplex</em> psychologists have used for decades. We rebuilt it on a different model,
    on both the base and the instruction-tuned checkpoint.</p></div>
    <div class="card"><h3>2. Why do the two models differ?</h3><p style="font-size:15px">They diverge
    sharply. Instruction tuning does not erase the emotion structure, it <em>demotes</em> it and puts
    something else in the top slot. We went looking for what.</p></div>
    <div class="card"><h3>3. Does it track a story?</h3><p style="font-size:15px">A vector that only
    labels static text is a dictionary. We wrote stories that move through three emotions and asked
    whether the model follows along, token by token.</p></div>
  </div>
  <h3 style="margin-top:38px">Why should anyone care whether this works?</h3>
  <p class="narrow" style="margin-bottom:20px">Five reasons, and they pull in different directions.
  Some want the representation to be strong; one of them is interesting precisely if it turns out weak.</p>
  <div class="grid3" style="gap:16px">
    <div class="card"><h3>A lever to pull</h3><p style="font-size:15px">A readable, causal
    representation of emotional state is something you can inspect and something you can
    <em>move</em>. That is interpretability (what is it representing?) and safety (what happens when
    you push on it?) in one object.</p></div>
    <div class="card"><h3>Models as listeners</h3><p style="font-size:15px">People already bring models
    their worst days, and products are being sold as companions and therapists. That job is not
    sentiment classification on one message: it is following how someone's state <em>moves</em>
    across a long conversation. Precisely what we tested, and precisely where it gets shaky.</p></div>
    <div class="card"><h3>Internal evidence, for the welfare question</h3><p style="font-size:15px">
    Arguments about model welfare mostly run on what a model <em>says</em>. Emotion vectors are one of
    the few handles on internal state instead. A caveat we take seriously: we measured the model
    reading emotions in a <em>story</em>, not having them. Those are different claims, and conflating
    them is the easiest mistake in this area.</p></div>
    <div class="card"><h3>How it carves up the world</h3><p style="font-size:15px">The emotion space we
    recover is the model's, not ours. Where it disagrees with human ratings, that gap is a map of what
    is idiosyncratic about its concepts, and instruction tuning visibly redraws it.</p></div>
    <div class="card" style="border-color:var(--orange)"><h3>Or: it may just not be very emotional</h3>
    <p style="font-size:15px">The interesting possibility we cannot rule out. Maybe emotion is simply
    not a load-bearing axis for this model, and we keep finding a weak signal because that is all there
    is. A null here would be a real finding about what these systems represent, not a failed
    experiment.</p></div>
  </div>
</div></section>

<section id="replication"><div class="wrap">
  <div class="kicker">Finding 1</div>
  <h2><span class="secno">2</span>The circumplex replicates on the base model, and collapses on the instruct model</h2>
  <p class="narrow">Each bar is one principal component of the 171 emotion vectors at layer 33. Height is
  how much of the spread it explains; the coloured bars are how strongly it correlates with human ratings
  of <b>valence</b> (pleasant/unpleasant), <b>arousal</b> (calm/excited), <b>dominance</b>, and with a
  confound we had to check: <b>how long the stories were</b>.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls">
      <button class="seg on" data-model="base">base model</button>
      <button class="seg" data-model="instruct">instruction-tuned</button>
      <span class="muted mono" style="margin-left:auto" id="pcVerdict"></span>
    </div>
    <div id="pcChart"></div>
    <div class="legend">
      <span><i style="background:var(--navy)"></i>valence</span>
      <span><i style="background:var(--teal)"></i>arousal</span>
      <span><i style="background:var(--green)"></i>dominance</span>
      <span><i style="background:var(--amber)"></i>story length (confound)</span>
      <span><i style="background:#D4D4D4"></i>variance explained</span>
    </div>
    <details class="howto"><summary>How to read this</summary>
      <p>A good replication looks like the base model: the <em>first</em> component is valence
      (|r| = 0.83) and the second is arousal (|r| = 0.55). That is the circumplex, recovered without
      being asked for. A failure looks like the instruct model: its first component correlates with
      valence at only 0.11, barely more than nothing, while explaining almost twice as much variance
      (28% against 15%). Valence has not disappeared, it has been pushed down to the third component
      (|r| = 0.72). Watch the amber bar too: the instruct model's second component tracks story
      <em>length</em> at 0.66, which is a reminder that not every strong axis is about emotion.</p>
    </details>
    <div class="src">source: notebooks/02_circumplex_geometry.ipynb, layer 33, post-fix instrument</div>
  </div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from this plot:</b> on the base model the biggest axis <em>is</em> valence, so the circumplex is recovered without asking for it. On the instruct model the biggest axis is something else, and valence has been pushed down to third.</div>
</div></section>

<section id="displaced"><div class="wrap">
  <div class="kicker">Finding 2</div>
  <h2><span class="secno">3</span>So what took valence's place?</h2>
  <p class="narrow">We compared every instruct component against every base component. If instruction
  tuning merely rotated the space, each instruct axis would find a partner. Hover any cell.</p>
  <div class="grid2" style="margin-top:22px">
    <div class="card">
      <div id="gridChart"></div>
      <div id="gridNote" class="mono muted" style="margin-top:10px;min-height:34px"></div>
    </div>
    <div class="card">
      <h3>The top axis has no partner</h3>
      <p style="font-size:15px">The instruct model's largest axis matches its best base counterpart at
      just <b>0.14</b>. It is not a rotation of anything the base model had: it is new structure,
      inserted by instruction tuning, and it is the single biggest direction in the space.</p>
      <p style="font-size:15px">Meanwhile the old valence axis is still there, intact, one floor down:
      instruct PC3 matches base PC1 at <b>0.83</b>.</p>
      <div style="margin-top:16px">
        <div class="mono muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.07em">
          what sits at each end of that new axis</div>
        <div style="margin-top:8px"><b>low:</b> <span class="mono" id="pc1low"></span></div>
        <div style="margin-top:6px"><b>high:</b> <span class="mono" id="pc1high"></span></div>
        <p class="muted" style="font-size:14px;margin-top:10px">Read the two lists. It is not
        pleasant-versus-unpleasant: <em>miserable</em> and <em>jubilant</em> sit together at one end.
        It looks more like a low-grade irritation or agitation axis. We do not yet know what it is.
        That is our open question.</p>
      </div>
    </div>
  </div>
  <div class="card" style="margin-top:20px">
    <div class="controls">
      <span class="mono muted">show</span><span id="rsaBtns"></span>
      <span class="mono muted" id="rsaNote" style="margin-left:auto"></span>
    </div>
    <div id="rsaChart"></div>
    <details class="howto"><summary>How to read this</summary>
      <p>Each square compares the <em>shape of emotion space</em> at two layers: take the full
      emotion-by-emotion similarity matrix at layer A and at layer B, and correlate them. Bright means
      the two layers organise the emotions the same way; dark means they disagree.</p>
      <p>A model with one stable emotion geometry is bright everywhere. The instruct model is not: it
      splits into blocks, which is the fragmentation. Switch to <b>top component removed</b> and the
      blocks largely merge, which is the evidence that one dominant axis was causing the split rather
      than the emotion structure being absent. The third view compares the two models directly.</p>
    </details>
    <div class="src">source: notebooks/02_circumplex_geometry.ipynb, section 9</div>
  </div>

  <div class="callout"><span class="k">The honest caveat</span>
  The first principal angle between the two models' top-3 subspaces is <b>86.1&deg;</b>, essentially a
  right angle. But when we remove just the top instruct component, cross-model similarity of the late
  layers jumps from <b>0.29</b> to <b>0.60</b>. The shared emotion geometry is not gone. It is buried
  under one dominant, non-affective direction.</div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from this grid:</b> the instruct model's top axis is not a rotation of anything the base model had. It is new structure, it is the largest thing in the space, and we cannot yet say what it encodes.</div>
</div></section>

<section id="story"><div class="wrap">
  <div class="kicker">Finding 3 &middot; the centrepiece</div>
  <h2><span class="secno">4</span>Watch it track a story, token by token</h2>
  <p class="narrow">Four real stories from our corpus. At every token we measure how close the model's
  internal state sits to each of the three emotion probes; scrub the token slider and watch both panels
  move together. Three of the four use the <b>same three emotions</b> and differ only in how they are
  written, and the model tracks them very differently. That is the point of this section: quality is not
  a property of the emotion alone.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls" style="margin-bottom:6px">
      <span class="mono muted">story</span><span id="storyBtns"></span>
      <span class="mono muted" id="storyQual" style="margin-left:auto"></span>
    </div>
    <div class="controls">
      <span class="mono muted">layer</span>
      <span id="layerBtns"></span>
      <span style="margin-left:auto;display:flex;align-items:center;gap:10px">
        <span class="mono muted">token</span>
        <input type="range" id="tokSlider" min="0" max="235" value="0" style="width:230px">
        <span class="mono" id="tokLabel" style="min-width:78px;display:inline-block"></span>
      </span>
    </div>
    <div class="grid2" style="gap:24px">
      <div><div class="mono muted" style="font-size:11.5px;margin-bottom:6px">
        the three readings over the story</div><div id="lineChart"></div></div>
      <div><div class="mono muted" style="font-size:11.5px;margin-bottom:6px">
        the same walk, inside the triangle of the three emotions</div><div id="ternChart"></div></div>
    </div>
    <div style="margin-top:20px">
      <div class="mono muted" style="font-size:11.5px;margin-bottom:6px">
        the story the model is reading (the phase you are scrubbing is highlighted)</div>
      <div class="storybox" id="storyBox"></div>
    </div>
    <details class="howto" open><summary>How to read this</summary>
      <p><b>Left:</b> one line per emotion probe. Higher means the model's state is closer to that
      emotion. The two vertical marks are where the story is written to turn. A perfect tracker would
      hand the lead from the red line to the blue to the green, right at those marks.</p>
      <p><b>Right:</b> the same three numbers as a position inside a triangle. Each corner is one
      emotion; the dot sits nearer the corner it most resembles. A perfect tracker walks corner to
      corner. A model that ignored the story would sit still in the middle.</p>
      <p><b>What actually happens:</b> the walk is real but sloppy. The handover happens, and it tends to
      happen <em>early</em>, before the written boundary: the model anticipates the turn. Switch layers
      and the picture changes character, which is section 6.</p>
      <p><b>Why four stories, and how "well tracked" is measured here.</b> The last three are the best,
      the middle and the worst of a random sample of 24 stories, scored on <em>exactly what this figure
      plots</em>: in each of the three phases, does the phase's own emotion have the highest curve, and
      by how much? Leading all three is the ceiling; one of three is what luck gives, since there are
      three curves.</p>
      <p>Note this is an easier question than the one section 5 asks. Here the model only has to pick
      the right emotion out of the story's own three. There it has to pick it out of twelve. A story can
      lead all three curves here and still place poorly out of twelve, so do not read this panel as the
      headline tracking number.</p>
    </details>
    <div class="src">story t000_seq_p2_2f9faf62, from notebooks/05_trajectories.ipynb; centered cosine against self-generated probes</div>
  </div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from this story:</b> the handover between emotions really happens, and it tends to happen <em>early</em>, before the written turn. The walk is real but noisy, and it looks different at different layers.</div>
</div></section>

<section id="emotions"><div class="wrap">
  <div class="kicker">Finding 4</div>
  <h2><span class="secno">5</span>It tracks some emotions well and others not at all</h2>
  <p class="narrow">For each tagged emotion, how often does its own probe win outright? Chance is 1 in
  12. Hover any bar for the full picture, and switch layers: the ranking is not stable across depth.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls">
      <button class="seg on" data-bank="selfgen">probes from stories Gemma wrote</button>
      <button class="seg" data-bank="deepseek">probes from stories DeepSeek wrote</button>
      <span style="margin-left:auto;display:flex;align-items:center;gap:8px">
        <span class="mono muted">layer</span><span id="emoLayerBtns"></span>
      </span>
    </div>
    <div id="emoChart"></div>
    <div id="emoNote" class="mono" style="margin-top:12px;min-height:36px;color:var(--body)"></div>
    <details class="howto"><summary>How to read this</summary>
      <p>The dashed line at 8% is what guessing gives you. At layer 33 in the Gemma-written bank,
      11 of the 12 emotions beat it, so something real is being tracked. But nothing clears 50%, and
      the spread is wide: <em>loving</em> and <em>guilty</em> win about half their phases while
      <em>nervous</em> (5.8%) sits below chance. Switch layers and the ranking moves, which is why
      no single layer should be quoted as "how well the model tracks emotion".</p>
      <p>The wrong answers are not random, and this is the part we did not expect. In the Gemma-written
      bank, wrong answers pile into two attractors: <b>guilty</b> and <b>happy</b> absorb the mistakes
      from emotion after emotion. Switch to the DeepSeek bank and the attractors vanish, and overall
      accuracy goes up. So a good chunk of what looks like "the model cannot track emotions" is
      actually a property of the probe set, not of the model.</p>
    </details>
    <div class="src">source: notebooks/11_tracking_taxonomy.ipynb, layer 33, instruct reader</div>
  </div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from these bars:</b> tracking is real but uneven, and a good part of the unevenness belongs to the probe set rather than the model. Switching the probe bank changes both the accuracy and which emotions absorb the mistakes.</div>
</div></section>

<section id="layers"><div class="wrap">
  <div class="kicker">Finding 5</div>
  <h2><span class="secno">6</span>Different layers are good at different things</h2>
  <p class="narrow">There is no single "emotion layer". Naming the current emotion is best early;
  anticipating the <em>next</em> one is best late. Those are different jobs, done in different places.</p>
  <div class="grid2" style="margin-top:22px">
    <div class="card"><div id="layerChart"></div>
      <div class="legend">
        <span><i style="background:var(--navy)"></i>names the right emotion (top-1 rate)</span>
        <span><i style="background:var(--orange)"></i>anticipates the size of the coming jump</span>
      </div>
    </div>
    <div class="card">
      <h3>Two different competences</h3>
      <p style="font-size:15px">Layer 6 names the current emotion best (<b>58%</b> top-1). By layer 33
      that has fallen to <b>27%</b>. But the ability to anticipate <em>how big</em> the next emotional
      jump will be climbs the other way, from <b>+0.03</b> at layer 6 to <b>+0.26</b> at layer 51.</p>
      <p style="font-size:15px">Even when it is wrong, it is wrong <em>nearby</em>: the emotion it
      picks instead is consistently closer in human valence-arousal space than a random emotion would
      be, at every single layer.</p>
      <div class="callout" style="margin-top:6px"><span class="k">A late surprise</span>
      At layer 33, the model's own probe geometry predicts its confusions better than human ratings do.
      At layer 51 that reverses. Our headline finding is layer-dependent, and we only noticed because
      every figure re-computes its verdict per layer.</div>
    </div>
  </div>
  <div class="src">source: notebooks/11_tracking_taxonomy.ipynb (per-layer reads)</div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from these curves:</b> there is no single "emotion layer". Naming the current emotion is an early-layer skill, anticipating the next one is a late-layer skill.</div>
</div></section>


<section id="probes"><div class="wrap">
  <div class="kicker">Finding 6</div>
  <h2><span class="secno">7</span>Where the vectors come from matters more than we expected</h2>
  <p class="narrow">Every emotion vector is built by averaging the model's activations over stories that
  evoke that emotion. So who <em>writes</em> those stories is a free parameter, and nobody reports it.
  We built four probe sets from four different authors and scored them identically.</p>
  <div class="card" style="margin-top:22px">
    <div id="lineageChart"></div>
    <div class="legend"><span><i style="background:var(--navy)"></i>layers where the probes pass the
    registered detection bar (8 of 12 scenarios, on two separate batteries)</span></div>
    <details class="howto"><summary>How to read this</summary>
      <p>Each bar is one probe set, differing only in who wrote the stories behind it. Height is how many
      of the 20 tested layers pass a bar that was registered before any of this was scored. Zero would
      mean the probes never work; the best we saw is 9.</p>
      <p><b>The ordering is the finding.</b> A stronger external writer (DeepSeek) gives probes that work
      at nine layers. The model's own writing gives five. A weaker model's writing gives one. So "the
      model understands its own emotions best" is false here: it is out-written.</p>
    </details>
    <div class="src">source: notebooks/07_generator_lineages.ipynb (experiments E11 and E12)</div>
  </div>
  <div class="grid2" style="margin-top:20px">
    <div class="card"><h3>Why its own stories are worse</h3>
      <p style="font-size:15px">Asked for 3,072 emotional stories, the model wrote a character named
      <b>Elias</b> in <b>98.2%</b> of them. Its self-written corpus repeats 5-word phrases
      <b>53&times;</b> more often than DeepSeek's. Narrow, repetitive text makes a narrow vector: the
      probe learns the model's writing habits alongside the emotion.</p>
      <p style="font-size:15px" class="muted">This is also the clearest thing we saw about what is
      idiosyncratic to this model. Ask it for variety and it gives you Elias, 3,000 times.</p></div>
    <div class="card"><h3>How many stories do you actually need?</h3>
      <div id="doseChart"></div>
      <p style="font-size:15px;margin-top:8px">About <b>64</b> per emotion. By then the probes reach
      8.6 of their 9-layer ceiling, and 4&times; more data buys almost nothing.</p></div>
  </div>
  <div class="takeaway"><span class="lbl">the read</span><b>What we take from these bars:</b> the quality of an emotion vector is set by who wrote the stories behind it. The model is out-written by a stronger external author, and its own writing is too repetitive to make a broad probe.</div>
</div></section>

<section id="next"><div class="wrap">
  <div class="kicker">Where this leaves us</div>
  <h2><span class="secno">8</span>What we would do next</h2>
  <div class="grid3" style="margin-top:24px">
    <div class="card"><h3>Name the new axis</h3><p style="font-size:15px">The largest direction in the
    instruct model is not valence, arousal, dominance, or story length. Identify it: steer along it and
    see what changes in the model's behaviour.</p></div>
    <div class="card"><h3>Separate probe from model</h3><p style="font-size:15px">Attractors appeared in
    one probe bank and not the other. Probe-set construction is doing more work than anyone
    reports. That deserves its own experiment.</p></div>
    <div class="card"><h3>Test the anticipation</h3><p style="font-size:15px">The model appears to lean
    into an emotional turn before it arrives. Is that reading the setup, or predicting the next token?
    A story whose turn is unforeseeable would separate the two.</p></div>
  </div>
  <div class="callout"><span class="k">What we are confident in, and what we are not</span>
  Confident: the base model reproduces the circumplex; the instruct model does not, and one non-affective
  axis dominates it. Emotion probes do track a story better than chance. Not confident: what the new axis
  <em>is</em>; how much of the tracking quality is the probe set rather than the model. Everything here
  is registered in a research tree with the evidence file behind each number, including the results that
  came out null.</div>
  <div class="callout" style="border-color:var(--navy)"><span class="k">The reading we cannot yet rule
  out, and would like to</span>
  Every positive result here is real but small: above chance, below any bar you would want before
  trusting it. The tidy story is that emotion is represented and we are measuring it imperfectly. The
  other story is that emotion is simply not a major axis of this model's world, and a weak signal is
  the honest answer rather than a measurement problem. Distinguishing those two is the experiment we
  would run next, and it is the more interesting question either way.</div>
</div></section>


<section id="methods"><div class="wrap">
  <div class="kicker">How we measured it</div>
  <h2><span class="secno">9</span>Method, in enough detail to argue with</h2>
  <p class="narrow">Each block is collapsed. Open the one you want to challenge.</p>

  <details class="howto" style="border-top:none;margin-top:18px">
    <summary style="font-size:15px">1. How an emotion becomes a vector</summary>
    <p>We take a corpus where every story is labelled with the emotion it is written to evoke, feed each
    story to the model, and record the activations at a chosen layer. For one emotion, we average those
    activations over all its stories. That gives one vector per emotion, 171 of them.</p>
    <div id="m1"></div>
    <p>Two details that change the numbers. <b>Centering:</b> we subtract the mean over all 12 or 171
    emotions, so what is left is what makes <em>this</em> emotion different rather than what all text has
    in common. Skipping this step is the difference between a result and an artefact.
    <b>Which tokens:</b> a story is many tokens, and we pool them: mask the padding, drop the first 50
    as narrative framing (the source paper's convention), and average over the rest, capped at 512
    tokens. Both choices were fixed before scoring. Notice this differs from the story-tracking read in
    block 3, where we deliberately keep every token instead of pooling.</p>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">2. What "PC1 = valence, |r| 0.83" actually means</summary>
    <p>This is the one worth spelling out, because the phrase hides four steps.</p>
    <p><b>Step one.</b> We have 171 emotion vectors, each thousands of numbers long. Principal component
    analysis finds the directions along which those 171 points differ most. The first component (PC1) is
    the single direction that captures the most spread. It is derived only from the model: no human
    labels are involved in finding it.</p>
    <p><b>Step two.</b> Each of the 171 emotions gets a <em>score</em> on PC1: how far along that
    direction it sits. So "elated" gets a number, "miserable" gets a number, and so on.</p>
    <p><b>Step three.</b> Separately, we look up each emotion word in the
    <a href="http://saifmohammad.com/WebPages/nrc-vad.html">NRC VAD lexicon</a>, a published set of
    human ratings where people scored thousands of words for valence, arousal and dominance. 164 of our
    171 words are in it.</p>
    <p><b>Step four.</b> We correlate the model's 164 PC1 scores against the humans' 164 valence ratings.
    That correlation is <b>0.83</b> on the base model. In words: the direction of largest variation in
    the model's emotion space, found without reference to any human judgement, lines up strongly with
    what people call pleasant versus unpleasant.</p>
    <div id="m2"></div>
    <p><b>Why |r| and not r.</b> The sign of a principal component is arbitrary, so we report absolute
    correlation. <b>What would count as failure:</b> |r| near 0, meaning the model's biggest axis has
    nothing to do with valence. That is close to what the instruct model gives (0.11).</p>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">3. How "does it track the story?" was scored</summary>
    <p>We wrote 173 three-emotion story sets, each moving through three tagged phases. Here nothing is
    pooled away: the model reads the story once and we keep the activation at <b>every token</b>, then
    take the cosine against each of 12 emotion probes at each position. A phase's score is the mean over
    the tokens inside that phase; phases shorter than 4 tokens are skipped rather than averaged. Two
    questions follow, both with pre-registered answers.</p>
    <p><b>Naming (the gate):</b> within a phase written as "afraid", where does the afraid probe rank
    among the 12? Rank 1 is perfect, 6.5 is guessing. <b>Anticipation (the lead):</b> in the window
    before a written turn, does the <em>next</em> emotion's probe already start to lead? Zero would mean
    the model does not see the turn coming.</p>
    <div id="m3"></div>
    <p>Both are compared against a wrong-emotion shuffle: the same computation with the labels permuted.
    That shuffle, not our intuition, is the floor.</p>
    <div id="m4"></div>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">4. What kept us honest</summary>
    <p>Every prediction and its pass bar was written into a research tree before the data existed. Claims
    only graduate after a falsification pass: permutation nulls, bootstrap confidence intervals clustered
    by story, random-direction controls, and base-rate checks. Results that failed are recorded as
    failed. Every number in this deck resolves to a file in the repository, and the notebooks re-run from
    those files on any machine.</p>
    <p class="muted">One convention change is worth flagging because it flipped a headline: our first
    detection campaign found nothing passing, and a later audit showed the readout had skipped the
    centering step in step 1. With centering, the same data passes on both models. We report both.</p>
  </details>
</div></section>

<section id="appendix"><div class="wrap">
  <div class="kicker">Appendix</div>
  <h2><span class="secno">10</span>Questions we expect, and our honest answers</h2>
  <p class="narrow">Collapsed, so we can jump to whichever one gets asked.</p>
  <div style="margin-top:18px">
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Isn't this just sentiment analysis with extra steps?</summary>
      <p>Sentiment analysis gives one label for a span of text. We are reading a direction in the model's internal state, at every token, without asking the model anything. That lets us do two things a classifier cannot: watch the representation move <em>through</em> a story, and intervene on it. The interesting result is not the label, it is that valence organises the space on its own.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Does the model actually feel anything?</summary>
      <p>Nothing here shows that, and we would resist the leap. We measured the model representing emotions <em>in a story it is reading</em>, which is closer to reading comprehension than to having a feeling. It is relevant as internal evidence, because most welfare arguments run on what a model says rather than on its internals, but reading an emotion and having one are different claims.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">The correlations are small. Isn't this noise?</summary>
      <p>Some of them are, and we say so. The geometry result is not small: |r| 0.83 on 164 words is a strong effect. The tracking results are modest and beat a shuffle null at p below 0.001. The intensity result we deliberately report as sign-only, because a control showed random directions reach the same correlation size, so only the direction is evidence.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Why Gemma 4 and not the model in the paper?</summary>
      <p>We could run both a base and an instruction-tuned checkpoint of the same size, which is exactly the comparison the paper cannot make. That constraint produced our main finding.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Could the instruct PC1 just be an artefact of your pipeline?</summary>
      <p>We checked the obvious ones. It is not story length (|r| 0.39, and PC2 tracks length harder at 0.66). It survives an extraction-convention audit that changed other numbers. It is not a sign flip or a rotation, since its best match against any base component is 0.14. What it <em>is</em> remains open, and we would rather say that than name it.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">How much of the tracking result is your probe set rather than the model?</summary>
      <p>More than we would like, and that is a finding in itself. Swapping who wrote the stories behind the probes moves passing layers from 1 to 9, and the confusion attractors present in one bank vanish in another. Any paper reporting emotion-probe accuracy without reporting probe provenance is under-specified.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Why does layer 6 name emotions better than layer 33?</summary>
      <p>We do not know, and it is our favourite loose end. One hypothesis: early layers stay closer to surface lexical cues, and our stories contain emotional words, so a shallow reader does well. The deciding test is a corpus where the emotion is implied but never lexically present.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">What would have changed your mind?</summary>
      <p>For the geometry: a base model whose PC1 did not correlate with valence. For the tracking: probe ranks indistinguishable from the wrong-emotion shuffle, which is what the constant-emotion control arm came close to. We ran that control precisely so a null had somewhere to show up.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">What is the single weakest part of this work?</summary>
      <p>The absolute effect sizes on tracking. Everything is above chance and below where you would want it before trusting it in a product. We cannot yet separate 'weak measurement of a real thing' from 'accurate measurement of a weak thing'.</p>
    </details>
  </div>
  <div class="callout" style="margin-top:24px"><span class="k">If you remember one thing</span>
  The base model reproduces the emotion circumplex almost exactly. Instruction tuning does not destroy
  that structure, it demotes it and puts one large non-affective axis on top. And whichever model you
  ask, how well its emotion probes work depends heavily on who wrote the stories you built them from.</div>
</div></section>

<footer><div class="wrap">
  <div>CAMBRIA capstone &middot; Gemma 4 31B, base and instruction-tuned &middot; every figure regenerated
  from the notebooks in the project repository.</div>
  <div class="src">Interactive figures built from the committed notebook outputs. Numbers shown here are
  the printed records of notebooks 02, 05, 08 and 11.</div>
</div></footer>

<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (t,a={})=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
/* the single palette every chart draws from: mirrors the CSS tokens above,
   so a colour can only be changed in one place */
const P = {text:"#0A0A0A", body:"#262626", muted:"#737373", border:"#E5E5E5",
  navy:"#1d3557", teal:"#457b9d", green:"#009E73", amber:"#E69F00", red:"#e63946",
  orange:"#CC785C", greySoft:"#D4D4D4", greyMid:"#BDBDBD", alert:"#b3202c"};
const C = {valence:P.navy, arousal:P.teal, dominance:P.green, length:P.amber, evr:P.greySoft};

/* one floating tooltip shared by every chart on the page */
const TIP=document.getElementById("tip");
function tipOn(node, html){
  node.style.cursor="default";
  node.addEventListener("mousemove",e=>{
    TIP.innerHTML=typeof html==="function"?html():html;
    TIP.style.opacity=1;
    const r=TIP.getBoundingClientRect();
    let x=e.clientX+14, y=e.clientY+14;
    if(x+r.width>innerWidth-8) x=e.clientX-r.width-14;
    if(y+r.height>innerHeight-8) y=e.clientY-r.height-14;
    TIP.style.left=x+"px"; TIP.style.top=y+"px";
  });
  node.addEventListener("mouseleave",()=>{TIP.style.opacity=0;});
}
const pct=v=>(v*100).toFixed(0)+"%";

/* ---------- Finding 1: PC bars ---------- */
function drawPCs(model){
  const host=document.getElementById("pcChart"); host.innerHTML="";
  const rows=D.pcs[model], W=640,H=290,L=52,R=14,T=16,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  // grading band: |r| above 0.5 is where a component is meaningfully aligned
  svg.appendChild(el("rect",{x:L,y:T,width:iw,height:ih*0.5,fill:P.green,opacity:.05}));
  [0,.25,.5,.75,1].forEach(v=>{
    const y=T+ih-v*ih;
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y,y2:y,stroke:P.border}));
    const tx=el("text",{x:L-8,y:y+4,"text-anchor":"end","font-size":10,fill:P.muted});
    tx.textContent=v.toFixed(2); svg.appendChild(tx);
  });
  // both ends of the scale, said in words rather than left to the reader
  [[1,"|r| = 1: the component IS that human rating",P.green],
   [0.5,"0.5: meaningfully aligned",P.muted],
   [0,"0: no relation at all",P.alert]].forEach(([v,label,col])=>{
    const y=T+ih-v*ih;
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y,y2:y,stroke:col,"stroke-dasharray":"4 3",
      "stroke-width":v===0.5?1:1.4,opacity:v===0.5?.5:.9}));
    const t=el("text",{x:W-R-2,y:y-5,"text-anchor":"end","font-size":10,fill:col});
    t.textContent=label; svg.appendChild(t);
  });
  rows.forEach((r,i)=>{
    const x0=L+i*bw;
    // variance explained as a soft backdrop bar
    const eh=r.evr*ih;
    svg.appendChild(el("rect",{x:x0+8,y:T+ih-eh,width:bw-16,height:eh,fill:C.evr,rx:3}));
    const keys=["valence","arousal","dominance","length"], sw=(bw-26)/keys.length;
    keys.forEach((k,j)=>{
      const h=r[k]*ih;
      const bar=el("rect",{x:x0+13+j*sw,y:T+ih-h,width:sw-3,height:h,fill:C[k],rx:2});
      const nm={valence:"valence",arousal:"arousal",dominance:"dominance",length:"story length"}[k];
      tipOn(bar,`<b>PC${r.pc} vs ${nm}</b>: |r| = ${r[k].toFixed(2)}`+
        `<span class="t-sub">this component explains ${(r.evr*100).toFixed(1)}% of the spread`+
        `<br>|r| of 1 = perfectly aligned, 0 = unrelated</span>`);
      svg.appendChild(bar);
    });
    const back=el("rect",{x:x0+8,y:T+ih-r.evr*ih,width:bw-16,height:r.evr*ih,fill:"transparent"});
    tipOn(back,`<b>PC${r.pc}</b> explains ${(r.evr*100).toFixed(1)}% of the variance`+
      `<span class="t-sub">grey backdrop = variance explained</span>`);
    svg.appendChild(back);
    const lab=el("text",{x:x0+bw/2,y:H-24,"text-anchor":"middle","font-size":11,fill:P.text});
    lab.textContent="PC"+r.pc; svg.appendChild(lab);
    const ev=el("text",{x:x0+bw/2,y:H-10,"text-anchor":"middle","font-size":9.5,fill:P.muted});
    ev.textContent=(r.evr*100).toFixed(1)+"% var"; svg.appendChild(ev);
  });
  host.appendChild(svg);
  document.getElementById("pcVerdict").textContent = model==="base"
    ? "PC1 = valence (|r| 0.83). the circumplex is right there."
    : "PC1 = ? (valence |r| 0.11). valence is demoted to PC3.";
}
document.querySelectorAll("[data-model]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-model]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); drawPCs(b.dataset.model);
});

/* ---------- Finding 2: correlation grid ---------- */
function drawGrid(){
  const host=document.getElementById("gridChart"); host.innerHTML="";
  const W=420,H=420,L=64,T=42,cell=62;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const note=document.getElementById("gridNote");
  D.grid.forEach((row,i)=>row.forEach((v,j)=>{
    const x=L+j*cell,y=T+i*cell;
    const g=el("rect",{x,y,width:cell-3,height:cell-3,rx:3,
      fill:`rgba(29,53,87,${(v*0.95).toFixed(3)})`,style:"cursor:pointer"});
    tipOn(g,`<b>instruct PC${i+1} vs base PC${j+1}</b>: |r| = ${v.toFixed(2)}`+
      `<span class="t-sub">`+
      (i===0 ? "instruct PC1 is the biggest axis in the instruct model. Its best match anywhere in the base model is 0.14, so it is new structure." :
       i===2&&j===0 ? "this is valence: the base model's top axis, still intact but demoted to third place." :
       "1 would mean the two components carry the same information; 0 means unrelated.")+
      `</span>`);
    svg.appendChild(g);
    const t=el("text",{x:x+(cell-3)/2,y:y+(cell-3)/2+4,"text-anchor":"middle","font-size":11,
      fill:v>0.45?"#fff":P.text}); t.textContent=v.toFixed(2); svg.appendChild(t);
  }));
  for(let j=0;j<5;j++){const t=el("text",{x:L+j*cell+(cell-3)/2,y:T-12,"text-anchor":"middle",
    "font-size":10.5,fill:P.muted});t.textContent="base PC"+(j+1);svg.appendChild(t);}
  for(let i=0;i<5;i++){const t=el("text",{x:L-8,y:T+i*cell+(cell-3)/2+4,"text-anchor":"end",
    "font-size":10.5,fill:P.muted});t.textContent="it PC"+(i+1);svg.appendChild(t);}
  // a scale strip, so a shade can be read without hovering
  const sx=L, sy=T+5*cell+14, sw=5*cell-3;
  for(let i=0;i<40;i++){
    svg.appendChild(el("rect",{x:sx+i*(sw/40),y:sy,width:sw/40+.5,height:9,
      fill:`rgba(29,53,87,${((i/39)*0.95).toFixed(3)})`}));
  }
  // only the two ends are labelled: a middle tick collided with the right label
  [["0 = unrelated",0,"start"],["1 = the same information",1,"end"]].forEach(([lab,f,anc])=>{
    const t=el("text",{x:sx+f*sw,y:sy+22,"text-anchor":anc,"font-size":10,fill:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  host.appendChild(svg);
  note.textContent="hover a cell";
}
document.getElementById("pc1low").textContent=D.itpc1.low.join(", ");
document.getElementById("pc1high").textContent=D.itpc1.high.join(", ");

/* ---------- Finding 3: the story ---------- */
const EC=[P.red,P.teal,P.green];
/* Every story this section can show: the walkthrough, plus three stories that
   share the SAME three emotions and differ only in how they are written. Those
   three were picked by measured tracking quality (best, median and worst mean
   gate rank at layer 33), not by eye. */
const STORIES=[
  {key:"orig", label:"the walkthrough", qual:null, id:D.story.story_id,
   emotions:D.story.emotions, boundaries:D.story.boundaries, n_tokens:D.story.n_tokens,
   lines_by_layer:D.story.lines_by_layer, text:D.storyText.text},
  ...D.three.map(t=>({
    key:t.quality,
    label:t.quality==="strong"?"tracked well":(t.quality==="mixed"?"tracked partly":"tracked badly"),
    qual:{wins:t.wins, perPhase:t.per_phase_win, margin:t.mean_margin}, id:t.id,
    emotions:t.emotions, boundaries:t.boundaries, n_tokens:t.n_tokens,
    lines_by_layer:t.lines_by_layer, text:t.text}))
];
let curStory=0, S=STORIES[0];
let curLayer=D.story.default_layer, curTok=0;
function drawStory(){
  const ys=S.lines_by_layer[curLayer];
  // line chart
  const host=document.getElementById("lineChart"); host.innerHTML="";
  const W=470,H=300,L=46,R=12,T=14,B=38;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, n=S.n_tokens;
  let lo=Infinity,hi=-Infinity; ys.forEach(s=>s.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
  const X=i=>L+(i/(n-1))*iw, Y=v=>T+ih-((v-lo)/(hi-lo))*ih;
  // the zero line is the story's own average: above it the model leans toward
  // that emotion, below it away. Labelled, because an unlabelled zero is noise.
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(0),y2:Y(0),stroke:P.greyMid,"stroke-dasharray":"3 3"}));
  const zl=el("text",{x:W-R-2,y:Y(0)+11,"text-anchor":"end","font-size":9,fill:P.muted});
  zl.textContent="0 = this story's average"; svg.appendChild(zl);
  S.boundaries.forEach(b=>{
    svg.appendChild(el("line",{x1:X(b),x2:X(b),y1:T,y2:T+ih,stroke:P.text,"stroke-width":1,"stroke-dasharray":"4 3",opacity:.45}));
  });
  ys.forEach((serie,k)=>{
    let d="";serie.forEach((v,i)=>{d+=(i?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1);});
    svg.appendChild(el("path",{d,fill:"none",stroke:EC[k],"stroke-width":1.9,opacity:.9}));
    const hit=el("path",{d,fill:"none",stroke:"transparent","stroke-width":11});
    tipOn(hit,()=>`<b>${S.emotions[k]}</b> at token ${curTok}: ${serie[curTok].toFixed(3)}`+
      `<span class="t-sub">how close the model's state sits to the ${S.emotions[k]} probe. `+
      `Higher means closer. Layer ${curLayer}.</span>`);
    svg.appendChild(hit);
  });
  svg.appendChild(el("line",{x1:X(curTok),x2:X(curTok),y1:T,y2:T+ih,stroke:P.orange,"stroke-width":2}));
  ys.forEach((serie,k)=>svg.appendChild(el("circle",{cx:X(curTok),cy:Y(serie[curTok]),r:4.5,
    fill:EC[k],stroke:"#fff","stroke-width":1.5})));
  S.emotions.forEach((nm,k)=>{
    const t=el("text",{x:L+2,y:T+12+k*14,"font-size":11,fill:EC[k],"font-weight":600});
    t.textContent=nm+"  "+ys[k][curTok].toFixed(3); svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent=`token in the story (turns marked at ${S.boundaries.join(" and ")})`;
  svg.appendChild(xl);
  const yl=el("text",{x:12,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 12 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="closeness to the emotion probe"; svg.appendChild(yl);
  host.appendChild(svg);

  // ternary
  const th=document.getElementById("ternChart"); th.innerHTML="";
  const TW=470,TH=300,cx=TW/2,top=26,side=232,hgt=side*Math.sin(Math.PI/3);
  const s2=el("svg",{viewBox:`0 0 ${TW} ${TH}`,width:"100%"});
  const A=[cx,top], B2=[cx-side/2,top+hgt], Cc=[cx+side/2,top+hgt];
  s2.appendChild(el("polygon",{points:`${A[0]},${A[1]} ${B2[0]},${B2[1]} ${Cc[0]},${Cc[1]}`,
    fill:"#FAFAFA",stroke:P.border}));
  const proj=(a,b,c)=>{const s=a+b+c||1;a/=s;b/=s;c/=s;
    return [a*A[0]+b*B2[0]+c*Cc[0], a*A[1]+b*B2[1]+c*Cc[1]];};
  // the triangle position comes from the same three curves, shifted to be
  // non-negative then normalised, so every story can be shown the same way
  const floor=Math.min(...ys.flat()), shift=v=>v-floor+1e-3;
  const mix=i=>[shift(ys[0][i]),shift(ys[1][i]),shift(ys[2][i])];
  let d="";
  for(let i=0;i<S.n_tokens;i++){const m=mix(i);const p=proj(m[0],m[1],m[2]);
    d+=(i?"L":"M")+p[0].toFixed(1)+","+p[1].toFixed(1);}
  s2.appendChild(el("path",{d,fill:"none",stroke:P.muted,"stroke-width":1.4,opacity:.55}));
  for(let i=0;i<=curTok;i+=3){const m=mix(i);const p=proj(m[0],m[1],m[2]);
    const ph=i<S.boundaries[0]?0:(i<S.boundaries[1]?1:2);
    s2.appendChild(el("circle",{cx:p[0],cy:p[1],r:2.4,fill:EC[ph],opacity:.5}));}
  const mc=mix(curTok); const pc=proj(mc[0],mc[1],mc[2]);
  const phase=curTok<S.boundaries[0]?0:(curTok<S.boundaries[1]?1:2);
  const dot=el("circle",{cx:pc[0],cy:pc[1],r:8,fill:EC[phase],stroke:"#fff","stroke-width":2.5});
  tipOn(dot,()=>{
    const m=mix(curTok), tot=m[0]+m[1]+m[2];
    const parts=S.emotions.map((nm,k)=>`${nm} ${pct(m[k]/tot)}`);
    return `<b>token ${curTok}, written as ${S.emotions[phase]}</b>`+
      `<span class="t-sub">the model's state reads as: ${parts.join(", ")}.`+
      `<br>A corner means it looks purely like that emotion; the middle means undecided.</span>`;
  });
  s2.appendChild(dot);
  const corners=[[A,S.emotions[0],"middle",-10],[B2,S.emotions[1],"end",16],[Cc,S.emotions[2],"start",16]];
  corners.forEach(([p,nm,anc,dy],k)=>{
    const t=el("text",{x:p[0]+(anc==="end"?-6:anc==="start"?6:0),y:p[1]+dy,"text-anchor":anc,
      "font-size":12,fill:EC[k],"font-weight":600}); t.textContent=nm; s2.appendChild(t);
  });
  // what the middle of the triangle means, said on the figure
  const mid=proj(1,1,1);
  s2.appendChild(el("circle",{cx:mid[0],cy:mid[1],r:3,fill:"none",stroke:P.muted,
    "stroke-dasharray":"2 2"}));
  const ml=el("text",{x:mid[0],y:mid[1]+16,"text-anchor":"middle","font-size":9.5,fill:P.muted});
  ml.textContent="centre = undecided"; s2.appendChild(ml);
  const cl2=el("text",{x:cx,y:TH-22,"text-anchor":"middle","font-size":9.5,fill:P.muted});
  cl2.textContent="a corner = reads purely as that emotion"; s2.appendChild(cl2);
  const cap=el("text",{x:cx,y:TH-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  cap.textContent="written to walk "+S.emotions.join(" → "); s2.appendChild(cap);
  th.appendChild(s2);

  document.getElementById("tokLabel").textContent="t = "+curTok+" / "+(S.n_tokens-1);
  paintStory(phase);
}

/* the story itself, split on its own <emotion> markers, dimmed except the live phase */
function storyPhases(raw){
  const out=[];
  const re=/<emotion>([^<]+)<\/emotion>/g; let m, last=null, lastIdx=0;
  while((m=re.exec(raw))!==null){
    if(last!==null) out.push({label:last,text:raw.slice(lastIdx,m.index).trim()});
    last=m[1]; lastIdx=re.lastIndex;
  }
  if(last!==null) out.push({label:last,text:raw.slice(lastIdx).trim()});
  return out;
}
function paintStory(active){
  const box=document.getElementById("storyBox");
  box.innerHTML=storyPhases(S.text).map((p,i)=>
    `<div class="ph${i===active?"":" dim"}">`+
    `<span class="phtag" style="color:${EC[i]}">phase ${i+1} &middot; written as ${p.label}</span>`+
    p.text.replace(/\n+/g,"<br>")+`</div>`).join("");
}
function selectStory(i){
  curStory=i; S=STORIES[i];
  curTok=0;
  const sl=document.getElementById("tokSlider");
  sl.max=S.n_tokens-1; sl.value=0;
  const q=S.qual
    ? `${S.qual.wins} of 3 phases led by the right emotion (per phase: `+
      `${S.qual.perPhase.map(w=>w?"yes":"no").join(", ")}); average margin `+
      `${S.qual.margin>=0?"+":""}${S.qual.margin.toFixed(3)}. One of three by luck.`
    : "the walkthrough story, shown first";
  document.getElementById("storyQual").textContent=q;
  drawStory();
}
(function(){
  const sb=document.getElementById("storyBtns");
  STORIES.forEach((st,i)=>{
    const b=document.createElement("button");
    b.className="seg"+(i===0?" on":""); b.textContent=st.label;
    b.onclick=()=>{sb.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); selectStory(i);};
    sb.appendChild(b);
  });
})();
(function(){
  const lb=document.getElementById("layerBtns");
  Object.keys(STORIES[0].lines_by_layer).forEach(L=>{
    const b=document.createElement("button"); b.className="seg"+(L===curLayer?" on":"");
    b.textContent=L; b.onclick=()=>{curLayer=L;
      lb.querySelectorAll("button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); drawStory();};
    lb.appendChild(b);
  });
  const sl=document.getElementById("tokSlider");
  sl.max=S.n_tokens-1;
  sl.oninput=()=>{curTok=+sl.value; drawStory();};
})();

/* ---------- Finding 4: per emotion, every layer ---------- */
let emoBank="selfgen", emoLayer="33";
function drawEmo(){
  const host=document.getElementById("emoChart"); host.innerHTML="";
  const rows=D.emoByLayer[emoBank][emoLayer];
  const W=880,H=310,L=26,R=26,T=20,B=70;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  const chance=1/12, top=Math.max(0.7,Math.ceil(rows[0].rate*10)/10+0.05), Y=v=>T+ih-(v/top)*ih;
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(chance),y2:Y(chance),stroke:P.alert,
    "stroke-dasharray":"5 4","stroke-width":1.5}));
  const ct=el("text",{x:L+4,y:Y(chance)-6,"text-anchor":"start","font-size":10.5,fill:P.alert});
  ct.textContent="guessing = 8%"; svg.appendChild(ct);
  rows.forEach((r,i)=>{
    const x=L+i*bw, h=Math.max(0,(T+ih)-Y(r.rate));
    const bar=el("rect",{x:x+7,y:r.rate>0?Y(r.rate):T+ih-3,width:bw-14,height:r.rate>0?h:3,
      rx:r.rate>0?3:1,fill:r.rate>=chance?P.navy:(r.rate>0?P.greyMid:P.alert)});
    const wrongHtml=r.wrong.length
      ? r.wrong.map(w=>`${w[0]} ${pct(w[1])}`).join(", ")
      : "no single dominant wrong answer";
    tipOn(bar,`<b>${r.e}</b> at layer ${emoLayer}`+
      `<span class="t-sub">its own probe wins <b>${pct(r.rate)}</b> of ${r.n} story phases`+
      ` (guessing would give 8%).<br>When it is wrong, the model says: ${wrongHtml}.</span>`);
    svg.appendChild(bar);
    const v=el("text",{x:x+bw/2,y:Y(r.rate)-6,"text-anchor":"middle","font-size":10.5,
      fill:r.rate>0?P.text:P.alert});
    v.textContent=r.rate>0?pct(r.rate):"never"; svg.appendChild(v);
    const t=el("text",{x:x+bw/2,y:T+ih+16,"text-anchor":"end","font-size":11,fill:P.body,
      transform:`rotate(-40 ${x+bw/2} ${T+ih+16})`}); t.textContent=r.e; svg.appendChild(t);
  });
  host.appendChild(svg);
  // the standing note restates the verdict for whichever layer and bank is showing
  const best=rows[0], nWin=rows.filter(r=>r.rate>=chance).length, nNever=rows.filter(r=>r.rate===0).length;
  document.getElementById("emoNote").innerHTML =
    `<b>layer ${emoLayer}, ${emoBank==="selfgen"?"Gemma-written":"DeepSeek-written"} probe bank:</b> `+
    `${nWin} of 12 emotions beat guessing, best is ${best.e} at ${pct(best.rate)}`+
    (nNever?`, and ${nNever} never win at all`:"")+". Hover any bar for its wrong answers.";
}
(function(){
  const host=document.getElementById("emoLayerBtns");
  Object.keys(D.emoByLayer.selfgen).forEach(L=>{
    const b=document.createElement("button");
    b.className="seg"+(L===emoLayer?" on":""); b.textContent=L;
    b.onclick=()=>{emoLayer=L;
      host.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); drawEmo();};
    host.appendChild(b);
  });
})();
document.querySelectorAll("[data-bank]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-bank]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); emoBank=b.dataset.bank; drawEmo();
});

/* ---------- Finding 5: layers ---------- */
function drawLayers(){
  const host=document.getElementById("layerChart"); host.innerHTML="";
  const rows=D.byLayer, W=470,H=320,L=44,R=44,T=26,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(rows.length-1))*iw;
  const Y1=v=>T+ih-(v/0.65)*ih, Y2=v=>T+ih-(v/0.30)*ih;
  // anchors: guessing for the naming curve, zero for the anticipation curve
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y1(1/12),y2:Y1(1/12),stroke:P.navy,
    "stroke-dasharray":"4 3",opacity:.55}));
  const a1=el("text",{x:L+2,y:Y1(1/12)-5,"font-size":9.5,fill:P.navy});
  a1.textContent="guessing = 8% of emotions named right"; svg.appendChild(a1);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y2(0),y2:Y2(0),stroke:P.orange,
    "stroke-dasharray":"4 3",opacity:.55}));
  const a2=el("text",{x:W-R-2,y:Y2(0)+13,"text-anchor":"end","font-size":9.5,fill:P.orange});
  a2.textContent="0 = sees no turn coming"; svg.appendChild(a2);
  let d1="",d2="";
  rows.forEach((r,i)=>{d1+=(i?"L":"M")+X(i)+","+Y1(r.top1);d2+=(i?"L":"M")+X(i)+","+Y2(r.r_dval);});
  svg.appendChild(el("path",{d:d1,fill:"none",stroke:P.navy,"stroke-width":2.4}));
  svg.appendChild(el("path",{d:d2,fill:"none",stroke:P.orange,"stroke-width":2.4,"stroke-dasharray":"5 3"}));
  rows.forEach((r,i)=>{
    const c1=el("circle",{cx:X(i),cy:Y1(r.top1),r:7,fill:P.navy});
    tipOn(c1,`<b>layer ${r.layer}: names the right emotion ${pct(r.top1)} of the time</b>`+
      `<span class="t-sub">when it is wrong, the emotion it picks sits ${r.vad.toFixed(2)} away in `+
      `valence-arousal space, against ${r.shuffle.toFixed(2)} for a random emotion: wrong, but nearby.</span>`);
    svg.appendChild(c1);
    const c2=el("circle",{cx:X(i),cy:Y2(r.r_dval),r:7,fill:P.orange});
    tipOn(c2,`<b>layer ${r.layer}: anticipation r = +${r.r_dval.toFixed(3)}</b>`+
      `<span class="t-sub">how well the lead before a turn tracks the size of the coming `+
      `valence jump. 0 would mean the model does not see the turn coming at all.</span>`);
    svg.appendChild(c2);
    const t=el("text",{x:X(i),y:H-24,"text-anchor":"middle","font-size":11,fill:P.text});
    t.textContent="L"+r.layer; svg.appendChild(t);
  });
  const a=el("text",{x:X(0),y:Y1(rows[0].top1)-10,"font-size":10.5,fill:P.navy});
  a.textContent="58% at layer 6"; svg.appendChild(a);
  const b=el("text",{x:X(5),y:Y2(rows[5].r_dval)-10,"text-anchor":"end","font-size":10.5,fill:P.orange});
  b.textContent="+0.26 at layer 51"; svg.appendChild(b);
  const xl=el("text",{x:L+iw/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="layer of the model"; svg.appendChild(xl);
  host.appendChild(svg);
}


/* ---------- Finding 6: probe lineage ---------- */
function drawLineage(){
  const host=document.getElementById("lineageChart"); host.innerHTML="";
  const rows=D.lineage, W=880,H=280,L=30,R=30,T=22,B=76;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length, Y=v=>T+ih-(v/10)*ih;
  [0,5,10].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-6,y:Y(v)+4,"text-anchor":"end","font-size":10,fill:P.muted});
    t.textContent=v; svg.appendChild(t);
  });
  rows.forEach((r,i)=>{
    const x=L+i*bw, h=(T+ih)-Y(r.layers);
    const bar=el("rect",{x:x+16,y:Y(r.layers),width:bw-32,height:h,rx:3,
      fill:r.key==="fixed"?P.navy:(r.key==="weak"?P.greyMid:P.teal)});
    tipOn(bar,`<b>stories by ${r.label}</b>`+
      `<span class="t-sub">${r.layers} of 20 layers pass the registered bar `+
      `(${D.bar} of 12 scenarios correct on both batteries).<br>`+
      `Best probe-to-preference correlation: |r| = ${r.pref.toFixed(3)}.<br>`+
      `Corpus: ${r.n.toLocaleString()} stories`+
      (r.overlap!==null?`, phrase overlap ${r.overlap}`:"")+`.</span>`);
    svg.appendChild(bar);
    const v=el("text",{x:x+bw/2,y:Y(r.layers)-8,"text-anchor":"middle","font-size":15,
      "font-weight":600,fill:P.text}); v.textContent=r.layers; svg.appendChild(v);
    const nm=el("text",{x:x+bw/2,y:T+ih+18,"text-anchor":"middle","font-size":11.5,fill:P.text});
    nm.textContent=r.label; svg.appendChild(nm);
    const sub=el("text",{x:x+bw/2,y:T+ih+33,"text-anchor":"middle","font-size":10.5,fill:P.muted});
    sub.textContent=r.sub; svg.appendChild(sub);
  });
  const yl=el("text",{x:L,y:T-8,"font-size":10.5,fill:P.muted});
  yl.textContent="layers (of 20) where these probes detect emotion above the registered bar";
  svg.appendChild(yl);
  host.appendChild(svg);
}

/* dose-response: how many stories per emotion you actually need */
function drawDose(){
  const host=document.getElementById("doseChart"); host.innerHTML="";
  const ns=Object.keys(D.dose), W=380,H=170,L=34,R=14,T=14,B=34;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(ns.length-1))*iw, Y=v=>T+ih-(v/10)*ih;
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(9),y2:Y(9),stroke:P.green,
    "stroke-dasharray":"4 3"}));
  const cl=el("text",{x:W-R,y:Y(9)-5,"text-anchor":"end","font-size":9.5,fill:P.green});
  cl.textContent="ceiling: 9 layers"; svg.appendChild(cl);
  let d=""; ns.forEach((n,i)=>{d+=(i?"L":"M")+X(i)+","+Y(D.dose[n]);});
  svg.appendChild(el("path",{d,fill:"none",stroke:P.navy,"stroke-width":2.2}));
  ns.forEach((n,i)=>{
    const c=el("circle",{cx:X(i),cy:Y(D.dose[n]),r:6,fill:P.navy});
    tipOn(c,`<b>${n} stories per emotion</b><span class="t-sub">gives ${D.dose[n]} passing layers `+
      `on average across 5 seeded subsamples (ceiling is 9).</span>`);
    svg.appendChild(c);
    const t=el("text",{x:X(i),y:H-16,"text-anchor":"middle","font-size":10,fill:P.muted});
    t.textContent=n; svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-3,"text-anchor":"middle","font-size":10,fill:P.muted});
  xl.textContent="stories per emotion"; svg.appendChild(xl);
  host.appendChild(svg);
}


/* ---------- tabbed math / pseudo-code blocks (site CodeTabs pattern) ---------- */
const PY_KW=new Set(["def","for","in","return","if","else","import","from","as","not","and","or",
  "None","True","False","while","with","lambda","assert","break","continue"]);
function hlPython(code){
  const re=/(#[^\n]*)|("(?:[^"\\]|\\.)*")|(\b\d+\.?\d*\b)|(\b[A-Za-z_]\w*\b)|(\s+)|([^\s\w])/g;
  let out="",m;
  while((m=re.exec(code))!==null){
    const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    if(m[1]) out+=`<span class="k-cm">${esc(m[1])}</span>`;
    else if(m[2]) out+=`<span class="k-str">${esc(m[2])}</span>`;
    else if(m[3]) out+=`<span class="k-num">${m[3]}</span>`;
    else if(m[4]) out+=PY_KW.has(m[4])?`<span class="k-kw">${m[4]}</span>`:esc(m[4]);
    else out+=esc(m[5]||m[6]||"");
  }
  return out;
}
function codeTabs(hostId, label, mathHtml, pyCode, symbols){
  const host=document.getElementById(hostId); if(!host) return;
  host.className="ct";
  const key = symbols && symbols.length
    ? `<details class="symkey"><summary>What every symbol means</summary>`+
      `<table class="symtab">`+
      symbols.map(([sym,mean])=>`<tr><td>${sym}</td><td>${mean}</td></tr>`).join("")+
      `</table></details>`
    : "";
  host.innerHTML=
    `<div class="ct-tabs"><button class="on" data-v="math">Math notation</button>`+
    `<button data-v="py">Pseudo-Python</button><span class="ct-label">${label}</span></div>`+
    `<div class="ct-body"><div class="ct-math mathblock">${mathHtml}${key}</div>`+
    `<pre class="code ct-py" style="display:none">${hlPython(pyCode)}</pre></div>`;
  host.querySelectorAll(".ct-tabs button").forEach(b=>b.onclick=()=>{
    host.querySelectorAll(".ct-tabs button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on");
    host.querySelector(".ct-math").style.display = b.dataset.v==="math"?"":"none";
    host.querySelector(".ct-py").style.display   = b.dataset.v==="py"?"":"none";
  });
}

/* --- 1. emotion vector --- */
codeTabs("m1","emotion_vectors/extraction",
 `<span class="eq">
    <i>v</i><sub>e</sub><span class="op">=</span>
    <span class="frac">
      <span class="num">1</span>
      <span class="den"><span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
        <span class="below"><i>s</i></span></span><i>T</i><sub>s</sub></span>
    </span>
    <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
      <span class="below"><i>s</i>&thinsp;&isin;&thinsp;<i>S</i><sub>e</sub></span></span>
    <i>T</i><sub>s</sub>
    <span class="frac"><span class="num">1</span><span class="den"><i>T</i><sub>s</sub></span></span>
    <span class="bigop"><span class="above"><i>n</i><sub>s</sub></span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;51</span></span>
    <i>h</i><sub><i>t</i></sub><sup>(&ell;)</sup><span class="paren">(</span><i>s</i><span class="paren">)</span>
  </span>
  <span class="where">where <i>n</i><sub>s</sub> is the story's token count after padding is masked and
  <i>T</i><sub>s</sub> <span class="op">=</span> <i>n</i><sub>s</sub> <span class="op">&minus;</span> 50
  is how many tokens survive.</span>
  <span class="gl">The vector for emotion <i>e</i> is a mean of means. <b>Within a story</b> we average
  the residual-stream activation at layer &ell; over every token <i>except the first 50</i>, which the
  source paper treats as narrative framing (padding is masked out too, and stories are capped at 512
  tokens). <b>Across stories</b> we take a token-weighted mean, so a long story counts for more than a
  short one. It is not read at a single token.</span>
  <span class="eq">
    <i>&#7805;</i><sub>e</sub>
    <span class="op">=</span> <i>v</i><sub>e</sub> <span class="op">&minus;</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>E</i></span></span>
    <span class="bigop"><span class="above"><i>E</i></span><span class="glyph">&sum;</span>
      <span class="below"><i>e</i>&prime;&thinsp;=&thinsp;1</span></span>
    <i>v</i><sub><i>e</i>&prime;</sub>
  </span>
  <span class="gl">Then centered: subtract the mean over all <i>E</i> emotions, so what remains is what
  makes this emotion different rather than what all text shares. Every result in this deck uses
  <i>&#7805;</i>, never the raw <i>v</i>.</span>`,
`TOKEN_OFFSET = 50            # the source paper's convention: skip narrative framing
MAX_LENGTH   = 512

def pool_story(hidden, attention_mask):
    # mask padding AND the first 50 tokens, then mean over what survives
    mask = attention_mask.clone()
    mask[:TOKEN_OFFSET] = 0
    return (hidden * mask[:, None]).sum(0) / mask.sum().clamp(min=1)

def emotion_vector(stories, layer):
    # token-weighted across stories: a long story counts for more
    means, counts = [], []
    for s in stories:
        out = forward(s, max_length=MAX_LENGTH)
        means.append(pool_story(out.hidden[layer], out.attention_mask))
        counts.append(out.attention_mask.sum() - TOKEN_OFFSET)
    return sum(m * n for m, n in zip(means, counts)) / sum(counts)

def contrast_vectors(vectors):
    # centering: what is specific to each emotion, not shared by all text
    pool_mean = mean(vectors, axis=0)
    return vectors - pool_mean`,
 [["<i>e</i>","one of the 171 emotion words, for example <i>elated</i>"],
  ["<i>S</i><sub>e</sub>","the set of stories written to evoke emotion <i>e</i>"],
  ["<i>s</i>","one story in that set"],
  ["&ell;","the layer we read from (we captured 20 of them)"],
  ["<i>h</i><sub><i>t</i></sub><sup>(&ell;)</sup>","the model's residual-stream activation at token <i>t</i>, layer &ell;: one vector of 5,376 numbers"],
  ["<i>n</i><sub>s</sub>","how many real (non-padding) tokens story <i>s</i> has"],
  ["<i>T</i><sub>s</sub>","how many of those survive the pooling mask, that is <i>n</i><sub>s</sub> &minus; 50"],
  ["<i>v</i><sub>e</sub>","the raw emotion vector: the token-weighted mean over the whole set"],
  ["<i>&#7805;</i><sub>e</sub>","the <b>centered</b> vector, what we actually use everywhere"],
  ["<i>E</i>","how many emotions are in the pool being centered over (12 or 171)"]]
);

/* --- 2. PCA + the |r| = 0.83 claim --- */
codeTabs("m2","geometry_report/_context, _displacement",
 `<span class="eq">
    <i>X</i> <span class="op">&isin;</span> &#8477;<sup>171&times;<i>d</i></sup>
    <span class="op">,</span> <i>X</i> <span class="op">=</span>
    <i>U</i>&thinsp;<i>&Sigma;</i>&thinsp;<i>W</i><sup>&#8868;</sup>
  </span>
  <span class="gl">Stack the 171 centered emotion vectors as the rows of <i>X</i> and take its singular
  value decomposition. The principal components are the right singular vectors
  <i>w</i><sub>1</sub>&thinsp;&hellip;&thinsp;<i>w</i><sub><i>k</i></sub>, ordered by how much spread
  they explain. No human label enters this step.</span>
  <span class="eq">
    <i>z</i><sub><i>e</i>,<i>k</i></sub> <span class="op">=</span>
    <i>&#7805;</i><sub>e</sub>
    <span class="op">&middot;</span> <i>w</i><sub><i>k</i></sub>
  </span>
  <span class="gl">Each emotion's score on component <i>k</i>: how far along that direction it sits.</span>
  <span class="eq">
    <i>r</i><sub><i>k</i></sub> <span class="op">=</span>
    <span class="frac">
      <span class="num">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i>&thinsp;&isin;&thinsp;<i>M</i></span></span>
        <span class="paren">(</span><i>z</i><sub><i>e</i>,<i>k</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>z</i></span><span class="paren">)</span>
        <span class="paren">(</span><i>a</i><sub><i>e</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>a</i></span><span class="paren">)</span>
      </span>
      <span class="den">
        &radic;<span style="border-top:1.1px solid currentColor;padding:0 .18em">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i></span></span>
        <span class="paren">(</span><i>z</i><sub><i>e</i>,<i>k</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>z</i></span><span class="paren">)</span><sup>2</sup>
        </span>
        &nbsp;&radic;<span style="border-top:1.1px solid currentColor;padding:0 .18em">
        <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
          <span class="below"><i>e</i></span></span>
        <span class="paren">(</span><i>a</i><sub><i>e</i></sub><span class="op">&minus;</span>
        <span class="bar"><i>a</i></span><span class="paren">)</span><sup>2</sup>
        </span>
      </span>
    </span>
  </span>
  <span class="where">where <i>a</i><sub><i>e</i></sub> is the human valence rating of emotion word
  <i>e</i> in the NRC VAD lexicon, and <i>M</i> is the 164 of our 171 words that appear in it.</span>
  <span class="gl">Pearson correlation between the model's component scores and published human
  ratings: two independently produced quantities. We report |<i>r</i>| because the sign of a principal
  component is arbitrary. <b>Base model: |<i>r</i><sub>1</sub>| = 0.83. Instruct model: 0.11 at
  <i>k</i> = 1, and 0.72 at <i>k</i> = 3.</b></span>`,
`def pc_valence_correlation(vectors, layer, vad_lexicon):
    contrasts = contrast_vectors(vectors[:, layer, :])   # 171 x d, centered
    components, variance_ratio = pca(contrasts, n=5)     # SVD, model-only

    scores = contrasts @ components.T                    # 171 x 5
    matched = [e for e in emotions if e in vad_lexicon]  # 164 of 171
    human_valence = [vad_lexicon[e].valence for e in matched]

    out = []
    for k in range(5):
        model_scores = [scores[index_of(e), k] for e in matched]
        # sign of a principal component is arbitrary -> absolute value
        out.append(abs(pearson_r(model_scores, human_valence)))
    return out, variance_ratio`,
 [["<i>X</i>","the 171 centered emotion vectors stacked as rows"],
  ["<i>d</i>","the model's hidden width, 5,376 for Gemma 4 31B"],
  ["<i>U</i>, <i>&Sigma;</i>, <i>W</i>","the three factors of the singular value decomposition; the columns of <i>W</i> are the principal components"],
  ["<i>w</i><sub><i>k</i></sub>","the <i>k</i>-th principal component: a direction in activation space"],
  ["<i>k</i>","which component we mean; <i>k</i> = 1 is the largest"],
  ["<i>z</i><sub><i>e</i>,<i>k</i></sub>","emotion <i>e</i>'s score on component <i>k</i>: how far along that direction it sits"],
  ["<span class='bar'><i>z</i></span>","the mean score across the matched emotions"],
  ["<i>a</i><sub><i>e</i></sub>","the <b>human</b> valence rating of word <i>e</i> from the NRC VAD lexicon"],
  ["<span class='bar'><i>a</i></span>","the mean human rating across those same words"],
  ["<i>M</i>","the 164 of our 171 emotion words that appear in the lexicon"],
  ["<i>r</i><sub><i>k</i></sub>","Pearson correlation between the model's scores and the human ratings"]]
);

/* --- 3. tracking: gate rank and anticipation lead --- */
codeTabs("m3","q3_conventions.py, score_q3_gate_r1.py",
 `<span class="eq">
    <i>c</i><sub><i>t</i>,<i>p</i></sub> <span class="op">=</span>
    <span class="frac">
      <span class="num"><i>h</i><sub><i>t</i></sub> <span class="op">&middot;</span> <i>p</i>
        <span class="op">&minus;</span> <i>&mu;</i><sub><i>p</i></sub></span>
      <span class="den">&#8214; <i>h</i><sub><i>t</i></sub> <span class="op">&minus;</span>
        <i>&mu;</i> &#8214;</span>
    </span>
  </span>
  <span class="where">with <i>&mu;</i><sub><i>p</i></sub> the token-weighted mean of probe <i>p</i>'s
  dot products over the whole story set.</span>
  <span class="gl">The centered cosine between token <i>t</i> and probe <i>p</i>. Centering on the
  <b>story-set</b> mean is what stops a probe that is simply large everywhere from winning by
  default.</span>
  <span class="eq">
    rank<sub><i>&phi;</i></sub> <span class="op">=</span> 1 <span class="op">+</span>
    <span class="paren">|</span>{ <i>p</i> <span class="op">:</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>
    <span class="op">&gt;</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>e</i>(<i>&phi;</i>)</sub>
    }<span class="paren">|</span>
    <span class="op">,</span>
    <span class="bar"><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>
    <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den">|<i>&phi;</i>|</span></span>
    <span class="bigop"><span class="above"></span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;&isin;&thinsp;<i>&phi;</i></span></span>
    <i>c</i><sub><i>t</i>,<i>p</i></sub>
  </span>
  <span class="gl"><b>The gate.</b> Here the model reads the story once and we keep <b>every token</b>:
  the cosine is computed at each position, then averaged over the tokens belonging to phase
  <i>&phi;</i> (phases shorter than 4 tokens are skipped as too short to average). Then count how many
  probes beat the tagged one. Rank 1 is perfect; with a bank of 12, guessing gives 6.5.</span>
  <span class="eq">
    lead<sub><i>b</i></sub> <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>W</i></span></span>
    <span class="bigop"><span class="above"><i>b</i>&minus;1</span><span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;<i>b</i>&minus;<i>W</i></span></span>
    <i>c</i><sub><i>t</i>,<i>q</i></sub> <span class="op">&minus;</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>W</i></span></span>
    <span class="bigop"><span class="above"><i>b</i>&minus;<i>W</i>&minus;1</span>
      <span class="glyph">&sum;</span>
      <span class="below"><i>t</i>&thinsp;=&thinsp;<i>b</i>&minus;2<i>W</i></span></span>
    <i>c</i><sub><i>t</i>,<i>q</i></sub>
  </span>
  <span class="where">at a written turn <i>b</i>, for the incoming emotion's probe <i>q</i>, with
  <i>W</i> = 16 tokens.</span>
  <span class="gl"><b>Anticipation.</b> The mean cosine in the 16 tokens before the turn minus the mean
  in the 16 before those. Above zero means the next emotion is already rising before the story turns.
  Both windows are referenced to the boundary, so story length cannot drive the effect.</span>`,
`W = 16                       # window, registered before scoring
LAYERS = [6, 15, 24, 33, 42, 51]
MIN_PHASE_TOKENS = 4         # shorter phases are skipped, not averaged

# NOTE the difference from probe extraction: there we pooled ONE vector per
# story (masked mean over tokens 50..end). Here we keep the per-token series,
# because the question is how the reading MOVES through the story.

def centered_cos(shard, story_set_mean):
    # story_set_mean is token-weighted over the whole corpus, not per story
    return (shard["dots"] - story_set_mean) / shard["norms_centered"][:, :, None]

def gate_rank(cos, phase, tagged_probe, bank):
    phase_mean = cos[phase.start:phase.end, :, bank].mean(axis=0)
    beaten_by = (phase_mean > phase_mean[tagged_probe]).sum()
    return beaten_by + 1          # 1 = the tagged probe wins outright

def anticipation_lead(cos, boundary, incoming_probe):
    near    = cos[boundary - W:boundary,       :, incoming_probe].mean(axis=0)
    earlier = cos[boundary - 2*W:boundary - W, :, incoming_probe].mean(axis=0)
    return near - earlier          # > 0 = the next emotion is already rising`,
 [["<i>t</i>","a token position in the story"],
  ["<i>p</i>","one probe, that is one emotion's centered vector"],
  ["<i>h</i><sub><i>t</i></sub>","the model's activation while reading token <i>t</i>"],
  ["<i>&mu;</i><sub><i>p</i></sub>","probe <i>p</i>'s mean dot product across the whole story set (the centering term)"],
  ["&#8214;&thinsp;&middot;&thinsp;&#8214;","vector length, so dividing by it turns a dot product into a cosine"],
  ["<i>c</i><sub><i>t</i>,<i>p</i></sub>","the centered cosine: how close token <i>t</i> reads to probe <i>p</i>"],
  ["<i>&phi;</i>","one phase of the story, that is one tagged emotion's stretch of tokens"],
  ["|<i>&phi;</i>|","how many tokens that phase has (fewer than 4 and we skip it)"],
  ["<i>e</i>(<i>&phi;</i>)","the emotion phase <i>&phi;</i> was written to express"],
  ["<span class='bar'><i>c</i></span><sub><i>&phi;</i>,<i>p</i></sub>","probe <i>p</i>'s average cosine over that phase's tokens"],
  ["rank<sub><i>&phi;</i></sub>","where the tagged emotion's probe places among the bank; 1 is best"],
  ["<i>b</i>","the token index where the story is written to turn"],
  ["<i>W</i>","the window length, fixed at 16 tokens before scoring"],
  ["<i>q</i>","the probe for the <b>incoming</b> emotion, the one after the turn"],
  ["lead<sub><i>b</i></sub>","how much the incoming emotion rises just before the turn"]]
);

/* --- 4. the nulls --- */
codeTabs("m4","score_q3_gate_r1.py (N1, N2)",
 `<span class="eq">
    <i>p</i> <span class="op">=</span>
    <span class="frac"><span class="num">1</span><span class="den"><i>B</i></span></span>
    <span class="bigop"><span class="above"><i>B</i></span><span class="glyph">&sum;</span>
      <span class="below"><i>b</i>&thinsp;=&thinsp;1</span></span>
    <b>1</b><span class="paren">[</span>
    median<span class="paren">(</span>rank<sup>(<i>b</i>)</sup><span class="paren">)</span>
    <span class="op">&le;</span>
    median<span class="paren">(</span>rank<sup>obs</sup><span class="paren">)</span>
    <span class="paren">]</span>
  </span>
  <span class="where">over <i>B</i> = 10,000 shuffles in which every phase is re-scored against a
  randomly assigned <b>wrong</b> emotion.</span>
  <span class="gl"><b>N2, the wrong-emotion shuffle.</b> How often does chance beat what we observed?
  This shuffle is the floor, not our intuition about what a good rank looks like.</span>
  <span class="eq">
    <i>&#7805;</i><sub>rand</sub>
    <span class="op">&sim;</span> span
    <span class="paren">{</span>
    <i>&#7805;</i><sub>e</sub>
    <span class="paren">}</span>
    <span class="op">,</span>
    lead <span class="op">&ge;</span>
    <span class="frac"><span class="num">1</span><span class="den">2</span></span>
    <i>&sigma;</i><sub>noise</sub><span class="paren">(</span>&ell;<span class="paren">)</span>
  </span>
  <span class="gl"><b>N1, the random-direction control.</b> Directions drawn from the span of the probe
  set, used to calibrate a per-layer noise scale. An effect must clear half that noise standard
  deviation, so a tiny but consistent drift cannot pass on a <i>p</i>-value alone. A claim graduates
  only when <b>both</b> conditions hold.</span>`,
`B = 10_000                    # shuffles, registered

def wrong_emotion_null(cos, phases, bank, rng):
    # rank EVERY probe once, then index the shuffled picks: 10k shuffles
    # without 10k re-sorts
    order = argsort(-phase_means, axis=1)
    rank_of_probe = argsort(order, axis=1) + 1

    null_medians = []
    for _ in range(B):
        wrong = rng.choice([p for p in bank if p != tagged], size=len(phases))
        null_medians.append(median(take_along(rank_of_probe, wrong)))
    return mean(null_medians <= observed_median)      # the p-value

# a claim graduates only if BOTH hold:
#   p < 0.001  and  effect >= 0.5 * calibrated_noise_sd[layer]`,
 [["<i>B</i>","how many shuffles we run, fixed at 10,000"],
  ["<i>b</i>","one of those shuffles"],
  ["rank<sup>(<i>b</i>)</sup>","the ranks obtained in shuffle <i>b</i>, where every phase was scored against a randomly picked <b>wrong</b> emotion"],
  ["rank<sup>obs</sup>","the ranks we actually observed"],
  ["<b>1</b>[&thinsp;&middot;&thinsp;]","the indicator: 1 when the statement inside is true, 0 otherwise"],
  ["<i>p</i>","the resulting p-value: the share of shuffles that did at least as well as we did"],
  ["<i>&#7805;</i><sub>rand</sub>","a random direction drawn from the span of the real probes"],
  ["<i>&sigma;</i><sub>noise</sub>(&ell;)","the spread those random directions produce at layer &ell;: our noise scale"]]
);

/* ---------- section index on the cover ---------- */
const SECTIONS=[...document.querySelectorAll("section[id]")].map((sec,i)=>({
  id:sec.id, n:i+1,
  title:sec.querySelector("h2").textContent.replace(/^\s*\d+\s*/,"").trim()}));
document.getElementById("coverIndex").innerHTML = SECTIONS.map(x=>
  `<a href="#${x.id}"><span class="n">${x.n}</span><span class="t">${x.title}</span></a>`).join("");

/* ---------- keyboard: next / previous / jump ---------- */
/* Track the section explicitly rather than re-deriving it from scrollY: a
   smooth scroll is still animating when the next key arrives, so deriving
   would make two quick presses land on the same section. Manual scrolling
   re-syncs the index. */
let navIdx = -1, navAnimating = false, scrollEndTimer = null;
function goToSection(i){
  navIdx = Math.max(0, Math.min(SECTIONS.length-1, i));
  navAnimating = true;   // cleared when scrolling actually stops, not on a guessed timeout
  document.getElementById(SECTIONS[navIdx].id).scrollIntoView({behavior:"smooth",block:"start"});
}
function sectionFromScroll(){
  const y=scrollY+140; let idx=-1;
  SECTIONS.forEach((x,i)=>{if(document.getElementById(x.id).offsetTop<=y) idx=i;});
  return idx;
}
// Sync the index only once scrolling has settled. Syncing during a smooth
// scroll would overwrite the target the user just asked for with wherever the
// animation happens to be, which made two fast presses land on one section.
addEventListener("scroll",()=>{
  clearTimeout(scrollEndTimer);
  scrollEndTimer = setTimeout(()=>{
    if(navAnimating) navAnimating = false;   // our own scroll finished; keep navIdx
    else navIdx = sectionFromScroll();       // the user scrolled by hand
  }, 130);
},{passive:true});
addEventListener("keydown",e=>{
  // never hijack typing, and leave modified keys to the browser
  const tag=(e.target.tagName||"").toLowerCase();
  if(tag==="input"||tag==="textarea"||e.metaKey||e.ctrlKey||e.altKey) return;
  if(e.key==="ArrowRight"||e.key===" "||e.key==="PageDown"){e.preventDefault();goToSection(navIdx+1);}
  else if(e.key==="ArrowLeft"||e.key==="PageUp"){e.preventDefault();goToSection(navIdx-1);}
  else if(e.key==="Home"){e.preventDefault();navIdx=-1;scrollTo({top:0,behavior:"smooth"});}
  else if(/^[0-9]$/.test(e.key)){
    // 1-9 are sections 1-9; 0 is section 10, the Q&A
    const n=e.key==="0"?10:parseInt(e.key,10);
    if(n<=SECTIONS.length){e.preventDefault();goToSection(n-1);}
  }
});

/* ---------- cross-layer RSA matrices ---------- */
const RSA_KEYS=Object.keys(D.rsa);
let rsaKey=RSA_KEYS[0];
function drawRsa(){
  const host=document.getElementById("rsaChart"); if(!host) return;
  host.innerHTML="";
  const m=D.rsa[rsaKey], z=m.z, layers=m.layers||z.map((_,i)=>i);
  const n=z.length, W=760,H=430,L=52,T=14,B=52,Rr=150;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const size=Math.min((W-L-Rr)/n,(H-T-B)/n);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const v=z[i][j];
    const cell=el("rect",{x:L+j*size,y:T+i*size,width:size+.5,height:size+.5,
      fill:`rgba(29,53,87,${Math.max(0,Math.min(1,v)).toFixed(3)})`});
    tipOn(cell,`<b>layer ${layers[i]} vs layer ${layers[j]}</b>: agreement ${v.toFixed(2)}`+
      `<span class="t-sub">1 means these two layers organise the 171 emotions the same way; `+
      `0 means they disagree completely.</span>`);
    svg.appendChild(cell);
  }
  [0,Math.floor(n/2),n-1].forEach(i=>{
    const a=el("text",{x:L-6,y:T+i*size+size/2+4,"text-anchor":"end","font-size":10,fill:P.muted});
    a.textContent="L"+layers[i]; svg.appendChild(a);
    const b=el("text",{x:L+i*size+size/2,y:T+n*size+16,"text-anchor":"middle","font-size":10,fill:P.muted});
    b.textContent="L"+layers[i]; svg.appendChild(b);
  });
  // the scale, with both ends named
  const sx=L+n*size+26, sy=T, sh=n*size;
  for(let i=0;i<40;i++)
    svg.appendChild(el("rect",{x:sx,y:sy+sh-(i+1)*(sh/40),width:13,height:sh/40+.5,
      fill:`rgba(29,53,87,${((i/39)).toFixed(3)})`}));
  [["1.0  same shape of emotion space",sy+6],
   ["0.5  partly agree",sy+sh/2],
   ["0.0  completely different",sy+sh]].forEach(([t,y])=>{
    const tx=el("text",{x:sx+18,y:y+4,"font-size":10,fill:P.muted}); tx.textContent=t; svg.appendChild(tx);
  });
  const xl=el("text",{x:L+(n*size)/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="layer of the model"; svg.appendChild(xl);
  host.appendChild(svg);
  document.getElementById("rsaNote").textContent=rsaKey;
}
(function(){
  const host=document.getElementById("rsaBtns"); if(!host) return;
  const short={"instruct RSA (unablated)":"instruct, as measured",
    "instruct RSA (top component removed)":"instruct, top component removed",
    "cross-model RSA: instruct vs base":"instruct vs base"};
  RSA_KEYS.forEach((k,i)=>{
    const b=document.createElement("button");
    b.className="seg"+(i===0?" on":""); b.textContent=short[k]||k;
    b.onclick=()=>{host.querySelectorAll("button").forEach(x=>x.classList.remove("on"));
      b.classList.add("on"); rsaKey=k; drawRsa();};
    host.appendChild(b);
  });
})();

/* ---------- progress: the nav is the tracker, no extra surface ---------- */
const NAV_LINKS=[...document.querySelectorAll("nav .wrap a")];
const POS=document.createElement("span");
POS.className="pos"; document.querySelector("nav .wrap").appendChild(POS);
function paintProgress(){
  const cur=sectionFromScroll();
  NAV_LINKS.forEach((a,i)=>{
    a.classList.toggle("on", i===cur);
    a.classList.toggle("done", i<cur);
  });
  POS.textContent = cur<0 ? "" : `${cur+1} / ${SECTIONS.length}`;
  const max=document.body.scrollHeight-innerHeight;
  document.getElementById("bar").style.width=(max>0?(scrollY/max)*100:0)+"%";
}
addEventListener("scroll",paintProgress,{passive:true});
paintProgress();

/* ---------- expand any figure ---------- */
const MODAL=document.getElementById("modal"), MODAL_BODY=document.getElementById("modalBody");
function closeModal(){MODAL.classList.remove("on");MODAL_BODY.innerHTML="";}
document.getElementById("modalClose").onclick=closeModal;
MODAL.onclick=e=>{if(e.target===MODAL) closeModal();};
addEventListener("keydown",e=>{if(e.key==="Escape") closeModal();});
// wrap every chart host so it gets an expand button; the modal shows a live clone
["pcChart","gridChart","lineChart","ternChart","emoChart","layerChart","lineageChart","doseChart",
 "rsaChart"].forEach(id=>{
  const host=document.getElementById(id); if(!host) return;
  const parent=host.parentElement;
  if(!parent.classList.contains("figwrap")) parent.classList.add("figwrap");
  if(parent.querySelector(".expand")) return;
  const b=document.createElement("button");
  b.className="expand"; b.textContent="expand";
  b.onclick=()=>{
    const svg=host.querySelector("svg"); if(!svg) return;
    MODAL_BODY.innerHTML="";
    MODAL_BODY.appendChild(svg.cloneNode(true));
    MODAL.classList.add("on");
  };
  parent.appendChild(b);
});

drawPCs("base"); drawGrid(); drawStory(); drawEmo(); drawLayers(); drawLineage(); drawDose(); drawRsa();
</script>
</body></html>"""

if __name__ == "__main__":
    out = HERE / "index.html"
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
