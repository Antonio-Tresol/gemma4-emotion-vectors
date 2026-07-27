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
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--display);font-size:12.5px;color:var(--muted);
margin-top:10px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.howto{margin-top:14px;border-top:1px dashed var(--border);padding-top:12px}
.howto summary{font-family:var(--display);font-size:13px;font-weight:600;color:var(--muted);cursor:pointer}
.howto[open] summary{color:var(--text);margin-bottom:8px}
.howto p{font-size:14.5px;color:var(--body)}
.callout{border-left:3px solid var(--orange);padding:14px 0 14px 18px;background:transparent;margin:18px 0}
/* the three parts of the talk, each announced by a full-width divider */
.partbar{border-top:2px solid var(--navy);margin-top:64px;padding-top:20px}
/* the deck sets a part off with a small letter-spaced caps label above the
   slide title, so the part reads as a SUPERheading and the section h2 below it
   stays the largest thing on screen. Same hierarchy here. */
.partbar .ptitle{font-family:var(--display);font-size:18px;font-weight:650;color:var(--navy);
  letter-spacing:.11em;text-transform:uppercase}
.partbar .pblurb{max-width:62ch;margin-top:12px;color:var(--muted);font-size:16px}
@media(max-width:700px){.partbar .ptitle{font-size:15px}}
.callout .k{font-family:var(--display);font-weight:650;color:var(--text);display:block;margin-bottom:4px}
/* Chart text is set in the UI sans, not the mono. Monospace advance widths are
   built for aligning columns of code; at 10px they make axis titles sprawl and
   wrap, which is most of why these figures read as unlabelled. */
svg text{font-family:var(--display)}
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
    story = json.loads(STORY_JSON.read_text(encoding="utf-8"))
    payload = {
        "story": story,
        "storyText": json.loads(STORY_TEXT_JSON.read_text(encoding="utf-8")),
        "emoByLayer": json.loads(EMO_BY_LAYER_JSON.read_text(encoding="utf-8")),
        "three": json.loads(THREE_STORIES_JSON.read_text(encoding="utf-8")),
        "rsa": json.loads(RSA_JSON.read_text(encoding="utf-8")),
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
<!-- Unreviewed sprint results, shared by link and deliberately kept out of search.
     This lives here rather than being hand-added to the published page: it was,
     once, and the next rebuild silently dropped it. -->
<meta name="robots" content="noindex, nofollow, noarchive">
<title>How are emotions represented in large language models?</title>
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
  <a href="#question">Why it matters</a><a href="#probes">Probe quality</a>
  <a href="#replication">Circumplex</a><a href="#displaced">What moved</a>
  <a href="#story">One story</a><a href="#emotions">Per emotion</a><a href="#layers">Layers</a><a href="#next">Next</a><a href="#methods">Methods</a><a href="#appendix">Appendix</a>
</div></nav>

<header><div class="wrap">
  <div class="kicker">CAMBRIA capstone &middot; Hannah Kim &middot; Peyton Li &middot; Antonio Badilla Olivas</div>
  <h1>How are emotions represented in<br>large language models?</h1>
  <p class="lede narrow">Anthropic found that a language model keeps a separate internal direction for
  each emotion, and that those directions arrange themselves the way psychologists arrange emotions.
  We rebuilt that result on Gemma&nbsp;4&nbsp;31B (a 31-billion-parameter model), in two versions of
  it. Then we asked a question the paper did not: can the model follow an emotion that
  <em>changes</em> partway through a story? Three findings, below.</p>
  <div class="grid3" style="margin-top:34px;gap:16px">
    <div class="card"><h3>The plain model reproduces the result</h3>
    <p style="font-size:15px">Sort 171 emotions by the model's internal readings alone and the biggest thing telling them apart
    is pleasant-versus-unpleasant &mdash; the same axis people use. Nobody asked for that. It is the
    published result, reproduced on a new model.</p></div>
    <div class="card"><h3>The chat-tuned version buries it</h3>
    <p style="font-size:15px">Train that same model to follow instructions and pleasantness drops
    from first place to third &mdash; still present, but no longer how the model mainly sorts
    emotions. A larger axis takes over, and it matches nothing in the plain model. We can measure
    it; we cannot yet say what it means.</p></div>
    <div class="card"><h3>It follows a story's emotion, roughly</h3>
    <p style="font-size:15px">In stories written to move through three emotions, the model's internal
    state hands over from one emotion to the next &mdash; better than guessing, often before the
    written turn, but never reliably, and much better for some emotions than others.</p></div>
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
  <h2><span class="secno">1</span>Why does this matter?</h2>
  <h3 style="margin-top:26px">Five words this page uses constantly</h3>
  <p class="narrow" style="margin-bottom:18px">Everything that follows is built out of these five
  ideas, so they are worth a minute up front.</p>
  <div class="grid3">
    <div class="card"><h3>An internal reading</h3><p style="font-size:15px">While the model reads text,
    we can pause it and write down the numbers it is holding at that moment &mdash; thousands of them.
    That list is the model's <b>internal reading</b>, and it is the only thing measured anywhere on
    this page. Nothing here comes from asking the model a question or reading what it writes. Two readings can be
    compared: readings taken while reading similar things point in similar directions.</p></div>
    <div class="card"><h3>A layer</h3><p style="font-size:15px">The model does not read text in one
    go. It passes the text through a stack of stages, one after another, and we can take an internal
    reading after any of them. Each stage is a <b>layer</b>. This model's stack runs to at least layer
    57, and the experiments here sample it rather than test every stage: twenty layers evenly spaced
    across the stack for most of the work, six of them for the story work. It matters because the answer often changes with depth,
    which is why nearly every number here comes with a layer attached.</p></div>
    <div class="card"><h3>An emotion vector</h3><p style="font-size:15px">Show the model short stories
    written to feel <em>afraid</em> and take an internal reading of each. Average those readings and
    you get one list of numbers &mdash; a direction &mdash; that stands for "afraid" inside the
    model. Do it for every emotion word and you have a set of directions to measure new text against.
    How many stories go into one vector matters: the 171-emotion set in part one averages only about
    <b>nine</b> stories per emotion, which is thin enough that we treat those directions as
    noise-limited; the twelve-emotion sets used from part three onwards average <b>256</b>.</p></div>
    <div class="card"><h3>The circumplex</h3><p style="font-size:15px">Ask people to rate hundreds of
    emotion words and the ratings collapse onto two axes: how <em>pleasant</em> the feeling is, and
    how <em>worked-up</em> it is. Furious and elated are both high-energy; content and miserable are
    both low. Psychologists have called this arrangement the <b>circumplex</b> for decades. The
    published result we set out to reproduce is that a model's emotion directions lay themselves out
    the same way, unprompted.</p></div>
    <div class="card"><h3>Plain vs. chat-tuned</h3><p style="font-size:15px">Two versions of one model.
    The <b>plain</b> one was only ever trained to predict the next word. The <b>chat-tuned</b> one is
    that same model after further training to follow instructions and behave like an assistant.
    Comparing them shows what that further training does to the emotion structure &mdash; the
    comparison the original paper could not run.</p></div>
  </div>
  <h3 style="margin-top:38px">Why should anyone care whether this works?</h3>
  <p class="narrow" style="margin-bottom:20px">Five reasons, and they pull in different directions.
  Four of them want the emotion signal to be strong. The last one is interesting precisely if it
  turns out to be weak.</p>
  <div class="grid3" style="gap:16px">
    <div class="card"><h3>Something you can read and push on</h3><p style="font-size:15px">If a
    direction really stands for an emotion, you can do two things with it: read it off while the model
    works, and add it back in to change how the model behaves. One object gives you both the
    "what is it representing?" question and the "what happens if we push?" question.</p></div>
    <div class="card"><h3>Models as listeners</h3><p style="font-size:15px">People already bring models
    their worst days, and companies are selling them as companions and therapists. Doing that well is
    not a matter of labelling one message as positive or negative. It means following how someone's
    state <em>moves</em> across a long conversation. Precisely what we tested, and precisely where it gets shaky.</p></div>
    <div class="card"><h3>Internal evidence, for the welfare question</h3><p style="font-size:15px">
    Arguments about model welfare mostly run on what a model <em>says</em>. Emotion vectors are one of
    the few handles on internal state instead. A caveat we take seriously: we measured the model
    reading emotions in a <em>story</em>, not having them. Those are different claims, and conflating
    them is the easiest mistake in this area.</p></div>
    <div class="card"><h3>Where the model's concepts differ from ours</h3><p style="font-size:15px">The
    emotion layout we recover belongs to the model, not to us. Wherever it disagrees with how humans
    rate the same words, that gap tells us something the model does not share with us &mdash; and the
    chat-tuning step visibly redraws it.</p></div>
    <div class="card" style="border-color:var(--orange)"><h3>Or: it may just not be very emotional</h3>
    <p style="font-size:15px">The possibility we cannot rule out, and the one we find most interesting.
    Maybe emotion is simply not one of the big organising ideas in this model, and we keep measuring a
    weak signal because a weak signal is all there is. That would be a real answer about what these
    systems represent, not a failed experiment.</p></div>
  </div>
</div></section>

<div class="partbar"><div class="wrap"><span class="ptitle">Part one &middot; Rebuilding the emotion vectors</span><p class="pblurb">Does the published result hold up on a different model? Before that question can be answered, a prior one: an emotion vector is only as good as the stories used to build it, and it turns out that matters a great deal.</p></div></div>
<section id="probes"><div class="wrap">
  <h2><span class="secno">2</span>Who writes the stories changes the answer</h2>
  <p class="narrow">To build an emotion vector you need stories written to feel like that emotion &mdash;
  so somebody has to write them. Nobody reports who. We built four sets of emotion vectors from four
  different story sources &mdash; three writers, with
  the strongest of them appearing twice under different prompting &mdash; and put all four through
  the same test.</p>
  <p class="narrow">The test: give the model a piece of text whose emotion we know, rank all twelve
  emotion vectors by how well each matches, and check whether the correct one lands in the top three.
  A layer of the model counts as <b>working</b> if it does that for at least 8 of 12 pieces of test
  text, on two separate sets. Guessing would put the right answer in the top three about a quarter of
  the time, so the bar is 8 of 12 against a chance rate of 3 of 12. We fixed that mark in advance,
  before looking at any results, so we could not move it to suit the answer.</p>
  <div class="card" style="margin-top:22px">
    <div id="lineageChart"></div>
    <div class="legend"><span><i style="background:var(--navy)"></i>each bar shows how many of the 20
    tested layers reach the 8-of-12 mark. Taller is better; 0 would mean the vectors never
    work.</span></div>
    <details class="howto"><summary>How to read this</summary>
      <p><b>Four story sources, one identical test.</b> Three different writers, plus a second attempt
      by the strongest of them under a different prompt recipe. The corpus sizes differ too, from 1,539
      stories to 12,262, so this compares sources as a whole rather than isolating one variable. Bar
      height is how many of the 20 tested layers give emotion vectors that work, by the standard above.
      <b>A bad result is 0</b> &mdash; the vectors never detect anything. <b>A good result is 20</b>,
      every layer working. Nothing we built came close: the best writer reached 9.</p>
      <p><b>The ordering is the finding.</b> The strongest outside writer here is DeepSeek, a different
      company's language model. Its stories produced vectors that work at nine layers. The model's own writing produced five. A smaller model's writing
      produced one. So the intuitive guess &mdash; that a model understands its own emotional writing
      best &mdash; is wrong here. It is out-written.</p>
    </details>
    <div class="src">source: notebooks/07_generator_lineages.ipynb (experiments E11 and E12)</div>
  </div>
  <div class="grid2" style="margin-top:20px">
    <div class="card"><h3>Why its own stories are worse</h3>
      <p style="font-size:15px">Asked for 3,072 stories about emotions, the model named its character
      <b>Elias</b> in <b>98.2%</b> of them. Its own writing reuses five-word phrases <b>53&times;</b>
      as often as DeepSeek's does. That matters because narrow, repetitive stories make a narrow
      vector: whatever the stories have in common gets averaged in, so the vector ends up standing for
      the model's writing habits as much as for the emotion.</p>
      <p style="font-size:15px" class="muted">It is also the sharpest thing we saw about what is
      peculiar to this particular model. Ask it for variety and it gives you Elias, 3,000 times.</p></div>
    <div class="card"><h3>How many stories do you actually need?</h3>
      <div id="doseChart"></div>
      <p style="font-size:15px;margin-top:8px">About <b>64</b> per emotion. At that point the vectors
      already reach 8.6 working layers out of the 9 this writer ever achieves &mdash; a fraction
      because it is an average over five random draws of that many stories &mdash; and quadrupling
      the stories buys almost nothing.</p></div>
  </div>
  <div class="takeaway"><span class="lbl">Takeaway</span>An emotion vector is only as good as the writer of the stories behind it &mdash; a nine-fold difference here, from the same model and the same test. The model is out-written by a stronger outside author, and its own writing is too repetitive to build a broad vector from.</div>
</div></section>

<section id="replication"><div class="wrap">
  <div class="kicker">Part one &middot; Rebuilding the emotion vectors</div>
  <h2><span class="secno">3</span>Pleasantness leads the plain model, and loses first place in the chat-tuned one</h2>
  <p class="narrow">We have 171 emotion vectors, one per emotion word, read at layer 33 of the model.
  Ask what separates them most and
  you get a ranked list of axes. The first axis is whatever splits the 171 emotions apart the most.
  The second is the largest split left over once the first is accounted for, and so on down the list. This is
  <b>principal component analysis</b>; the axes are called <b>PC1, PC2</b> and so on, and they come
  from the model's internal readings alone, with no human input.</p>
  <p class="narrow">The question is then whether those model-found axes mean anything to us. So we take
  each axis and check how closely it lines up with published human ratings of the same 171 words.
  People rated three things: how <b>pleasant</b> the emotion is (psychologists call this
  <em>valence</em>), how <b>worked-up</b> (<em>arousal</em>), and how <b>in control</b> the person feels
  (<em>dominance</em>). We also check it
  against one thing that has nothing to do with emotion but could fake the whole result: <b>how long the
  stories were</b>. A perfect match scores 1.0; no relationship scores 0.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls">
      <button class="seg on" data-model="base">plain model</button>
      <button class="seg" data-model="instruct">chat-tuned model</button>
      <span class="muted" style="margin-left:auto;font-size:13px" id="pcVerdict"></span>
    </div>
    <div id="pcChart"></div>
    <div class="legend">
      <span><i style="background:var(--navy)"></i>pleasantness (valence)</span>
      <span><i style="background:var(--teal)"></i>energy (arousal)</span>
      <span><i style="background:var(--green)"></i>control (dominance)</span>
      <span><i style="background:var(--amber)"></i>story length &mdash; the fake-result check</span>
      <span><i style="background:#D4D4D4"></i>grey backdrop: how much of the total spread this axis
      accounts for</span>
    </div>
    <details class="howto"><summary>How to read this</summary>
      <p><b>What a success looks like:</b> the plain model. Its biggest axis matches human pleasantness
      ratings at 0.83, and its second matches energy at 0.55. Those are the two axes of the circumplex,
      in order, found by a method that never saw a human rating. That is the published result,
      reproduced.</p>
      <p><b>What a failure looks like:</b> the chat-tuned model. Its biggest axis matches pleasantness
      at 0.11 &mdash; near zero. The 171 emotions differ from one another in many ways at once, and
      this single axis accounts for 28% of all of that variation, against 15% for the plain model's
      top axis. So it is not merely a big axis, it is nearly twice as dominant &mdash; and it is not
      pleasantness.</p>
      <p>Pleasantness has not vanished, it has moved down: the chat-tuned model's <em>third</em> axis
      matches it at 0.72. And watch the amber bar. The chat-tuned model's second axis matches story
      length at 0.66, harder than it matches any emotion rating &mdash; a standing reminder that a big
      axis is not automatically an emotional one.</p>
    </details>
    <div class="src">source: notebooks/02_circumplex_geometry.ipynb, layer 33, measured after the centering
    correction described in Method block 4</div>
  </div>
  <div class="takeaway"><span class="lbl">Takeaway</span>Sort the plain model's emotions by what separates them most and the answer is pleasantness &mdash; the human circumplex, recovered without being asked for. Do the same on the chat-tuned model and the top axis is something else entirely, with pleasantness pushed down to third.</div>
</div></section>

<div class="partbar"><div class="wrap"><span class="ptitle">Part two &middot; Plain model vs. chat-tuned model</span><p class="pblurb">The two versions of the model organise emotions differently. Working out what pushed pleasantness out of first place is the part of this project we are least able to explain and most interested in.</p></div></div>
<section id="displaced"><div class="wrap">
  <h2><span class="secno">4</span>What took pleasantness's place?</h2>
  <p class="narrow">Suppose chat-tuning had simply shuffled the same information around. Then every axis
  in the chat-tuned model would have a partner somewhere in the plain model: the same emotions
  separated the same way, just renumbered. So we checked all 25 pairings &mdash; each of the chat-tuned
  model's top five axes against each of the plain model's top five. <b>1.0 means the two axes carry the
  same information; 0 means they have nothing in common.</b> Hover any square.</p>
  <div class="grid2" style="margin-top:22px">
    <div class="card">
      <div id="gridChart"></div>
      <div id="gridNote" class="muted" style="margin-top:10px;min-height:34px;font-size:13.5px"></div>
    </div>
    <div class="card">
      <h3>The top row is the finding: nothing matches</h3>
      <p style="font-size:15px">Read the top row of the grid. The chat-tuned model's biggest axis
      scores <b>0.14</b> against its closest relative among the plain model's five biggest axes
      &mdash; near the bottom of the scale. It is not a renaming or a reshuffle of anything the plain model had. It is
      new, chat-tuning put it there, and it is now the largest single thing separating the model's
      emotions.</p>
      <p style="font-size:15px">Now read down the first column. Pleasantness is still in there, intact,
      one floor down: the chat-tuned model's third axis scores <b>0.83</b> against the plain model's
      first. Chat-tuning did not delete the circumplex. It demoted it.</p>
      <div style="margin-top:16px">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.07em;
          font-family:var(--display);font-weight:600">
          the emotions at each end of that new axis</div>
        <div style="margin-top:8px;font-size:14.5px"><b>one end:</b> <span id="pc1low"></span></div>
        <div style="margin-top:6px;font-size:14.5px"><b>the other:</b> <span id="pc1high"></span></div>
        <p class="muted" style="font-size:14px;margin-top:10px">Reading the two lists is the honest way
        to guess what an axis means, and here the guess fails. It is clearly not pleasant-versus-
        unpleasant, because <em>miserable</em> and <em>jubilant</em> sit at the same end. If anything
        it looks like low-grade irritation. But we do not know, and naming it without evidence is
        exactly the mistake we are trying to avoid. It stays an open question.</p>
      </div>
    </div>
  </div>
  <h3 style="margin-top:38px">The same question, asked without any of that machinery</h3>
  <p class="narrow" style="margin-bottom:14px">Principal components are one way to look, and a reader
  is right to wonder whether the finding is a by-product of that particular method. So here is a second look that
  uses no components at all. A model has many layers, stacked from input to output. At each layer, ask
  which emotions it treats as similar to which; that gives one emotion-by-emotion similarity table per
  layer. Every square below compares two of those tables, for every possible pairing of layers. <b>Dark means the two layers sort the
  171 emotions the same way; pale means they disagree.</b></p>
  <div class="card">
    <div class="controls">
      <span class="muted" style="font-size:13px">showing</span><span id="rsaBtns"></span>
      <span class="muted" id="rsaNote" style="margin-left:auto;font-size:12.5px"></span>
    </div>
    <div id="rsaChart"></div>
    <details class="howto"><summary>How to read this</summary>
      <p><b>What a good result looks like:</b> dark everywhere. That is a model whose layers all agree
      on what emotions are, one stable picture from bottom to top. <b>What a bad result looks like:</b>
      pale patches and visible blocks &mdash; groups of layers that agree with each other but not with
      the rest, meaning the model has no single account of emotion.</p>
      <p>Every square is scored from 0 to 1: 1 means the two layers sort the 171 emotions identically,
      0 means they share nothing. Start on <b>plain model</b> and you get close to the good case: its
      late layers agree with each other at <b>0.94</b> out of 1. Now switch to <b>chat-tuned
      model</b>. The same measure drops to <b>0.79</b> and the grid visibly breaks into blocks. The plain view matters here as the control:
      without it, you could not tell a genuinely broken-up model from a measure that just always looks
      blotchy.</p>
      <p>Now switch to <b>chat-tuned, biggest axis subtracted</b>. We remove just the biggest axis at each
      layer &mdash; and the blocks largely merge back together. That is the evidence that the one dominant
      axis <em>caused</em> the fragmentation, rather than the emotion structure having gone missing.</p>
      <p><b>plain vs. chat-tuned</b> is the odd one out: it is the only view with a different model on
      each axis, so it is the only one that is not a mirror image about its diagonal. Rows are
      chat-tuned layers, columns are plain layers, and the diagonal answers "how much did chat-tuning
      change this particular depth?". If chat-tuning changed nothing the diagonal would read 1.0
      throughout. It reads 0.97 at layer 3 and 0.21 at layer 57: the early layers came through intact,
      the late half was rebuilt.</p>
    </details>
    <div class="src">source: notebooks/02_circumplex_geometry.ipynb, section 9</div>
  </div>

  <div class="callout"><span class="k">The caveat that cuts the other way</span>
  Taken at face value, the two models look largely unrelated. Comparing each model's top three axes
  as a set gives three angles &mdash; <b>86.1&deg;</b>, 56.0&deg; and 46.1&deg; &mdash; where 90&deg;
  means two directions share nothing and 0&deg; that they coincide. The widest is almost a right
  angle; even the closest pair is only partly aligned.
  But subtract that one mystery axis and the late layers of the two models jump from <b>0.29</b>
  agreement to <b>0.60</b>, on the same 0-to-1 scale as the grid above. So the shared emotion structure is not gone. It is sitting underneath one
  large direction that has nothing to do with emotion.</div>
  <div class="takeaway"><span class="lbl">Takeaway</span>The biggest thing separating emotions in the chat-tuned model is not a rearrangement of anything the plain model had. It is new, chat-tuning put it there, and it breaks the model's layers into disagreeing blocks. We can show all of that. We still cannot say what it encodes.</div>
</div></section>

<div class="partbar"><div class="wrap"><span class="ptitle">Part three &middot; Following an emotion as it changes</span><p class="pblurb">The step past the published work: stories written to move through three emotions, read one word at a time, to see whether the model changes its mind when the story does.</p></div></div>
<section id="story"><div class="wrap">
  <h2><span class="secno">5</span>Watch it follow one story, word by word</h2>
  <p class="narrow">Everything so far has been about static text: one story, one emotion, one reading.
  Here the story <em>moves</em>. Each of these four was written to pass through three emotions in turn,
  with the turns marked. We feed one to the model and, at every single word, measure how close its
  internal state sits to each of the three emotion vectors. Drag the word slider and both panels
  advance together.</p>
  <p class="narrow">Three of the four stories use the <b>same three emotions</b> and differ only in how
  they are written &mdash; and the model follows them very differently. That is the point of this
  section: how well the model follows an emotion is not a property of the emotion alone.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls" style="margin-bottom:6px">
      <span class="muted" style="font-size:13px">story</span><span id="storyBtns"></span>
      <span class="muted" id="storyQual" style="margin-left:auto;font-size:12.5px"></span>
    </div>
    <div class="controls">
      <span class="muted" style="font-size:13px">layer</span>
      <span id="layerBtns"></span>
      <span style="margin-left:auto;display:flex;align-items:center;gap:10px">
        <span class="muted" style="font-size:13px">word</span>
        <input type="range" id="tokSlider" min="0" max="235" value="0" style="width:230px">
        <span class="mono" id="tokLabel" style="min-width:78px;display:inline-block"></span>
      </span>
    </div>
    <!-- the live readout doubles as the colour legend: it is the only thing on the
         figure that says which line is which emotion, so it sits above the panels
         rather than inside the plot where the lines used to run through it -->
    <div id="storyReadout" class="legend" style="margin:10px 0 2px"></div>
    <div class="grid2" style="gap:24px">
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">
        one line per emotion, over the length of the story</div><div id="lineChart"></div></div>
      <div><div class="muted" style="font-size:12.5px;margin-bottom:6px">
        the same three numbers, as a position between the three emotions</div>
        <div id="ternChart"></div></div>
    </div>
    <div style="margin-top:20px">
      <div class="muted" style="font-size:12.5px;margin-bottom:6px">
        the story the model is reading &mdash; the part you are currently on is highlighted</div>
      <div class="storybox" id="storyBox"></div>
    </div>
    <details class="howto"><summary>How to read this</summary>
      <p><b>Left panel.</b> One line per emotion. Higher means the model's internal state is closer to
      that emotion right now. The two vertical marks are where the story was written to turn.
      <b>A good result</b> is the lead passing cleanly from the red line to the blue to the green, at
      those marks. <b>A bad result</b> is three flat lines that never change order &mdash; a model
      reading the same emotion all the way through.</p>
      <p><b>Right panel.</b> The same three numbers, drawn as a position inside a triangle. Each corner
      is one emotion, and the dot sits nearest whichever it currently resembles most. <b>A good
      result</b> is a walk from corner to corner. <b>A bad result</b> is a dot that never leaves the
      middle, which is what a model ignoring the story would give you.</p>
      <p><b>What actually happens.</b> The handover is real, and it is messy. It also tends to arrive
      <em>early</em> &mdash; the lead changes before the written turn, so the model is picking up the
      shift while still reading the previous part. Change the layer and the character of the picture
      changes, which is what section 7 is about.</p>
      <p><b>Why these four stories, and what "followed well" means here.</b> The last three come out of a
      random sample of 24: the best, the middle and the worst. We scored them on exactly what this
      figure draws &mdash; in each of the three parts, is the part's own emotion the highest line, and by
      how much?
      Leading in all three parts is the best possible; leading in one is what luck alone gives, since
      there are three lines.</p>
      <p><b>This is an easier test than the next section's, deliberately.</b> Here the model only has
      to pick the right emotion out of the story's own three. In section 6 it has to pick out of
      twelve. A story can lead all three lines here and still do poorly out of twelve, so this panel
      illustrates the mechanism rather than carrying the headline number.</p>
    </details>
    <div class="src">story <span id="storyId"></span>, from notebooks/05_trajectories.ipynb. Closeness is how nearly the
    model's internal reading points the same way as each emotion vector (cosine similarity), measured
    against vectors built from the model's own stories. The average across the whole story corpus is
    subtracted first, so 0 means "typical for these stories" rather than "no emotion" &mdash; a line
    above 0 is leaning towards that emotion more than the corpus does on average.</div>
  </div>
  <div class="takeaway"><span class="lbl">Takeaway</span>The model really does hand over from one emotion to the next as the story turns &mdash; and it tends to do so <em>early</em>, before the written turn arrives. The signal is real and it is noisy, and it looks different depending which layer you read.</div>
</div></section>

<section id="emotions"><div class="wrap">
  <div class="kicker">Part three &middot; Following an emotion as it changes</div>
  <h2><span class="secno">6</span>Some emotions are followed well, others not at all</h2>
  <p class="narrow">Now the harder test, run over all the stories rather than one. Take every part of every
  story and find which emotion vector the model's internal reading sits closest to. How often is
  the winner the emotion that part was written to convey?</p>
  <p class="narrow">Note the shift from 171 emotions to <b>twelve</b>. The stories were written to
  move between these twelve, so twelve is what there is to choose from &mdash; and picking one of
  twelve is a test we can actually score. Picking at random would be right 1 time in 12, or 8%.
  Each bar is one emotion's score.</p>
  <p class="narrow">The two buttons swap in a different set of emotion vectors, built from a different
  writer's stories &mdash; the same swap as section 2. The layer buttons read the same measurement at a
  different depth of the model. Hover any bar to see where its wrong answers went.</p>
  <p class="narrow"><b>Why the chat-tuned model, when parts one and two just showed its emotion
  structure is the messier of the two?</b> Because the twelve-emotion vector sets only exist for it: an
  emotion vector has to be built from stories, and these were built from stories the chat-tuned model
  wrote about itself. Running the same test on the plain model would mean rebuilding the whole set from
  plain-model stories, which we did not do. So this section reads the model people actually talk to,
  and the plain model is absent here rather than losing.</p>
  <div class="card" style="margin-top:22px">
    <div class="controls">
      <button class="seg on" data-bank="selfgen">vectors from Gemma's own stories</button>
      <button class="seg" data-bank="deepseek">vectors from DeepSeek's stories</button>
      <span style="margin-left:auto;display:flex;align-items:center;gap:8px">
        <span class="muted" style="font-size:13px">layer</span><span id="emoLayerBtns"></span>
      </span>
    </div>
    <div id="emoChart"></div>
    <div id="emoNote" style="margin-top:12px;min-height:36px;color:var(--body);font-size:14px"></div>
    <details class="howto"><summary>How to read this</summary>
      <p><b>The grading scale.</b> The dashed line at 8% is guessing &mdash; a bar at or below it means
      that emotion is not being followed at all. 100% would mean perfect. Neither extreme happens.</p>
      <p><b>What we see.</b> At layer 33 with Gemma's own vectors, 11 of the 12 emotions beat guessing,
      so we are measuring something real. But nothing reaches even 50%, and the spread is enormous:
      <em>loving</em> and <em>guilty</em> win about half the time while <em>nervous</em> (5.8%) sits
      below guessing. Change the layer and the order shuffles &mdash; which is exactly why nobody should
      quote a single layer as "how well the model follows emotion".</p>
      <p><b>The part we did not expect.</b> The wrong answers are not spread evenly. With Gemma's own
      vectors, mistakes pile up on two emotions: whatever the true answer was, the model keeps
      answering <b>guilty</b> or <b>happy</b>. Switch to DeepSeek's vectors and that pile-up
      disappears, and overall accuracy rises. So a real share of what looks like "the model cannot
      follow emotions" is a property of the vectors we handed it, not of the model.</p>
    </details>
    <div class="src">source: notebooks/11_tracking_taxonomy.ipynb, layer 33, chat-tuned model reading</div>
  </div>
  <div class="takeaway"><span class="lbl">Takeaway</span>The model follows emotion better than guessing, but unevenly and never reliably &mdash; and a large part of that unevenness comes from the emotion vectors rather than the model. Swap in a different writer's vectors and both the accuracy and the pattern of mistakes change.</div>
</div></section>

<section id="layers"><div class="wrap">
  <div class="kicker">Part three &middot; Following an emotion as it changes</div>
  <h2><span class="secno">7</span>Different layers are good at different things</h2>
  <p class="narrow">There is no single "emotion layer". We measured two different skills at six depths of
  the model, and they peak in different places. The first is the one from section 6:
  <b>does it name the emotion right now?</b> The second asks something subtler.
  Before each turn the model leans slightly towards the emotion about to arrive; this measures
  whether the <em>size</em> of that lean tracks the <em>size</em> of the coming emotional change.
  A big swing from cheerful to devastated should produce a bigger lean than a small shift from
  content to calm. The two lines are on different scales, so each has its own axis and its own
  failure line &mdash; read them as two charts sharing an x-axis.</p>
  <div class="grid2" style="margin-top:22px">
    <div class="card"><div id="layerChart"></div>
      <div class="legend">
        <span><i style="background:var(--navy)"></i>names the current emotion right (left axis, 8% is
        guessing)</span>
        <span><i style="background:var(--orange)"></i>the lean tracks how big the coming change is
        (right axis, 0 means the size of the lean says nothing about the size of the change)</span>
      </div>
    </div>
    <div class="card">
      <h3>Two different skills, in two different places</h3>
      <p style="font-size:15px">Naming peaks early and falls: averaged over all twelve emotions, layer 6
      gets the current emotion right <b>58%</b> of the time, and by layer 33 that is down to
      <b>27%</b>. This curve is the average of section 6's twelve bars, on
      the Gemma-written vectors: at layer 33 the best single emotion reaches 49% and the worst
      sits at 6%, below guessing. The other measure runs the other way,
      climbing from <b>+0.03</b> at layer 6 to <b>+0.26</b> at layer 51 &mdash; but
      not smoothly: it reaches +0.143 by layer 24, collapses to +0.001 at layer 33, then climbs again.
      Whatever the model is doing, it is not doing it all in one place, and the middle of the stack
      behaves differently from either end.</p>
      <p style="font-size:15px">One consolation in the errors: when the model names the wrong emotion,
      it names a <em>near</em> one. The emotion it picks instead sits closer to the true one in the
      human pleasantness-and-energy layout than a randomly chosen emotion would &mdash; and that holds
      at every layer we looked at.</p>
      <div class="callout" style="margin-top:6px"><span class="k">A result that reverses with depth</span>
      The model's own emotion layout explains its mistakes better at layer 33; human ratings explain
      them better at layer 51. So the answer to "does it confuse emotions the way people do?" depends
      on where you look. We only caught this because every figure recomputes
      its verdict per layer instead of quoting one.</div>
    </div>
  </div>
  <div class="src">source: notebooks/11_tracking_taxonomy.ipynb (measurements repeated at each layer)</div>
  <div class="takeaway"><span class="lbl">Takeaway</span>There is no single "emotion layer". Naming the emotion in front of it is an early-layer skill. Registering <em>how big</em> the coming emotional change is grows with depth and peaks at the deepest layer we read &mdash; unevenly, with a collapse at layer 33 we cannot explain. Any result quoted at one depth is a result about that depth.</div>
</div></section>



<section id="next"><div class="wrap">
  <div class="kicker">Where this leaves us</div>
  <h2><span class="secno">8</span>What we would do next</h2>
  <p class="narrow" style="margin-bottom:4px">Three experiments, in the order we would run them. Each
  one attacks a specific thing we could not settle.</p>
  <div class="grid3" style="margin-top:24px">
    <div class="card"><h3>Find out what the mystery axis is</h3><p style="font-size:15px">We can
    measure the chat-tuned model's biggest emotion axis and we cannot say what it encodes. It is not
    pleasantness, energy, control, or story length &mdash; we checked all four. The next test is to
    stop looking and start pushing: add that direction into the model while it writes, and see what
    changes in the output. Whatever changes is what the axis was for.</p></div>
    <div class="card"><h3>Find out why the layers divide the work</h3><p style="font-size:15px">Naming
    the emotion in front of it is an early-layer skill; registering the size of the change about
    to arrive is a late-layer one. We can show the split exists and we have no account of why. One guess worth
    testing: early layers may be reading the emotion words on the page rather than the situation, in
    which case a story that implies an emotion without ever naming it should break them.</p></div>
    <div class="card"><h3>Separate the vectors from the model</h3><p style="font-size:15px">Our
    stories switch emotions by label. A sharper test varies the <em>intensity</em> of one emotion
    instead, which is a harder thing to fake. And since the pile-up of wrong answers appeared with one
    writer's vectors and vanished with another's, telling "the model cannot do this" apart from "these
    vectors cannot do this" deserves an experiment of its own.</p></div>
  </div>
  <div class="callout"><span class="k">Three results, three different grades of evidence</span>
  We track claims in a research tree that records how far each one has been tested, and the three
  results on this page are not equally settled. <b>Only the first has finished the process.</b>
  <br><br>
  <b>1. The plain model reproduces the circumplex &mdash; tested and survived.</b> It has been through
  the full falsification pass and came out the other side, including a re-run after we corrected an
  extraction bug.
  <br><br>
  <b>2. Chat-tuning demotes it and puts a large non-emotional axis on top &mdash; measured, not yet
  formally tested.</b> Every number behind it is real and reproducible, but the claim is still marked
  unvalidated in our tree: its falsification pass is owed. Read it as a strong observation, not a
  settled result.
  <br><br>
  <b>3. Emotion vectors follow a story better than guessing &mdash; exploratory.</b> Parts three's
  measurements are registered as hypothesis-generating work with no pass bar attached, and one thing
  should be said plainly: the detection bar we <em>did</em> register in advance was written for the
  171-emotion vector set, and that set failed it at every layer. What passes is the twelve-emotion
  sets, which is a substitution we made after seeing the results. It may well be the right call. It is
  not the thing we pre-registered, and we are not going to describe it as though it were.
  <br><br>
  <b>Not confident about, in any of the three:</b> what the new axis is, and how much of the
  story-following quality belongs to the model rather than to the stories we built the vectors from.
  Every number on
  this page, including the ones that came out null, is registered in the project's research tree with
  the evidence file behind it.</div>
  <div class="callout" style="border-color:var(--navy)"><span class="k">The explanation we cannot rule
  out</span>
  Every positive result here is real and small: better than guessing, worse than anything you would
  want to rely on. There are two readings and we cannot yet separate them. The comfortable one is that
  emotion is genuinely represented and our instruments are blunt. The other is that emotion is simply
  not one of the big organising ideas in this model, and a weak signal is the correct answer rather
  than a measurement failure. Telling those apart is the experiment we would most like to run, and
  either outcome is worth having.</div>
</div></section>


<section id="methods"><div class="wrap">
  <div class="kicker">How we measured it</div>
  <h2><span class="secno">9</span>How we measured it, in enough detail to argue with</h2>
  <p class="narrow">Four blocks, all collapsed. Open whichever one you want to challenge; each carries
  the formula, the pseudo-code and a glossary for every symbol in it.</p>

  <details class="howto" style="border-top:none;margin-top:18px">
    <summary style="font-size:15px">1. How an emotion becomes a vector</summary>
    <p>We start from a collection of stories, each labelled with the emotion it was written to evoke.
    Each story goes through the model and we take an internal reading at a chosen layer. For one
    emotion, we average those readings across all of its stories. That average is the emotion vector.
    Repeat for every emotion word and you have 171 of them.</p>
    <div id="m1"></div>
    <p>Two details that change the numbers. <b>Subtracting the average (centering):</b> we subtract the
    mean across all the emotions in the set &mdash; all 12 or all 171, and which pool we use changes
    the answer &mdash; so what remains is what makes <em>this</em> emotion different rather than what all
    English prose has in common. Skipping this step is the difference between a result
    and a number produced by the method rather than by the model &mdash; and we know, because we
    skipped it once (block 4). <b>Which words count:</b>
    a story is hundreds of words and we combine them into one recording: ignore the padding, drop the
    first 50 as scene-setting (the convention the original paper used), and average the rest, up to 512.
    (Strictly these are <em>tokens</em>, the word-fragments the model actually reads &mdash; roughly
    three-quarters of a word each. We say "word" elsewhere on this page for readability.) We fixed both choices before any scoring. Note this is the opposite of what block 3 does,
    where we deliberately keep every word separate instead of averaging.</p>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">2. What "the biggest axis is pleasantness, 0.83" actually means</summary>
    <p>This is the one worth spelling out, because the phrase hides four steps.</p>
    <p><b>Step one.</b> We have 171 emotion vectors, each thousands of numbers long. Principal component
    analysis finds the directions along which those 171 points differ most. The first component (PC1) is
    the single direction that captures the most spread. It comes only from the model: we use no human
    labels to find it.</p>
    <p><b>Step two.</b> Each of the 171 emotions gets a <em>score</em> on PC1: how far along that
    direction it sits. So "elated" gets a number, "miserable" gets a number, and so on.</p>
    <p><b>Step three.</b> Separately, we look up each emotion word in the
    <a href="http://saifmohammad.com/WebPages/nrc-vad.html">NRC VAD lexicon</a>, a published set of
    human ratings where people scored thousands of words for valence, arousal and dominance. 164 of our
    171 words are in it.</p>
    <p><b>Step four.</b> We correlate the model's 164 PC1 scores against the humans' 164 valence ratings.
    That correlation is <b>0.83</b> on the plain model. In words: the direction of largest variation in
    the model's emotion space, found without reference to any human judgement, lines up strongly with
    what people call pleasant versus unpleasant.</p>
    <div id="m2"></div>
    <p><b>Why we ignore the sign.</b> A principal component points along an axis, but which of its two
    directions gets called "positive" is arbitrary &mdash; flip it and the correlation flips sign
    without anything changing. So we report the size of the correlation and not its sign.
    <b>What would count as failure:</b> a number near 0, meaning the model's biggest axis has nothing
    to do with pleasantness. That is close to what the chat-tuned model gives (0.11).</p>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">3. How "does it follow the story?" was scored</summary>
    <p>We wrote 173 three-emotion recipes and generated stories from each, giving a corpus of
    8,938 labelled story parts and 5,934 transitions between them. Nothing is averaged away here:
    the model reads the story once, we keep its state at <b>every single word</b>, and at each position
    we measure the angle between that state and each of the 12 emotion vectors. A part's score is the
    average over the words inside it; we skip parts shorter than 4 words rather than averaging them in.
    Two questions follow, and we wrote down what would count as passing before running either.</p>
    <p><b>Naming:</b> inside a part written as "afraid", where does the afraid vector place among the
    12? First place is perfect; 6.5th on average is what guessing gives. <b>Anticipation:</b> in the
    words just before a written turn, has the <em>next</em> emotion's vector already begun to pull
    ahead? Zero would mean the model does not see the turn coming.</p>
    <div id="m3"></div>
    <p>We compare both against a shuffle: the identical computation with the emotion labels randomly
    reassigned. That shuffle, not our intuition about what looks good, is the floor.</p>
    <div id="m4"></div>
  </details>

  <details class="howto" style="border-top:none">
    <summary style="font-size:15px">4. What kept us honest</summary>
    <p>We wrote down every prediction, and the mark it had to beat, before the data existed. Nothing
    is called a finding until it has survived a round of deliberate attempts to destroy it. We re-run
    every measurement with the emotion labels shuffled. We put error bars on it that treat each story as
    one unit rather than each word, because words within a story are not independent. We check whether a
    randomly chosen direction scores just as well. And we check that the result is not simply reproducing
    how common each answer already is. We record what failed as failed. The notebooks re-run from those files on any
    machine.</p>
    <p class="muted">One correction is worth flagging because it reversed a headline. Our first round of
    detection tests found nothing passing anywhere. A later audit found that the readout had skipped the
    subtract-the-average step from block 1. With that step restored, the same data passes on both
    models. We report both, and the section-3 figure is the corrected one.</p>
  </details>
</div></section>

<section id="appendix"><div class="wrap">
  <div class="kicker">Appendix</div>
  <h2><span class="secno">10</span>Questions we expect, and our honest answers</h2>
  <p class="narrow">All collapsed, so we can open whichever one actually gets asked.</p>
  <div style="margin-top:18px">
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Isn't this just sentiment analysis with extra steps?</summary>
      <p>Sentiment analysis attaches one label to a piece of text. We read the model's internal state directly, at every word, without asking the model any question at all. That buys two things a classifier cannot give you: you can watch the state move <em>through</em> a story, and you can push on it. And the interesting result is not the labelling accuracy &mdash; it is that pleasantness turns out to be the axis the model sorts emotions along, with nobody having asked for that.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Does the model actually feel anything?</summary>
      <p>Nothing here shows that, and we would push back on the leap. What we measured is the model representing the emotions <em>of a story it is reading</em> &mdash; closer to reading comprehension than to feeling anything. It matters as evidence about internals, because most arguments in this area run entirely on what a model says about itself. But recognising an emotion and having one are different claims, and running them together is the easiest mistake available here.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">The correlations are small. Isn't this noise?</summary>
      <p>Some of them are, and we say so where they are. The layout result is not small: 0.83 across 164 words is a strong effect. (The original paper's comparable figure was computed over 45 emotions against a different set of human ratings, so ours is directionally but not procedurally comparable to it.) The story-following results are modest, but they beat the shuffled-label version of themselves at p below 0.001: if the emotion labels carried no information at all, a gap this large would turn up in fewer than 1 shuffle in 1,000. One measurement not shown on this page shows how we handle a weak result. We also asked whether the vectors track how <em>strongly</em> an emotion is felt. For that one we report only which way the effect pointed, never how big it was, because a control found that randomly chosen directions scored just as high &mdash; so the size carries no information and only the sign does.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Why Gemma 4 and not the model in the paper?</summary>
      <p>Because both versions of it are available at the same size: the plain model and the chat-tuned model, otherwise identical. That is exactly the comparison the original paper could not run, and it is where our main finding came from.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Could the mystery axis just be a bug in your pipeline?</summary>
      <p>Fair question, and we checked the obvious ways it could be. Story length is the fake-result check we worried about most, and it does not account for this axis. On the 0-to-1 scale of section 3 the mystery axis matches length at 0.39, below the 0.5 we treat as clearly related. The <em>second</em> axis matches length at 0.66 and plainly is about length. It survived an audit of our extraction conventions that did change other numbers on this page. It is not a sign flip or a rearrangement, because its best match against any of the plain model's five biggest axes is 0.14. What it <em>is</em> stays open, and we would rather leave it open than name it.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">How much of the story-following result comes from your emotion vectors rather than the model?</summary>
      <p>More than we would like, and that is itself a finding. Changing only who wrote the stories moves the number of working layers from 1 to 9, and the pile-up of wrong answers that appears with one writer's vectors disappears with another's. Any paper reporting how accurately emotion vectors detect emotion, without saying who wrote the stories those vectors were built from, is under-specified.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">Why does layer 6 name emotions better than layer 33?</summary>
      <p>We do not know, and it is our favourite loose end. One guess: the early layers stay closer to the actual words on the page, and our stories do contain emotional words, so a shallow reading does well on them. The test that would settle it is a set of stories where the emotion is unmistakable but never named.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">What would have changed your mind?</summary>
      <p>For the layout result: a plain model whose biggest axis had nothing to do with pleasantness. For the story-following result: scores indistinguishable from the shuffled-label version &mdash; which is roughly what our control stories, the ones holding a single emotion throughout, actually gave. We ran that control precisely so that a null would have somewhere to show up.</p>
    </details>
    <details class="howto" style="border-top:none">
      <summary style="font-size:15px">What is the single weakest part of this work?</summary>
      <p>The size of the story-following effects. Everything is above guessing and below anywhere you would want to be before relying on it. We cannot yet tell a weak measurement of a real thing apart from an accurate measurement of a weak thing &mdash; and those two have very different implications.</p>
    </details>
  </div>
  <div class="callout" style="margin-top:24px"><span class="k">If you remember one thing</span>
  The plain model lays its emotions out the way psychologists do, without being asked. Chat-tuning does
  not destroy that layout &mdash; it demotes it, and puts one large axis on top that we cannot explain.
  And whichever model you ask, how well any of this works depends heavily on who wrote the stories the
  emotion vectors were built from.</div>
</div></section>

<footer><div class="wrap">
  <div>CAMBRIA capstone &middot; Gemma 4 31B, plain and chat-tuned &middot; every figure regenerated
  from the notebooks in the project repository.</div>
  <div class="src">A short sprint, not a peer-reviewed paper: no external review, and the null results are included and labelled as such. Every number here is transcribed from the printed output of notebooks 02, 05, 08 and 11.</div>
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

/* ---------- circumplex: principal-component bars ---------- */
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
  [[1,"1.0 — this axis is exactly that human rating",P.green],
   [0.5,"0.5 — clearly related",P.muted],
   [0,"0 — nothing to do with it",P.alert]].forEach(([v,label,col])=>{
    const y=T+ih-v*ih;
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y,y2:y,stroke:col,"stroke-dasharray":"4 3",
      "stroke-width":v===0.5?1:1.4,opacity:v===0.5?.5:.9}));
    // the zero line sits at the foot of the bars, so its label goes below the
    // axis rather than on top of PC5's columns
    const t=el("text",{x:W-R-2,y:v===0?y+13:y-5,"text-anchor":"end","font-size":10,fill:col});
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
      const nm={valence:"pleasantness",arousal:"energy",dominance:"control",
        length:"story length"}[k];
      tipOn(bar,`<b>axis PC${r.pc} vs human ${nm} ratings</b>: ${r[k].toFixed(2)}`+
        `<span class="t-sub">1.0 would mean this axis is exactly that rating, 0 that it has nothing `+
        `to do with it.<br>PC${r.pc} accounts for ${(r.evr*100).toFixed(1)}% of everything that `+
        `separates the 171 emotions.</span>`);
      svg.appendChild(bar);
    });
    const back=el("rect",{x:x0+8,y:T+ih-r.evr*ih,width:bw-16,height:r.evr*ih,fill:"transparent"});
    tipOn(back,`<b>axis PC${r.pc}</b> accounts for ${(r.evr*100).toFixed(1)}% of everything that `+
      `separates the 171 emotions<span class="t-sub">that is what the grey backdrop bar shows</span>`);
    svg.appendChild(back);
    const lab=el("text",{x:x0+bw/2,y:H-24,"text-anchor":"middle","font-size":11,fill:P.text});
    lab.textContent=r.pc===1?"PC1 (biggest axis)":"PC"+r.pc; svg.appendChild(lab);
    const ev=el("text",{x:x0+bw/2,y:H-10,"text-anchor":"middle","font-size":9.5,fill:P.muted});
    ev.textContent=(r.evr*100).toFixed(1)+"% of the spread"; svg.appendChild(ev);
  });
  // the y axis had tick numbers but never said what they measured
  const yl=el("text",{x:14,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="match with the human rating (higher is closer)"; svg.appendChild(yl);
  host.appendChild(svg);
  document.getElementById("pcVerdict").textContent = model==="base"
    ? "biggest axis = pleasantness, 0.83. the circumplex, recovered."
    : "biggest axis = unknown. pleasantness scores 0.11 here, and leads PC3 instead.";
}
document.querySelectorAll("[data-model]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-model]").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); drawPCs(b.dataset.model);
});

/* ---------- what moved in: correlation grid ---------- */
function drawGrid(){
  const host=document.getElementById("gridChart"); host.innerHTML="";
  const W=420,H=420,L=64,T=42,cell=62;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const note=document.getElementById("gridNote");
  D.grid.forEach((row,i)=>row.forEach((v,j)=>{
    const x=L+j*cell,y=T+i*cell;
    const g=el("rect",{x,y,width:cell-3,height:cell-3,rx:3,
      fill:`rgba(29,53,87,${(v*0.95).toFixed(3)})`,style:"cursor:pointer"});
    tipOn(g,`<b>chat-tuned axis ${i+1} vs plain axis ${j+1}</b>: ${v.toFixed(2)}`+
      `<span class="t-sub">`+
      (i===0 ? "This row is the finding. The chat-tuned model's biggest axis scores at most 0.14 against any of the plain model's five biggest axes, so it is new structure rather than a rearrangement of them." :
       i===2&&j===0 ? "This is pleasantness: the plain model's top axis, still intact, but demoted to third place by chat-tuning." :
       "1 would mean the two axes carry the same information; 0 that they have nothing in common.")+
      `</span>`);
    svg.appendChild(g);
    const t=el("text",{x:x+(cell-3)/2,y:y+(cell-3)/2+4,"text-anchor":"middle","font-size":11,
      fill:v>0.45?"#fff":P.text}); t.textContent=v.toFixed(2); svg.appendChild(t);
  }));
  // column and row headers, plus one line naming which model each side belongs to,
  // because "it PC1" read as an abbreviation nobody on the page had defined
  const ch=el("text",{x:L,y:14,"font-size":10.5,fill:P.muted,"font-weight":600});
  ch.textContent="the plain model's five biggest axes"; svg.appendChild(ch);
  for(let j=0;j<5;j++){const t=el("text",{x:L+j*cell+(cell-3)/2,y:T-12,"text-anchor":"middle",
    "font-size":10.5,fill:P.muted});t.textContent="plain "+(j+1);svg.appendChild(t);}
  for(let i=0;i<5;i++){const t=el("text",{x:L-8,y:T+i*cell+(cell-3)/2+4,"text-anchor":"end",
    "font-size":10.5,fill:P.muted});t.textContent="chat "+(i+1);svg.appendChild(t);}
  const rh=el("text",{x:12,y:T+(5*cell)/2,"font-size":10.5,fill:P.muted,"font-weight":600,
    transform:`rotate(-90 12 ${T+(5*cell)/2})`,"text-anchor":"middle"});
  rh.textContent="the chat-tuned model's five biggest axes"; svg.appendChild(rh);
  // a scale strip, so a shade can be read without hovering
  const sx=L, sy=T+5*cell+14, sw=5*cell-3;
  for(let i=0;i<40;i++){
    svg.appendChild(el("rect",{x:sx+i*(sw/40),y:sy,width:sw/40+.5,height:9,
      fill:`rgba(29,53,87,${((i/39)*0.95).toFixed(3)})`}));
  }
  // only the two ends are labelled: a middle tick collided with the right label
  [["0 — nothing in common",0,"start"],["1 — the same information",1,"end"]].forEach(([lab,f,anc])=>{
    const t=el("text",{x:sx+f*sw,y:sy+22,"text-anchor":anc,"font-size":10,fill:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  host.appendChild(svg);
  note.textContent="hover any square for what that pairing means";
}
document.getElementById("pc1low").textContent=D.itpc1.low.join(", ");
document.getElementById("pc1high").textContent=D.itpc1.high.join(", ");

/* ---------- one story, token by token ---------- */
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
  const W=470,H=300,L=62,R=12,T=14,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, n=S.n_tokens;
  let lo=Infinity,hi=-Infinity; ys.forEach(s=>s.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v);}));
  const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
  const X=i=>L+(i/(n-1))*iw, Y=v=>T+ih-((v-lo)/(hi-lo))*ih;
  // the zero line is the story's own average: above it the model leans toward
  // that emotion, below it away. Labelled, because an unlabelled zero is noise.
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(0),y2:Y(0),stroke:P.greyMid,"stroke-dasharray":"3 3"}));
  // in the left margin, not over the plot: the three lines cross the zero line
  // constantly, so an inline label sat on top of the data for most stories
  [[hi,hi.toFixed(2)],[0,"0"],[lo,lo.toFixed(2)]].forEach(([v,lab])=>{
    const t=el("text",{x:L-6,y:Y(v)+3,"text-anchor":"end","font-size":9,
      fill:v===0?P.text:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  const zl2=el("text",{x:L-6,y:Y(0)+12,"text-anchor":"end","font-size":8,fill:P.muted});
  zl2.textContent="corpus avg"; svg.appendChild(zl2);
  S.boundaries.forEach(b=>{
    svg.appendChild(el("line",{x1:X(b),x2:X(b),y1:T,y2:T+ih,stroke:P.text,"stroke-width":1,"stroke-dasharray":"4 3",opacity:.45}));
  });
  ys.forEach((serie,k)=>{
    let d="";serie.forEach((v,i)=>{d+=(i?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1);});
    svg.appendChild(el("path",{d,fill:"none",stroke:EC[k],"stroke-width":1.9,opacity:.9}));
    const hit=el("path",{d,fill:"none",stroke:"transparent","stroke-width":11});
    tipOn(hit,()=>`<b>${S.emotions[k]}</b> at word ${curTok}: ${serie[curTok].toFixed(3)}`+
      `<span class="t-sub">how close the model's state sits to the ${S.emotions[k]} vector. `+
      `Higher means closer. Read at layer ${curLayer}.</span>`);
    svg.appendChild(hit);
  });
  svg.appendChild(el("line",{x1:X(curTok),x2:X(curTok),y1:T,y2:T+ih,stroke:P.orange,"stroke-width":2}));
  ys.forEach((serie,k)=>svg.appendChild(el("circle",{cx:X(curTok),cy:Y(serie[curTok]),r:4.5,
    fill:EC[k],stroke:"#fff","stroke-width":1.5})));
  const readout=document.getElementById("storyReadout");
  if(readout) readout.innerHTML = S.emotions.map((nm,k)=>
    `<span><i style="background:${EC[k]}"></i>${nm} <b style="color:${EC[k]}">`+
    `${ys[k][curTok].toFixed(3)}</b></span>`).join("")+
    `<span style="margin-left:auto">reading at word ${curTok}, layer ${curLayer}</span>`;
  [[0,"0"],[S.boundaries[0],"turn 1 · "+S.boundaries[0]],
   [S.boundaries[1],"turn 2 · "+S.boundaries[1]],[n-1,String(n-1)]].forEach(([i,lab],k)=>{
    const t=el("text",{x:X(i),y:T+ih+13,"text-anchor":k===0?"start":(k===3?"end":"middle"),
      "font-size":9,fill:k===1||k===2?P.text:P.muted});
    t.textContent=lab; svg.appendChild(t);
  });
  const xl=el("text",{x:L+iw/2,y:H-6,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent="word in the story";
  svg.appendChild(xl);
  const yl=el("text",{x:12,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 12 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="closeness to each emotion vector"; svg.appendChild(yl);
  // the y range is re-fitted per layer, so say so rather than let the reader
  // assume the shapes are comparable across the layer buttons
  const rn=el("text",{x:W-R,y:T+8,"text-anchor":"end","font-size":8.5,fill:P.muted});
  rn.textContent="y range re-fitted per layer"; svg.appendChild(rn);
  host.appendChild(svg);

  // ternary
  const th=document.getElementById("ternChart"); th.innerHTML="";
  const TW=470,TH=316,cx=TW/2,top=26,side=232,hgt=side*Math.sin(Math.PI/3);
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
  // Only the path already walked is drawn. Drawing the whole story from t=0
  // showed the reader the future and made the walk look directionless.
  let dPast="", dRest="";
  for(let i=0;i<S.n_tokens;i++){const m=mix(i);const p=proj(m[0],m[1],m[2]);
    const seg=p[0].toFixed(1)+","+p[1].toFixed(1);
    if(i<=curTok) dPast+=(i?"L":"M")+seg;
    if(i>=curTok) dRest+=(i===curTok?"M":"L")+seg;}
  // what is still to come, barely visible, so the shape is not a surprise
  s2.appendChild(el("path",{d:dRest,fill:"none",stroke:P.border,"stroke-width":1.2,
    "stroke-dasharray":"2 3",opacity:.6}));
  s2.appendChild(el("path",{d:dPast,fill:"none",stroke:P.muted,"stroke-width":1.6,opacity:.75}));
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
  const cl2=el("text",{x:cx,y:TH-38,"text-anchor":"middle","font-size":9.5,fill:P.muted});
cl2.textContent="corner = reads purely as that emotion · middle = undecided"; s2.appendChild(cl2);
  const cap=el("text",{x:cx,y:TH-23,"text-anchor":"middle","font-size":9.5,fill:P.muted});
cap.textContent="solid line = the walk so far · dotted = still to come · "+
    "big dot = where it is now"; s2.appendChild(cap);
  const cap2=el("text",{x:cx,y:TH-7,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  cap2.textContent="written to walk "+S.emotions.join(" → "); s2.appendChild(cap2);
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
    : "the worked example; the other three are scored, best / middle / worst of a random 24";
  document.getElementById("storyQual").textContent=q;
  document.getElementById("storyId").textContent=S.id;
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

/* ---------- per emotion, every layer ---------- */
let emoBank="selfgen", emoLayer="33";
function drawEmo(){
  const host=document.getElementById("emoChart"); host.innerHTML="";
  const rows=D.emoByLayer[emoBank][emoLayer];
  const W=880,H=330,L=64,R=150,T=20,B=70;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length;
  // Fixed 0-100% domain. It used to rescale to the tallest bar, so switching
  // vector set redrew 49% and 83% at almost the same length — on the one
  // control the section exists to make you compare.
  const chance=1/12, Y=v=>T+ih-v*ih;
  [0,0.25,0.5,0.75,1].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-7,y:Y(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.muted});
    t.textContent=(v*100)+"%"; svg.appendChild(t);
  });
  const yl=el("text",{x:16,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 16 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="story parts where this emotion’s own vector wins"; svg.appendChild(yl);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(chance),y2:Y(chance),stroke:P.alert,
    "stroke-dasharray":"5 4","stroke-width":1.5}));
  // in the right margin, clear of the bars: inside the plot this label was
  // drawn in dark red across the two tallest navy bars
  const ct=el("text",{x:W-R+6,y:Y(chance)+3.5,"font-size":10.5,fill:P.alert});
  ct.textContent="guessing (8%)"; svg.appendChild(ct);
  const ct2=el("text",{x:W-R+6,y:Y(chance)+16,"font-size":9.5,fill:P.muted});
  ct2.textContent="at or below this line: no signal"; svg.appendChild(ct2);
  const gt=el("text",{x:W-R+6,y:Y(1)+3.5,"font-size":10.5,fill:P.green});
  gt.textContent="100% = always right"; svg.appendChild(gt);
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
    const nearChance=Math.abs(r.rate-chance)<0.03;
    const v=el("text",{x:x+bw/2,y:Y(r.rate)+(nearChance?-14:-6),"text-anchor":"middle",
      "font-size":10.5,fill:r.rate>0?P.text:P.alert});
    v.textContent=r.rate>0?pct(r.rate):"never"; svg.appendChild(v);
    const t=el("text",{x:x+bw/2,y:T+ih+16,"text-anchor":"end","font-size":11,fill:P.body,
      transform:`rotate(-40 ${x+bw/2} ${T+ih+16})`}); t.textContent=r.e; svg.appendChild(t);
  });
  host.appendChild(svg);
  // the standing note restates the verdict for whichever layer and bank is showing
  const best=rows[0], nWin=rows.filter(r=>r.rate>=chance).length, nNever=rows.filter(r=>r.rate===0).length;
  document.getElementById("emoNote").innerHTML =
    `<b>layer ${emoLayer}, vectors from ${emoBank==="selfgen"?"Gemma's own":"DeepSeek's"} stories:</b> `+
`${nWin} of 12 emotions beat guessing, best is ${best.e} at ${pct(best.rate)}`+
    ` (${rows[0].n} story parts per emotion)`+
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

/* ---------- naming vs anticipating, by layer ---------- */
function drawLayers(){
  const host=document.getElementById("layerChart"); host.innerHTML="";
  const rows=D.byLayer, W=470,H=320,L=52,R=52,T=34,B=44;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(rows.length-1))*iw;
  const Y1=v=>T+ih-(v/0.65)*ih, Y2=v=>T+ih-(v/0.30)*ih;
  // Two measurements on two scales share this chart, so each line gets its own
  // axis title in its own colour and its own labelled failure line. Without the
  // colour pairing a reader cannot tell which y-value belongs to which curve.
  const yLeft=el("text",{x:13,y:T+ih/2,"font-size":10,fill:P.navy,
    transform:`rotate(-90 13 ${T+ih/2})`,"text-anchor":"middle"});
  yLeft.textContent="how often it names the emotion right"; svg.appendChild(yLeft);
  const yRight=el("text",{x:W-13,y:T+ih/2,"font-size":10,fill:P.orange,
    transform:`rotate(90 ${W-13} ${T+ih/2})`,"text-anchor":"middle"});
  yRight.textContent="lean size vs size of the coming change"; svg.appendChild(yRight);
  [[0,"0%"],[0.3,"30%"],[0.6,"60%"]].forEach(([v,lab])=>{
    const t=el("text",{x:L-7,y:Y1(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.navy});
    t.textContent=lab; svg.appendChild(t);
  });
  [[0,"0"],[0.15,"+0.15"],[0.30,"+0.30"]].forEach(([v,lab])=>{
    const t=el("text",{x:W-R+7,y:Y2(v)+3.5,"font-size":9.5,fill:P.orange});
    t.textContent=lab; svg.appendChild(t);
  });
  // anchors: guessing for the naming curve, zero for the anticipation curve
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y1(1/12),y2:Y1(1/12),stroke:P.navy,
    "stroke-dasharray":"4 3",opacity:.55}));
  const a1=el("text",{x:W-R-2,y:Y1(1/12)-6,"text-anchor":"end","font-size":9.5,fill:P.navy});
  a1.textContent="guessing (8%)"; svg.appendChild(a1);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y2(0),y2:Y2(0),stroke:P.orange,
    "stroke-dasharray":"4 3",opacity:.55}));
  // above its own line, not below: below puts it on the x-axis tick labels, and
  // both zero lines land at the same height because the two scales share a floor
  const a2=el("text",{x:W-R-2,y:Y2(0)-6,"text-anchor":"end","font-size":9.5,fill:P.orange});
  // kept short on purpose: right-anchored, this label extends leftwards into the
  // plot, and layer 33 sits exactly ON the zero line — a longer string runs
  // straight under that point. The axis title carries the full meaning.
  a2.textContent="0 — no relation"; svg.appendChild(a2);
  let d1="",d2="";
  rows.forEach((r,i)=>{d1+=(i?"L":"M")+X(i)+","+Y1(r.top1);d2+=(i?"L":"M")+X(i)+","+Y2(r.r_dval);});
  svg.appendChild(el("path",{d:d1,fill:"none",stroke:P.navy,"stroke-width":2.4}));
  svg.appendChild(el("path",{d:d2,fill:"none",stroke:P.orange,"stroke-width":2.4,"stroke-dasharray":"5 3"}));
  rows.forEach((r,i)=>{
    const c1=el("circle",{cx:X(i),cy:Y1(r.top1),r:7,fill:P.navy});
    tipOn(c1,`<b>layer ${r.layer}: names the right emotion ${pct(r.top1)} of the time</b>`+
      `<span class="t-sub">when it is wrong, the emotion it picks instead sits ${r.vad.toFixed(2)} `+
      `away in the human pleasantness-and-energy layout, against ${r.shuffle.toFixed(2)} for a `+
      `randomly chosen emotion: wrong, but nearby.</span>`);
    svg.appendChild(c1);
    const c2=el("circle",{cx:X(i),cy:Y2(r.r_dval),r:7,fill:P.orange});
    tipOn(c2,`<b>layer ${r.layer}: anticipation +${r.r_dval.toFixed(3)}</b>`+
      `<span class="t-sub">in the words before a turn, how well the model's lean towards the next `+
      `emotion matches how big the coming change in pleasantness will be. 0 would mean it does not `+
      `see the turn coming at all.</span>`);
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


/* ---------- probe lineage: who wrote the stories ---------- */
function drawLineage(){
  const host=document.getElementById("lineageChart"); host.innerHTML="";
  const rows=D.lineage, W=880,H=290,L=58,R=30,T=34,B=76;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B, bw=iw/rows.length, Y=v=>T+ih-(v/20)*ih;
  [0,5,10,15,20].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-6,y:Y(v)+4,"text-anchor":"end","font-size":10,fill:P.muted});
    t.textContent=v; svg.appendChild(t);
  });
  // The failure anchor (0 of 20) is named in the legend below the chart rather
  // than on the baseline: the shortest bar is 1, so a baseline label sits under it.
  rows.forEach((r,i)=>{
    const x=L+i*bw, h=(T+ih)-Y(r.layers);
    const bar=el("rect",{x:x+16,y:Y(r.layers),width:bw-32,height:h,rx:3,
      fill:P.navy});
    tipOn(bar,`<b>stories written by ${r.label}</b>`+
      `<span class="t-sub">${r.layers} of the 20 tested layers reach the mark set in advance: `+
      `${D.bar} of 12 checks with the right emotion in the top three, on both sets of test text `+
      `(guessing gives about 3 of 12).<br>`+
      `Written from ${r.n.toLocaleString()} stories`+
      (r.overlap!==null?`. Five-word phrase repetition: ${r.overlap} — higher means the stories keep `+
        `reusing the same phrasing`:"")+`.</span>`);
    svg.appendChild(bar);
    const v=el("text",{x:x+bw/2,y:Y(r.layers)-8,"text-anchor":"middle","font-size":15,
      "font-weight":600,fill:P.text}); v.textContent=r.layers; svg.appendChild(v);
    const nm=el("text",{x:x+bw/2,y:T+ih+18,"text-anchor":"middle","font-size":11.5,fill:P.text});
    nm.textContent=r.label; svg.appendChild(nm);
    const sub=el("text",{x:x+bw/2,y:T+ih+33,"text-anchor":"middle","font-size":10.5,fill:P.muted});
    sub.textContent=r.sub; svg.appendChild(sub);
  });
  // One title saying what the bars count, and one rotated axis label saying what
  // the numbers are. The old single line tried to be both and was neither.
  const ttl=el("text",{x:L,y:16,"font-size":13,fill:P.text,"font-weight":600});
  ttl.textContent="Who wrote the stories decides whether the emotion vectors work at all";
  svg.appendChild(ttl);
  // the success anchor, drawn rather than left in the prose: nothing we built
  // came close to it, and the old 0-10 axis hid that
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(20),y2:Y(20),stroke:P.green,
    "stroke-dasharray":"4 3","stroke-width":1.4}));
  const gl=el("text",{x:W-R-2,y:Y(20)+12,"text-anchor":"end","font-size":10,fill:P.green});
  gl.textContent="20 = every tested layer works"; svg.appendChild(gl);
  const yl=el("text",{x:16,y:T+ih/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 16 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="layers that work, out of 20 tested"; svg.appendChild(yl);
  host.appendChild(svg);
}

/* dose-response: how many stories per emotion you actually need */
function drawDose(){
  const host=document.getElementById("doseChart"); host.innerHTML="";
  const ns=Object.keys(D.dose), W=380,H=190,L=52,R=14,T=18,B=42;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const iw=W-L-R, ih=H-T-B;
  const X=i=>L+(i/(ns.length-1))*iw, Y=v=>T+ih-(v/20)*ih;
  // This chart shares its y-scale with the bar chart beside it but used to draw
  // no axis at all, so its dots floated free of the "9 of 20" they refer to.
  [0,10,20].forEach(v=>{
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:P.border}));
    const t=el("text",{x:L-6,y:Y(v)+3.5,"text-anchor":"end","font-size":9.5,fill:P.muted});
    t.textContent=v; svg.appendChild(t);
  });
  const yl=el("text",{x:14,y:T+ih/2,"font-size":9.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+ih/2})`,"text-anchor":"middle"});
  yl.textContent="layers that work, of 20"; svg.appendChild(yl);
  const src=el("text",{x:L,y:H-3,"font-size":8.5,fill:P.muted});
  src.textContent="notebooks/07_generator_lineages.ipynb (E12)"; svg.appendChild(src);
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:Y(9),y2:Y(9),stroke:P.green,
    "stroke-dasharray":"4 3"}));
  // left-anchored: right-anchored, the trailing "9" sat under the last marker
  const cl=el("text",{x:L+4,y:Y(9)-5,"font-size":9.5,fill:P.green});
  cl.textContent="best this writer ever reaches: 9 of 20"; svg.appendChild(cl);
  let d=""; ns.forEach((n,i)=>{d+=(i?"L":"M")+X(i)+","+Y(D.dose[n]);});
  svg.appendChild(el("path",{d,fill:"none",stroke:P.navy,"stroke-width":2.2}));
  ns.forEach((n,i)=>{
    const c=el("circle",{cx:X(i),cy:Y(D.dose[n]),r:6,fill:P.navy});
    tipOn(c,`<b>${n} stories per emotion</b><span class="t-sub">gives ${D.dose[n]} working layers, `+
      `averaged over 5 random draws of that many stories. The most this writer ever reaches is 9.</span>`);
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
`W = 16                       # window size, fixed before any scoring
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
`B = 10_000                    # shuffles, count fixed in advance

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
/* Button label AND figure title come from here. The raw keys are the notebook's
   own names and carry jargon ("unablated", "RSA"); they never reach the page. */
const RSA_LABEL={"instruct RSA (unablated)":"chat-tuned model",
  "base RSA (unablated)":"plain model",
  "instruct RSA (top component removed)":"chat-tuned, biggest axis subtracted",
  "cross-model RSA: instruct vs base":"plain vs. chat-tuned"};
/* the one-line reminder under the buttons: what THIS view is for */
const RSA_NOTE={"instruct RSA (unablated)":"do the chat-tuned model's layers agree with each other?",
  "base RSA (unablated)":"the control: the same question on the plain model",
  "instruct RSA (top component removed)":"the same layers, with only the mystery axis removed",
  "cross-model RSA: instruct vs base":"one model on each axis — what did chat-tuning change?"};
let rsaKey=RSA_KEYS[0];
function drawRsa(){
  const host=document.getElementById("rsaChart"); if(!host) return;
  host.innerHTML="";
  const m=D.rsa[rsaKey], z=m.z, layers=m.layers||z.map((_,i)=>i);
  const n=z.length, W=760,H=452,L=62,T=36,B=52,Rr=150;
  // Only the cross-model view has a different model on each axis, and it is the
  // one view that is NOT symmetric — so orientation has to be on the figure.
  const isCross=rsaKey.indexOf("cross-model")===0;
  const rowName=isCross?"layer of the chat-tuned model"
                       :"layer of the model (same model on both axes)";
  const colName=isCross?"layer of the plain model"
                       :"layer of the model (same model on both axes)";
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%"});
  const size=Math.min((W-L-Rr)/n,(H-T-B)/n);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const v=z[i][j];
    const cell=el("rect",{x:L+j*size,y:T+i*size,width:size+.5,height:size+.5,
      fill:`rgba(29,53,87,${Math.max(0,Math.min(1,v)).toFixed(3)})`});
    tipOn(cell,`<b>${isCross?"chat-tuned":""} layer ${layers[i]} vs ${isCross?"plain":""} `+
      `layer ${layers[j]}</b>: agreement ${v.toFixed(2)}`+
      `<span class="t-sub">1 means these two layers sort the 171 emotions the same way; `+
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
  [["1.0  sort emotions identically",sy+6],
   ["0.5  partly agree",sy+sh/2],
   ["0.0  no agreement at all",sy+sh]].forEach(([t,y])=>{
    const tx=el("text",{x:sx+18,y:y+4,"font-size":10,fill:P.muted}); tx.textContent=t; svg.appendChild(tx);
  });
  const xl=el("text",{x:L+(n*size)/2,y:H-8,"text-anchor":"middle","font-size":10.5,fill:P.muted});
  xl.textContent=colName; svg.appendChild(xl);
  const ylab=el("text",{x:14,y:T+(n*size)/2,"font-size":10.5,fill:P.muted,
    transform:`rotate(-90 14 ${T+(n*size)/2})`,"text-anchor":"middle"});
  ylab.textContent=rowName; svg.appendChild(ylab);
  // the title lives INSIDE the svg: an exported png of this chart is otherwise
  // three near-identical matrices with nothing saying which one you are seeing
  const ttl=el("text",{x:L,y:20,"font-size":12.5,fill:P.text,"font-weight":600});
  ttl.textContent=RSA_LABEL[rsaKey]||rsaKey; svg.appendChild(ttl);
  host.appendChild(svg);
  document.getElementById("rsaNote").textContent=RSA_NOTE[rsaKey]||"";
}
(function(){
  const host=document.getElementById("rsaBtns"); if(!host) return;
  RSA_KEYS.forEach((k,i)=>{
    const b=document.createElement("button");
    b.className="seg"+(i===0?" on":""); b.textContent=RSA_LABEL[k]||k;
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
// the element holding each chart's live state, to carry into the expanded view
const FIG_NOTE={pcChart:"pcVerdict", gridChart:"gridNote", lineChart:"tokLabel",
  ternChart:"tokLabel", emoChart:"emoNote", rsaChart:"rsaNote"};
function addModalLine(text,marginBottom){
  const d=document.createElement("div");
  d.className="mono muted";
  d.style.cssText=`font-size:13px;margin:${marginBottom==="0"?"12px 0 0":"0 0 "+marginBottom}`;
  d.textContent=text; MODAL_BODY.appendChild(d);
}
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
    // A chart's caption sits in a sibling div and its live state (which token,
    // which layer, which probe bank) in the controls row - both outside the svg.
    // Cloning the svg alone dropped them, so an expanded figure arrived with no
    // title and no way to tell which state you were looking at.
    const cap=host.previousElementSibling;
    if(cap && cap.classList && cap.classList.contains("mono")) addModalLine(cap.textContent,"12px");
    MODAL_BODY.appendChild(svg.cloneNode(true));
    const note=FIG_NOTE[id] && document.getElementById(FIG_NOTE[id]);
    if(note && note.textContent.trim()) addModalLine(note.textContent,"0");
    MODAL.classList.add("on");
  };
  parent.appendChild(b);
});

drawPCs("base"); drawGrid(); drawStory(); drawEmo(); drawLayers(); drawLineage(); drawDose(); drawRsa();
</script>
</body></html>"""

if __name__ == "__main__":
    out = HERE / "index.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
