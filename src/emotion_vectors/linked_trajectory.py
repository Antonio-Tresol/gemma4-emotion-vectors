"""Q3 linked trajectory view: one self-contained figure where the cosine plot
and the story's own text are wired together.

``linked_trajectory_html`` returns a single HTML/JS blob (wrap it in
``IPython.display.HTML``) that puts three linked controls in one output:

- **Bidirectional hover.** Hover a point on the cosine plot and the matching
  token lights up in the text below; hover a token in the text and the plot
  shows the hover readout at that token. The highlight unit is the *token*
  (Gemma subwords, e.g. ``'Kw'`` ``'ame'``), because that is exactly what the
  trajectory samples — concatenating the token spans reconstructs the story.
- **Emotion-opacity toggle.** Buttons paint each token's background with an
  opacity set by one emotion's per-token cosine (min-max normalized within the
  shown layer), so the text becomes a heat-over-words for the selected emotion.
- **Layer slider.** The native Plotly layer slider drives the plot lines *and*
  the text opacity together, so scrubbing layers re-reads both views at once.

Everything is vanilla JS attached to an embedded Plotly figure (Plotly.js from
CDN via ``to_html``): no ipywidgets, no live kernel, so it survives a
statically-viewed or exported notebook — the module philosophy shared with
``interactive`` and ``trajectory_plots``. Python-side ``FigureWidget`` hover
callbacks were the alternative and were rejected for exactly that reason.
"""

from __future__ import annotations

import html as htmllib
import json
import uuid
from string import Template

import numpy as np
import plotly.graph_objects as go

SERIES_COLORS = ("#6366f1", "#dc2626", "#d97706")  # one palette for lines AND text

# JS wiring. $-placeholders (string.Template) so the many literal {} in the JS
# body are left untouched; the figure HTML is concatenated separately, never
# passed through Template or an f-string, so its JSON braces are safe too.
_JS = Template(
    """
(function() {
  var LAYERS = $layers;          // model layer number per slider step
  var OPAC = $opacity;           // [layerIdx][token][emotion] in [0,1]
  var RGB = $rgb;                // [[r,g,b], ...] one per emotion
  var plotId = "$plot_id";
  var textId = "$text_id";
  var ctrlId = "$ctrl_id";
  var curLayer = $default_idx;   // index into LAYERS
  var curEmo = 0;                // emotion index, or -1 for "off"
  var spans = null, plotDiv = null, hoverTok = null;

  function paint() {
    if (!spans) return;
    for (var i = 0; i < spans.length; i++) {
      if (curEmo < 0) { spans[i].style.backgroundColor = "transparent"; continue; }
      var a = OPAC[curLayer][i][curEmo];
      var c = RGB[curEmo];
      spans[i].style.backgroundColor = "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + (0.85 * a).toFixed(3) + ")";
    }
  }

  function markSpan(i, on) {
    if (!spans || i < 0 || i >= spans.length) return;
    spans[i].style.outline = on ? "2px solid #111" : "none";
  }

  function selectEmotion(e, buttons) {
    curEmo = e;
    for (var k = 0; k < buttons.length; k++) {
      var active = parseInt(buttons[k].dataset.emo, 10) === e;
      buttons[k].style.fontWeight = active ? "700" : "400";
      buttons[k].style.outline = active ? "2px solid #111" : "none";
    }
    paint();
  }

  function wire() {
    plotDiv = document.getElementById(plotId);
    spans = document.querySelectorAll("#" + textId + " span.tok");
    if (!plotDiv || !window.Plotly || !plotDiv.on || spans.length === 0) return false;

    // point -> word
    plotDiv.on("plotly_hover", function(d) {
      var i = d.points[0].pointNumber;
      markSpan(i, true);
      hoverTok = i;
    });
    plotDiv.on("plotly_unhover", function() {
      if (hoverTok != null) { markSpan(hoverTok, false); hoverTok = null; }
    });

    // layer slider -> plot (native) + text opacity (here)
    plotDiv.on("plotly_sliderchange", function(e) {
      var idx = LAYERS.indexOf(parseInt(e.step.label, 10));
      if (idx >= 0) { curLayer = idx; paint(); }
    });

    // word -> point
    for (var i = 0; i < spans.length; i++) {
      (function(i) {
        spans[i].addEventListener("mouseenter", function() {
          markSpan(i, true);
          try {
            Plotly.Fx.hover(plotDiv, [
              {curveNumber: 0, pointNumber: i},
              {curveNumber: 1, pointNumber: i},
              {curveNumber: 2, pointNumber: i}
            ]);
          } catch (err) {}
        });
        spans[i].addEventListener("mouseleave", function() {
          markSpan(i, false);
          try { Plotly.Fx.unhover(plotDiv); } catch (err) {}
        });
      })(i);
    }

    // emotion opacity toggle
    var buttons = document.querySelectorAll("#" + ctrlId + " button.emo");
    for (var b = 0; b < buttons.length; b++) {
      (function(b) {
        buttons[b].addEventListener("click", function() {
          selectEmotion(parseInt(buttons[b].dataset.emo, 10), buttons);
        });
      })(b);
    }
    selectEmotion(0, buttons);  // start on the first emotion so the feature is visible
    return true;
  }

  var tries = 0;
  var iv = setInterval(function() {
    if (wire() || ++tries > 100) clearInterval(iv);
  }, 100);
})();
"""
)


def _opacity(cos_by_layer: dict[int, np.ndarray], layers: list[int]) -> list:
    """Per-(layer, emotion) min-max normalized cosine in [0,1] — 'where is this
    emotion strongest in this story at this layer'. Flat columns map to 0."""
    out = []
    for layer in layers:
        cos = np.asarray(cos_by_layer[layer], dtype=np.float32)
        cols = []
        for e in range(cos.shape[1]):
            col = cos[:, e]
            lo, hi = float(col.min()), float(col.max())
            norm = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
            cols.append(norm)
        out.append(np.stack(cols, axis=1).round(4).tolist())  # [tokens, emotions]
    return out


def _figure(
    cos_by_layer: dict[int, np.ndarray],
    emotions: list[str],
    phase_starts: list[int],
    layers: list[int],
    default_layer: int,
    tokens: list[str],
    colors: tuple[str, ...],
    title: str,
) -> go.Figure:
    """The cosine-vs-token lines with a layer slider (same shape as
    interactive.trajectory_lines_scrubber, rebuilt here so the trace colors
    match the text palette exactly and the div id is ours to wire)."""
    n = len(cos_by_layer[default_layer])
    lo = min(float(np.min(c)) for c in cos_by_layer.values())
    hi = max(float(np.max(c)) for c in cos_by_layer.values())
    pad = 0.1 * (hi - lo)
    hover = [f"t={t} {tokens[t]!r}" for t in range(n)]
    fig = go.Figure()
    for k, name in enumerate(emotions):
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=np.asarray(cos_by_layer[default_layer])[:, k].tolist(),
                name=name,
                line={"color": colors[k % len(colors)], "width": 2},
                text=hover,
                hoverinfo="text+y+name",
            )
        )
    for start in phase_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="gray")
    steps = [
        {
            "method": "restyle",
            "label": str(layer),
            "args": [{"y": [np.asarray(cos_by_layer[layer])[:, k].tolist() for k in range(3)]}],
        }
        for layer in layers
    ]
    fig.update_layout(
        title=title or "Per-token probe trajectories (linked to the text below)",
        xaxis_title="token",
        yaxis_title="centered cosine",
        yaxis_range=[lo - pad, hi + pad],
        width=940,
        height=460,
        margin={"b": 10},
        sliders=[
            {
                "active": layers.index(default_layer),
                "currentvalue": {"prefix": "layer "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )
    return fig


def _inner_html(
    cos_by_layer: dict[int, np.ndarray],
    tokens: list[str],
    emotions: list[str],
    phase_starts: list[int],
    *,
    default_layer: int,
    title: str,
    colors: tuple[str, ...],
) -> str:
    """The controls + Plotly figure + text + wiring script, as one HTML
    fragment. Rendered inside an iframe document by ``linked_trajectory_html``
    (see there for why the iframe is required)."""
    layers = sorted(cos_by_layer)
    if default_layer not in cos_by_layer:
        raise ValueError(f"default_layer {default_layer} not in {layers}")
    if len(emotions) != 3:
        raise ValueError("linked view expects exactly three emotions (the triple)")

    plot_id = "plot_" + uuid.uuid4().hex[:10]
    text_id = "text_" + uuid.uuid4().hex[:10]
    ctrl_id = "ctrl_" + uuid.uuid4().hex[:10]

    fig = _figure(
        cos_by_layer, emotions, phase_starts, layers, default_layer, tokens, colors, title
    )
    # Inline Plotly.js rather than loading it from a CDN: the figure lives in a
    # sandboxed iframe, and a notebook host with a strict content-security
    # policy (VS Code's output webview) can block an external <script src>,
    # blanking the iframe. Inlining makes it fully self-contained — nothing
    # external to block, and it works offline too.
    fig_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id=plot_id)

    rgb = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors[:3]]
    js = _JS.substitute(
        layers=json.dumps(layers),
        opacity=json.dumps(_opacity(cos_by_layer, layers)),
        rgb=json.dumps(rgb),
        plot_id=plot_id,
        text_id=text_id,
        ctrl_id=ctrl_id,
        default_idx=layers.index(default_layer),
    )

    phase_set = set(phase_starts)
    span_html = "".join(
        f'<span class="tok" data-tok="{i}" style="'
        + ("border-left:2px solid #999;margin-left:1px;" if (i in phase_set and i > 0) else "")
        + f'">{htmllib.escape(tk)}</span>'
        for i, tk in enumerate(tokens)
    )

    buttons = "".join(
        f'<button class="emo" data-emo="{k}" style="cursor:pointer;margin-right:6px;'
        f"padding:2px 8px;border:1px solid #ccc;border-radius:4px;background:#fff;"
        f'border-bottom:3px solid {colors[k % len(colors)]}">{htmllib.escape(name)}</button>'
        for k, name in enumerate(emotions)
    )
    buttons += (
        '<button class="emo" data-emo="-1" style="cursor:pointer;padding:2px 8px;'
        'border:1px solid #ccc;border-radius:4px;background:#fff">off</button>'
    )

    controls = (
        f'<div id="{ctrl_id}" style="font-family:sans-serif;font-size:13px;margin:6px 0">'
        f'<span style="color:#555;margin-right:8px">shade text by emotion:</span>{buttons}</div>'
    )
    text_div = (
        f'<div id="{text_id}" style="font-family:ui-monospace,monospace;line-height:2.0;'
        f'max-width:940px;white-space:pre-wrap;padding:10px 4px">{span_html}</div>'
    )
    caption = (
        '<div style="font-size:11px;color:gray;max-width:940px;margin-top:2px">'
        "How to read: hover a point to light its token in the text, or hover a token to read "
        "the plot at that point. The emotion buttons shade every token by how strongly that "
        "emotion reads there (min-max within the shown layer). The layer slider moves the plot "
        "and the text shading together.</div>"
    )

    return (
        '<div style="max-width:960px">'
        + controls
        + fig_html
        + text_div
        + caption
        + "<script>"
        + js
        + "</script></div>"
    )


def _iframe_height(tokens: list[str]) -> int:
    """Over-estimate a fitting iframe height so nothing clips: plot + controls +
    caption chrome, plus a text estimate that low-balls chars-per-line (so the
    line count high-balls). If the estimate is still short the iframe body
    scrolls rather than truncating."""
    text = "".join(tokens)
    lines = -(-len(text) // 90) + text.count("\n") + 3  # ceil(chars/90) + breaks + slack
    return min(1500, 470 + 60 + max(6, lines) * 30)


def linked_trajectory_html(
    cos_by_layer: dict[int, np.ndarray],
    tokens: list[str],
    emotions: list[str],
    phase_starts: list[int],
    *,
    default_layer: int = 33,
    title: str = "",
    colors: tuple[str, ...] = SERIES_COLORS,
) -> str:
    """Build the linked plot+text view for one story as a sandboxed iframe.
    Wrap the return value in ``IPython.display.HTML`` to render it.

    ``cos_by_layer`` maps each captured layer to a ``[tokens, 3]`` cosine array
    (column j = phase j's emotion), exactly the dict the notebook already
    builds for the scrubbers; ``tokens`` are the decoded per-token strings
    (``story_tokens`` in the notebook). The three emotions must be in
    phase/column order.

    Why an iframe: the wiring is custom JS, and inline ``<script>`` in a
    ``display(HTML(...))`` output does NOT execute in JupyterLab (its HTML
    renderer inserts via ``innerHTML``, and browsers never run scripts inserted
    that way). A sandboxed ``<iframe srcdoc>`` is a real document the browser
    parses and runs, so the linked interactivity works the same in JupyterLab,
    VS Code, and exported/static HTML — the cross-environment portability the
    'self-contained JS' choice is about.
    """
    if len(emotions) != 3:
        raise ValueError("linked view expects exactly three emotions (the triple)")
    inner = _inner_html(
        cos_by_layer,
        tokens,
        emotions,
        phase_starts,
        default_layer=default_layer,
        title=title,
        colors=colors,
    )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;padding:6px;overflow:auto;"
        "font-family:sans-serif}</style></head><body>" + inner + "</body></html>"
    )
    srcdoc = htmllib.escape(doc, quote=True)
    height = _iframe_height(tokens)
    return (
        f'<iframe sandbox="allow-scripts" srcdoc="{srcdoc}" loading="lazy" '
        f'style="width:100%;max-width:1000px;height:{height}px;border:1px solid #eee;'
        f'border-radius:6px"></iframe>'
    )
