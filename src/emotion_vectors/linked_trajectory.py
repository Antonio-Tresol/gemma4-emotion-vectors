"""Q3 linked trajectory view: one self-contained figure where the cosine plot,
the emotion triangle, and the story's own text are wired together.

``linked_trajectory_html`` returns a single HTML/JS blob (wrap it in
``IPython.display.HTML``) with linked controls in one output:

- **Bidirectional hover.** Hover a point on the lines plot *or* a point on the
  emotion-triangle (ternary) plot and the matching token lights up in the text
  below; hover a token in the text and both plots mark that token (a vertical
  cue on the lines plot, a ring on the triangle). The highlight unit is the
  *token* (Gemma subwords, e.g. ``'Kw'`` ``'ame'``) — what the trajectory
  samples; concatenating the token spans reconstructs the story.
- **Emotion-opacity toggle.** Buttons shade each token's background by one
  emotion's per-token cosine (min-max normalized within the shown layer), so
  the text becomes a heat-over-words for the selected emotion.
- **Layer slider.** The native Plotly layer slider on the lines plot drives the
  lines, the triangle, *and* the text shading together.

Everything is vanilla JS attached to embedded Plotly figures (Plotly.js inlined
into the iframe): no ipywidgets, no live kernel, so it survives a
statically-viewed or exported notebook — the philosophy shared with
``interactive`` and ``trajectory_plots``.

Why an iframe: the wiring is custom JS, and inline ``<script>`` in a
``display(HTML(...))`` output does NOT execute in JupyterLab (its HTML renderer
inserts via ``innerHTML``, and browsers never run scripts inserted that way). A
sandboxed ``<iframe srcdoc>`` is a real document the browser parses and runs, so
the linked interactivity works the same in JupyterLab, VS Code, and exported
HTML. Plotly.js is inlined rather than loaded from a CDN so a host with a strict
content-security policy (VS Code's output webview) can't blank the frame by
blocking an external ``<script src>``.
"""

from __future__ import annotations

import html as htmllib
import json
import uuid
from string import Template

import numpy as np
import plotly.graph_objects as go

from emotion_vectors.trajectory_plots import barycentric

SERIES_COLORS = ("#6366f1", "#dc2626", "#d97706")  # one palette for lines, triangle, AND text
TEXT_COLOR = "#000000"  # black text on the enforced white panel (theme-independent)

_LINES_HEIGHT = 500  # plot + native layer slider
_TERN_HEIGHT = 580
_CHROME = 154  # controls + caption + padding

# JS wiring. $-placeholders (string.Template) so the many literal {} in the JS
# body are untouched; figure HTML is concatenated separately, never through
# Template or an f-string, so its JSON braces are safe too.
_JS = Template(
    """
(function() {
  var LAYERS = $layers;
  var OPAC = $opacity;            // [layerIdx][token][emotion] in [0,1]
  var RGB = $rgb;                 // [[r,g,b], ...] per emotion
  var HAS_TERN = $has_tern;
  var BARY_PATH = $bary_path;     // [layerIdx][3][token]  (a,b,c) or null
  var BARY_STARTS = $bary_starts; // [layerIdx][3][phaseStart] or null
  var PHASE = $phase_starts;      // token index of each phase start
  var YRANGE = $yrange;           // [lo, hi] for the lines highlight cue
  var linesId = "$lines_id", ternId = "$tern_id", textId = "$text_id", ctrlId = "$ctrl_id";
  var HL_LINE = $hl_line, HL_TERN = $hl_tern;  // highlight trace indices
  var curLayer = $default_idx, curEmo = 0;
  var spans = null, linesDiv = null, ternDiv = null, hoverTok = null;

  function paint() {
    if (!spans) return;
    for (var i = 0; i < spans.length; i++) {
      if (curEmo < 0) { spans[i].style.backgroundColor = "transparent"; continue; }
      var a = OPAC[curLayer][i][curEmo], c = RGB[curEmo];
      spans[i].style.backgroundColor = "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + (0.85 * a).toFixed(3) + ")";
    }
  }

  function markSpan(i, on) {
    if (!spans || i < 0 || i >= spans.length) return;
    spans[i].style.outline = on ? "2px solid #111" : "none";
  }

  function showHL(i) {
    try { Plotly.restyle(linesDiv, {x: [[i, i]], y: [[YRANGE[0], YRANGE[1]]]}, [HL_LINE]); } catch (e) {}
    if (HAS_TERN && ternDiv) {
      try {
        Plotly.restyle(ternDiv, {
          a: [[BARY_PATH[curLayer][0][i]]],
          b: [[BARY_PATH[curLayer][1][i]]],
          c: [[BARY_PATH[curLayer][2][i]]]
        }, [HL_TERN]);
      } catch (e) {}
    }
  }

  function hideHL() {
    try { Plotly.restyle(linesDiv, {x: [[]], y: [[]]}, [HL_LINE]); } catch (e) {}
    if (HAS_TERN && ternDiv) {
      try { Plotly.restyle(ternDiv, {a: [[]], b: [[]], c: [[]]}, [HL_TERN]); } catch (e) {}
    }
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
    linesDiv = document.getElementById(linesId);
    spans = document.querySelectorAll("#" + textId + " span.tok");
    if (!linesDiv || !window.Plotly || !linesDiv.on || spans.length === 0) return false;
    if (HAS_TERN) { ternDiv = document.getElementById(ternId); if (!ternDiv || !ternDiv.on) return false; }

    // point -> word (lines)
    linesDiv.on("plotly_hover", function(d) { var i = d.points[0].pointNumber; markSpan(i, true); hoverTok = i; });
    linesDiv.on("plotly_unhover", function() { if (hoverTok != null) { markSpan(hoverTok, false); hoverTok = null; } });

    // layer slider (native, on lines) -> triangle restyle + text opacity
    linesDiv.on("plotly_sliderchange", function(e) {
      var idx = LAYERS.indexOf(parseInt(e.step.label, 10));
      if (idx < 0) return;
      curLayer = idx;
      if (HAS_TERN && ternDiv) {
        try {
          Plotly.restyle(ternDiv, {
            a: [BARY_PATH[idx][0], BARY_STARTS[idx][0]],
            b: [BARY_PATH[idx][1], BARY_STARTS[idx][1]],
            c: [BARY_PATH[idx][2], BARY_STARTS[idx][2]]
          }, [0, 1]);
        } catch (err) {}
      }
      paint();
    });

    // point -> word (triangle): path points map 1:1 to tokens; the phase-start
    // marker trace maps through PHASE
    if (HAS_TERN) {
      ternDiv.on("plotly_hover", function(d) {
        var p = d.points[0];
        var i = (p.curveNumber === 1) ? PHASE[p.pointNumber] : p.pointNumber;
        markSpan(i, true); hoverTok = i;
      });
      ternDiv.on("plotly_unhover", function() { if (hoverTok != null) { markSpan(hoverTok, false); hoverTok = null; } });
    }

    // word -> point (both plots)
    for (var i = 0; i < spans.length; i++) {
      (function(i) {
        spans[i].addEventListener("mouseenter", function() { markSpan(i, true); showHL(i); });
        spans[i].addEventListener("mouseleave", function() { markSpan(i, false); hideHL(); });
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
    selectEmotion(0, buttons);
    return true;
  }

  var tries = 0;
  var iv = setInterval(function() { if (wire() || ++tries > 100) clearInterval(iv); }, 100);
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


def _bary_payload(
    cos_by_layer: dict[int, np.ndarray], layers: list[int], phase_starts: list[int]
) -> tuple[list, list]:
    """Barycentric (softmax) coords per layer for the triangle: (path, starts),
    each [layerIdx][abc][point], so the JS can restyle the triangle on a layer
    change and place the word->point highlight from stored data alone."""
    path, starts = [], []
    for layer in layers:
        b = barycentric(np.asarray(cos_by_layer[layer], dtype=np.float32))  # [tokens, 3]
        path.append([b[:, j].round(5).tolist() for j in range(3)])
        s = b[phase_starts]
        starts.append([s[:, j].round(5).tolist() for j in range(3)])
    return path, starts


def _lines_figure(
    cos_by_layer: dict[int, np.ndarray],
    emotions: list[str],
    phase_starts: list[int],
    layers: list[int],
    default_layer: int,
    tokens: list[str],
    colors: tuple[str, ...],
    yrange: tuple[float, float],
    title: str,
) -> go.Figure:
    """Cosine-vs-token lines with a layer slider, plus a hidden highlight trace
    (index len(emotions)) the word->point hover repositions as a vertical cue."""
    n = len(cos_by_layer[default_layer])
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
    fig.add_trace(  # highlight cue (empty until a word is hovered)
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": "#888", "width": 1, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
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
        yaxis_range=list(yrange),
        width=940,
        height=440,
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


def _ternary_figure(
    cos_by_layer: dict[int, np.ndarray],
    emotions: list[str],
    phase_starts: list[int],
    default_layer: int,
    tokens: list[str],
) -> go.Figure:
    """The emotion triangle at the default layer: path (token-colored), phase
    starts, and a hidden highlight ring (trace 2) the word->point hover moves."""
    b = barycentric(np.asarray(cos_by_layer[default_layer], dtype=np.float32))
    n = len(b)
    hover = [f"t={t} {tokens[t]!r}" for t in range(n)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterternary(
            a=b[:, 0],
            b=b[:, 1],
            c=b[:, 2],
            mode="lines+markers",
            marker={"size": 4, "color": list(range(n)), "colorscale": "Viridis"},
            line={"width": 1, "color": "rgba(120,120,120,0.4)"},
            text=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    s = b[phase_starts]
    fig.add_trace(
        go.Scatterternary(
            a=s[:, 0],
            b=s[:, 1],
            c=s[:, 2],
            mode="markers",
            marker={"size": 12, "symbol": "diamond", "color": "black"},
            text=[f"phase start t={t}" for t in phase_starts],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_trace(  # highlight ring (empty until a word is hovered)
        go.Scatterternary(
            a=[],
            b=[],
            c=[],
            mode="markers",
            marker={"size": 16, "symbol": "circle-open", "color": "#111", "line": {"width": 3}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        ternary={
            "aaxis": {"title": emotions[0]},
            "baxis": {"title": emotions[1]},
            "caxis": {"title": emotions[2]},
        },
        title="Emotion phase space (softmax display transform, T=28)",
        showlegend=False,
        width=620,
        height=560,
        margin={"t": 40, "b": 10},
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
    show_ternary: bool,
) -> str:
    """Controls + Plotly figure(s) + text + wiring script, as one HTML fragment
    (rendered inside the iframe document by ``linked_trajectory_html``)."""
    layers = sorted(cos_by_layer)
    lo = min(float(np.min(c)) for c in cos_by_layer.values())
    hi = max(float(np.max(c)) for c in cos_by_layer.values())
    pad = 0.1 * (hi - lo)
    yrange = (lo - pad, hi + pad)

    lines_id = "lines_" + uuid.uuid4().hex[:10]
    tern_id = "tern_" + uuid.uuid4().hex[:10]
    text_id = "text_" + uuid.uuid4().hex[:10]
    ctrl_id = "ctrl_" + uuid.uuid4().hex[:10]

    fig_lines = _lines_figure(
        cos_by_layer, emotions, phase_starts, layers, default_layer, tokens, colors, yrange, title
    )
    # Inline Plotly.js once (on the lines figure); the triangle reuses it.
    lines_html = fig_lines.to_html(include_plotlyjs=True, full_html=False, div_id=lines_id)
    hl_line = len(emotions)  # highlight trace index on the lines figure

    if show_ternary:
        fig_tern = _ternary_figure(cos_by_layer, emotions, phase_starts, default_layer, tokens)
        tern_html = fig_tern.to_html(include_plotlyjs=False, full_html=False, div_id=tern_id)
        bary_path, bary_starts = _bary_payload(cos_by_layer, layers, phase_starts)
    else:
        tern_html = ""
        bary_path = bary_starts = None

    rgb = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors[:3]]
    js = _JS.substitute(
        layers=json.dumps(layers),
        opacity=json.dumps(_opacity(cos_by_layer, layers)),
        rgb=json.dumps(rgb),
        has_tern=json.dumps(show_ternary),
        bary_path=json.dumps(bary_path),
        bary_starts=json.dumps(bary_starts),
        phase_starts=json.dumps(list(phase_starts)),
        yrange=json.dumps([round(yrange[0], 5), round(yrange[1], 5)]),
        lines_id=lines_id,
        tern_id=tern_id,
        text_id=text_id,
        ctrl_id=ctrl_id,
        hl_line=hl_line,
        hl_tern=2,
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
        f"padding:2px 8px;border:1px solid #ccc;border-radius:4px;background:#fff;color:#222;"
        f'border-bottom:3px solid {colors[k % len(colors)]}">{htmllib.escape(name)}</button>'
        for k, name in enumerate(emotions)
    )
    buttons += (
        '<button class="emo" data-emo="-1" style="cursor:pointer;padding:2px 8px;'
        'border:1px solid #ccc;border-radius:4px;background:#fff;color:#222">off</button>'
    )

    controls = (
        f'<div id="{ctrl_id}" style="font-family:sans-serif;font-size:13px;margin:6px 0">'
        f'<span style="color:#aaa;margin-right:8px">shade text by emotion:</span>{buttons}</div>'
    )
    text_div = (
        f'<div id="{text_id}" style="font-family:ui-monospace,monospace;line-height:2.0;'
        f"color:{TEXT_COLOR};background:#ffffff;max-width:940px;white-space:pre-wrap;"
        f'padding:10px 12px;border-radius:4px">{span_html}</div>'
    )
    triangle_note = " (or a point on the triangle)" if show_ternary else ""
    caption = (
        '<div style="font-size:11px;color:#9a9a9a;max-width:940px;margin-top:2px">'
        f"How to read: hover a point on the lines plot{triangle_note} to light its token in the "
        "text, or hover a token to mark that point on the plot(s). The emotion buttons shade "
        "every token by how strongly that emotion reads there (min-max within the shown layer). "
        "The layer slider moves the plot(s) and the text shading together.</div>"
    )

    return (
        '<div style="max-width:960px">'
        + controls
        + lines_html
        + tern_html
        + text_div
        + caption
        + "<script>"
        + js
        + "</script></div>"
    )


def _iframe_height(tokens: list[str], show_ternary: bool) -> int:
    """Over-estimate a fitting iframe height so nothing clips: figure + chrome,
    plus a text estimate that low-balls chars-per-line (so line count high-balls).
    If still short, the iframe body scrolls rather than truncating."""
    text = "".join(tokens)
    lines = -(-len(text) // 90) + text.count("\n") + 3
    chrome = _LINES_HEIGHT + _CHROME + (_TERN_HEIGHT if show_ternary else 0)
    return min(2600, chrome + max(6, lines) * 30)


def linked_trajectory_html(
    cos_by_layer: dict[int, np.ndarray],
    tokens: list[str],
    emotions: list[str],
    phase_starts: list[int],
    *,
    default_layer: int = 33,
    title: str = "",
    colors: tuple[str, ...] = SERIES_COLORS,
    show_ternary: bool = False,
) -> str:
    """Build the linked plot(s)+text view for one story as a sandboxed iframe.
    Wrap the return value in ``IPython.display.HTML`` to render it.

    ``cos_by_layer`` maps each captured layer to a ``[tokens, 3]`` cosine array
    (column j = phase j's emotion), exactly the dict the notebook already builds
    for the scrubbers; ``tokens`` are the decoded per-token strings
    (``story_tokens`` in the notebook). The three emotions must be in
    phase/column order. Set ``show_ternary=True`` to add the emotion-triangle
    plot, cross-wired to the same text and layer slider.
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
        show_ternary=show_ternary,
    )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;padding:6px;overflow:auto;"
        "font-family:sans-serif}</style></head><body>" + inner + "</body></html>"
    )
    srcdoc = htmllib.escape(doc, quote=True)
    height = _iframe_height(tokens, show_ternary)
    return (
        f'<iframe sandbox="allow-scripts" srcdoc="{srcdoc}" loading="lazy" '
        f'style="width:100%;max-width:1000px;height:{height}px;border:1px solid #eee;'
        f'border-radius:6px"></iframe>'
    )
