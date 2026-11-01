"""Cricket Batting AI Coach - Gradio web app.

    python app.py            # then open http://127.0.0.1:7860

A thin UI over ``src.pipeline.analyze``: upload or record a side-on clip of one
shot, pick the shot type, get a score out of 100, a skeleton video, key-frame
images and one or two fixes with drills. All analysis lives in ``src/`` - nothing
here depends on Gradio except this file.
"""
from __future__ import annotations

import html

import gradio as gr

from src.pipeline import SHOT_LABELS, SHOT_TYPES, AnalysisResult, analyze

# ----------------------------------------------------------------------------- text
TITLE = "Cricket Batting AI Coach"
INTRO = ("Film **one batting shot** from the side (3-6 seconds), upload it, and get a score, "
         "a skeleton overlay of your technique and the one or two things to work on first.")
FILMING_TIPS = """
**For the best result**
- Put the camera **side-on**: level with the batter, looking straight across the pitch.
- Keep the **whole body in view**, head to feet, for the whole shot. Move the camera back if needed.
- Hold the phone **still** (a tripod or a prop helps) and film in good light.
- Only the batter in shot. Start filming a second **before** the shot and stop after the follow-through.
- One shot per clip, **3-6 seconds**.
"""
DISCLAIMER = ("Demo only: the scoring rules are placeholders until a coach has set them, so treat the "
              "numbers as a guide, not a verdict.")
STAGE_ORDER = ["Checking the video", "Finding the batter", "Finding the key moments",
               "Checking the clip quality", "Scoring the shot", "Drawing the skeleton video"]

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px 20px;
        background: var(--background-fill-secondary); color: var(--body-text-color); }
.score-card { text-align: center; }
.score-number { font-size: clamp(3.5rem, 18vw, 6rem); font-weight: 800; line-height: 1; }
.score-number small { font-size: 0.3em; font-weight: 600; opacity: .65; }
.score-verdict { font-size: 1.15rem; font-weight: 600; margin-top: 4px; }
.chip { display: inline-block; font-size: .8rem; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--border-color-primary); margin: 8px 4px 0; opacity: .9; }
.note { font-size: .82rem; opacity: .75; margin-top: 10px; }
.bar-row { margin: 12px 0; text-align: left; }
.bar-label { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 4px; }
.bar-track { height: 14px; border-radius: 8px; background: var(--border-color-primary); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 8px; }
.good { background: #2e9e5b; } .ok { background: #d99a1c; } .low { background: #d2493f; }
.fix { text-align: left; margin-bottom: 14px; }
.fix h3 { margin: 0 0 6px; font-size: 1.05rem; }
.fix .drill { margin-top: 8px; padding: 10px 12px; border-left: 4px solid #2e9e5b;
              background: var(--background-fill-primary); border-radius: 6px; }
.fix .measure { font-size: .85rem; opacity: .75; margin-top: 8px; }
.retry { border-color: #d99a1c; text-align: left; }
.retry h2 { margin: 0 0 8px; }
.retry li { margin: 6px 0; }
@media (max-width: 640px) { .card { padding: 14px; } }
"""


# ----------------------------------------------------------------------------- rendering
def _esc(text) -> str:
    return html.escape(str(text))


def score_band(score: float) -> tuple[str, str]:
    """(css class, plain-words verdict) for a 0-100 score."""
    if score >= 85:
        return "good", "Strong technique"
    if score >= 65:
        return "ok", "Getting there"
    return "low", "Needs work"


def confidence_label(confidence: float) -> str:
    return "high" if confidence >= 0.7 else "medium" if confidence >= 0.45 else "low"


def render_summary(result: AnalysisResult) -> str:
    """The big score card with sub-score bars - or the friendly retry/error card."""
    if result.status != "ok":
        return render_retry(result)
    score = result.score
    _, verdict = score_band(score.total)
    bars = []
    for name, sub in score.sub_scores.items():
        if sub.score is None:
            bars.append(f'<div class="bar-row"><div class="bar-label"><span>{_esc(name)}</span>'
                        '<span>not measured</span></div><div class="bar-track"></div></div>')
            continue
        bar_cls, band = score_band(sub.score)
        bars.append(
            f'<div class="bar-row"><div class="bar-label"><span>{_esc(name)}</span>'
            f'<span>{sub.score:.0f} / 100 &middot; {band}</span></div>'
            f'<div class="bar-track" role="progressbar" aria-label="{_esc(name)}" aria-valuemin="0" '
            f'aria-valuemax="100" aria-valuenow="{sub.score:.0f}">'
            f'<div class="bar-fill {bar_cls}" style="width:{max(2.0, sub.score):.0f}%"></div></div></div>')
    shot = SHOT_LABELS.get(result.shot_type, result.shot_type)
    note = f'<div class="note">{_esc(DISCLAIMER)}</div>'
    return (
        f'<div class="card score-card"><div class="chip">{_esc(shot)}</div>'
        f'<div class="score-number">{score.total:.0f}<small>/100</small></div>'
        f'<div class="score-verdict">{verdict}</div>'
        f'<div class="chip">Confidence: {confidence_label(score.confidence)} ({score.confidence:.2f})</div>'
        f'<div style="margin-top:16px">{"".join(bars)}</div>{note}</div>')


def render_retry(result: AnalysisResult) -> str:
    """Friendly 'please film again' card for rejected clips and errors."""
    if result.status == "error":
        return (f'<div class="card retry"><h2>We hit a problem</h2><p>{_esc(result.message)}</p></div>')
    steps = result.quality.instructions if result.quality else [result.message]
    items = "".join(f"<li>{_esc(s)}</li>" for s in steps)
    return (f'<div class="card retry"><h2>Let&rsquo;s try that clip again</h2>'
            f"<p>We couldn&rsquo;t score this one, so no numbers this time:</p><ul>{items}</ul>"
            f'<p class="note">Tip: the filming tips above show what works best.</p></div>')


def render_fixes(result: AnalysisResult) -> str:
    """Top fixes with drills (or a well-done message)."""
    if result.status != "ok":
        return ""
    fixes = result.score.fixes
    if not fixes:
        return ('<div class="card fix"><h3>Nice work</h3><p>Everything we measured is in the target range. '
                "Keep repeating the shot and film again to check it stays consistent.</p></div>")
    cards = []
    for i, f in enumerate(fixes, start=1):
        cards.append(
            f'<div class="card fix"><h3>Fix {i}: {_esc(f.label)}</h3><p>{_esc(f.text)}</p>'
            f'<div class="drill"><strong>Try this drill:</strong> {_esc(f.drill)}</div>'
            f'<div class="measure">You: {f.value:.2f} &middot; Target: {_esc(f.ideal_text)} '
            f"&middot; {_esc(f.category)}</div></div>")
    return "".join(cards)


def render_timing(result: AnalysisResult) -> str:
    t = result.timings
    if not t:
        return ""
    parts = {"pose": "finding the batter", "analysis": "key moments and metrics", "render": "drawing the video"}
    detail = " &middot; ".join(f"{label} {t[key]:.1f} s" for key, label in parts.items() if key in t)
    return (f'<div class="note">Processed in <strong>{t.get("total", 0):.1f} s</strong>'
            f'{" (" + detail + ")" if detail else ""}.</div>')


def metrics_rows(result: AnalysisResult) -> list[list]:
    if not result.metrics:
        return []
    return [[name, round(m.value, 3), round(m.confidence, 2)] for name, m in result.metrics.values.items()]


def rules_rows(result: AnalysisResult) -> list[list]:
    if not result.score:
        return []
    rows = []
    for r in result.score.rule_results:
        rows.append([r.rule.label, r.rule.category, r.status,
                     "" if r.value is None else round(r.value, 3), r.rule.ideal_text,
                     "" if r.score is None else round(r.score, 2), r.rule.weight])
    return rows


def gallery_items(result: AnalysisResult) -> list[tuple[str, str]]:
    moments = result.key_moments
    if not result.ok or not moments:
        return []
    # Short captions: three thumbnails share a half-width column on desktop.
    return [(str(path), label.title()) for label, path in result.outputs.get("key_frames", {}).items()]


# ----------------------------------------------------------------------------- handler
_PROGRESS = gr.Progress()   # Gradio injects the live tracker into parameters defaulting to this


def run_analysis(video, shot_type, progress=_PROGRESS):
    """Gradio handler: returns the values for every output component."""
    empty = ("", None, [], "", "", [], [], None)
    if not video:
        gr.Warning("Please add a video first.")
        return empty

    def on_stage(name: str) -> None:
        index = STAGE_ORDER.index(name) if name in STAGE_ORDER else 0
        progress(index / len(STAGE_ORDER), desc=name)

    try:
        result = analyze(video, shot_type, on_stage=on_stage)
    except Exception as exc:  # noqa: BLE001 - last resort: never show a stack trace to the user
        gr.Warning("Something went wrong while analysing that clip.")
        print(f"analysis failed: {exc!r}")
        error = AnalysisResult("error", shot_type, "Something unexpected went wrong. Please try another clip.")
        return (render_retry(error), None, [], "", "", [], [], None)

    overlay = result.outputs.get("overlay_video")
    return (render_summary(result), str(overlay) if overlay else None, gallery_items(result),
            render_fixes(result), render_timing(result), metrics_rows(result), rules_rows(result),
            result.to_dict())


# ----------------------------------------------------------------------------- layout
def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE, fill_width=False) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Accordion("How to film your shot", open=False):
            gr.Markdown(FILMING_TIPS)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                video = gr.Video(label="Your clip (upload or record)", sources=["upload", "webcam"])
                shot = gr.Dropdown(choices=[(SHOT_LABELS[s], s) for s in SHOT_TYPES], value=SHOT_TYPES[0],
                                   label="Shot type", interactive=True)
                go = gr.Button("Analyse my shot", variant="primary", size="lg")
            with gr.Column(scale=1, min_width=300):
                summary = gr.HTML(label="Result")
                timing = gr.HTML()

        fixes = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                overlay = gr.Video(label="Skeleton overlay", interactive=False)
            with gr.Column(scale=1, min_width=300):
                gallery = gr.Gallery(label="Key moments", columns=3, height="auto", object_fit="contain",
                                     interactive=False)

        with gr.Accordion("Details: measurements and rules", open=False):
            metrics = gr.Dataframe(headers=["Metric", "Value", "Confidence"], label="Raw metrics",
                                   interactive=False, wrap=True)
            rules = gr.Dataframe(headers=["Rule", "Category", "Result", "Value", "Target", "Score", "Weight"],
                                 label="Rule results", interactive=False, wrap=True)
            raw = gr.JSON(label="Full result (JSON)", open=False)

        gr.Markdown(f"<small>{_esc(DISCLAIMER)}</small>")
        go.click(run_analysis, inputs=[video, shot],
                 outputs=[summary, overlay, gallery, fixes, timing, metrics, rules, raw],
                 api_name="analyze")
    demo.queue(max_size=8, default_concurrency_limit=2)
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
