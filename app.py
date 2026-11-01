"""Cricket Batting AI Coach - Gradio web app.

    python app.py            # then open http://127.0.0.1:7860

A thin UI over ``src.pipeline.analyze``: upload or record a side-on clip of one
shot, pick the shot type, get a score out of 100, a skeleton video, key-frame
images and one or two fixes with drills. All analysis lives in ``src/`` - nothing
here depends on Gradio except this file.
"""
from __future__ import annotations

import gradio as gr

from labelling_tab import build_labelling_tab
from src.pipeline import SHOT_LABELS, SHOT_TYPES, AnalysisResult, analyze
from ui_cards import (  # noqa: F401  (re-exported for tests and the labelling tab)
    DISCLAIMER,
    _esc,
    confidence_label,
    gallery_items,
    metrics_rows,
    render_fixes,
    render_retry,
    render_summary,
    render_timing,
    rules_rows,
    score_band,
)

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


# ----------------------------------------------------------------------------- handler
_PROGRESS = gr.Progress()   # Gradio injects the live tracker into parameters defaulting to this


def run_analysis(video: str | None, shot_type: str, progress: gr.Progress = _PROGRESS) -> tuple:
    """Gradio handler: returns the values for every output component."""
    empty = ("", None, [], "", "", [], [], None)
    if not video:
        gr.Warning("Please add a video first.")
        return empty

    def on_stage(name: str) -> None:
        """Show the pipeline's current stage on the Gradio progress bar."""
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
def _analysis_tab() -> None:
    """The main tab: upload a clip, get the score, fixes and overlay."""
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

    go.click(run_analysis, inputs=[video, shot],
             outputs=[summary, overlay, gallery, fixes, timing, metrics, rules, raw],
             api_name="analyze")


def build_app() -> gr.Blocks:
    """Assemble the two-tab Gradio app: analysis and coach labelling."""
    with gr.Blocks(title=TITLE, fill_width=False) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Tabs():
            with gr.Tab("Analyse a shot"):
                _analysis_tab()
            with gr.Tab("Coach labelling"):
                build_labelling_tab(STAGE_ORDER, _PROGRESS)
        gr.Markdown(f"<small>{_esc(DISCLAIMER)}</small>")
    demo.queue(max_size=8, default_concurrency_limit=2)
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
