"""The "Coach labelling" tab of the web app.

A coach loads a clip, sees what the AI made of it, corrects the three key
frames with sliders, marks each technique check pass/fail and gives an overall
score. Labels go to ``labels/labels.csv`` (see ``src/labels.py``); the
``scripts/agreement_report.py`` script compares them with the AI.

The handlers (``load_clip``, ``show_frame``, ``save_clip``) are plain functions
that take and return plain data, so they can be tested without a browser;
``build_labelling_tab`` only wires them to Gradio components.
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable

import gradio as gr

from src import labels
from src.labels import MAX_LABELLABLE_FRAMES, MOMENT_NAMES
from src.pipeline import SHOT_LABELS, SHOT_TYPES, analyze
from ui_cards import _esc, render_summary

MAX_RULE_SLOTS = 16     # radio buttons available for rules; the rule files have 9 each today
MOMENT_TEXT = {
    "stance": ("Stance", "The last still frame before the front foot starts to move."),
    "front_foot_landing": ("Front foot landing", "The front foot is planted at the end of the stride."),
    "contact": ("Contact", "The bat meets the ball (or the instant of the shot)."),
}
RULE_CHOICES = [("Pass", "pass"), ("Fail", "fail"), ("Skip", "skip")]
INSTRUCTIONS = """
**How to label a clip**
1. Enter your **coach ID**, add a clip, choose the shot and press **Load**.
2. Check the three **key frames**. Move a slider if the AI picked the wrong frame, then tick the box.
3. For each technique check choose **Pass** (good enough) or **Fail** (needs work). Leave it alone or choose
   *Skip* if you are not sure: only the checks you answer are saved.
4. Give an **overall score** out of 100 if you can, then **Save**. Saving again replaces your earlier labels for this clip.
"""


def _frame_info(index: int, state: dict, moment: str) -> str:
    fps = state.get("fps") or 25.0
    ai = state["moments"].get(moment)
    text = f"Frame **{index}** of {len(state['frames']) - 1} &middot; {index / fps:.2f} s"
    if ai is None:
        return f"{text} &middot; the AI did not find this moment."
    diff = ai - index
    verdict = ("you agree with the AI" if diff == 0 else
               f"the AI chose frame {ai}, {abs(diff)} frame{'s' if abs(diff) != 1 else ''} "
               f"{'later' if diff > 0 else 'earlier'} ({abs(diff) / fps * 1000:.0f} ms)")
    return f"{text} &middot; {verdict}"


def show_frame(index, state: dict | None, moment: str) -> tuple[str | None, str]:
    """Image path and caption for the frame under a slider."""
    if not state or not state.get("frames"):
        return None, ""
    index = max(0, min(int(index or 0), len(state["frames"]) - 1))
    return state["frames"][index], _frame_info(index, state, moment)


def rule_choice_info(rule: dict) -> str:
    if rule["status"] == "skipped":
        return "AI: could not measure this on the clip."
    return (f"AI says {rule['status'].upper()} - measured {rule['value']:.2f}, "
            f"target {rule['ideal']}.")


def load_clip(video, shot_type: str, coach_id: str, previous_state: dict | None = None,
              on_stage: Callable[[str], None] | None = None) -> dict:
    """Analyse the clip and prepare the form. Returns {component name: new value}."""
    out: dict = {"form": gr.update(visible=False)}
    coach = labels.clean_coach_id(coach_id)
    if not coach:
        out["status"] = "Please enter your coach ID first."
        return out
    if not video:
        out["status"] = "Please add a clip first."
        return out

    if previous_state and previous_state.get("frame_dir"):
        shutil.rmtree(previous_state["frame_dir"], ignore_errors=True)
    result = analyze(video, shot_type, render=False, on_stage=on_stage)
    if result.status == "error":
        out["status"] = result.message
        return out
    if result.video_info.get("frames", 0) > MAX_LABELLABLE_FRAMES:
        out["status"] = (f"That clip has {result.video_info['frames']} frames. Please label clips of "
                         f"{MAX_LABELLABLE_FRAMES} frames (about 24 s) or fewer.")
        return out

    clip_id = labels.clip_id_for(video)
    frame_dir = tempfile.mkdtemp(prefix="cricket_label_")
    frames = labels.extract_frames(video, frame_dir)
    if not frames:
        shutil.rmtree(frame_dir, ignore_errors=True)
        out["status"] = "We couldn't read any frames from that clip."
        return out

    snapshot = labels.ai_snapshot(result, clip_id)
    state = {**snapshot, "frames": [str(p) for p in frames], "frame_dir": frame_dir,
             "fps": result.video_info.get("fps") or snapshot["fps"] or 25.0, "coach_id": coach}
    prior = labels.load_coach_labels(clip_id, shot_type, coach)

    out.update(
        state=state, form=gr.update(visible=True), ai_card=render_summary(result),
        confirm=gr.update(value=bool(prior["moments"])),
        overall=gr.update(value=prior["overall"]), notes=gr.update(value=prior["notes"]),
    )
    for moment in MOMENT_NAMES:
        start = prior["moments"].get(moment, snapshot["moments"][moment])
        start = 0 if start is None else min(int(start), len(frames) - 1)
        image, info = show_frame(start, state, moment)
        out[f"{moment}_slider"] = gr.update(minimum=0, maximum=len(frames) - 1, value=start, step=1)
        out[f"{moment}_image"], out[f"{moment}_info"] = image, info
    for i in range(MAX_RULE_SLOTS):
        if i < len(snapshot["rules"]):
            rule = snapshot["rules"][i]
            out[f"rule_{i}"] = gr.update(visible=True, label=rule["label"], info=rule_choice_info(rule),
                                         value=prior["rules"].get(rule["id"]))
        else:
            out[f"rule_{i}"] = gr.update(visible=False, value=None)

    extra = " Your earlier labels for this clip are filled in." if prior["moments"] or prior["rules"] \
        or prior["overall"] is not None else ""
    ai = "scored" if result.ok else "not scored (the clip failed the quality check)"
    out["status"] = f"Loaded **{clip_id}** as coach **{_esc(coach)}**. The AI {ai}.{extra}"
    return out


def save_clip(state: dict | None, coach_id: str, confirmed: bool, stance, landing, contact,
              overall, notes: str, *rule_values) -> str:
    """Save the coach's labels; returns the message to show."""
    coach = labels.clean_coach_id(coach_id)
    if not state or not state.get("clip_id"):
        return "Load a clip first."
    if not coach:
        return "Please enter your coach ID first."
    moments = {"stance": stance, "front_foot_landing": landing, "contact": contact} if confirmed else None
    coach_rules = {rule["id"]: value for rule, value in zip(state["rules"], rule_values, strict=False)}
    rows = labels.build_rows(state, coach, moments, coach_rules,
                             None if overall is None else float(overall), notes or "")
    if not rows:
        return ("Nothing to save yet. Tick the key-frame box, answer at least one technique check, "
                "or give an overall score.")
    try:
        saved = labels.save_labels(rows)
    except PermissionError:
        return "Could not write labels/labels.csv. If it is open in Excel, close it and press Save again."
    all_rows = labels.load_labels()
    clips = len({(r["clip_id"], r["shot_type"], r["coach_id"]) for r in all_rows})
    return (f"Saved {saved} label{'s' if saved != 1 else ''} for **{state['clip_id']}** as coach "
            f"**{_esc(coach)}**. The file now holds labels for {clips} clip{'s' if clips != 1 else ''} in total.")


# --------------------------------------------------------------------------- Gradio wiring
def build_labelling_tab(stage_order: list[str], progress: gr.Progress) -> None:
    """Create the tab's components and connect them to the handlers above (call inside gr.Tab)."""
    gr.Markdown(INSTRUCTIONS)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            coach = gr.Textbox(label="Your coach ID", placeholder="e.g. alex",
                               info="Keeps each coach's labels separate.")
            video = gr.Video(label="Clip to label", sources=["upload"])
            shot = gr.Dropdown(choices=[(SHOT_LABELS[s], s) for s in SHOT_TYPES], value=SHOT_TYPES[0],
                               label="Shot type", interactive=True)
            load = gr.Button("Load clip and AI result", variant="primary")
        with gr.Column(scale=1, min_width=300):
            status = gr.Markdown()
            ai_card = gr.HTML()

    state = gr.State(None)
    comps: dict = {"state": state, "status": status, "ai_card": ai_card}
    with gr.Column(visible=False) as form:
        comps["form"] = form
        gr.Markdown("### 1. Key frames")
        sliders = {}
        for moment in MOMENT_NAMES:
            title, hint = MOMENT_TEXT[moment]
            gr.Markdown(f"**{title}** - {hint}")
            with gr.Row(equal_height=False):
                image = gr.Image(label=title, interactive=False, height=320, scale=2, min_width=300)
                with gr.Column(scale=1, min_width=260):
                    slider = gr.Slider(0, 1, value=0, step=1, label=f"{title}: frame number")
                    info = gr.Markdown()
            comps[f"{moment}_image"], comps[f"{moment}_slider"], comps[f"{moment}_info"] = image, slider, info
            sliders[moment] = (slider, image, info)
        confirm = gr.Checkbox(label="I have checked these three key frames (corrected where needed)")
        comps["confirm"] = confirm

        gr.Markdown("### 2. Technique checks\n*Pass* = good enough, *Fail* = needs work.")
        radios = []
        for i in range(MAX_RULE_SLOTS):
            radio = gr.Radio(choices=RULE_CHOICES, value=None, label=f"Check {i + 1}", visible=False)
            comps[f"rule_{i}"] = radio
            radios.append(radio)

        gr.Markdown("### 3. Overall")
        overall = gr.Number(label="Your overall score (0-100)", value=None, minimum=0, maximum=100, precision=0)
        notes = gr.Textbox(label="Notes (optional)", lines=2)
        comps["overall"], comps["notes"] = overall, notes
        save = gr.Button("Save my labels", variant="primary")
        saved = gr.Markdown()

    names = list(comps)

    def on_load(video_path, shot_type, coach_id, previous, prog=progress):
        def on_stage(name: str) -> None:
            prog(stage_order.index(name) / len(stage_order) if name in stage_order else 0, desc=name)
        values = load_clip(video_path, shot_type, coach_id, previous, on_stage=on_stage)
        return [values.get(name, gr.update()) for name in names]

    load.click(on_load, inputs=[video, shot, coach, state], outputs=[comps[n] for n in names],
               api_name="load_for_labelling")
    for moment, (slider, image, info) in sliders.items():
        slider.change(lambda i, st, m=moment: show_frame(i, st, m), inputs=[slider, state],
                      outputs=[image, info], show_progress="hidden")
    save.click(save_clip,
               inputs=[state, coach, confirm, *(sliders[m][0] for m in MOMENT_NAMES), overall, notes, *radios],
               outputs=saved, api_name="save_labels")
