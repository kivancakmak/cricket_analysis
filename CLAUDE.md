# Cricket Batting AI Coach — Demo

## What this is
A portfolio proof-of-concept for an AI cricket coach. A player films a 3–6 second
side-on clip of one batting shot (forward defence or front-foot drive). The system
tracks body keypoints, finds the key moments of the shot, computes measurable
technique checks, scores the shot out of 100 against a coach-editable rulebook,
and returns 1–2 plain-English fixes plus a skeleton-overlay video. If the clip is
not good enough, it returns a friendly "please film again" message instead.

## Stack
- Python 3.10+
- MediaPipe Pose Landmarker (Tasks API, VIDEO mode, `pose_landmarker_full.task`)
- OpenCV + ffmpeg for video I/O (browser-playable H.264 output)
- NumPy, PyYAML
- Gradio for the web UI (deployable to Hugging Face Spaces, CPU only)
- pytest, ruff

## Structure
```
src/
  pose.py          # video -> PoseSequence (33 landmarks/frame + visibility), smoothing
  orientation.py   # facing direction, front/back leg, handedness
  phases.py        # key moments: STANCE, FRONT_FOOT_LANDING, CONTACT
  metrics.py       # normalized technique metrics at key moments
  scoring.py       # YAML rule engine -> score /100, sub-scores, fixes
  quality.py       # confidence gate -> "please film again"
  overlay.py       # skeleton overlay video + key-frame images
  pipeline.py      # analyze(video, shot_type) -> AnalysisResult
rules/
  forward_defence.yaml
  front_foot_drive.yaml
scripts/           # model download, debug plots, agreement report
labels/            # coach labels (CSV)
tests/
data/              # local sample clips (raw videos git-ignored)
app.py             # Gradio UI (analysis tab + coach labelling tab)
```

## Principles
- **Modular:** each stage is a pure function with typed dataclass inputs/outputs,
  independent of the UI, so it can later run behind a cloud API or on-device.
- **Shot-agnostic engine:** adding a new shot (or bowling in Phase 2) means a new
  phase detector, new metrics and a new rules YAML — not a rewrite.
- **Coach-editable rules:** all thresholds, weights and feedback text live in YAML.
  Heuristic thresholds for phase detection and quality checks live in config dicts.
- **Scale-invariant metrics:** normalize by the batter's body size; never rely on
  raw pixel distances.
- **Every output carries confidence**, derived from landmark visibility.
- Side-on clips occlude the far limbs: prefer near-side joints, lower confidence
  when visibility is low.
- Keep code readable and documented; this repo is shown to clients.

## Workflow
- Update PROGRESS.md after each step.
- Run tests (`pytest`) and `ruff check .` before finishing a step.
- Don't commit raw videos, model files or outputs.

## Progress checklist (mirror in PROGRESS.md)
1. Project setup  2. Pose extraction  3. Overlay  4. Orientation
5. Key moments  6. Metrics  7. Rule engine & scoring  8. Quality gate
9. Pipeline + Gradio UI  10. Coach labelling + agreement report
11. Tests, cleanup, README  12. Hugging Face Spaces deployment
