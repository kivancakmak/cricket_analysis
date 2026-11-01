# Progress

- [x] 1. Project setup
- [x] 2. Pose extraction
- [x] 3. Overlay
- [ ] 4. Orientation
- [ ] 5. Key moments
- [ ] 6. Metrics
- [ ] 7. Rule engine & scoring
- [ ] 8. Quality gate
- [ ] 9. Pipeline + Gradio UI
- [ ] 10. Coach labelling + agreement report
- [ ] 11. Tests, cleanup, README
- [ ] 12. Hugging Face Spaces deployment

## Notes
- Step 2: `src/pose.py` uses the MediaPipe Tasks API (mediapipe 1.2.0; legacy `mp.solutions` no longer exists). Smoothing is `smooth_pose()` (One-Euro, numpy only), separate from raw `extract_pose()`. Verified on real clips in `data/raw/` (front-foot defence tutorial, 1920x1080, 25 fps): the two trimmed clips (~100 frames) gave 100% detection, mean visibility 0.94–0.95; the full 2252-frame video gave 84.2% detection, 0.877 visibility.
- Step 3: `src/overlay.py` — `render_overlay()` pipes frames into ffmpeg (libx264, yuv420p, faststart; downscaled to max 1280 px wide) and `export_key_frames()` writes PNGs. ffmpeg is taken from PATH, else from the `imageio-ffmpeg` package (added to requirements; none was installed on this machine). Checked on the trimmed clip: skeleton sits correctly on the batter in a rendered frame. Key-frame labels in the CLI (`--key LABEL=FRAME`) are manual until step 5 detects them. Not yet played in an actual browser.
