# Example clips

Put one or two **of your own** short clips here. The app lists them under *Try an example clip*, and
visitors can load one with a click.

Rules:

- Only footage you filmed yourself (or have written permission to publish). **No YouTube or other
  downloaded clips.** Everything in this folder is published with the Space.
- Side-on, whole body in frame, one shot, 3-6 seconds, no audio, under 15 MB (files bigger than that are ignored).
- Name them `forward_defence_<name>.mp4` or `front_foot_drive_<name>.mp4`; the prefix decides which shot
  type the example selects.

The easiest way is the helper script, which trims, shrinks and checks the clip for you:

```bash
python scripts/prepare_example.py path/to/my_clip.mov forward_defence --name nets_1 --start 1.0 --seconds 5
```

It writes the file here and tells you whether the app would score it or ask for a new clip.
