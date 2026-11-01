# Deploying to Hugging Face Spaces

Exact steps to publish this app as a Space. Takes about 15 minutes, plus the first build (a few minutes).

## What is already set up in the repository

| File | Why it matters on a Space |
|---|---|
| `README.md` (the `---` block at the top) | The Space configuration: Gradio SDK, `app_file: app.py`, Python 3.12, `sdk_version` (kept equal to the Gradio you develop with; a test checks this). |
| `requirements.txt` | Runtime dependencies only, so the build stays small and fast. Test tools are in `requirements-dev.txt`. |
| `packages.txt` | `libgl1` and `libglib2.0-0`, system libraries OpenCV needs on a bare Linux image. |
| `src/model.py` + `app.py` | The 9 MB pose model is downloaded when the app starts (and again on the first analysis if that failed), so nothing manual is needed. |
| `app.py` settings | One analysis at a time (the free tier has 2 CPU cores), uploads deleted after about three hours, coach-labelling tab hidden. |
| `examples/` | Your own clips, offered as "Try an example clip". |

The model file, raw videos in `data/raw/` and outputs are git-ignored, so they are not pushed.

## 0. Before you start

1. **Add your own example clips** (one or two). From the repository folder:

   ```bash
   python scripts/prepare_example.py path/to/your_clip.mov forward_defence --name nets_1 --start 1.0 --seconds 5
   ```

   It writes `examples/forward_defence_nets_1.mp4` (a few MB, no audio) and tells you whether the app would score it. Use only footage you filmed yourself and may publish. Repeat with `front_foot_drive` for a drive if you have one.
2. Check everything is green and commit:

   ```bash
   pytest
   ruff check .
   git add -A
   git commit -m "Add example clips"
   ```

## 1. Create the Hugging Face account, token and Space

1. Sign up or log in at <https://huggingface.co>.
2. **Create an access token**: Settings -> Access Tokens -> *Create new token* -> type **Write** -> copy it somewhere safe (you see it once). It is used as the *password* when you push.
3. **Create the Space**: open <https://huggingface.co/new-space> and choose:
   - *Space name*: e.g. `cricket-batting-ai-coach`
   - *License*: your choice
   - *Select the Space SDK*: **Gradio** (leave the template **blank**)
   - *Space hardware*: **CPU basic - 2 vCPU - 16 GB - FREE**
   - *Visibility*: **Private** while you test, switch to Public in the Space's Settings when you are happy.
4. Note your username (`<USER>`) and the Space name (`<SPACE>`).

> **Check this when you create it:** the Hugging Face documentation I read says that creating a Gradio Space can require a paid plan (PRO) for personal accounts, with free accounts limited to ZeroGPU Gradio Spaces. I could not test this with an account. If the form will not let you pick Gradio on the free tier, read the current rules at <https://huggingface.co/docs/hub/spaces-overview> before continuing.

## 2. Push the repository to the Space

A Space is a git repository. From your project folder (it is already a git repository on branch `main`):

```bash
git remote add space https://huggingface.co/spaces/<USER>/<SPACE>
git push space main
```

When asked, enter your Hugging Face **username** and use the **token** as the password (Git Credential Manager on Windows offers to remember it).

**If the push is rejected as "non-fast-forward / fetch first":** a new Space is created with a boilerplate commit (a README and `.gitattributes`). Because this repository's `README.md` already contains the Space header, replacing that boilerplate is correct for a brand-new Space:

```bash
git push space main --force
```

(Only force-push to a Space you just created. Never do it on a Space that holds work you want to keep.)

**If the push is rejected because of binary files** (this can happen for your example `.mp4` or the screenshots in `docs/`): Hugging Face wants binary files tracked with its large-file storage. Follow the current instructions at <https://huggingface.co/docs/hub/xet>; in short, install `git-xet` (`git xet install`), run `git xet track "*.mp4" "*.png" "*.gif"`, then re-add the files and commit before pushing again. If files were already committed without tracking, tell me the error and I will give you the exact commands to rewrite that commit.

## 3. Watch it build

1. Open `https://huggingface.co/spaces/<USER>/<SPACE>`. The status badge shows *Building* and then *Running*.
2. Click **Logs** (top right of the Space) to follow the build. The first build installs mediapipe, OpenCV and Gradio and typically takes a few minutes; later pushes are faster.
3. When the app starts you should see a line like `Running on local URL: http://0.0.0.0:7860` and the app appears. The pose model downloads at that point (a couple of seconds).

## 4. Test the live Space

1. Click an **example clip**, then **Analyse my shot**. The first analysis can take 15-30 seconds (the free CPU is a few times slower than a laptop and the app is warming up).
2. Try a bad clip too (for example a very short one) to confirm the friendly "please film again" card appears.
3. Open the Space on your phone to check the layout.

## 5. Settings you may want

In the Space: **Settings -> Variables and secrets -> New variable**.

| Variable | Value | Effect |
|---|---|---|
| `ENABLE_LABELLING` | `1` | Shows the *Coach labelling* tab. On the free tier the labels are written to a **temporary disk and are lost on restart**, and anyone with the link can add labels, so use this only on a **private** Space for a short session and download `labels/labels.csv` afterwards (or run labelling locally). |

Free Spaces **go to sleep** after a period of inactivity and wake up (about a minute) on the next visit.

## 6. Updating later

```bash
git add -A
git commit -m "Describe the change"
git push space main
```

The Space rebuilds automatically.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Build fails with `libGL.so.1: cannot open shared object file` | `packages.txt` was not pushed or is not at the repository root. |
| *Runtime error* on start, `sdk_version` complaint | The version in the README header is not an available Gradio release. Set it to a released version and push again (locally, `pytest tests/test_deploy.py` checks it matches the Gradio you have installed). |
| "The pose model is missing and could not be downloaded" | The Space could not reach `storage.googleapis.com` at startup. Restart the Space (Settings -> Factory reboot); the app retries on the next analysis. |
| Analysis very slow or times out | Expected on 2 shared CPU cores; keep clips to 3-6 seconds. Hardware can be upgraded in the Space settings (paid). |
| Examples do not appear | `examples/` is empty or the files are over 15 MB. Run `prepare_example.py` again. |
| Videos will not play in the browser | Never seen with the bundled ffmpeg (it writes H.264), but if it happens, check the Space logs for `ffmpeg failed`. |

## Privacy note for a public Space

Visitors upload video of themselves. The app keeps uploads only temporarily (deleted within about three hours), does not store them in the repository, and does not log their contents, but the Space is still a public web service: say so on any page that links to it, and do not publish it as a product without a proper privacy policy.
