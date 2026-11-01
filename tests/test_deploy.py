"""Deployment readiness: model download, free-tier settings, examples, Space metadata."""
import io
import os
import time
from pathlib import Path

import gradio as gr
import pytest
import yaml

import app
from src import model, pipeline
from src.pipeline import analyze

ROOT = Path(__file__).resolve().parent.parent


class FakeResponse(io.BytesIO):
    """Stands in for what ``urlopen`` returns (a readable context manager)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_download(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return FakeResponse(b"x" * 2_000_000)

    monkeypatch.setattr(model.urllib.request, "urlopen", fake_urlopen)
    return calls


# ------------------------------------------------------------------ model download
def test_missing_model_is_downloaded_once(tmp_path, fake_download):
    target = tmp_path / "models" / "m.task"
    assert model.ensure_model(target) == target and target.stat().st_size == 2_000_000
    assert model.ensure_model(target) == target
    assert len(fake_download) == 1                                   # second call found the file
    assert [p.name for p in target.parent.iterdir()] == ["m.task"]    # no .part file left behind


def test_force_and_truncated_files_trigger_a_new_download(tmp_path, fake_download):
    target = tmp_path / "m.task"
    target.write_bytes(b"tiny")                                      # e.g. an interrupted earlier download
    model.ensure_model(target)
    model.ensure_model(target, force=True)
    assert len(fake_download) == 2


def test_failed_or_bogus_download_raises_and_cleans_up(tmp_path, monkeypatch):
    def broken(url, timeout=None):
        raise OSError("no network")

    monkeypatch.setattr(model.urllib.request, "urlopen", broken)
    with pytest.raises(OSError, match="no network"):
        model.ensure_model(tmp_path / "m.task")

    monkeypatch.setattr(model.urllib.request, "urlopen", lambda url, timeout=None: FakeResponse(b"<html>404</html>"))
    with pytest.raises(OSError, match="expected the model"):
        model.ensure_model(tmp_path / "m.task")
    assert list(tmp_path.iterdir()) == []


def test_pipeline_fetches_a_missing_model_and_retries(monkeypatch, make_sequence, blank_video):
    attempts, fetched = [], []

    def flaky_extract(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("model not found")
        return make_sequence()

    monkeypatch.setattr(pipeline, "extract_pose", flaky_extract)
    monkeypatch.setattr(pipeline, "ensure_model", lambda: fetched.append(1))
    assert analyze(blank_video(), "forward_defence", render=False).ok
    assert (len(attempts), len(fetched)) == (2, 1)


def test_unavailable_model_gives_a_clear_error_not_a_crash(monkeypatch, blank_video):
    def missing(*args, **kwargs):
        raise FileNotFoundError("model not found")

    def offline():
        raise OSError("no network")

    monkeypatch.setattr(pipeline, "extract_pose", missing)
    monkeypatch.setattr(pipeline, "ensure_model", offline)
    result = analyze(blank_video(), "forward_defence", render=False)
    assert result.status == "error" and "could not be downloaded" in result.message


def test_startup_survives_a_failed_download(monkeypatch, capsys):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(app, "ensure_model", offline)
    app.prepare()                                                    # must not raise
    assert "will retry" in capsys.readouterr().out


# ------------------------------------------------------------------ free-tier settings
def test_space_detection_and_defaults(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("ENABLE_LABELLING", raising=False)
    assert (app.on_hf_space(), app.labelling_enabled(), app.concurrency_limit()) == (False, True, 2)

    monkeypatch.setenv("SPACE_ID", "someone/cricket-coach")
    assert (app.on_hf_space(), app.labelling_enabled(), app.concurrency_limit()) == (True, False, 1)

    monkeypatch.setenv("ENABLE_LABELLING", "1")
    assert app.labelling_enabled() is True
    monkeypatch.delenv("SPACE_ID")
    monkeypatch.setenv("ENABLE_LABELLING", "0")
    assert app.labelling_enabled() is False


def tab_labels(demo) -> list[str]:
    return [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]


def test_public_space_hides_the_labelling_tab_unless_enabled(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "someone/cricket-coach")
    monkeypatch.delenv("ENABLE_LABELLING", raising=False)
    assert tab_labels(app.build_app()) == []
    monkeypatch.setenv("ENABLE_LABELLING", "1")
    assert tab_labels(app.build_app()) == ["Analyse a shot", "Coach labelling"]


def test_stale_output_folders_are_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.tempfile, "tempdir", str(tmp_path))
    old, fresh, other = tmp_path / "cricket_old", tmp_path / "cricket_label_new", tmp_path / "unrelated_old"
    for folder in (old, fresh, other):
        folder.mkdir()
        (folder / "file.bin").write_bytes(b"x")
    long_ago = time.time() - 10 * 3600
    for folder in (old, other):
        os.utime(folder, (long_ago, long_ago))
    assert pipeline.cleanup_stale_runs(3 * 3600) == 1
    assert not old.exists() and fresh.exists() and other.exists()      # only our own, only the old


# ------------------------------------------------------------------ example clips
def test_examples_are_found_by_name_and_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_EXAMPLE_BYTES", 100)
    for name, size in [("front_foot_drive_nets.mp4", 50), ("forward_defence_a.MP4", 50), ("my_clip.mov", 50),
                       ("huge.mp4", 500), ("notes.txt", 5)]:
        (tmp_path / name).write_bytes(b"x" * size)
    found = app.discover_examples(tmp_path)
    assert [(Path(p).name, shot) for p, shot in found] == [
        ("forward_defence_a.MP4", "forward_defence"), ("front_foot_drive_nets.mp4", "front_foot_drive"),
        ("my_clip.mov", "forward_defence")]                              # unknown prefix -> forward defence
    assert app.discover_examples(tmp_path / "missing") == []

    for i in range(10):
        (tmp_path / f"forward_defence_extra{i}.mp4").write_bytes(b"x")
    assert len(app.discover_examples(tmp_path)) == app.MAX_EXAMPLES


def test_the_app_builds_with_examples_present(tmp_path, monkeypatch):
    clip = tmp_path / "front_foot_drive_demo.mp4"
    clip.write_bytes(b"not a real video, only the file name matters here")
    monkeypatch.setattr(app, "discover_examples", lambda: [[str(clip), "front_foot_drive"]])
    kinds = {type(b).__name__ for b in app.build_app().blocks.values()}
    assert "Dataset" in kinds                                        # the examples gallery is on the page


# ------------------------------------------------------------------ Space metadata and dependencies
def front_matter() -> dict:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "README.md must start with the Space YAML header"
    return yaml.safe_load(text.split("---\n")[1])


def test_space_header_is_valid():
    meta = front_matter()
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py" and (ROOT / meta["app_file"]).exists()
    assert meta["sdk_version"] == gr.__version__                      # Space and local Gradio must agree
    assert isinstance(meta["python_version"], str) and meta["python_version"].startswith("3.")
    assert meta["colorFrom"] in {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    assert meta["colorTo"] in {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    assert len(meta["short_description"]) <= 60 and meta["title"]


def test_runtime_requirements_are_lean_and_complete():
    runtime = [line.split("#")[0].strip().lower() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    runtime = [line for line in runtime if line]
    for needed in ("mediapipe", "numpy", "pyyaml", "imageio-ffmpeg", "opencv-contrib-python"):
        assert needed in runtime, needed
    assert any(line.startswith("gradio") for line in runtime)
    assert not {"pytest", "ruff", "matplotlib", "opencv-python"} & set(runtime)   # dev tools stay out of the Space
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert "-r requirements.txt" in dev and "pytest" in dev and "ruff" in dev


def test_system_packages_for_opencv_are_listed():
    packages = (ROOT / "packages.txt").read_text().split()
    assert "libgl1" in packages and "libglib2.0-0" in packages


def test_no_large_or_forbidden_files_would_be_pushed():
    tracked_dirs = [ROOT / "examples", ROOT / "docs", ROOT / "rules"]
    for folder in tracked_dirs:
        for path in folder.rglob("*"):
            assert not path.is_file() or path.stat().st_size < 15 * 1024 * 1024, path
    ignored = (ROOT / ".gitignore").read_text()
    assert "models/*.task" in ignored and "data/raw/" in ignored
