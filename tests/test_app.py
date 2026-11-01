"""UI rendering tests: HTML cards are built from a real pipeline result; no browser needed."""
import dataclasses

import pytest

import app
from src import pipeline
from src.pipeline import analyze
from src.scoring import Fix


@pytest.fixture
def good_result(monkeypatch, make_sequence, blank_video, tmp_path):
    monkeypatch.setattr(pipeline, "extract_pose", lambda *a, **k: make_sequence())
    return analyze(blank_video(), "forward_defence", tmp_path / "out")


@pytest.fixture
def rejected_result(blank_video):
    return analyze(blank_video(frames=300), "forward_defence")


def test_score_bands_and_confidence_words():
    assert [app.score_band(s)[1] for s in (95, 85, 70, 65, 40)] == [
        "Strong technique", "Strong technique", "Getting there", "Getting there", "Needs work"]
    assert app.score_band(90)[0] == "good" and app.score_band(70)[0] == "ok" and app.score_band(10)[0] == "low"
    assert [app.confidence_label(c) for c in (0.9, 0.5, 0.1)] == ["high", "medium", "low"]


def test_summary_card_has_score_bars_and_placeholder_warning(good_result):
    card = app.render_summary(good_result)
    assert f"{good_result.score.total:.0f}<small>/100" in card
    for name in ("Head &amp; Balance", "Footwork", "Body Shape"):          # category names are escaped
        assert name in card
    assert card.count('class="bar-fill') == 3 and 'role="progressbar"' in card
    assert "Confidence:" in card and "placeholders" in card and "Forward defence" in card


def test_rejected_clip_gets_the_retry_card_not_a_score(rejected_result):
    card = app.render_summary(rejected_result)
    assert "try that clip again" in card and "too long" in card
    assert "score-number" not in card
    assert app.render_fixes(rejected_result) == ""
    assert app.gallery_items(rejected_result) == [] and app.metrics_rows(rejected_result) == []


def test_error_card(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    card = app.render_summary(analyze(bad, "forward_defence"))
    assert "We hit a problem" in card and "couldn&#x27;t open" in card


def test_fix_text_is_html_escaped(good_result):
    fix = Fix("r", "Label", "Footwork", "t", "d", 1.0, "0 to 1", 1.0)
    evil = dataclasses.replace(fix, text="<script>alert(1)</script>", drill="<img src=x onerror=1>")
    result = dataclasses.replace(good_result, score=dataclasses.replace(good_result.score, fixes=(evil,)))
    html = app.render_fixes(result)
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "Try this drill" in html and "Fix 1:" in html


def test_no_fixes_message(good_result):
    result = dataclasses.replace(good_result, score=dataclasses.replace(good_result.score, fixes=()))
    assert "Nice work" in app.render_fixes(result)


def test_tables_gallery_and_timing(good_result):
    names = [row[0] for row in app.metrics_rows(good_result)]
    assert "stride_length_landing" in names and "head_stability" in names
    assert len(app.rules_rows(good_result)) == 9
    assert all(len(row) == 7 for row in app.rules_rows(good_result))
    items = app.gallery_items(good_result)
    assert [caption for _, caption in items] == ["Stance", "Front Foot Down", "Contact"]
    assert "Processed in" in app.render_timing(good_result)


def test_handler_returns_a_value_for_every_output(monkeypatch, make_sequence, blank_video):
    monkeypatch.setattr(pipeline, "extract_pose", lambda *a, **k: make_sequence())
    outputs = app.run_analysis(str(blank_video()), "forward_defence")
    assert len(outputs) == 8
    summary, overlay, gallery, _fixes, _timing, metrics, rules, raw = outputs
    assert "score-number" in summary and overlay.endswith("overlay.mp4") and len(gallery) == 3
    assert raw["status"] == "ok" and metrics and rules


def test_handler_without_a_video_does_not_crash(monkeypatch):
    monkeypatch.setattr(app.gr, "Warning", lambda *a, **k: None)
    outputs = app.run_analysis(None, "forward_defence")
    assert len(outputs) == 8 and outputs[1] is None


def test_every_shot_type_has_a_dropdown_label():
    assert set(app.SHOT_LABELS) == set(app.SHOT_TYPES)
