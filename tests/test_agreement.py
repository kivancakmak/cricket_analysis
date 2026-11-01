"""Agreement statistics: small hand-checkable examples, then the markdown report."""
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src import agreement as ag
from src import labels

NOW = datetime(2026, 1, 1, 12, 0)


def row(kind, item, coach, ai, shot="forward_defence", coach_id="alex", clip="c1", fps="25", rules_hash="h1"):
    return {"timestamp": "", "clip_id": clip, "shot_type": shot, "coach_id": coach_id, "kind": kind, "item": item,
            "coach_label": coach, "ai_label": ai, "ai_value": "", "ai_confidence": "", "fps": fps,
            "rules_hash": rules_hash, "notes": ""}


def rule_rows(item, pairs, **kw):
    return [row("rule", item, c, a, clip=f"c{i}", **kw) for i, (c, a) in enumerate(pairs)]


# 10 clips: both pass x5, coach pass / AI fail x1, coach fail / AI pass x1, both fail x3
KAPPA_PAIRS = [("pass", "pass")] * 5 + [("pass", "fail")] + [("fail", "pass")] + [("fail", "fail")] * 3


def test_kappa_known_values():
    coach, ai = zip(*KAPPA_PAIRS, strict=True)
    # observed 0.8; chance = 0.6*0.6 + 0.4*0.4 = 0.52  ->  (0.8 - 0.52) / 0.48
    assert ag.cohens_kappa(coach, ai) == pytest.approx(0.28 / 0.48)
    assert ag.cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert ag.cohens_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    assert math.isnan(ag.cohens_kappa(["a", "a"], ["a", "a"]))        # undefined: no variation at all
    assert math.isnan(ag.cohens_kappa([], []))
    with pytest.raises(ValueError, match="same length"):
        ag.cohens_kappa(["a"], ["a", "b"])


def test_kappa_wording():
    assert [ag.kappa_words(k) for k in (-0.2, 0.1, 0.3, 0.5, 0.7, 0.9, float("nan"))] == [
        "worse than chance", "slight", "fair", "moderate", "substantial", "almost perfect", "n/a"]


def test_correlations():
    x, y = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
    assert ag.pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert ag.spearman(x, [v ** 3 for v in x]) == pytest.approx(1.0)       # any increasing curve
    assert ag.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)   # ties
    assert math.isnan(ag.pearson([1, 2], [1, 2])) and math.isnan(ag.pearson([1, 1, 1], [1, 2, 3]))


def test_rule_agreement_counts_and_pooled_row():
    rows = rule_rows("stride", KAPPA_PAIRS) + rule_rows("knee", [("pass", "pass"), ("fail", "skipped")])
    stats = {s.item: s for s in ag.rule_agreement(rows)}
    s = stats["stride"]
    assert (s.n, s.accuracy, s.ai_too_harsh, s.ai_too_lenient) == (10, 0.8, 1, 1)
    assert s.kappa == pytest.approx(0.28 / 0.48)
    assert (s.coach_fail_rate, s.ai_fail_rate) == (0.4, 0.4)
    knee = stats["knee"]
    assert (knee.n, knee.ai_skipped) == (1, 1)                              # the skipped pair is not scored
    assert stats["(all rules)"].n == 11 and stats["(all rules)"].ai_skipped == 1
    assert next(reversed(stats)) == "(all rules)"


def test_unlabelled_rules_are_ignored():
    rows = [row("rule", "x", "", "pass"), row("rule", "x", "skip", "fail"), row("moment", "stance", "3", "4")]
    assert ag.rule_agreement(rows) == []


def test_timing_error_in_milliseconds():
    rows = [row("moment", "contact", "10", "11", clip="a"), row("moment", "contact", "20", "18", clip="b"),
            row("moment", "contact", "5", "", clip="c"), row("moment", "stance", "3", "3", fps="")]
    (stat,) = ag.timing_agreement(rows)          # missing AI frame and missing fps rows are skipped
    assert stat.item == "contact" and stat.n == 2
    assert stat.mean_error_ms == pytest.approx(-20)                         # (+40 ms, -80 ms) at 25 fps
    assert (stat.mae_ms, stat.max_abs_ms) == (pytest.approx(60), pytest.approx(80))
    assert (stat.within_100ms, stat.within_200ms) == (1.0, 1.0)


def test_overall_score_agreement():
    pairs = [(60, 65), (70, 68), (80, 85), (90, 95)]
    rows = [row("overall", "overall", c, a, clip=f"c{i}") for i, (c, a) in enumerate(pairs)]
    s = ag.score_agreement(rows)
    assert s.n == 4 and s.mae == pytest.approx(4.25) and s.bias == pytest.approx(3.25)
    assert s.rmse == pytest.approx(math.sqrt(19.75))
    assert s.pearson == pytest.approx(np.corrcoef([60, 70, 80, 90], [65, 68, 85, 95])[0, 1])
    assert ag.score_agreement([]) is None


def full_rows():
    return (rule_rows("stride_length", KAPPA_PAIRS)
            + [row("moment", "contact", "10", "12", clip="c0"), row("overall", "overall", "70", "75", clip="c0")]
            + rule_rows("spine_tilt", [("pass", "pass")], shot="front_foot_drive", coach_id="sam", rules_hash="h9"))


def test_report_contents():
    report = ag.build_report(full_rows(), source="labels.csv", now=NOW)
    assert report.startswith("# AI vs coach agreement report")
    for needle in ("Generated 2026-01-01 12:00", "## Forward Defence", "## Front Foot Drive", "stride_length",
                   "0.58", "moderate", "(too few)", "Key-moment timing", "+80", "Pearson", "**2** coach(es)",
                   "Rules to look at first", "How to read this"):
        assert needle in report, needle


def test_report_filters_and_empty_and_mixed_versions():
    only_fd = ag.build_report(full_rows(), shot="front_foot_drive", now=NOW)
    assert "Forward Defence" not in only_fd and "spine_tilt" in only_fd
    assert "No labels yet" in ag.build_report([], now=NOW)
    assert "No labels yet" in ag.build_report(full_rows(), coach="nobody", now=NOW)
    mixed = [*full_rows(), row("rule", "x", "pass", "pass", clip="z", rules_hash="h2")]
    assert "different versions of the rules" in ag.build_report(mixed, now=NOW)


def test_report_script_writes_markdown(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    snapshot = {"clip_id": "c-1", "shot_type": "forward_defence", "rules_hash": "h", "fps": 25.0, "total": 80,
                "confidence": 0.9, "moments": {"stance": 3, "front_foot_landing": 9, "contact": 12},
                "rules": [{"id": "stride_length", "status": "pass", "value": 0.7, "confidence": 0.9}]}
    rows = labels.build_rows(snapshot, "alex", {"stance": 4, "front_foot_landing": 9, "contact": 14},
                             {"stride_length": "pass"}, 75.0)
    labels.save_labels(rows, labels_csv)
    out = tmp_path / "report.md"
    script = Path(__file__).resolve().parent.parent / "scripts" / "agreement_report.py"
    done = subprocess.run([sys.executable, str(script), "--labels", str(labels_csv), "--out", str(out)],
                          capture_output=True, text=True, check=True)
    text = out.read_text(encoding="utf-8")
    assert "AI vs coach agreement report" in text and "stride_length" in text and "contact" in text
    assert "Saved to" in done.stdout
