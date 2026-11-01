"""HTML cards and table rows for the web UI, built from an ``AnalysisResult``.

Pure functions (string/list in, string/list out) shared by the analysis tab and the
coach-labelling tab, so they can be tested without a browser.
"""
from __future__ import annotations

import html

from src.pipeline import SHOT_LABELS, AnalysisResult

DISCLAIMER = ("Demo only: the scoring rules are placeholders until a coach has set them, so treat the "
              "numbers as a guide, not a verdict.")


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
