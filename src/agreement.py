"""AI-vs-coach agreement statistics and the markdown report built from them.

Input is the rows of ``labels/labels.csv`` (see ``src.labels``). Output is a
markdown report with, per shot type:

* **per-rule agreement** - accuracy and Cohen's kappa of the AI's pass/fail
  against the coach's, plus how the disagreements split (AI too harsh / too lenient),
* **key-moment timing error** in milliseconds (AI frame minus coach frame, so a
  positive number means the AI is *late*),
* **overall score agreement** - Pearson and Spearman correlation, mean absolute
  error and bias (AI minus coach).

Each coach labelling of a clip is one sample; if several coaches label the same
clip they are pooled unless you filter by coach. Sample sizes are always shown,
and small ones are flagged: with fewer than ``MIN_RELIABLE`` samples the numbers
are anecdotes, not evidence.

Pure functions only (no file or UI access) so they are easy to test.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

MIN_RELIABLE = 10          # fewer samples than this are flagged as too few to rely on
NAN = float("nan")


# --------------------------------------------------------------------------- basic statistics
def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two equal-length label sequences; NaN when undefined.

    kappa = (observed agreement - agreement expected by chance) / (1 - expected).
    It is undefined when chance agreement is already 1 (both raters always give
    the same single label), and for empty input.
    """
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    n = len(a)
    if n == 0:
        return NAN
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] / n * cb[label] / n for label in set(ca) | set(cb))
    return NAN if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def kappa_words(kappa: float) -> str:
    """Landis & Koch (1977) wording for a kappa value."""
    if math.isnan(kappa):
        return "n/a"
    if kappa < 0:
        return "worse than chance"
    for limit, words in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= limit:
            return words
    return "almost perfect"


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN with fewer than 3 points or no spread."""
    if len(x) < 3:
        return NAN
    x_arr, y_arr = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x_arr) == 0 or np.ptp(y_arr) == 0:
        return NAN
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        tied = values == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    if len(x) < 3:
        return NAN
    return pearson(_ranks(np.asarray(x, float)), _ranks(np.asarray(y, float)))


# --------------------------------------------------------------------------- per-kind statistics
@dataclass(frozen=True)
class RuleStat:
    item: str
    n: int
    accuracy: float
    kappa: float
    coach_fail_rate: float
    ai_fail_rate: float
    ai_too_harsh: int         # AI said fail, coach said pass
    ai_too_lenient: int       # AI said pass, coach said fail
    ai_skipped: int           # coach labelled it but the AI could not measure it


@dataclass(frozen=True)
class MomentStat:
    item: str
    n: int
    mean_error_ms: float      # signed: positive = AI is later than the coach
    mae_ms: float
    median_abs_ms: float
    max_abs_ms: float
    within_100ms: float
    within_200ms: float


@dataclass(frozen=True)
class ScoreStat:
    n: int
    pearson: float
    spearman: float
    mae: float
    bias: float               # mean(AI - coach)
    rmse: float


def _float(text) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rule_stat(item: str, pairs: list[tuple[str, str]], skipped: int) -> RuleStat:
    coach, ai = [c for c, _ in pairs], [a for _, a in pairs]
    n = len(pairs)
    return RuleStat(
        item, n,
        accuracy=sum(c == a for c, a in pairs) / n if n else NAN,
        kappa=cohens_kappa(coach, ai),
        coach_fail_rate=coach.count("fail") / n if n else NAN,
        ai_fail_rate=ai.count("fail") / n if n else NAN,
        ai_too_harsh=sum(c == "pass" and a == "fail" for c, a in pairs),
        ai_too_lenient=sum(c == "fail" and a == "pass" for c, a in pairs),
        ai_skipped=skipped,
    )


def rule_agreement(rows: list[dict]) -> list[RuleStat]:
    """One stat per rule (in first-seen order) plus a final pooled ``(all rules)`` entry."""
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["kind"] != "rule" or r["coach_label"] not in ("pass", "fail"):
            continue
        if r["ai_label"] in ("pass", "fail"):
            pairs[r["item"]].append((r["coach_label"], r["ai_label"]))
        else:
            skipped[r["item"]] += 1
            pairs.setdefault(r["item"], [])
    stats = [_rule_stat(item, p, skipped[item]) for item, p in pairs.items()]
    if stats:
        pooled = [pair for p in pairs.values() for pair in p]
        stats.append(_rule_stat("(all rules)", pooled, sum(skipped.values())))
    return stats


def timing_agreement(rows: list[dict]) -> list[MomentStat]:
    """Key-moment timing error in milliseconds, per moment."""
    errors: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["kind"] != "moment":
            continue
        coach, ai, fps = _float(r["coach_label"]), _float(r["ai_label"]), _float(r.get("fps"))
        if coach is None or ai is None or not fps:
            continue
        errors[r["item"]].append((ai - coach) / fps * 1000.0)
    stats = []
    for item, errs in errors.items():
        a = np.abs(errs)
        stats.append(MomentStat(item, len(errs), float(np.mean(errs)), float(a.mean()), float(np.median(a)),
                                float(a.max()), float((a <= 100).mean()), float((a <= 200).mean())))
    return stats


def score_agreement(rows: list[dict]) -> ScoreStat | None:
    """Overall-score agreement (None when no clip has both a coach and an AI score)."""
    pairs = [(c, a) for r in rows if r["kind"] == "overall"
             for c, a in [(_float(r["coach_label"]), _float(r["ai_label"]))] if c is not None and a is not None]
    if not pairs:
        return None
    coach, ai = [c for c, _ in pairs], [a for _, a in pairs]
    diff = np.asarray(ai) - np.asarray(coach)
    return ScoreStat(len(pairs), pearson(coach, ai), spearman(coach, ai), float(np.abs(diff).mean()),
                     float(diff.mean()), float(np.sqrt((diff ** 2).mean())))


# --------------------------------------------------------------------------- markdown report
def _num(value: float, fmt: str = ".2f") -> str:
    return "n/a" if value is None or math.isnan(value) else format(value, fmt)


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def _flag(n: int) -> str:
    return " (too few)" if n < MIN_RELIABLE else ""


def _rule_table(stats: list[RuleStat]) -> list[str]:
    lines = ["| Rule | n | Agreement | Kappa | Reading | Coach fails | AI fails | AI too harsh | AI too lenient | AI skipped |",
             "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
    for s in stats:
        lines.append(f"| {s.item} | {s.n}{_flag(s.n)} | {_pct(s.accuracy)} | {_num(s.kappa)} | {kappa_words(s.kappa)} | "
                     f"{_pct(s.coach_fail_rate)} | {_pct(s.ai_fail_rate)} | {s.ai_too_harsh} | {s.ai_too_lenient} | "
                     f"{s.ai_skipped} |")
    return lines


def _timing_table(stats: list[MomentStat]) -> list[str]:
    lines = ["| Key moment | n | Mean error (ms) | Mean abs error (ms) | Median abs (ms) | Worst (ms) | Within 100 ms | Within 200 ms |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for s in stats:
        lines.append(f"| {s.item} | {s.n}{_flag(s.n)} | {s.mean_error_ms:+.0f} | {s.mae_ms:.0f} | "
                     f"{s.median_abs_ms:.0f} | {s.max_abs_ms:.0f} | {_pct(s.within_100ms)} | {_pct(s.within_200ms)} |")
    return lines


def build_report(rows: list[dict], source: str = "labels.csv", shot: str | None = None,
                 coach: str | None = None, now: datetime | None = None) -> str:
    """Markdown agreement report for the given label rows (optionally filtered by shot/coach)."""
    if shot:
        rows = [r for r in rows if r["shot_type"] == shot]
    if coach:
        rows = [r for r in rows if r["coach_id"] == coach]
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    out = ["# AI vs coach agreement report", "",
           f"Generated {stamp} from `{source}`"
           + (f", shot: **{shot}**" if shot else "") + (f", coach: **{coach}**" if coach else "") + ".", ""]
    if not rows:
        empty = ("**No labels yet.** Use the *Coach labelling* tab in the app to add some, "
                 "then run this report again.")
        return "\n".join([*out, empty]) + "\n"

    labellings = {(r["clip_id"], r["shot_type"], r["coach_id"]) for r in rows}
    counts = (f"- **{len(labellings)}** labelling(s) of **{len({r['clip_id'] for r in rows})}** clip(s) "
              f"by **{len({r['coach_id'] for r in rows})}** coach(es)")
    out += [counts, f"- Samples below {MIN_RELIABLE} are marked *(too few)*: treat them as anecdotes, not evidence.", ""]

    for shot_type in sorted({r["shot_type"] for r in rows}):
        shot_rows = [r for r in rows if r["shot_type"] == shot_type]
        out += [f"## {shot_type.replace('_', ' ').title()}", ""]
        hashes = sorted({r["rules_hash"] for r in shot_rows if r.get("rules_hash")})
        if len(hashes) > 1:
            warning = (f"> **Warning:** the AI values come from {len(hashes)} different versions of the rules "
                       f"file ({', '.join(hashes)}). Agreement mixes old and new thresholds; relabel or filter "
                       "before tuning.")
            out += [warning, ""]

        rules = rule_agreement(shot_rows)
        out += ["### Per-rule pass/fail agreement", ""]
        out += _rule_table(rules) if rules else ["No per-rule labels."]
        if rules:
            worst = sorted((s for s in rules[:-1] if s.n and s.accuracy < 1), key=lambda s: s.accuracy)[:3]
            if worst:
                out += ["", "Rules to look at first (lowest agreement): "
                        + ", ".join(f"`{s.item}` ({_pct(s.accuracy)}, "
                                    f"{'AI too harsh' if s.ai_too_harsh >= s.ai_too_lenient else 'AI too lenient'})"
                                    for s in worst) + "."]

        out += ["", "### Key-moment timing", ""]
        timing = timing_agreement(shot_rows)
        out += _timing_table(timing) if timing else ["No confirmed key-frame labels."]
        if timing:
            out += ["", "Error = AI frame minus coach frame in milliseconds: positive means the AI is late, negative early."]

        out += ["", "### Overall score", ""]
        score = score_agreement(shot_rows)
        if score is None:
            out += ["No overall scores labelled."]
        else:
            row = (f"| {score.n}{_flag(score.n)} | {_num(score.pearson)} | {_num(score.spearman)} | "
                   f"{score.mae:.1f} | {score.bias:+.1f} | {score.rmse:.1f} |")
            out += ["| n | Pearson r | Spearman rho | Mean abs error (points) | Bias, AI - coach (points) | RMSE |",
                    "|--:|--:|--:|--:|--:|--:|", row]
        out.append("")

    howto = ("**How to read this.** Kappa corrects agreement for chance: 0 is no better than guessing, 1 is "
             "perfect; the *Reading* column uses the common Landis & Koch bands. A rule with high agreement but low "
             "kappa usually means nearly everyone passes (or fails) it. A high correlation with a large bias means "
             "the AI ranks clips like the coach but its scale is shifted - fix the rule thresholds, not the weights.")
    out += ["---", howto]
    return "\n".join(out) + "\n"
