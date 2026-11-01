"""Stage 7: rule engine - metrics -> score out of 100, sub-scores and fixes.

All coaching judgement lives in YAML files under ``rules/`` (one per shot), so
coaches can change numbers and wording without touching code. See the header of
``rules/forward_defence.yaml`` for the file format.

Scoring maths
-------------
Each rule looks at one metric and has an *ideal range* (``min`` and/or ``max``)
and a *tolerance*::

    score = 1                                  inside the ideal range
    score = 1 - distance_outside / tolerance   in the tolerance band (linear falloff)
    score = 0                                  tolerance or more outside the range

``distance_outside`` and ``tolerance`` are in the metric's own units (torso
lengths, degrees, ...). A rule *passes* when ``score >= pass_score``
(default 0.9). Rules whose metric could not be measured are *skipped*: they are
left out of the score and lower the confidence instead.

* total score and each category sub-score = 100 * weighted mean of rule scores
  (weights from the YAML),
* confidence = weighted mean of the metrics' confidences over *all* rules, with a
  skipped rule counting as 0, so missing data lowers it,
* fixes: failed rules ranked by ``weight * (1 - score)``; the top ``max_fixes``
  (default 2) are returned with their fix and drill text. A rule can give
  different text when the value is too low or too high.

Rule files are validated when loaded; every problem found is reported at once,
with the rule it belongs to.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.metrics import METRIC_NAMES, Metrics

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
SHOT_TYPES = ("forward_defence", "front_foot_drive")
DEFAULT_PASS_SCORE = 0.9
DEFAULT_MAX_FIXES = 2

_TOP_KEYS = {"shot", "title", "status", "categories", "pass_score", "rules"}
_REQUIRED_RULE_KEYS = {"id", "label", "category", "metric", "ideal", "tolerance", "weight",
                       "fix", "drill"}
_OPTIONAL_RULE_KEYS = {"moment"}


class RuleFileError(ValueError):
    """A rules file is malformed. ``problems`` lists everything that is wrong."""

    def __init__(self, source: str, problems: list[str]):
        self.source, self.problems = source, problems
        super().__init__(f"{source}: {len(problems)} problem(s) found:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Rule:
    """One coach-defined rule: what to measure, the ideal range, tolerance, weight and fix text."""
    id: str
    label: str
    category: str
    metric_key: str             # full metric name, e.g. "head_over_front_knee_landing"
    ideal_min: float | None
    ideal_max: float | None
    tolerance: float
    weight: float
    fix: Mapping[str, str]      # "low" / "high" -> text
    drill: Mapping[str, str]

    @property
    def ideal_text(self) -> str:
        """The ideal range in words, e.g. "0.5 to 0.9" or "at most 0.15"."""
        if self.ideal_min is not None and self.ideal_max is not None:
            return f"{self.ideal_min:g} to {self.ideal_max:g}"
        if self.ideal_min is not None:
            return f"at least {self.ideal_min:g}"
        return f"at most {self.ideal_max:g}"


@dataclass(frozen=True)
class RuleSet:
    """All rules for one shot type, as loaded from its YAML file."""
    shot: str
    title: str
    status: str | None          # e.g. "PLACEHOLDER - to be set by coaches"
    categories: tuple[str, ...]
    pass_score: float
    rules: tuple[Rule, ...]
    source: str = "<string>"


@dataclass(frozen=True)
class RuleResult:
    """How one rule scored on a clip (or that it was skipped)."""
    rule: Rule
    value: float | None         # None when the metric could not be measured
    confidence: float
    score: float | None         # 0..1; None when skipped
    side: str | None            # "low" / "high" when outside the ideal range
    passed: bool | None

    @property
    def status(self) -> str:
        """"passed", "failed" or "skipped"."""
        return "skipped" if self.score is None else ("passed" if self.passed else "failed")


@dataclass(frozen=True)
class SubScore:
    """Score of one category (e.g. Footwork) out of 100."""
    category: str
    score: float | None         # 0..100; None if none of its rules could be measured
    confidence: float
    rules_scored: int
    rules_total: int


@dataclass(frozen=True)
class Fix:
    """One recommended fix with its drill, ranked by weight times deficit."""
    rule_id: str
    label: str
    category: str
    text: str
    drill: str
    value: float
    ideal_text: str
    priority: float             # weight * deficit, the ranking key


@dataclass(frozen=True)
class ScoreResult:
    """The full scoring outcome: total, sub-scores, per-rule results and top fixes."""
    shot: str
    title: str
    status: str | None
    total: float | None         # 0..100; None if no rule could be scored
    confidence: float
    sub_scores: dict[str, SubScore]
    rule_results: tuple[RuleResult, ...]
    fixes: tuple[Fix, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> list[RuleResult]:
        """Rules that met their target."""
        return [r for r in self.rule_results if r.status == "passed"]

    @property
    def failed(self) -> list[RuleResult]:
        """Rules that missed their target."""
        return [r for r in self.rule_results if r.status == "failed"]

    @property
    def skipped(self) -> list[RuleResult]:
        """Rules whose metric could not be measured."""
        return [r for r in self.rule_results if r.status == "skipped"]


# --------------------------------------------------------------------------- loading
def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _parse_text(value: object, where: str, field_name: str, ideal_min: float | None,
                ideal_max: float | None,
                problems: list[str]) -> dict[str, str]:
    """A fix/drill is one string, or {low: ..., high: ...} for direction-specific wording."""
    if isinstance(value, str) and value.strip():
        return {"low": value.strip(), "high": value.strip()}
    if isinstance(value, dict):
        extra = set(value) - {"low", "high"}
        if extra:
            problems.append(f"{where}: '{field_name}' may only have 'low' and 'high', found {sorted(extra)}")
        needed = [side for side, bound in (("low", ideal_min), ("high", ideal_max)) if bound is not None]
        out = {}
        for side in needed:
            text = value.get(side)
            if isinstance(text, str) and text.strip():
                out[side] = text.strip()
            else:
                problems.append(f"{where}: '{field_name}' needs a '{side}' text "
                                f"(the rule can fail by being too {side})")
        for side in ("low", "high"):                    # unused side: harmless, keep for completeness
            out.setdefault(side, out.get("high" if side == "low" else "low") or "")
        return out
    problems.append(f"{where}: '{field_name}' must be text, or a mapping with 'low'/'high' text")
    return {"low": "", "high": ""}


def _parse_rule(raw: object, index: int, categories: tuple[str, ...], seen_ids: set[str],
                problems: list[str]) -> Rule | None:
    if not isinstance(raw, dict):
        problems.append(f"rule #{index}: each rule must be a mapping of fields")
        return None
    rule_id = raw.get("id")
    where = f"rule '{rule_id}'" if isinstance(rule_id, str) and rule_id else f"rule #{index}"
    start = len(problems)

    missing = _REQUIRED_RULE_KEYS - set(raw)
    if missing:
        problems.append(f"{where}: missing field(s) {sorted(missing)}")
    unknown = set(raw) - _REQUIRED_RULE_KEYS - _OPTIONAL_RULE_KEYS
    if unknown:
        problems.append(f"{where}: unknown field(s) {sorted(unknown)} (typo?)")

    if "id" in raw:
        if not (isinstance(rule_id, str) and rule_id.strip()):
            problems.append(f"{where}: 'id' must be non-empty text")
        elif rule_id in seen_ids:
            problems.append(f"{where}: duplicate id")
        else:
            seen_ids.add(rule_id)
    if "label" in raw and not (isinstance(raw["label"], str) and raw["label"].strip()):
        problems.append(f"{where}: 'label' must be non-empty text")
    if "category" in raw and raw["category"] not in categories:
        problems.append(f"{where}: category {raw['category']!r} is not one of {list(categories)}")

    metric_key = ""
    if "metric" in raw:
        metric, moment = raw["metric"], raw.get("moment")
        metric_key = f"{metric}_{moment}" if moment else str(metric)
        if metric_key not in METRIC_NAMES:
            stems = [n for n in METRIC_NAMES if n == metric or n.startswith(f"{metric}_")]
            known = sorted({n.removesuffix("_landing").removesuffix("_contact") for n in METRIC_NAMES})
            hint = (f"valid names for metric '{metric}': {stems}" if stems
                    else f"unknown metric; known metrics: {known}")
            problems.append(f"{where}: no metric named {metric_key!r} ({hint})")

    ideal_min = ideal_max = None
    ideal = raw.get("ideal")
    if "ideal" in raw:
        if not isinstance(ideal, dict) or not ideal or set(ideal) - {"min", "max"}:
            problems.append(f"{where}: 'ideal' must be a mapping with 'min' and/or 'max'")
        else:
            ideal_min, ideal_max = ideal.get("min"), ideal.get("max")
            for name, v in (("min", ideal_min), ("max", ideal_max)):
                if v is not None and not _is_number(v):
                    problems.append(f"{where}: ideal '{name}' must be a number, got {v!r}")
            if _is_number(ideal_min) and _is_number(ideal_max) and ideal_min > ideal_max:
                problems.append(f"{where}: ideal min ({ideal_min}) is greater than max ({ideal_max})")
    for name in ("tolerance", "weight"):
        if name in raw and not (_is_number(raw[name]) and raw[name] > 0):
            problems.append(f"{where}: '{name}' must be a number greater than 0, got {raw[name]!r}")

    ideal_min = ideal_min if _is_number(ideal_min) else None
    ideal_max = ideal_max if _is_number(ideal_max) else None
    fix = _parse_text(raw.get("fix"), where, "fix", ideal_min, ideal_max, problems) \
        if "fix" in raw else {"low": "", "high": ""}
    drill = _parse_text(raw.get("drill"), where, "drill", ideal_min, ideal_max, problems) \
        if "drill" in raw else {"low": "", "high": ""}

    if len(problems) > start:
        return None
    return Rule(rule_id, raw["label"].strip(), raw["category"], metric_key, ideal_min, ideal_max,
                float(raw["tolerance"]), float(raw["weight"]), fix, drill)


def parse_rules(text: str, source: str = "<string>") -> RuleSet:
    """Parse and validate rules YAML text; raises ``RuleFileError`` listing every problem."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise RuleFileError(source, [f"not valid YAML{where}: {getattr(exc, 'problem', exc)}"]) from exc
    if not isinstance(raw, dict):
        raise RuleFileError(source, ["the file must contain a mapping with 'shot', 'categories' and 'rules'"])

    problems: list[str] = []
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        problems.append(f"unknown top-level field(s) {sorted(unknown)} (typo?)")
    for name in ("shot", "categories", "rules"):
        if name not in raw:
            problems.append(f"missing top-level field '{name}'")

    categories = raw.get("categories")
    if not (isinstance(categories, list) and categories and all(isinstance(c, str) and c for c in categories)):
        if "categories" in raw:
            problems.append("'categories' must be a non-empty list of names")
        categories = []
    pass_score = raw.get("pass_score", DEFAULT_PASS_SCORE)
    if not (_is_number(pass_score) and 0 < pass_score <= 1):
        problems.append(f"'pass_score' must be a number in (0, 1], got {pass_score!r}")
        pass_score = DEFAULT_PASS_SCORE
    for name in ("shot", "title", "status"):
        if name in raw and not (isinstance(raw[name], str) and raw[name].strip()):
            problems.append(f"'{name}' must be non-empty text")

    rules: list[Rule] = []
    raw_rules = raw.get("rules")
    if "rules" in raw and not (isinstance(raw_rules, list) and raw_rules):
        problems.append("'rules' must be a non-empty list")
    else:
        seen: set[str] = set()
        for i, item in enumerate(raw_rules or [], start=1):
            rule = _parse_rule(item, i, tuple(categories), seen, problems)
            if rule:
                rules.append(rule)

    if problems:
        raise RuleFileError(source, problems)
    shot = raw["shot"].strip()
    return RuleSet(shot, (raw.get("title") or shot).strip(), (raw.get("status") or "").strip() or None,
                   tuple(categories), float(pass_score), tuple(rules), source)


def load_rules(shot_or_path: str | Path) -> RuleSet:
    """Load a rule set by shot name (``"forward_defence"``) or by file path."""
    if isinstance(shot_or_path, str) and shot_or_path in SHOT_TYPES:
        path = RULES_DIR / f"{shot_or_path}.yaml"
    else:
        path = Path(shot_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path} (known shots: {list(SHOT_TYPES)})")
    return parse_rules(path.read_text(encoding="utf-8"), source=str(path))


# --------------------------------------------------------------------------- scoring
def rule_score(value: float, ideal_min: float | None, ideal_max: float | None,
               tolerance: float) -> tuple[float, str | None]:
    """Score one value in [0, 1] and say which side it missed on ("low", "high" or None)."""
    if ideal_min is not None and value < ideal_min:
        gap, side = ideal_min - value, "low"
    elif ideal_max is not None and value > ideal_max:
        gap, side = value - ideal_max, "high"
    else:
        return 1.0, None
    return max(0.0, 1.0 - gap / tolerance), side


def _weighted(results: list[RuleResult]) -> tuple[float | None, float]:
    """(score 0..100 over scored rules, confidence over all given rules incl. skipped)."""
    scored = [r for r in results if r.score is not None]
    total_weight = sum(r.rule.weight for r in results)
    confidence = (sum(r.rule.weight * r.confidence for r in scored) / total_weight) if total_weight else 0.0
    if not scored:
        return None, 0.0
    scored_weight = sum(r.rule.weight for r in scored)
    return 100.0 * sum(r.rule.weight * r.score for r in scored) / scored_weight, confidence


def score_shot(metrics: Metrics, ruleset: RuleSet, max_fixes: int = DEFAULT_MAX_FIXES) -> ScoreResult:
    """Apply ``ruleset`` to measured ``metrics``; see the module docstring for the maths."""
    results: list[RuleResult] = []
    for rule in ruleset.rules:
        metric = metrics.get(rule.metric_key)
        if metric is None or not math.isfinite(metric.value):
            results.append(RuleResult(rule, None, 0.0, None, None, None))
            continue
        score, side = rule_score(metric.value, rule.ideal_min, rule.ideal_max, rule.tolerance)
        results.append(RuleResult(rule, metric.value, metric.confidence, score, side,
                                  score >= ruleset.pass_score))

    total, confidence = _weighted(results)
    sub_scores = {}
    for category in ruleset.categories:
        group = [r for r in results if r.rule.category == category]
        score, conf = _weighted(group)
        sub_scores[category] = SubScore(category, score, conf,
                                        sum(r.score is not None for r in group), len(group))

    failed = [r for r in results if r.status == "failed"]
    failed.sort(key=lambda r: r.rule.weight * (1 - r.score), reverse=True)   # stable: file order breaks ties
    fixes = tuple(
        Fix(r.rule.id, r.rule.label, r.rule.category, r.rule.fix[r.side], r.rule.drill[r.side],
            r.value, r.rule.ideal_text, r.rule.weight * (1 - r.score))
        for r in failed[:max_fixes]
    )
    return ScoreResult(ruleset.shot, ruleset.title, ruleset.status, total, confidence, sub_scores,
                       tuple(results), fixes)
