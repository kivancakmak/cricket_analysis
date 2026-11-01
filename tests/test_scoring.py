"""Scoring-engine tests: the maths, ranking, aggregation and YAML validation."""
import textwrap

import pytest

from src.metrics import Metric, Metrics
from src.scoring import (
    SHOT_TYPES,
    RuleFileError,
    load_rules,
    parse_rules,
    rule_score,
    score_shot,
)


# ------------------------------------------------------------------ the per-rule maths
@pytest.mark.parametrize(("value", "expected", "side"), [
    (0.5, 1.0, None),        # inside
    (0.2, 1.0, None),        # on the lower edge
    (0.8, 1.0, None),        # on the upper edge
    (0.1, 0.5, "low"),       # halfway down the falloff band (tolerance 0.2)
    (0.9, 0.5, "high"),
    (0.0, 0.0, "low"),       # a full tolerance outside
    (-5.0, 0.0, "low"),      # never below zero
    (3.0, 0.0, "high"),
])
def test_rule_score_linear_falloff(value, expected, side):
    score, got_side = rule_score(value, 0.2, 0.8, tolerance=0.2)
    assert score == pytest.approx(expected)
    assert got_side == side


def test_one_sided_rules():
    assert rule_score(-100, None, 0.15, 0.25) == (1.0, None)          # max only: low is fine
    assert rule_score(0.275, None, 0.15, 0.25) == (pytest.approx(0.5), "high")
    assert rule_score(1000, 100, None, 30) == (1.0, None)             # min only: high is fine
    assert rule_score(85, 100, None, 30) == (pytest.approx(0.5), "low")


# ------------------------------------------------------------------ helpers
def rules_yaml(rules_body: str, header: str = "") -> str:
    return textwrap.dedent(f"""\
        shot: test_shot
        title: Test shot
        status: "PLACEHOLDER – to be set by coaches"
        categories: [Head, Feet]
        {header}
        rules:
        """) + textwrap.indent(textwrap.dedent(rules_body), "  ")


def rule(id_, metric="stride_length", moment="landing", category="Feet", ideal="{min: 0.4, max: 0.6}",
         tolerance=0.2, weight=1, fix='"Fix {id}."', drill='"Drill {id}."'):
    moment_line = f"moment: {moment}" if moment else ""
    return textwrap.dedent(f"""\
        - id: {id_}
          label: Label {id_}
          category: {category}
          metric: {metric}
          {moment_line}
          ideal: {ideal}
          tolerance: {tolerance}
          weight: {weight}
          fix: {fix.replace("{id}", id_)}
          drill: {drill.replace("{id}", id_)}
        """)


def metrics(**values):
    """Metrics from name=value (confidence 0.9 each)."""
    return Metrics({name: Metric(v, 0.9) for name, v in values.items()})


# ------------------------------------------------------------------ aggregation
def test_total_and_sub_scores_are_weighted_means():
    rs = parse_rules(rules_yaml(
        rule("a", category="Head", metric="balance", moment="contact", weight=3)
        + rule("b", category="Feet", weight=1)))
    # a: in range -> 1.0; b: 0.1 below 0.4 with tolerance 0.2 -> 0.5
    result = score_shot(metrics(balance_contact=0.5, stride_length_landing=0.3), rs)
    assert result.total == pytest.approx((3 * 1.0 + 1 * 0.5) / 4 * 100)      # 87.5
    assert result.sub_scores["Head"].score == pytest.approx(100.0)
    assert result.sub_scores["Feet"].score == pytest.approx(50.0)


def test_passed_and_failed_lists_carry_measured_values():
    rs = parse_rules(rules_yaml(
        rule("a", category="Head", metric="balance", moment="contact")
        + rule("b", category="Feet")))
    result = score_shot(metrics(balance_contact=0.5, stride_length_landing=0.3), rs)
    assert [r.rule.id for r in result.passed] == ["a"]
    assert [r.rule.id for r in result.failed] == ["b"]
    failed = result.failed[0]
    assert failed.value == pytest.approx(0.3) and failed.side == "low"
    assert failed.rule.ideal_text == "0.4 to 0.6"


def test_pass_threshold_decides_pass_vs_fail():
    body = rule("b", ideal="{min: 0.4, max: 0.6}", tolerance=0.2)
    value = 0.38                       # score 0.9 exactly
    strict = score_shot(metrics(stride_length_landing=value), parse_rules(rules_yaml(body, "pass_score: 0.95")))
    lenient = score_shot(metrics(stride_length_landing=value), parse_rules(rules_yaml(body, "pass_score: 0.85")))
    assert [r.rule.id for r in strict.failed] == ["b"]
    assert [r.rule.id for r in lenient.passed] == ["b"]


# ------------------------------------------------------------------ fixes
def test_fixes_are_ranked_by_weight_times_deficit_and_limited():
    rs = parse_rules(rules_yaml(
        rule("small_deficit_big_weight", weight=10, metric="balance", moment="contact")      # 10 * 0.25
        + rule("big_deficit_small_weight", weight=2, metric="stride_length")                # 2 * 1.0
        + rule("medium", weight=6, metric="spine_tilt", moment="landing",
               ideal="{min: 10, max: 20}", tolerance=10)                                    # 6 * 0.5
        + rule("passing", weight=50, metric="front_knee_angle", moment="landing",
               ideal="{min: 100, max: 150}", tolerance=30)))
    result = score_shot(metrics(
        balance_contact=0.35,            # 0.05 below 0.4 -> score 0.75 -> deficit 0.25
        stride_length_landing=0.0,       # deficit 1.0
        spine_tilt_landing=5.0,          # 5 below 10 -> score 0.5
        front_knee_angle_landing=120.0), rs)
    assert [f.rule_id for f in result.fixes] == ["medium", "small_deficit_big_weight"]
    # 'big_deficit_small_weight' (2.0) is cut by the two-fix limit; 'passing' never appears
    assert [f.priority for f in result.fixes] == pytest.approx([3.0, 2.5])


def test_fixes_ordering_by_priority_value():
    rs = parse_rules(rules_yaml(
        rule("a", weight=1, metric="stride_length")
        + rule("b", weight=5, metric="balance", moment="contact")
        + rule("c", weight=3, metric="spine_tilt", moment="landing", ideal="{min: 10, max: 20}", tolerance=10)))
    result = score_shot(metrics(stride_length_landing=0.0, balance_contact=0.3, spine_tilt_landing=5.0), rs,
                        max_fixes=3)
    # a: 1 * 1.0 = 1.0; b: 5 * 0.5 = 2.5; c: 3 * 0.5 = 1.5
    assert [f.rule_id for f in result.fixes] == ["b", "c", "a"]
    assert [f.priority for f in result.fixes] == pytest.approx([2.5, 1.5, 1.0])
    assert len(score_shot(metrics(stride_length_landing=0.0, balance_contact=0.3, spine_tilt_landing=5.0),
                          rs).fixes) == 2


def test_fix_text_depends_on_which_side_was_missed():
    body = rule("s", fix='{low: "too short {id}", high: "too long {id}"}',
                drill='{low: "drill short", high: "drill long"}')
    rs = parse_rules(rules_yaml(body))
    low = score_shot(metrics(stride_length_landing=0.1), rs).fixes[0]
    high = score_shot(metrics(stride_length_landing=0.9), rs).fixes[0]
    assert (low.text, low.drill) == ("too short s", "drill short")
    assert (high.text, high.drill) == ("too long s", "drill long")
    assert low.value == pytest.approx(0.1) and low.ideal_text == "0.4 to 0.6"


def test_no_fixes_when_everything_passes():
    rs = parse_rules(rules_yaml(rule("a")))
    result = score_shot(metrics(stride_length_landing=0.5), rs)
    assert result.total == pytest.approx(100.0) and result.fixes == ()


# ------------------------------------------------------------------ missing data and confidence
def test_skipped_rules_are_excluded_from_score_but_lower_confidence():
    rs = parse_rules(rules_yaml(
        rule("a", category="Head", metric="balance", moment="contact", weight=1)
        + rule("b", category="Feet", weight=1)))
    result = score_shot(metrics(balance_contact=0.5), rs)          # stride_length_landing missing
    assert result.total == pytest.approx(100.0)                    # only 'a' counts
    assert [r.rule.id for r in result.skipped] == ["b"]
    assert result.confidence == pytest.approx(0.9 * 1 / 2)         # skipped rule counts as 0
    assert result.sub_scores["Feet"].score is None
    assert result.sub_scores["Feet"].rules_scored == 0 and result.sub_scores["Feet"].rules_total == 1


def test_confidence_is_weight_averaged():
    rs = parse_rules(rules_yaml(
        rule("a", category="Head", metric="balance", moment="contact", weight=3)
        + rule("b", category="Feet", weight=1)))
    m = Metrics({"balance_contact": Metric(0.5, 1.0), "stride_length_landing": Metric(0.5, 0.2)})
    assert score_shot(m, rs).confidence == pytest.approx((3 * 1.0 + 1 * 0.2) / 4)


def test_no_measurable_metrics_gives_no_score():
    rs = parse_rules(rules_yaml(rule("a")))
    result = score_shot(Metrics({}), rs)
    assert result.total is None and result.confidence == 0.0 and result.fixes == ()


# ------------------------------------------------------------------ validation
def problems_of(text: str) -> list[str]:
    with pytest.raises(RuleFileError) as info:
        parse_rules(text, source="coach_file.yaml")
    assert "coach_file.yaml" in str(info.value)
    return info.value.problems


def test_all_problems_are_reported_at_once_with_the_rule_name():
    problems = problems_of(rules_yaml(
        rule("bad_metric", metric="head_height", moment=None)
        + rule("bad_moment", moment="stance")
        + rule("bad_tolerance", tolerance=-1)
        + rule("bad_weight", weight=0)
        + rule("bad_category", category="Mind")
        + rule("bad_range", ideal="{min: 5, max: 1}")
        + rule("dup")
        + rule("dup")))
    text = "\n".join(problems)
    assert "rule 'bad_metric'" in text and "unknown metric" in text
    assert "rule 'bad_moment'" in text and "stride_length_landing" in text      # suggests valid names
    assert "rule 'bad_tolerance'" in text and "'tolerance' must be a number greater than 0" in text
    assert "rule 'bad_weight'" in text and "'weight'" in text
    assert "rule 'bad_category'" in text and "'Mind'" in text
    assert "rule 'bad_range'" in text and "greater than max" in text
    assert "duplicate id" in text
    assert len(problems) >= 7


def test_typos_and_missing_fields_are_caught():
    problems = problems_of(rules_yaml("""\
        - id: oops
          label: Oops
          category: Feet
          metric: stride_length
          moment: landing
          ideal: {min: 0.4}
          tolrance: 0.2
          weight: 1
          fix: "x"
          drill: "y"
        """))
    text = "\n".join(problems)
    assert "tolrance" in text and "missing field" in text and "tolerance" in text


def test_direction_specific_text_must_cover_every_failing_side():
    problems = problems_of(rules_yaml(rule("r", fix='{low: "only low {id}"}')))
    assert any("'fix' needs a 'high' text" in p for p in problems)
    # one-sided rule: only the side that can fail needs text
    parse_rules(rules_yaml(rule("r", ideal="{max: 0.2}", fix='{high: "too much {id}"}')))


def test_top_level_problems():
    assert any("valid YAML" in p for p in problems_of("rules: [unclosed"))
    assert any("mapping" in p for p in problems_of("- just\n- a list"))
    assert any("'rules'" in p for p in problems_of("shot: x\ncategories: [A]\nrules: []"))
    assert any("pass_score" in p for p in problems_of(rules_yaml(rule("a"), "pass_score: 1.5")))
    assert any("unknown top-level" in p for p in problems_of(rules_yaml(rule("a"), "colour: red")))


# ------------------------------------------------------------------ shipped coach files
@pytest.mark.parametrize("shot", SHOT_TYPES)
def test_shipped_rule_files_are_valid_and_marked_as_placeholders(shot):
    rs = load_rules(shot)
    assert rs.shot == shot
    assert rs.status and "PLACEHOLDER" in rs.status
    assert set(rs.categories) == {"Head & Balance", "Footwork", "Body Shape"}
    assert {r.category for r in rs.rules} == set(rs.categories)
    assert all(r.fix["low"] and r.fix["high"] and r.drill["low"] and r.drill["high"] for r in rs.rules)


@pytest.mark.parametrize("shot", SHOT_TYPES)
def test_ideal_values_score_100_and_extreme_values_fail_every_rule(shot):
    rs = load_rules(shot)

    def midpoint(r):
        lo = r.ideal_min if r.ideal_min is not None else r.ideal_max - 1
        hi = r.ideal_max if r.ideal_max is not None else r.ideal_min + 1
        return (lo + hi) / 2

    ideal = Metrics({r.metric_key: Metric(midpoint(r), 1.0) for r in rs.rules})
    perfect = score_shot(ideal, rs)
    assert perfect.total == pytest.approx(100.0) and perfect.fixes == ()
    assert all(s.score == pytest.approx(100.0) for s in perfect.sub_scores.values())

    terrible = score_shot(Metrics({r.metric_key: Metric(1e6, 1.0) for r in rs.rules}), rs)
    assert len(terrible.fixes) == 2          # capped at two, with real text
    assert all(f.text and f.drill for f in terrible.fixes)


def test_missing_rules_file_names_the_known_shots(tmp_path):
    with pytest.raises(FileNotFoundError, match="forward_defence"):
        load_rules(tmp_path / "nope.yaml")
