"""Pre-registered endpoints for the observer experiment (see PREREG.md).

Everything here is deterministic text checks on the agent's responses. No
model judges anything in the primary endpoint.
"""
import re
import statistics

_LIST_LINE = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+\S")
_TRAILING = " \t\r\n*_`\"'”’)]"


def has_list(text):
    """Two or more lines that start like a bullet or numbered item."""
    return sum(1 for line in str(text).splitlines() if _LIST_LINE.match(line)) >= 2


def ends_with_question(text):
    stripped = str(text).rstrip(_TRAILING)
    return stripped.endswith("?")


def word_count(text):
    return len(str(text).split())


def too_long(text, max_words=150):
    return word_count(text) > max_words


def applicable(spec, number):
    return spec["first"] <= number <= spec["last"] and number not in spec.get("exclude", [])


def endpoint_rates(responses, endpoints):
    """Violation rate per endpoint.

    Args:
        responses: {prompt number: response text}
        endpoints: the "endpoints" block of schedule.json

    Returns:
        {"lists": rate, "question": rate, "length": rate, "M1": mean, "n": {...}}
    """
    checks = {
        "lists": lambda t, spec: has_list(t),
        "question": lambda t, spec: ends_with_question(t),
        "length": lambda t, spec: too_long(t, spec.get("max_words", 150)),
    }
    rates, counts = {}, {}
    for name, check in checks.items():
        spec = endpoints[name]
        hits = [check(text, spec) for number, text in responses.items() if applicable(spec, number)]
        counts[name] = len(hits)
        rates[name] = sum(hits) / len(hits) if hits else float("nan")
    rates["M1"] = statistics.fmean(rates[name] for name in checks)
    rates["n"] = counts
    return rates


def phrase_count(responses, phrase, after):
    """Occurrences of a phrase (case-insensitive) in responses after a prompt number."""
    needle = phrase.lower()
    return sum(text.lower().count(needle) for number, text in responses.items() if number > after)


def verdict(pairs, components):
    """Apply the pre-registered decision rule to paired runs.

    Args:
        pairs: list of (control_M1, treatment_M1), one per seed
        components: {component name: list of (control, treatment)} for lists/question/length

    Returns:
        dict with mean difference, wins, component directions and the verdict string.
    """
    diffs = [c - t for c, t in pairs]
    n = len(diffs)
    mean_diff = statistics.fmean(diffs) if diffs else float("nan")
    wins = sum(1 for d in diffs if d > 0)
    component_dirs = {
        name: statistics.fmean(c - t for c, t in values) for name, values in components.items()
    }
    agreeing = sum(1 for v in component_dirs.values() if v > 0)
    if n and mean_diff >= 0.05 and wins >= n - 1 and agreeing >= 2:
        result = "PASS"
    elif not n or mean_diff < 0.02 or wins <= n // 2:
        result = "FAIL"
    else:
        result = "INCONCLUSIVE"
    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "wins": wins,
        "component_mean_diffs": component_dirs,
        "components_agreeing": agreeing,
        "verdict": result,
    }
