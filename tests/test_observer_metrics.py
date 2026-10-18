import json
import math
import os

from experiments.observer import metrics

SCHEDULE = os.path.join(os.path.dirname(__file__), "..", "experiments", "observer", "schedule.json")


def endpoints():
    with open(SCHEDULE) as f:
        return json.load(f)["endpoints"]


def test_has_list_needs_two_list_lines():
    assert metrics.has_list("Intro\n- one\n- two")
    assert metrics.has_list("1. a\n2) b")
    assert metrics.has_list("• a\n• b")
    assert not metrics.has_list("- only one\nprose")
    assert not metrics.has_list("A sentence - with a dash.\nAnother - here.")
    assert not metrics.has_list("In 2026. Then 2027.")


def test_ends_with_question_ignores_trailing_markup():
    assert metrics.ends_with_question("Done. Want more?")
    assert metrics.ends_with_question("Want more?**  \n")
    assert metrics.ends_with_question('He asked "why?"')
    assert not metrics.ends_with_question("Is it? No.")
    assert not metrics.ends_with_question("")


def test_word_count_and_length():
    assert metrics.word_count("a b  c\nd") == 4
    assert metrics.too_long("w " * 151)
    assert not metrics.too_long("w " * 150)


def test_endpoint_rates_respects_windows_and_exclusions():
    eps = endpoints()
    responses = {n: "Fine." for n in range(1, 101)}
    responses[40] = "- a\n- b"           # list, applicable
    responses[38] = "- a\n- b"           # list, before the window
    responses[78] = "What?"              # excluded everywhere
    responses[20] = "Shall I go on?"     # question, applicable
    responses[50] = "w " * 200           # excluded from length
    responses[60] = "w " * 200           # length violation
    rates = metrics.endpoint_rates(responses, eps)
    assert rates["n"]["lists"] == 61 and math.isclose(rates["lists"], 1 / 61)
    assert rates["n"]["question"] == 94 and math.isclose(rates["question"], 1 / 94)
    assert rates["n"]["length"] == 63 and math.isclose(rates["length"], 1 / 63)
    assert math.isclose(rates["M1"], (1 / 61 + 1 / 94 + 1 / 63) / 3)


def test_phrase_count_after():
    responses = {39: "crucial", 41: "Crucial and crucial", 50: "fine"}
    assert metrics.phrase_count(responses, "crucial", after=40) == 2


def test_verdict_pass_fail_inconclusive():
    comps = {"lists": [(0.3, 0.1)] * 8, "question": [(0.5, 0.3)] * 8, "length": [(0.2, 0.2)] * 8}
    passing = [(0.40, 0.30)] * 7 + [(0.30, 0.31)]
    assert metrics.verdict(passing, comps)["verdict"] == "PASS"

    flat = [(0.30, 0.29)] * 8
    assert metrics.verdict(flat, comps)["verdict"] == "FAIL"

    split = [(0.40, 0.30)] * 4 + [(0.30, 0.31)] * 4
    assert metrics.verdict(split, comps)["verdict"] == "FAIL"

    middling = [(0.40, 0.36)] * 6 + [(0.30, 0.31)] * 2
    assert metrics.verdict(middling, comps)["verdict"] == "INCONCLUSIVE"


def test_verdict_needs_two_components_agreeing():
    one_component = {"lists": [(0.6, 0.0)] * 8, "question": [(0.3, 0.3)] * 8, "length": [(0.2, 0.25)] * 8}
    pairs = [(0.40, 0.30)] * 8
    assert metrics.verdict(pairs, one_component)["verdict"] == "INCONCLUSIVE"
