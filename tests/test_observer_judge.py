import json

from experiments.observer import judge


def make_run(root, name, directives):
    d = root / name
    d.mkdir()
    with open(d / "progress.jsonl", "w") as f:
        f.writelines(json.dumps({"number": n, "response": f"reply {n}"}) + "\n" for n in range(1, 101))
    (d / "summary.json").write_text(json.dumps({"final_directives": directives}))


PROMPTS = [{"number": n, "prompt": f"prompt {n}", "expected_behavior": f"expect {n}"} for n in range(1, 101)]
SCHEDULE = {"corrections": [{"type": "Freetext", "content": "No bullets."},
                            {"type": "Stop saying...", "content": "'crucial'"}]}


def test_items_are_blind_and_cover_probes_and_system_directives(tmp_path):
    make_run(tmp_path, "PEER-s1", [{"content": "Use lists", "origin": "system"},
                                   {"content": "Prefer prose", "origin": "human"}])
    make_run(tmp_path, "SELF-s1", [])
    (tmp_path / "SELF-s2").mkdir()  # incomplete: skipped
    items = judge.collect_items(str(tmp_path), PROMPTS, SCHEDULE)
    assert len([i for i in items if i["kind"] == "J1"]) == 16
    j2 = [i for i in items if i["kind"] == "J2"]
    assert len(j2) == 1 and "Use lists" in j2[0]["text"]
    assert "No bullets." in j2[0]["text"] and "Stop saying 'crucial'" in j2[0]["text"]
    for item in items:
        assert "PEER" not in item["text"] and "SELF" not in item["text"]


def test_parse_and_summarize(tmp_path):
    assert judge.parse("PASS\nfine", "PASS", "FAIL") == "PASS"
    assert judge.parse("**FAIL** because", "PASS", "FAIL") == "FAIL"
    assert judge.parse("maybe", "PASS", "FAIL") is None
    assert judge.parse("", "CONFLICT", "OK") is None

    make_run(tmp_path, "PEER-s1", [{"content": "Use lists", "origin": "system"}])
    items = judge.collect_items(str(tmp_path), PROMPTS, SCHEDULE)
    seen = []

    def ask(text):
        seen.append(text)
        return "CONFLICT\nlists" if "Directive" in text else ("PASS" if "prompt 34" in text else "unclear")

    result = judge.summarize(judge.judge(items, ask))
    arm = result["per_arm"]["PEER"]
    assert arm["J1_pass"] == 1 and arm["unparsed"] == 7 and arm["J2_rate"] == 1.0
    assert len(seen) == 9
