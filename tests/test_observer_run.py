"""End-to-end runner test on fake models: the whole protocol, every arm."""
import json
import os

import pytest

from experiments.observer import run as runner
from tests.conftest import FakeEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


@pytest.fixture
def system_prompt(tmp_path):
    path = tmp_path / "system.txt"
    path.write_text("You are Sable.")
    return str(path)


def reply(prompt):
    return "- one\n- two" if "transformer" in prompt.split("User:")[-1] else "A short answer. Anything else?"


@pytest.mark.parametrize("arm", sorted(runner.ARMS))
def test_full_protocol_runs_for_every_arm(tmp_path, system_prompt, arm):
    engine = FakeEngine(reply=reply, structured_reply="Keep answers under four sentences")
    writer = FakeEngine(structured_reply="Keep answers under four sentences") if arm == "OTHER" else None
    summary = runner.run(arm, 1, str(tmp_path / arm), load("data/test_prompts_100.json"),
                         load("experiments/observer/schedule.json"), None, None, system_prompt,
                         engine=engine, writer=writer)
    assert summary["completed"] == 100
    ep = summary["endpoints"]
    assert ep["question"] > 0.9          # every reply ends with a question
    assert 0 < ep["lists"] < 0.1         # only the transformer prompt (39) lists
    for name in ("summary.json", "writer_log.json", "memories.json", "progress.jsonl"):
        assert (tmp_path / arm / name).exists()

    log = json.loads((tmp_path / arm / "writer_log.json").read_text())
    assert log and all(entry["interaction"] >= 1 for entry in log)
    observed = any("observing" in e["prompt"] for e in log)
    assert observed == (arm in ("PEER", "OTHER"))
    if arm == "OTHER":
        assert engine.batch_prompts == []  # the subject never wrote its own self-model


def test_corrections_reach_the_observer_and_the_store(tmp_path, system_prompt):
    engine = FakeEngine(reply=reply)
    runner.run("PEER", 2, str(tmp_path / "p"), load("data/test_prompts_100.json"),
               load("experiments/observer/schedule.json"), None, None, system_prompt,
               engine=engine, limit=45)
    memories = json.loads((tmp_path / "p" / "memories.json").read_text())
    human = [m for m in memories if m.get("origin") == "human"]
    assert any("question back to me" in m["content"] for m in human)
    assert any(m["content"] == "Stop saying 'crucial'" for m in human)
    log = json.loads((tmp_path / "p" / "writer_log.json").read_text())
    later = [e["prompt"] for e in log if e["interaction"] >= 42 and "patterns" in e["prompt"]]
    assert later and "Stop saying 'crucial'" in later[-1]


def test_partial_run_is_not_silently_overwritten(tmp_path, system_prompt):
    out = tmp_path / "x"
    out.mkdir()
    (out / "progress.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit):
        runner.run("SELF", 1, str(out), [], {"corrections": []}, None, None, system_prompt,
                   engine=FakeEngine())


def test_same_seed_scope_for_the_subject_in_every_arm(tmp_path, system_prompt):
    scopes = {}
    for arm in ("SELF", "PEER"):
        engine = FakeEngine()
        seen = []
        engine.set_seed_scope = seen.append
        runner.run(arm, 7, str(tmp_path / arm), load("data/test_prompts_100.json"),
                   load("experiments/observer/schedule.json"), None, None, system_prompt,
                   engine=engine, limit=3)
        scopes[arm] = seen
    assert scopes["SELF"] == scopes["PEER"] == ["7:1", "7:2", "7:3"]


def test_structured_corrections_follow_the_ui_semantics(tmp_path, fake_engine, memory_store):
    from core.context_assembler import ContextAssembler
    from core.corrections import apply_correction
    from core.interaction_loop import InteractionLoop
    from core.memory_store import MemoryType

    loop = InteractionLoop(engine=fake_engine, memory_store=memory_store,
                           assembler=ContextAssembler(engine=fake_engine, memory_store=memory_store))
    loop.chat("Contact Front has four suits.")
    assert "Prefer" in apply_correction(loop, "Prefer...", "prose over lists")
    assert loop.reflection_engine.directives[-1] == "Prefer prose over lists"
    assert "Superseded" in apply_correction(loop, "Actually...", "Contact Front has four suits is wrong")
    assert "Forgot" in apply_correction(loop, "Forget...", "Contact Front")
    assert "Ignored" in apply_correction(loop, "Ignore...", "suits")
    with pytest.raises(ValueError):
        apply_correction(loop, "Whisper...", "x")
    procedural = memory_store.retrieve_by_type(MemoryType.PROCEDURAL)
    assert all(m.origin == "human" for m in procedural)
    assert len(loop.reflection_engine._corrections) == 4
