"""The observer experiment: who writes the agent's self-model, from where."""
import pytest

from core.memory_store import MemoryType, MemoryUnit
from core.reflection import PERSPECTIVES, ReflectionEngine
from tests.conftest import FakeEngine

ORIGINAL_D0 = (
    "[INST] You are a self-reflection module analyzing an interaction.\n\n"
    "Interaction:\nUser: Hi\nAssistant: Hello.\n\n"
    "Analyze this interaction factually:\n"
    "1. Was the response accurate and relevant?\n"
    "2. Did it address what the user actually asked?\n"
    "3. Were there any errors or misunderstandings?\n\n"
    "Be brief and specific. One paragraph. [/INST]"
)


def seed_memories(memory, n=3):
    for i in range(n):
        memory.store(MemoryUnit(content=f"Reflection D1: I notice I tend to elaborate {i}",
                                memory_type=MemoryType.REFLECTIVE, tags=["reflection"]))


def run_full_pass(engine, user="Hi", reply="Hello."):
    for count in (1, 2, 3):
        engine.reflect(user, reply, f"id{count}", interaction_count=count)


def test_perspectives_are_the_four_arms():
    assert set(PERSPECTIVES) == {"self", "self_informed", "observer"}
    with pytest.raises(ValueError):
        ReflectionEngine(FakeEngine(), None, perspective="mirror")


def test_self_perspective_keeps_the_original_prompts(fake_engine, memory_store):
    engine = ReflectionEngine(fake_engine, memory_store)
    engine.reflect("Hi", "Hello.", "id1", interaction_count=1)
    assert fake_engine.batch_prompts == [ORIGINAL_D0]


def test_writer_does_all_generation_and_subject_is_untouched(fake_engine, memory_store):
    writer = FakeEngine(structured_reply="Keep answers under four sentences")
    seed_memories(memory_store)
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    run_full_pass(engine)
    assert fake_engine.prompts == [] and fake_engine.batch_prompts == []
    assert writer.batch_prompts and writer.prompts


def test_observer_prompts_are_third_person_and_never_see_the_memory_store(fake_engine, memory_store):
    writer = FakeEngine()
    seed_memories(memory_store)
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    engine.note_feedback("Stop ending answers with a question")
    run_full_pass(engine)
    everything = "\n".join(writer.batch_prompts + writer.prompts)
    assert "self-reflection" not in everything
    assert "observing" in everything
    assert "I tend to elaborate" not in everything  # inner self-narrative stays private
    d1 = next(p for p in writer.batch_prompts if "patterns" in p)
    assert "Stop ending answers with a question" in d1
    assert d1.count("User: Hi") == 3  # saw the whole window since the last pass


def test_self_informed_gets_same_window_and_corrections_plus_its_memories(fake_engine, memory_store):
    seed_memories(memory_store)
    engine = ReflectionEngine(fake_engine, memory_store, perspective="self_informed")
    engine.note_feedback("Stop ending answers with a question")
    run_full_pass(engine)
    d1 = next(p for p in fake_engine.batch_prompts if "patterns" in p)
    assert "self-reflection" in d1
    assert "Stop ending answers with a question" in d1
    assert d1.count("User: Hi") == 3
    assert "I tend to elaborate" in d1  # keeps access to its own memories


def test_observer_and_self_informed_d1_differ_only_in_framing_and_memories(fake_engine, memory_store):
    """The D1 evidence block (window + corrections + the corrections question) is shared."""
    from core.reflection import _evidence_block
    block = _evidence_block([("Hi", "Hello.")], ["Be brief"])
    assert "Recent interactions" in block and "Be brief" in block


def test_stored_memories_look_identical_across_perspectives(fake_engine, memory_store):
    writer = FakeEngine()
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    engine.reflect("Hi", "Hello.", "id1", interaction_count=1)
    stored = memory_store.retrieve_by_type(MemoryType.REFLECTIVE)
    assert stored[0].content.startswith("Reflection D0: ")
    assert "observer" in stored[0].tags


def test_observer_directives_pass_through_the_same_gates(fake_engine, memory_store):
    writer = FakeEngine(structured_reply="Offer encouragement and wellness resources")
    seed_memories(memory_store)
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    run_full_pass(engine)
    assert engine.directives == []  # platitude rejected by the heuristic gate

    writer.structured_reply = lambda p: "OK" if "contradict" in p else "Keep answers under four sentences"
    engine.reflect("Hi", "Hello.", "id6", interaction_count=6)
    assert engine.directives == ["Keep answers under four sentences"]


def test_window_only_holds_interactions_since_last_full_pass(fake_engine, memory_store):
    writer = FakeEngine()
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    for count in range(1, 7):
        engine.reflect(f"Q{count}", "A.", f"id{count}", interaction_count=count)
    d1s = [p for p in writer.batch_prompts if "patterns" in p]
    assert "Q4" in d1s[-1] and "Q6" in d1s[-1] and "Q3" not in d1s[-1]


def test_contradiction_gate_and_coherence_check_run_on_the_writer(fake_engine, memory_store):
    def answer(prompt):
        if "contradict" in prompt:
            return "CONTRADICTS: 1"
        if "coherence checker" in prompt:
            return "ISSUE: redundant\nREMOVE: 2"
        return "Keep answers under four sentences"

    writer = FakeEngine(structured_reply=answer)
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer")
    engine.directives = ["Avoid long answers entirely", "Prefer prose to lists"]
    engine.directive_ids = ["a", "b"]
    run_full_pass(engine)
    assert "Keep answers under four sentences" not in engine.directives  # rejected as contradiction
    assert engine.directives == ["Avoid long answers entirely"]          # D3 pruned number 2
    assert any("coherence checker" in p for p in writer.prompts)
    assert fake_engine.prompts == []


def test_cap_displaces_weakest_directive(fake_engine, memory_store):
    writer = FakeEngine(structured_reply=lambda p: "OK" if "contradict" in p else
                        ("COHERENT" if "coherence" in p else "Verify every number before stating it"))
    engine = ReflectionEngine(fake_engine, memory_store, writer=writer, perspective="observer",
                              max_directives=1)
    old = MemoryUnit(content="Limit answers to two sentences", memory_type=MemoryType.PROCEDURAL,
                     salience_score=0.4, tags=["directive"])
    memory_store.store(old)
    engine.directives, engine.directive_ids = [old.content], [old.id]
    run_full_pass(engine)
    assert engine.directives == ["Verify every number before stating it"]
    assert [d.content for d in engine.get_directives()] == ["Verify every number before stating it"]


def test_correction_kills_matching_directive(fake_engine, memory_store):
    engine = ReflectionEngine(fake_engine, memory_store)
    engine.directives, engine.directive_ids = ["Use bullet points for lists of steps"], ["x"]
    killed = engine.check_correction_conflicts("Never use bullet points for lists of steps")
    assert killed == ["Use bullet points for lists of steps"] and engine.directives == []
