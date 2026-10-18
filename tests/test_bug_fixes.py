"""Regression tests for harness bugs found while setting up the observer experiment."""
from core.bias_builder import BiasBuilder
from core.memory_store import MemoryType, MemoryUnit
from core.reflection import ReflectionEngine


def store_correction(memory, content, tags):
    memory.store(MemoryUnit(content=content, memory_type=MemoryType.SEMANTIC,
                            salience_score=0.9, confidence=1.0, tags=tags, origin="human"))


def captured_phrases(fake_engine, memory):
    captured = []
    fake_engine.suppress_tokens = lambda phrases: captured.extend(phrases) or ({}, [])
    BiasBuilder(fake_engine, memory).build_bias_map()
    return captured


def test_structured_suppression_strips_quotes_and_adds_mid_sentence_form(fake_engine, memory_store):
    # The protocol's template is "Stop saying 'crucial'". Before the fix the target
    # was the quoted string "'crucial'", which the model never writes, and the
    # space-prefixed form the model does write (" crucial") was never suppressed.
    store_correction(memory_store, "Stop saying 'crucial'", ["user_correction", "suppress_phrase"])
    phrases = captured_phrases(fake_engine, memory_store)
    assert "'crucial'" not in phrases
    assert "crucial" in phrases
    assert " crucial" in phrases
    assert " Crucial" in phrases or "Crucial" in phrases


def test_freetext_suppression_also_strips_quotes(fake_engine, memory_store):
    store_correction(memory_store, 'CORRECTION: stop saying "it is important to note"',
                     ["user_correction"])
    phrases = captured_phrases(fake_engine, memory_store)
    assert "it is important to note" in phrases
    assert " it is important to note" in phrases
    assert not any(p.startswith('"') or p.endswith('"') for p in phrases)


def test_base_suppressions_unchanged(fake_engine, memory_store):
    phrases = captured_phrases(fake_engine, memory_store)
    assert "Certainly!" in phrases
    assert " Certainly!" not in phrases


def test_get_directives_never_returns_a_superseded_directive(fake_engine, memory_store):
    # InteractionLoop.feedback() supersedes conflicting system directives in the
    # store but did not remove them from the reflection engine's live list, and
    # get_directives() fetched by id without checking superseded_by, so a killed
    # directive kept appearing in the prompt.
    engine = ReflectionEngine(fake_engine, memory_store)
    mem = MemoryUnit(content="Use bullet points to structure answers",
                     memory_type=MemoryType.PROCEDURAL, salience_score=0.7, tags=["directive"])
    memory_store.store(mem)
    engine.directives.append(mem.content)
    engine.directive_ids.append(mem.id)
    memory_store.supersede(mem.id, "correction-id")
    assert [d.content for d in engine.get_directives()] == []
    assert engine.directives == [] and engine.directive_ids == []


def test_feedback_supersession_removes_directive_from_live_list(fake_engine, memory_store):
    from core.context_assembler import ContextAssembler
    from core.interaction_loop import InteractionLoop

    assembler = ContextAssembler(engine=fake_engine, memory_store=memory_store)
    loop = InteractionLoop(engine=fake_engine, memory_store=memory_store, assembler=assembler)
    loop.chat("Explain transformers.")
    mem = MemoryUnit(content="Use bullet points for every answer",
                     memory_type=MemoryType.PROCEDURAL, salience_score=0.7, tags=["directive"])
    memory_store.store(mem)
    loop.reflection_engine.directives.append(mem.content)
    loop.reflection_engine.directive_ids.append(mem.id)
    loop.feedback(3, correction="Never use bullet points for every answer")
    live = [d.content for d in loop.reflection_engine.get_directives()]
    assert "Use bullet points for every answer" not in live
    assert len(loop.reflection_engine.directives) == len(loop.reflection_engine.directive_ids)
