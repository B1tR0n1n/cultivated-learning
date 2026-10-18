"""Correction templates, shared by the UI and the scripted test runner.

This is the template registry and side-effect logic from ui/app.py
(_STRUCTURED_CORRECTIONS, _apply_structured_correction and the Forget/Ignore
branches of apply_correction), moved here so a scripted run takes exactly the
code path a human's click took. Every correction is also reported to the
reflection engine through note_feedback(), so whoever writes the self-model
can see it.
"""
from core.memory_store import MemoryType, MemoryUnit

# (template label) → (content prefix, memory_type, tags, salience_score)
STRUCTURED_CORRECTIONS = {
    # Suppression — stored as SEMANTIC, picked up by BiasBuilder for logit bias
    "Stop saying...":  ("Stop saying",  MemoryType.SEMANTIC,   ["user_correction", "suppress_phrase"],   0.9),
    "Stop being...":   ("Stop being",   MemoryType.SEMANTIC,   ["user_correction", "suppress_behavior"], 0.9),
    "Don't...":        ("Don't",        MemoryType.SEMANTIC,   ["user_correction", "suppress_phrase"],   0.9),
    "Never...":        ("Never",        MemoryType.SEMANTIC,   ["user_correction", "suppress_phrase"],   0.9),
    "Less...":         ("Less",         MemoryType.SEMANTIC,   ["user_correction", "suppress_behavior"], 0.9),
    # Enforcement — stored as PROCEDURAL, injected into live directive list
    "Always...":       ("Always",       MemoryType.PROCEDURAL, ["user_correction", "directive"],         0.8),
    "More...":         ("More",         MemoryType.PROCEDURAL, ["user_correction", "directive"],         0.8),
    "Start...":        ("Start",        MemoryType.PROCEDURAL, ["user_correction", "directive"],         0.8),
    "Prefer...":       ("Prefer",       MemoryType.PROCEDURAL, ["user_correction", "directive"],         0.8),
    "When...":         ("When",         MemoryType.PROCEDURAL, ["user_correction", "directive"],         0.8),
    # Correction — stored as SEMANTIC + triggers supersession of similar memories
    "Actually...":     ("Actually",     MemoryType.SEMANTIC,   ["user_correction", "fact_correction"],   0.9),
    "That's wrong...": ("That's wrong", MemoryType.SEMANTIC,   ["user_correction", "fact_correction"],   0.9),
    "Not X...":        ("Not",          MemoryType.SEMANTIC,   ["user_correction", "fact_correction"],   0.9),
    "Instead...":      ("Instead",      MemoryType.SEMANTIC,   ["user_correction", "fact_correction"],   0.9),
    "Correct:":        ("Correct:",     MemoryType.SEMANTIC,   ["user_correction", "fact_correction"],   0.9),
    # Meta — stored as SEMANTIC with descriptive tags
    "Remember...":     ("Remember",     MemoryType.SEMANTIC,   ["user_correction", "high_priority"],     0.9),
    "Focus on...":     ("Focus on",     MemoryType.SEMANTIC,   ["user_correction", "focus"],             0.8),
}

# Correction templates that trigger supersede_by_correction() in addition to storage
CORRECTION_TRIGGERS_SUPERSESSION = {
    "Actually...", "That's wrong...", "Not X...", "Instead...", "Correct:",
}


def apply_correction(loop, correction_type, content):
    """Apply one correction the way the UI does. Returns a short description."""
    memory = loop.memory
    reflection = loop.reflection_engine
    content = content.strip()

    if correction_type == "Freetext":
        loop.feedback(rating=3, correction=content)
        return f"Freetext correction stored: '{content[:70]}'"

    if correction_type == "Forget...":
        similar = memory.retrieve(content, top_k=5)
        forget_mem = MemoryUnit(
            content=f"Forget {content}",
            memory_type=MemoryType.SEMANTIC,
            salience_score=0.5,
            confidence=1.0,
            tags=["user_correction", "forget"],
            origin="human",
        )
        memory.store(forget_mem)
        for m in similar:
            memory.supersede(m.id, forget_mem.id)
        if reflection:
            reflection.note_feedback(f"Forget {content}")
        return f"Forgot: superseded {len(similar)} memories matching '{content[:50]}'."

    if correction_type == "Ignore...":
        similar = memory.retrieve(content, top_k=5)
        for m in similar:
            memory.adjust_salience(m.id, -0.4)
        if reflection:
            reflection.note_feedback(f"Ignore {content}")
        return f"Ignored: reduced salience of {len(similar)} memories matching '{content[:50]}'."

    if correction_type not in STRUCTURED_CORRECTIONS:
        raise ValueError(f"unknown correction type {correction_type!r}")

    prefix, mem_type, tags, salience = STRUCTURED_CORRECTIONS[correction_type]
    full_content = f"{prefix} {content}".strip()
    mem = MemoryUnit(
        content=full_content,
        memory_type=mem_type,
        salience_score=salience,
        confidence=1.0,
        tags=list(tags),
        origin="human",
    )
    memory.store(mem)
    if reflection:
        reflection.note_feedback(full_content)

    suffix = ""
    if correction_type in CORRECTION_TRIGGERS_SUPERSESSION:
        superseded = memory.supersede_by_correction(full_content, mem.id, similarity_threshold=0.45)
        if reflection:
            reflection.check_correction_conflicts(full_content)
        if superseded:
            suffix = f" Superseded {len(superseded)} conflicting memories."

    # Enforcement category updates the live directive list immediately.
    if mem_type == MemoryType.PROCEDURAL and reflection and full_content not in reflection.directives:
        reflection.directives.append(full_content)
        reflection.directive_ids.append(mem.id)

    label = correction_type.rstrip(".").rstrip(":").strip()
    return f"[{label}] stored: '{full_content[:70]}'{suffix}"
