import sys
import os

BASE_DIR = os.environ.get("CL_BASE_DIR", "/workspace/Projects/cultivated-learning")
sys.path.insert(0, BASE_DIR)

import gradio as gr
from engine.inference import InferenceEngine
from core.memory_store import MemoryStore, MemoryType, MemoryUnit
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop
from core.consolidation import ConsolidationEngine
from core.cold_storage import ColdStorage
from core.bias_builder import BiasBuilder
from evaluation.metrics import EvaluationMetrics

# Initialize
print("Loading models...")
engine = InferenceEngine(
    "/workspace/models/results/Mistral-7B-Instruct-v0.3",
    adapter_path="/workspace/models/results/shell-native-combined"
)
engine.load()

memory = MemoryStore(
    persist_dir=os.path.join(BASE_DIR, "data/memory_db"),
    engine=engine
)

assembler = ContextAssembler(
    engine=engine,
    memory_store=memory,
    max_context=engine.max_context,
    system_prompt_path=os.path.join(BASE_DIR, "sable_system_prompt.txt"),
    prompt_format=engine.prompt_format,
)

metrics = EvaluationMetrics(
    log_dir=os.path.join(BASE_DIR, "data/evaluation"),
    memory_store=memory
)

bias_builder = BiasBuilder(engine, memory)

loop = InteractionLoop(
    engine=engine,
    memory_store=memory,
    assembler=assembler,
    log_dir=os.path.join(BASE_DIR, "data/interaction_log"),
    metrics_dir=os.path.join(BASE_DIR, "data/evaluation"),
    bias_builder=bias_builder,
)

consolidator = ConsolidationEngine(engine, memory)

cold = ColdStorage(
    engine=engine,
    memory_store=memory,
    archive_dir=os.path.join(BASE_DIR, "data/cold_storage")
)

print("All systems ready.")


# --- Functions ---

import re as _re


def split_sentences(text):
    """Split a response into individual sentences, capped at 8."""
    text = text.strip()
    if not text:
        return []
    parts = _re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 8][:8]


def chat(message, history):
    """Process a chat message through the full pipeline.

    Returns 6 values: clears message box, updates chatbot, stores response
    in state, and resets the feedback controls to defaults.
    """
    response = loop.chat(message)
    gradio_history = [
        {"role": h["role"], "content": h["content"]}
        for h in loop.history
    ]
    return "", gradio_history, response, 3, "Freetext", "", quick_status()


# ── Correction template registry ─────────────────────────────────────────────
# (template label) → (content prefix, memory_type, tags, salience_score)
# Special cases "Forget...", "Ignore...", "Freetext" are handled separately.

_STRUCTURED_CORRECTIONS = {
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
_CORRECTION_TRIGGERS_SUPERSESSION = {
    "Actually...", "That's wrong...", "Not X...", "Instead...", "Correct:",
}

# Full ordered list for dropdowns (same in both whole-response and segment sections)
_ALL_CORRECTION_CHOICES = [
    "Stop saying...", "Stop being...", "Don't...", "Never...", "Less...",
    "Always...", "More...", "Start...", "Prefer...", "When...",
    "Actually...", "That's wrong...", "Not X...", "Instead...", "Correct:",
    "Remember...", "Forget...", "Ignore...", "Focus on...", "Freetext",
]


def _apply_structured_correction(correction_type, content):
    """Store a structured correction memory and run any side-effects.

    Returns (mem, full_content, result_suffix) where result_suffix is a short
    string describing what happened (for appending to the feedback message).
    """
    prefix, mem_type, tags, salience = _STRUCTURED_CORRECTIONS[correction_type]
    full_content = f"{prefix} {content}".strip()

    mem = MemoryUnit(
        content=full_content,
        memory_type=mem_type,
        salience_score=salience,
        confidence=1.0,
        tags=tags,
    )
    memory.store(mem)

    suffix = ""

    # Correction category triggers supersession + directive conflict check
    if correction_type in _CORRECTION_TRIGGERS_SUPERSESSION:
        superseded = memory.supersede_by_correction(full_content, mem.id, similarity_threshold=0.45)
        if loop.reflection_engine:
            loop.reflection_engine.check_correction_conflicts(full_content)
        if superseded:
            suffix = f" Superseded {len(superseded)} conflicting memories."

    # Enforcement category updates the live directive list immediately.
    # Both lists updated together — directive_ids[i] must always equal directives[i].
    if mem_type == MemoryType.PROCEDURAL and loop.reflection_engine:
        if full_content not in loop.reflection_engine.directives:
            loop.reflection_engine.directives.append(full_content)
            loop.reflection_engine.directive_ids.append(mem.id)

    return mem, full_content, suffix


def give_feedback(rating, correction_type, correction_content, target_query):
    rating = int(rating)
    content = correction_content.strip()
    tq = target_query.strip() if target_query.strip() else None

    # Freetext uses the full legacy path: supersession + directive conflict check + salience
    if not content or correction_type == "Freetext":
        loop.feedback(rating=rating, correction=content or None, target_query=tq)
        msg = f"Rating {rating} recorded."
        if tq:
            msg += f" Targeted at: '{tq[:50]}'"
        if content:
            msg += " Freetext correction stored."

    else:
        # All structured types: record rating for salience adjustment, then handle correction
        loop.feedback(rating=rating, correction=None, target_query=tq)

        if correction_type == "Forget...":
            similar = memory.retrieve(content, top_k=5)
            forget_mem = MemoryUnit(
                content=f"Forget {content}",
                memory_type=MemoryType.SEMANTIC,
                salience_score=0.5,
                confidence=1.0,
                tags=["user_correction", "forget"],
            )
            memory.store(forget_mem)
            for m in similar:
                memory.supersede(m.id, forget_mem.id)
            msg = (f"Rating {rating} recorded. Forgot: superseded {len(similar)} memories "
                   f"matching '{content[:50]}'.")

        elif correction_type == "Ignore...":
            similar = memory.retrieve(content, top_k=5)
            for m in similar:
                memory.adjust_salience(m.id, -0.4)
            msg = (f"Rating {rating} recorded. Ignored: reduced salience of {len(similar)} "
                   f"memories matching '{content[:50]}'.")

        else:
            _, full_content, suffix = _apply_structured_correction(correction_type, content)
            label = correction_type.rstrip(".").rstrip(":").strip()
            msg = f"Rating {rating} recorded. [{label}] stored: '{full_content[:70]}'{suffix}"

    log_update = {"rating": rating}
    if content:
        log_update["corrections"] = [{"type": correction_type, "content": content[:200]}]
    loop.update_log_feedback(loop.last_interaction_id, log_update)
    return msg


def give_feedback_main(rating, correction_type, correction_content):
    """Main-view feedback wrapper — defaults target_query to last interaction."""
    return give_feedback(rating, correction_type, correction_content, ""), quick_status()


def load_segments(response_text):
    """Split the last response into numbered sentences for segment feedback."""
    if not response_text.strip():
        return (
            "No response loaded. Send a message in the Chat tab first.",
            gr.update(visible=False, minimum=1, maximum=1, value=1),
            "",
        )
    sentences = split_sentences(response_text)
    if not sentences:
        return (
            "Could not parse sentences from response.",
            gr.update(visible=False, minimum=1, maximum=1, value=1),
            "",
        )
    display = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
    return (
        display,
        gr.update(visible=True, minimum=1, maximum=len(sentences), value=1),
        sentences[0],
    )


def select_sentence(response_text, sentence_idx):
    """Return the sentence at the given 1-based index."""
    sentences = split_sentences(response_text)
    i = int(sentence_idx) - 1
    if 0 <= i < len(sentences):
        return sentences[i]
    return ""


def submit_segment_feedback(response_text, sentence_idx, seg_rating):
    """Adjust salience of memories that influenced a specific sentence."""
    sentences = split_sentences(response_text)
    i = int(sentence_idx) - 1
    if not sentences or not (0 <= i < len(sentences)):
        return "Invalid sentence index — load the response first."

    sentence = sentences[i]
    delta = (int(seg_rating) - 3) * 0.1

    # Adjust salience of memories that likely influenced this sentence
    similar = memory.retrieve(sentence, top_k=5)
    for m in similar:
        memory.adjust_salience(m.id, delta)

    # Store the sentence as a tagged episodic memory
    interaction_id = loop.last_interaction_id or "unknown"
    seg_mem = MemoryUnit(
        content=f"[Segment {i+1}] {sentence}",
        memory_type=MemoryType.EPISODIC,
        salience_score=max(0.1, min(1.0, 0.5 + delta)),
        source_interaction_id=interaction_id,
        tags=["segment_feedback", f"segment_{i+1}", f"interaction_{interaction_id}"],
    )
    memory.store(seg_mem)

    loop.update_log_feedback(loop.last_interaction_id, {
        "segment_ratings": [{"sentence_index": i + 1, "sentence": sentence[:150], "rating": int(seg_rating)}],
    })
    return (f"Sentence {i+1} rated {seg_rating}: "
            f"adjusted {len(similar)} related memories by {delta:+.2f}.")


def apply_correction(correction_type, content):
    """Apply a standalone correction not tied to any specific response or sentence."""
    content = content.strip()
    if not content:
        return "Enter correction content."

    if correction_type == "Freetext":
        corr_mem = MemoryUnit(
            content=f"CORRECTION: {content}",
            memory_type=MemoryType.SEMANTIC,
            salience_score=0.9,
            confidence=1.0,
            tags=["user_correction", "high_priority"],
        )
        memory.store(corr_mem)
        memory.supersede_by_correction(content, corr_mem.id, similarity_threshold=0.45)
        msg = f"Freetext correction stored: '{content[:70]}'"

    elif correction_type == "Forget...":
        similar = memory.retrieve(content, top_k=5)
        forget_mem = MemoryUnit(
            content=f"Forget {content}",
            memory_type=MemoryType.SEMANTIC,
            salience_score=0.5,
            confidence=1.0,
            tags=["user_correction", "forget"],
        )
        memory.store(forget_mem)
        for m in similar:
            memory.supersede(m.id, forget_mem.id)
        msg = f"Forgot: superseded {len(similar)} memories matching '{content[:50]}'."

    elif correction_type == "Ignore...":
        similar = memory.retrieve(content, top_k=5)
        for m in similar:
            memory.adjust_salience(m.id, -0.4)
        msg = f"Ignored: reduced salience of {len(similar)} memories matching '{content[:50]}'."

    else:
        _, full_content, suffix = _apply_structured_correction(correction_type, content)
        label = correction_type.rstrip(".").rstrip(":").strip()
        msg = f"[{label}] stored: '{full_content[:70]}'{suffix}"

    loop.update_log_feedback(loop.last_interaction_id, {
        "corrections": [{"type": correction_type, "content": content[:200]}],
    })
    return msg


def log_adaptation_event(event_type, description):
    if not description.strip():
        return "Enter a description."
    loop.log_adaptation(event_type, description.strip())
    return f"Logged: {event_type} — {description.strip()[:80]}"


def get_status():
    stats = memory.get_stats()
    status = loop.status()
    cold_stats = cold.get_stats()
    lines = [
        f"INTERACTION COUNT:  {status.get('interaction_count', '?')}",
        f"CONTEXT WINDOW:     {status.get('context_window', '?')} tokens",
        f"ACTIVE MEMORIES:    {stats['total']}",
        f"COLD STORAGE:       {cold_stats['archived_count']}",
        f"ACTIVE DIRECTIVES:  {status.get('active_directives', '?')}/{status.get('max_directives', '?')}",
        f"REFLECTION:         {'ENABLED' if status.get('reflection_enabled') else 'DISABLED'}",
        f"METRICS:            {'ENABLED' if status.get('metrics_enabled') else 'DISABLED'}",
        "",
    ]
    if stats["total"] > 0:
        lines.append(f"BY TYPE:  {stats['by_type']}")
        lines.append(f"SALIENCE: avg {stats['avg_salience']:.3f}  min {stats['min_salience']:.3f}  max {stats['max_salience']:.3f}")
    return "\n".join(lines)


def get_evaluation_report():
    report = metrics.full_report()
    lines = []
    
    # Retrieval
    r = report["retrieval"]
    lines.append("── RETRIEVAL PRECISION ──")
    if r.get("entries", 0) > 0:
        lines.append(f"  Entries: {r['entries']}  |  Avg precision: {r['avg_precision']}")
        lines.append(f"  Irrelevant rate: {r['avg_irrelevant_rate']}  |  Trend: {r['trend']}")
    else:
        lines.append(f"  {r.get('message', 'No data')}")
    lines.append("")

    # Directives
    d = report["directives"]
    lines.append("── DIRECTIVE STABILITY ──")
    if d.get("snapshots", 0) >= 2:
        lines.append(f"  Snapshots: {d['snapshots']}  |  Avg churn: {d['avg_churn']}")
        lines.append(f"  Current count: {d['current_count']}  |  Assessment: {d['assessment']}")
    else:
        lines.append(f"  {d.get('message', 'No data')}")
    lines.append("")

    # Quality
    q = report["quality"]
    lines.append("── RESPONSE QUALITY DELTA ──")
    if q.get("comparisons", 0) > 0:
        lines.append(f"  Comparisons: {q['comparisons']}  |  Avg delta: {q['avg_delta']:+.2f}")
        lines.append(f"  Framework wins: {q['framework_wins']}  |  Vanilla wins: {q['vanilla_wins']}")
    else:
        lines.append(f"  {q.get('message', 'No data')}")
    lines.append("")

    # Adaptation
    a = report["adaptation"]
    lines.append("── ADAPTATION EVIDENCE ──")
    if a.get("total_events", 0) > 0:
        lines.append(f"  Total: {a['total_events']}  |  Helps: {a['help_count']}  |  Harms: {a['harm_count']}")
        lines.append(f"  Harm rate: {a['harm_rate']}")
    else:
        lines.append(f"  {a.get('message', 'No data')}")
    lines.append("")

    # Compute
    c = report["compute"]
    lines.append("── COMPUTE COST ──")
    if c.get("total_interactions", 0) > 0:
        lines.append(f"  Interactions: {c['total_interactions']}  |  Avg tokens: {c['avg_tokens']}")
        lines.append(f"  Avg time: {c['avg_time_seconds']}s  |  Total tokens: {c['total_tokens_all_time']:,}")
    else:
        lines.append(f"  {c.get('message', 'No data')}")
    lines.append("")

    # Drift
    dr = report["drift"]
    lines.append("── BEHAVIORAL DRIFT ──")
    if dr.get("samples", 0) >= 3:
        lines.append(f"  Samples: {dr['samples']}  |  Avg words: {dr['avg_response_words']}")
        lines.append(f"  Avg sentence: {dr['avg_sentence_length']} words  |  Vocab: {dr['avg_vocab_richness']}")
        lines.append(f"  Length trend: {dr['length_trend']}")
    else:
        lines.append(f"  {dr.get('message', 'No data')}")

    return "\n".join(lines)


def run_consolidation():
    new_mems = consolidator.consolidate()
    if new_mems:
        results = [f"  [{m.salience_score:.2f}] {m.content[:100]}" for m in new_mems]
        return f"Consolidated {len(new_mems)} new semantic memories:\n" + "\n".join(results)
    return "No memories ready for consolidation."


def run_decay():
    memory.decay_pass()
    stats = memory.get_stats()
    return f"Decay pass complete. Avg salience: {stats['avg_salience']:.3f}"


def run_archive():
    count = cold.archive_pass()
    return f"Archive pass complete. {count} memories moved to cold storage."


def run_resurface(query):
    if not query.strip():
        return "Enter a query to check cold storage for matches."
    resurfaced = cold.resurface(query.strip())
    if resurfaced:
        results = [f"  [{m.salience_score:.2f}] {m.content[:100]}" for m in resurfaced]
        return f"Resurfaced {len(resurfaced)} memories:\n" + "\n".join(results)
    return "No matches in cold storage."


def browse_memories(mem_type):
    type_map = {
        "All": None,
        "Episodic": MemoryType.EPISODIC,
        "Semantic": MemoryType.SEMANTIC,
        "Procedural": MemoryType.PROCEDURAL,
        "Reflective": MemoryType.REFLECTIVE,
    }
    mt = type_map.get(mem_type)
    if mt:
        mems = memory.retrieve_by_type(mt, limit=30)
    else:
        all_data = memory.collection.get(limit=30)
        mems = []
        for i in range(len(all_data["ids"])):
            mem = MemoryUnit.from_chroma(
                id=all_data["ids"][i],
                document=all_data["documents"][i],
                metadata=all_data["metadatas"][i],
            )
            # Also filter superseded in the "All" view
            if mem.superseded_by is None:
                mems.append(mem)

    mems.sort(key=lambda m: m.salience_score, reverse=True)
    lines = []
    for m in mems:
        lines.append(f"[{m.memory_type.value:11s}] sal:{m.salience_score:.2f}  {m.content[:90]}")
    return "\n".join(lines) if lines else "No memories found."


def get_directives():
    if loop.reflection_engine:
        directives = loop.reflection_engine.get_directives()
        count, cap = loop.reflection_engine.get_directive_count()
        if directives:
            header = f"Active directives ({count}/{cap}):\n"
            return header + "\n".join([f"{i+1}. {d}" for i, d in enumerate(directives)])
    return "No active directives."


def reload_directives():
    """Reload directives from memory store into the reflection engine.

    Fixes the stale directive problem: if you manually prune directives
    through ChromaDB directly, the in-memory self.directives list goes
    stale until restart. This forces a reload without restarting.
    """
    if loop.reflection_engine:
        loop.reflection_engine._load_directives()
        count = len(loop.reflection_engine.get_directives())
        return f"Reloaded {count} directives from memory store."
    return "Reflection engine not enabled."


def upload_document(file, source_label, chunk_mode):
    """Chunk a file and store each chunk as a semantic memory.

    JSON (.json): expects an array of block objects with fields:
        text      (str)  — block content
        blockType (str)  — "heading", "subheading", "body", "notes", ...
        section   (str)  — section title (optional; prepended to body/note content)
        page      (int)  — page number (stored as tag "page:N")
        index     (int)  — block position within document
        xrefs     (list) — cross-references (ignored during import)
    Salience: heading=0.8, body/notes=0.7, subheading=0.6, other=0.7.

    Plain text (.txt / .md): split by paragraph or line per chunk_mode.
    """
    if file is None:
        return "No file selected."

    source = source_label.strip() or "uploaded_document"
    stored = 0
    skipped = 0

    if file.name.lower().endswith(".json"):
        import json as _json

        with open(file.name, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = _json.load(f)
            except Exception as e:
                return f"JSON parse error: {e}"

        # Format 1: flat top-level array of blocks
        if isinstance(data, list):
            blocks = data
        # Format 2: GameForge-style { "pages": [ { "blocks": [...] }, ... ] }
        elif isinstance(data, dict) and "pages" in data:
            blocks = []
            for page in data["pages"]:
                blocks.extend(page.get("blocks", []))
        else:
            return "JSON must be a flat array of blocks or an object with a 'pages' key."

        _SALIENCE_BY_TYPE = {
            "heading":    0.8,
            "body":       0.7,
            "notes":      0.7,
            "subheading": 0.6,
        }

        for block in blocks:
            text = block.get("text", "").strip()
            if not text or len(text) < 10:
                skipped += 1
                continue

            block_type = block.get("blockType", "body").lower()
            section = (block.get("section") or "").strip()
            page = block.get("page")

            # Prepend section title to non-heading blocks for retrieval context
            if section and block_type != "heading":
                content = f"{section}: {text}"
            else:
                content = text

            if len(content) > 1000:
                content = content[:1000]

            tags = ["uploaded_document", source, block_type]
            if section:
                tags.append(f"section:{section}")
            if page is not None:
                tags.append(f"page:{page}")

            mem = MemoryUnit(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                salience_score=_SALIENCE_BY_TYPE.get(block_type, 0.7),
                confidence=0.9,
                tags=tags,
            )
            memory.store(mem)
            stored += 1

        if stored == 0:
            return "No blocks met the minimum length (10 chars)."
        return (
            f"Stored {stored} blocks from '{source}' "
            f"({skipped} skipped). "
            f"Total memories: {memory.collection.count()}"
        )

    else:  # Plain text / markdown fallback
        with open(file.name, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        if chunk_mode == "Paragraph":
            chunks = [c.strip() for c in _re.split(r'\n\s*\n', raw) if c.strip()]
        else:  # Line
            chunks = [line.strip() for line in raw.splitlines()]

        for i, chunk in enumerate(chunks):
            if len(chunk) < 20:
                skipped += 1
                continue
            if len(chunk) > 1000:
                chunk = chunk[:1000]
            mem = MemoryUnit(
                content=chunk,
                memory_type=MemoryType.SEMANTIC,
                salience_score=0.7,
                confidence=0.9,
                tags=["uploaded_document", source, f"chunk_{i + 1}"],
            )
            memory.store(mem)
            stored += 1

        if stored == 0:
            return "No chunks met the minimum length (20 chars). Check the file or try 'Line' mode."
        return (
            f"Stored {stored} chunks from '{source}' "
            f"({skipped} skipped, min 20 chars). "
            f"Total memories: {memory.collection.count()}"
        )


def export_session_report():
    """Generate a markdown session report and save it to data/exports/.

    Filename format: report_YYYY-MM-DD_interactions_1-N.md

    Contents:
        - Timestamp and interaction count
        - Memory stats (total, by type, avg/min/max salience)
        - Active directives
        - Cold storage count
        - Full interaction log (prompt, response, tokens, elapsed)
        - Active logit bias suppressions (token IDs decoded to strings)
    """
    import datetime
    import glob

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    interaction_count = loop.interaction_count

    exports_dir = os.path.join(BASE_DIR, "data/exports")
    os.makedirs(exports_dir, exist_ok=True)
    filename = f"report_{date_str}_interactions_1-{interaction_count}.md"
    filepath = os.path.join(exports_dir, filename)

    lines = []
    lines.append("# Cultivated Learning — Session Report")
    lines.append("")
    lines.append(f"**Generated:** {date_str} {time_str}  ")
    lines.append(f"**Interaction count:** {interaction_count}  ")
    lines.append("")

    # ── Memory stats ──────────────────────────────────────────────────────────
    stats = memory.get_stats()
    lines.append("## Memory")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total memories | {stats['total']} |")
    if stats["total"] > 0:
        for mtype, count in stats.get("by_type", {}).items():
            lines.append(f"| {mtype} | {count} |")
        lines.append(f"| Avg salience | {stats['avg_salience']:.3f} |")
        lines.append(f"| Min salience | {stats['min_salience']:.3f} |")
        lines.append(f"| Max salience | {stats['max_salience']:.3f} |")
    lines.append("")

    cold_stats = cold.get_stats()
    lines.append(f"**Cold storage:** {cold_stats['archived_count']} archived  ")
    lines.append("")

    # ── Active directives ─────────────────────────────────────────────────────
    lines.append("## Active Directives")
    lines.append("")
    if loop.reflection_engine:
        directives = loop.reflection_engine.get_directives()
        if directives:
            for i, d in enumerate(directives):
                lines.append(f"{i + 1}. {d}")
        else:
            lines.append("None.")
    else:
        lines.append("Reflection disabled.")
    lines.append("")

    # ── Logit bias suppressions ───────────────────────────────────────────────
    lines.append("## Active Logit Bias Suppressions")
    lines.append("")
    flat_biases = engine._flat_biases
    seq_suppressions = engine._sequence_suppressions
    if flat_biases or seq_suppressions:
        for token_id, bias in flat_biases.items():
            token_str = engine.tokenizer.decode([token_id])
            lines.append(f"- `{token_str}` (token {token_id}, bias {bias:+.1f}) [single-token]")
        for seq in seq_suppressions:
            phrase = engine.tokenizer.decode(seq)
            lines.append(f"- `{phrase}` ({len(seq)} tokens) [sequence]")
    else:
        lines.append("None active.")
    lines.append("")

    # ── Interaction log ───────────────────────────────────────────────────────
    lines.append("## Interaction Log")
    lines.append("")
    log_dir = os.path.join(BASE_DIR, "data/interaction_log")
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))

    if not log_files:
        lines.append("No interaction logs found.")
    else:
        import datetime as _dt
        import json as _json
        for idx, log_path in enumerate(log_files):
            try:
                with open(log_path, "r") as f:
                    entry = _json.load(f)
            except Exception:
                continue

            ts = _dt.datetime.fromtimestamp(entry.get("timestamp", 0)).strftime("%H:%M:%S")
            lines.append(f"### Interaction {idx + 1} — {ts}")
            lines.append("")
            lines.append(f"**Prompt tokens:** {entry.get('prompt_tokens', '?')}  "
                         f"**Response tokens:** {entry.get('response_tokens', '?')}  "
                         f"**Elapsed:** {entry.get('elapsed_seconds', '?')}s  ")
            lines.append("")
            lines.append("**User:**")
            lines.append("")
            user_msg = entry.get("user_message", "").replace("\n", "  \n> ")
            lines.append(f"> {user_msg}")
            lines.append("")
            lines.append("**Response:**")
            lines.append("")
            response_text = entry.get("response", "").replace("\n", "  \n> ")
            lines.append(f"> {response_text}")
            lines.append("")

            rating_val = entry.get("rating")
            corrections_val = entry.get("corrections", [])
            seg_ratings_val = entry.get("segment_ratings", [])
            if rating_val is not None or corrections_val or seg_ratings_val:
                lines.append("**Feedback:**")
                lines.append("")
                if rating_val is not None:
                    lines.append(f"- Rating: {rating_val}/5")
                for c in corrections_val:
                    lines.append(f"- Correction [{c.get('type', '?')}]: {c.get('content', '')[:100]}")
                for s in seg_ratings_val:
                    lines.append(
                        f"- Segment {s.get('sentence_index', '?')} "
                        f"rated {s.get('rating', '?')}/5: "
                        f"\"{s.get('sentence', '')[:80]}\""
                    )
                lines.append("")

    content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Saved: {filepath}\n({len(log_files)} interactions logged, {stats['total']} memories)"


# --- Test Runner ---

_test_prompts = []
_test_cursor = 0

_TR_DECAY = {25, 50, 75, 100}
_TR_CONSOLIDATION = {50, 100}


def _format_researcher_notes(entry):
    """Build the researcher notes display for a prompt entry."""
    if entry is None:
        return ""
    lines = []
    num = entry.get("number", "?")
    cat = entry.get("category", "")
    lines.append(f"#{num} \u2014 {cat}")
    expected = entry.get("expected_behavior")
    if expected:
        lines.append(f"Expected: {expected}")
    notes = entry.get("researcher_notes")
    if notes:
        lines.append(f"Researcher: {notes}")
    correction = entry.get("suggested_correction")
    if correction:
        lines.append(f"Suggested correction: {correction}")
    if not expected and not notes and not correction:
        lines.append("(no researcher notes)")
    return "\n".join(lines)


def load_test_prompts():
    """Load the 100-prompt evaluation file and reset cursor."""
    global _test_prompts, _test_cursor
    import json as _json
    path = os.path.join(BASE_DIR, "data/test_prompts_100.json")
    try:
        with open(path, "r") as f:
            _test_prompts = _json.load(f)
    except FileNotFoundError:
        return "Not loaded \u2014 file not found", ""
    _test_cursor = 0
    first_notes = _format_researcher_notes(_test_prompts[0]) if _test_prompts else ""
    return f"Prompt 0/{len(_test_prompts)} \u2014 Ready ({len(_test_prompts)} loaded)", first_notes


def _advance_test_prompt():
    """Run the next test prompt. Returns 7-element UI tuple, or None if done."""
    global _test_cursor
    if not _test_prompts or _test_cursor >= len(_test_prompts):
        return None

    entry = _test_prompts[_test_cursor]
    prompt_num = entry.get("number", _test_cursor + 1)
    prompt_text = entry.get("prompt", "")

    response = loop.chat(prompt_text)

    # --- Checkpoint automation ---
    if prompt_num in _TR_DECAY:
        memory.decay_pass()

    if prompt_num in _TR_CONSOLIDATION:
        consolidator.consolidate()

    _test_cursor += 1

    # Progress
    if _test_cursor >= len(_test_prompts):
        progress = f"Complete: {len(_test_prompts)}/{len(_test_prompts)} \u2014 All prompts finished"
    else:
        next_cat = _test_prompts[_test_cursor].get("category", "")
        progress = f"Prompt {_test_cursor}/{len(_test_prompts)} \u2014 {next_cat}"

    notes = _format_researcher_notes(entry)
    gradio_history = [{"role": h["role"], "content": h["content"]} for h in loop.history]

    return (gradio_history, response, progress, notes, 3, "Freetext", "", quick_status())


def run_next_test_prompt():
    """Run one test prompt. Conversation appears in the chatbox."""
    result = _advance_test_prompt()
    if result is None:
        done = f"Complete: {len(_test_prompts)}/{len(_test_prompts)}" if _test_prompts else "No prompts loaded."
        return (gr.update(), gr.update(), done, "Run finished.",
                gr.update(), gr.update(), gr.update(), quick_status())
    return result


def run_all_test_prompts():
    """Generator: run all remaining prompts, yielding after each."""
    if not _test_prompts or _test_cursor >= len(_test_prompts):
        done = f"Complete: {len(_test_prompts)}/{len(_test_prompts)}" if _test_prompts else "No prompts loaded."
        yield (gr.update(), gr.update(), done, "No prompts remaining.",
               gr.update(), gr.update(), gr.update(), quick_status())
        return
    while _test_cursor < len(_test_prompts):
        result = _advance_test_prompt()
        if result is None:
            break
        yield result


def stop_test_run():
    """Update progress to show the run was stopped."""
    if _test_prompts and _test_cursor < len(_test_prompts):
        return f"Stopped at {_test_cursor}/{len(_test_prompts)} -- click Run Next or Run All to continue"
    return gr.update()


def quick_status():
    """Compact status readout for the main view."""
    stats = memory.get_stats()
    dc = 0
    if loop.reflection_engine:
        dc, _ = loop.reflection_engine.get_directive_count()
    refl = "ON" if loop.reflect_enabled else "OFF"
    return (f"Interactions: {loop.interaction_count}  |  Memories: {stats['total']}\n"
            f"Directives: {dc}  |  Reflection: {refl}")


# --- Theme ---

b1tr0n1n = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fdf8e8", c100="#f5ebc4", c200="#edde9f", c300="#e4d07b",
        c400="#d4b83e", c500="#c9a227", c600="#a88520", c700="#8b7320",
        c800="#6e5a19", c900="#514213", c950="#3a2f0e"
    ),
    neutral_hue=gr.themes.Color(
        c50="#ede5d0", c100="#c8bda0", c200="#9a9080", c300="#7a7060",
        c400="#5a5245", c500="#3d362c", c600="#2a2620", c700="#1a1814",
        c800="#0f0e0b", c900="#0a0908", c950="#050504"
    ),
    font=gr.themes.GoogleFont("JetBrains Mono"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0908",
    body_text_color="#c8bda0",
    block_background_fill="#1a1814",
    block_border_color="#2a2620",
    block_label_text_color="#7a7060",
    block_title_text_color="#ede5d0",
    input_background_fill="#0f0e0b",
    input_border_color="#2a2620",
    input_placeholder_color="#5a5245",
    button_primary_background_fill="#c9a227",
    button_primary_text_color="#0a0908",
    button_primary_background_fill_hover="#e4d07b",
    button_secondary_background_fill="#1a1814",
    button_secondary_text_color="#c8bda0",
    button_secondary_border_color="#3d362c",
    button_secondary_background_fill_hover="#2a2620",
    checkbox_background_color="#0f0e0b",
    checkbox_border_color="#3d362c",
    slider_color="#c9a227",
)


# --- Layout ---

_S = "<span style='font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#7a7060'>"
_RULE = "<div style='border-top:1px solid #2a2620;margin:16px 0 12px'></div>"

_CSS = """
    .gradio-container {
        max-width: 1280px !important;
        padding: 16px 24px !important;
    }
    footer { display: none !important; }

    /* Header */
    .cl-header { margin-bottom: 4px !important; }
    .cl-header h1 {
        font-size: 18px !important;
        letter-spacing: 4px !important;
        font-weight: 400 !important;
        color: #c9a227 !important;
        margin: 0 !important;
    }
    .cl-subtitle {
        font-size: 10px !important;
        letter-spacing: 2px !important;
        color: #5a5245 !important;
        margin: -4px 0 12px 0 !important;
        font-style: italic;
    }

    /* Tab bar */
    .tab-nav button {
        font-size: 9px !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
        padding: 6px 14px !important;
    }

    /* Compact buttons */
    .cl-btn-row button {
        min-height: 34px !important;
        max-height: 34px !important;
        font-size: 11px !important;
        letter-spacing: 1px !important;
    }

    /* Compact slider */
    .cl-rating .label-wrap { font-size: 11px !important; }

    /* Quick status: monospace readout */
    .cl-qs textarea {
        font-size: 10px !important;
        line-height: 1.5 !important;
        color: #7a7060 !important;
        border: 1px solid #1a1814 !important;
        background: #0a0908 !important;
    }

    /* Feedback result line */
    .cl-fb-result textarea {
        font-size: 10px !important;
        color: #7a7060 !important;
        border: none !important;
        background: transparent !important;
        padding: 2px 0 !important;
    }

    /* Progress line */
    .cl-progress textarea {
        font-size: 11px !important;
        color: #c9a227 !important;
        border: 1px solid #2a2620 !important;
    }

    /* Section labels */
    .cl-section-label p {
        font-size: 9px !important;
        text-transform: uppercase !important;
        letter-spacing: 2.5px !important;
        color: #5a5245 !important;
        margin: 0 0 6px 0 !important;
    }

    /* Notes accordion */
    .cl-notes .label-wrap {
        font-size: 10px !important;
        padding: 4px 8px !important;
    }

    /* Chatbox breathing room */
    .cl-chat { margin-bottom: 0 !important; }

    /* Send row alignment */
    .cl-send-row { margin-top: 4px !important; }

    /* Right panel internal spacing */
    .cl-right-panel > div { margin-bottom: 0 !important; }

    /* Correction row compactness */
    .cl-corr-row .wrap { gap: 4px !important; }
"""

with gr.Blocks(title="Cultivated Learning") as app:

    # Header
    gr.Markdown("# \U0001F331 CULTIVATED LEARNING", elem_classes=["cl-header"])
    gr.Markdown("Frozen model. Growing mind.", elem_classes=["cl-subtitle"])

    last_response_state = gr.State("")

    with gr.Row():

        # LEFT: Conversation (dominant)
        with gr.Column(scale=7, min_width=500):
            chatbox = gr.Chatbot(
                label="Conversation", height=560,
                elem_classes=["cl-chat"],
            )
            with gr.Row(elem_classes=["cl-send-row"]):
                msg = gr.Textbox(
                    show_label=False, placeholder="Speak...",
                    scale=7, lines=1, max_lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # RIGHT: Test Runner + Feedback + Status (stacked)
        with gr.Column(scale=3, min_width=300, elem_classes=["cl-right-panel"]):

            # -- Test Runner --
            gr.Markdown("TEST", elem_classes=["cl-section-label"])
            with gr.Row(elem_classes=["cl-btn-row"]):
                tr_load_btn = gr.Button("Load", variant="secondary", scale=1, min_width=60)
                tr_next_btn = gr.Button("Next", variant="primary", scale=1, min_width=60)
                tr_all_btn = gr.Button("All", variant="secondary", scale=1, min_width=60)
                tr_stop_btn = gr.Button("Stop", variant="secondary", scale=1, min_width=60)
            tr_progress = gr.Textbox(
                show_label=False, value="Not loaded", interactive=False,
                lines=1, max_lines=1, elem_classes=["cl-progress"],
            )
            with gr.Accordion("Notes", open=False, elem_classes=["cl-notes"]):
                tr_notes = gr.Textbox(
                    show_label=False, interactive=False, lines=3,
                    placeholder="Load prompts to see researcher notes.",
                )

            gr.HTML(_RULE)

            # -- Feedback --
            gr.Markdown("FEEDBACK", elem_classes=["cl-section-label"])
            rating = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Rating", elem_classes=["cl-rating"],
            )
            with gr.Row(elem_classes=["cl-corr-row"]):
                correction_type = gr.Dropdown(
                    choices=_ALL_CORRECTION_CHOICES, value="Freetext",
                    show_label=False, scale=2, min_width=100,
                )
                correction_content = gr.Textbox(
                    show_label=False, placeholder="Correction...",
                    scale=5, lines=1, max_lines=1,
                )
            fb_btn = gr.Button("Submit Feedback", variant="primary", elem_classes=["cl-btn-row"])
            fb_output = gr.Textbox(
                show_label=False, interactive=False,
                lines=1, max_lines=1, elem_classes=["cl-fb-result"],
            )

            gr.HTML(_RULE)

            # -- Quick Status --
            gr.Markdown("STATUS", elem_classes=["cl-section-label"])
            qs_display = gr.Textbox(
                show_label=False, interactive=False,
                lines=2, max_lines=2,
                value=quick_status(),
                elem_classes=["cl-qs"],
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  TABS (secondary tools, full width below workspace)
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tabs():

        with gr.Tab("Memory"):
            with gr.Row():
                mem_type = gr.Dropdown(
                    choices=["All", "Episodic", "Semantic", "Procedural", "Reflective"],
                    value="All", label="Filter by type", scale=3,
                )
                browse_btn = gr.Button("Browse", variant="primary", scale=1)
                dir_btn = gr.Button("Directives", variant="secondary", scale=1)
                reload_btn = gr.Button("Reload", variant="secondary", scale=1)
            mem_output = gr.Textbox(label="Memories", interactive=False, lines=14)
            browse_btn.click(fn=browse_memories, inputs=[mem_type], outputs=mem_output)
            dir_output = gr.Textbox(label="Directives", interactive=False, lines=5)
            dir_btn.click(fn=get_directives, outputs=dir_output)
            reload_btn.click(fn=reload_directives, outputs=dir_output)

        with gr.Tab("Evaluation"):
            eval_btn = gr.Button("Generate Report", variant="primary")
            eval_output = gr.Textbox(label="Evaluation Report", interactive=False, lines=28)
            eval_btn.click(fn=get_evaluation_report, outputs=eval_output)

        with gr.Tab("Maintenance"):
            with gr.Row():
                consol_btn = gr.Button("Consolidate", variant="primary", scale=1)
                decay_btn = gr.Button("Decay", variant="secondary", scale=1)
                archive_btn = gr.Button("Archive", variant="secondary", scale=1)
            maint_output = gr.Textbox(label="Result", interactive=False, lines=4)
            consol_btn.click(fn=run_consolidation, outputs=maint_output)
            decay_btn.click(fn=run_decay, outputs=maint_output)
            archive_btn.click(fn=run_archive, outputs=maint_output)

            gr.HTML(_RULE)
            with gr.Row():
                resurface_query = gr.Textbox(
                    label="Resurface Query", placeholder="Search cold storage...", scale=5,
                )
                resurface_btn = gr.Button("Resurface", variant="secondary", scale=1)
            resurface_output = gr.Textbox(label="Result", interactive=False, lines=3)
            resurface_btn.click(fn=run_resurface, inputs=[resurface_query], outputs=resurface_output)

            gr.HTML(_RULE)
            gr.Markdown(f"{_S}SEGMENT FEEDBACK</span>")
            load_seg_btn = gr.Button("Load Response Segments", variant="secondary")
            sentences_display = gr.Textbox(label="Sentences", interactive=False, lines=6)
            seg_idx = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Sentence #", visible=False)
            selected_sentence = gr.Textbox(label="Selected", interactive=False, lines=2)
            seg_rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Segment rating")
            submit_seg_btn = gr.Button("Submit Segment Feedback", variant="primary")
            seg_output = gr.Textbox(label="Result", interactive=False)
            load_seg_btn.click(fn=load_segments, inputs=[last_response_state],
                               outputs=[sentences_display, seg_idx, selected_sentence])
            seg_idx.change(fn=select_sentence, inputs=[last_response_state, seg_idx],
                           outputs=[selected_sentence])
            submit_seg_btn.click(fn=submit_segment_feedback,
                                 inputs=[last_response_state, seg_idx, seg_rating],
                                 outputs=[seg_output])

            gr.HTML(_RULE)
            gr.Markdown(f"{_S}STANDALONE CORRECTIONS</span>")
            with gr.Row():
                standalone_corr_type = gr.Dropdown(
                    choices=_ALL_CORRECTION_CHOICES, value="Freetext",
                    label="Type", scale=2,
                )
                standalone_corr_content = gr.Textbox(label="Content", scale=4)
            standalone_corr_btn = gr.Button("Apply", variant="primary")
            standalone_corr_output = gr.Textbox(label="Result", interactive=False)
            standalone_corr_btn.click(fn=apply_correction,
                                      inputs=[standalone_corr_type, standalone_corr_content],
                                      outputs=standalone_corr_output)

            gr.HTML(_RULE)
            gr.Markdown(f"{_S}ADAPTATION EVIDENCE</span>")
            with gr.Row():
                adapt_type = gr.Dropdown(choices=["help", "harm"], value="help", label="Event", scale=1)
                adapt_desc = gr.Textbox(label="Description", scale=4)
            adapt_btn = gr.Button("Log Event", variant="secondary")
            adapt_output = gr.Textbox(label="Result", interactive=False)
            adapt_btn.click(fn=log_adaptation_event, inputs=[adapt_type, adapt_desc], outputs=adapt_output)

        with gr.Tab("Status"):
            status_output = gr.Textbox(label="System Status", interactive=False, lines=12)
            status_btn = gr.Button("Refresh", variant="primary")
            status_btn.click(fn=get_status, outputs=status_output)

        with gr.Tab("Upload"):
            upload_file = gr.File(label="File (.json, .txt, or .md)", file_types=[".json", ".txt", ".md"])
            with gr.Row():
                source_label_input = gr.Textbox(label="Source label", placeholder="e.g. Fields of Fire Rulebook", scale=4)
                chunk_mode = gr.Radio(choices=["Paragraph", "Line"], value="Paragraph", label="Chunk by", scale=2)
            upload_btn = gr.Button("Upload & Store", variant="primary")
            upload_output = gr.Textbox(label="Result", interactive=False, lines=2)
            upload_btn.click(fn=upload_document, inputs=[upload_file, source_label_input, chunk_mode],
                             outputs=upload_output)

        with gr.Tab("Export"):
            export_btn = gr.Button("Export Session Report", variant="primary")
            export_output = gr.Textbox(label="Result", interactive=False, lines=3)
            export_btn.click(fn=export_session_report, outputs=export_output)

    # ══════════════════════════════════════════════════════════════════════════
    #  EVENT WIRING
    # ══════════════════════════════════════════════════════════════════════════

    # Chat: 7 outputs
    _chat_outputs = [msg, chatbox, last_response_state, rating, correction_type, correction_content, qs_display]
    send_btn.click(fn=chat, inputs=[msg, chatbox], outputs=_chat_outputs)
    msg.submit(fn=chat, inputs=[msg, chatbox], outputs=_chat_outputs)

    # Feedback: 2 outputs
    fb_btn.click(fn=give_feedback_main,
                 inputs=[rating, correction_type, correction_content],
                 outputs=[fb_output, qs_display])

    # Test Runner: 8 outputs
    tr_load_btn.click(fn=load_test_prompts, outputs=[tr_progress, tr_notes])

    _tr_outputs = [chatbox, last_response_state, tr_progress, tr_notes,
                   rating, correction_type, correction_content, qs_display]
    tr_next_btn.click(fn=run_next_test_prompt, outputs=_tr_outputs)
    _tr_all_event = tr_all_btn.click(fn=run_all_test_prompts, outputs=_tr_outputs)
    tr_stop_btn.click(fn=stop_test_run, outputs=[tr_progress], cancels=[_tr_all_event])


app.launch(server_name="0.0.0.0", server_port=7880, share=False, theme=b1tr0n1n, css=_CSS)
