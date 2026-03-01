import sys
sys.path.insert(0, "/workspace/Projects/cultivated-learning")

import gradio as gr
from engine.inference import InferenceEngine
from core.memory_store import MemoryStore, MemoryType
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop
from core.consolidation import ConsolidationEngine
from core.cold_storage import ColdStorage
from evaluation.metrics import EvaluationMetrics

# Initialize
print("Loading models...")
engine = InferenceEngine("/workspace/models/results/Mistral-7B-Instruct-v0.3")
engine.load()

memory = MemoryStore(
    persist_dir="/workspace/Projects/cultivated-learning/data/memory_db",
    engine=engine
)

assembler = ContextAssembler(engine=engine, memory_store=memory)

metrics = EvaluationMetrics(
    log_dir="/workspace/Projects/cultivated-learning/data/evaluation",
    memory_store=memory
)

loop = InteractionLoop(
    engine=engine,
    memory_store=memory,
    assembler=assembler,
    log_dir="/workspace/Projects/cultivated-learning/data/interaction_log",
    metrics_dir="/workspace/Projects/cultivated-learning/data/evaluation",
)

consolidator = ConsolidationEngine(engine, memory)

cold = ColdStorage(
    engine=engine,
    memory_store=memory,
    archive_dir="/workspace/Projects/cultivated-learning/data/cold_storage"
)

print("All systems ready.")


# --- Functions ---

def chat(message, history):
    response = loop.chat(message)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def give_feedback(rating, correction, target_query):
    rating = int(rating)
    corr = correction.strip() if correction.strip() else None
    tq = target_query.strip() if target_query.strip() else None
    loop.feedback(rating=rating, correction=corr, target_query=tq)
    msg = f"Rating {rating} recorded."
    if tq:
        msg += f" Targeted at: '{tq[:50]}'"
    if corr:
        msg += f" Correction stored at salience 0.9."
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
        from core.memory_store import MemoryUnit
        mems = []
        for i in range(len(all_data["ids"])):
            mems.append(MemoryUnit.from_chroma(
                id=all_data["ids"][i],
                document=all_data["documents"][i],
                metadata=all_data["metadatas"][i],
            ))

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

with gr.Blocks(title="Cultivated Learning", theme=b1tr0n1n, css="""
    .gradio-container { max-width: 900px !important; }
    footer { display: none !important; }
    .tab-nav button { font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
""") as app:

    gr.Markdown("# 🌱 CULTIVATED LEARNING")
    gr.Markdown("*Frozen model. Growing mind. — b1tr0n1n*")

    with gr.Tab("Chat"):
        chatbox = gr.Chatbot(label="Conversation", height=420)
        with gr.Row():
            msg = gr.Textbox(label="Message", placeholder="Speak...", scale=5)
            send_btn = gr.Button("Send", variant="primary", scale=1)
        send_btn.click(fn=chat, inputs=[msg, chatbox], outputs=[msg, chatbox])
        msg.submit(fn=chat, inputs=[msg, chatbox], outputs=[msg, chatbox])

    with gr.Tab("Feedback"):
        gr.Markdown("### Signal")
        rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating")
        correction = gr.Textbox(label="Correction", placeholder="Behavioral correction (optional)")
        target_query = gr.Textbox(label="About (optional)", placeholder="What topic is this feedback about? Leave blank for last interaction.")
        fb_btn = gr.Button("Submit", variant="primary")
        fb_output = gr.Textbox(label="Result", interactive=False)
        fb_btn.click(fn=give_feedback, inputs=[rating, correction, target_query], outputs=fb_output)

        gr.Markdown("---")
        gr.Markdown("### Adaptation Evidence")
        adapt_type = gr.Dropdown(choices=["help", "harm"], value="help", label="Event Type")
        adapt_desc = gr.Textbox(label="Description", placeholder="What happened?")
        adapt_btn = gr.Button("Log Event", variant="secondary")
        adapt_output = gr.Textbox(label="Result", interactive=False)
        adapt_btn.click(fn=log_adaptation_event, inputs=[adapt_type, adapt_desc], outputs=adapt_output)

    with gr.Tab("Memory"):
        gr.Markdown("### Browse")
        with gr.Row():
            mem_type = gr.Dropdown(
                choices=["All", "Episodic", "Semantic", "Procedural", "Reflective"],
                value="All", label="Filter by type"
            )
            browse_btn = gr.Button("Browse", variant="primary")
        mem_output = gr.Textbox(label="Memories", interactive=False, lines=15)
        browse_btn.click(fn=browse_memories, inputs=[mem_type], outputs=mem_output)

        gr.Markdown("---")
        gr.Markdown("### Directives")
        dir_btn = gr.Button("Show Active Directives", variant="secondary")
        dir_output = gr.Textbox(label="Directives", interactive=False, lines=6)
        dir_btn.click(fn=get_directives, outputs=dir_output)

    with gr.Tab("Evaluation"):
        gr.Markdown("### Longitudinal Metrics")
        eval_btn = gr.Button("Generate Report", variant="primary")
        eval_output = gr.Textbox(label="Evaluation Report", interactive=False, lines=30)
        eval_btn.click(fn=get_evaluation_report, outputs=eval_output)

    with gr.Tab("Maintenance"):
        gr.Markdown("### Consolidation")
        gr.Markdown("Distill fading episodic memories into durable semantic memories.")
        consol_btn = gr.Button("Run Consolidation", variant="primary")
        consol_output = gr.Textbox(label="Result", interactive=False, lines=6)
        consol_btn.click(fn=run_consolidation, outputs=consol_output)

        gr.Markdown("---")
        gr.Markdown("### Decay")
        gr.Markdown("Apply salience decay to all memories.")
        decay_btn = gr.Button("Run Decay Pass", variant="secondary")
        decay_output = gr.Textbox(label="Result", interactive=False)
        decay_btn.click(fn=run_decay, outputs=decay_output)

        gr.Markdown("---")
        gr.Markdown("### Cold Storage")
        gr.Markdown("Archive faded memories. Resurface on strong semantic match.")
        with gr.Row():
            archive_btn = gr.Button("Run Archive Pass", variant="secondary")
            archive_output = gr.Textbox(label="Result", interactive=False)
        archive_btn.click(fn=run_archive, outputs=archive_output)

        with gr.Row():
            resurface_query = gr.Textbox(label="Resurface Query", placeholder="Search cold storage...")
            resurface_btn = gr.Button("Resurface", variant="secondary")
        resurface_output = gr.Textbox(label="Result", interactive=False, lines=4)
        resurface_btn.click(fn=run_resurface, inputs=[resurface_query], outputs=resurface_output)

    with gr.Tab("Status"):
        status_output = gr.Textbox(label="System Status", interactive=False, lines=12)
        status_btn = gr.Button("Refresh", variant="primary")
        status_btn.click(fn=get_status, outputs=status_output)


app.launch(server_name="0.0.0.0", server_port=7880, share=False)
