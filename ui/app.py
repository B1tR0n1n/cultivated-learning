import sys
sys.path.insert(0, "/workspace/Projects/cultivated-learning")

import gradio as gr
from engine.inference import InferenceEngine
from core.memory_store import MemoryStore
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop

# Initialize
print("Loading models...")
engine = InferenceEngine("/workspace/models/results/Mistral-7B-Instruct-v0.3")
engine.load()

memory = MemoryStore(
    persist_dir="/workspace/Projects/cultivated-learning/data/memory_db",
    engine=engine
)

assembler = ContextAssembler(engine=engine, memory_store=memory)

loop = InteractionLoop(
    engine=engine,
    memory_store=memory,
    assembler=assembler,
    log_dir="/workspace/Projects/cultivated-learning/data/interaction_log"
)
print("Ready.")


def chat(message, history):
    """Send message through the full cognitive pipeline."""
    response = loop.chat(message)
    return response


def give_feedback(rating, correction):
    """Submit feedback on the last interaction."""
    rating = int(rating)
    corr = correction.strip() if correction.strip() else None
    loop.feedback(rating=rating, correction=corr)
    msg = f"Feedback recorded: rating {rating}"
    if corr:
        msg += f", correction stored"
    return msg


def get_status():
    """Show memory stats."""
    stats = memory.get_stats()
    status = loop.status()
    lines = [
        f"Interactions: {status.get('interaction_count', '?')}",
        f"Memories: {stats['total']}",
    ]
    if stats["total"] > 0:
        lines.append(f"By type: {stats['by_type']}")
        lines.append(f"Avg salience: {stats['avg_salience']:.3f}")
        lines.append(f"Range: {stats['min_salience']:.3f} — {stats['max_salience']:.3f}")
    return "\n".join(lines)


with gr.Blocks(title="Cultivated Learning") as app:
    gr.Markdown("# 🌱 Cultivated Learning")
    gr.Markdown("*Frozen model. Growing mind.*")
    
    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(
            fn=chat,

        )
    
    with gr.Tab("Feedback"):
        gr.Markdown("### Rate the last response")
        rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating (1-5)")
        correction = gr.Textbox(
            label="Correction (optional)", 
            placeholder="e.g. Be more concise next time"
        )
        fb_btn = gr.Button("Submit Feedback")
        fb_output = gr.Textbox(label="Result", interactive=False)
        fb_btn.click(fn=give_feedback, inputs=[rating, correction], outputs=fb_output)
    
    with gr.Tab("Status"):
        status_output = gr.Textbox(label="System Status", interactive=False, lines=6)
        status_btn = gr.Button("Refresh")
        status_btn.click(fn=get_status, outputs=status_output)

app.launch(server_name="0.0.0.0", server_port=7880, share=False)
