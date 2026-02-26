# 🌱 Cultivated Learning

**Longitudinal Behavioral Development in Frozen Language Models Through Memory-Augmented Reflective Interaction**

> The model is the organism. The human is the gardener. Traditional training changes the brain. Cultivated Learning changes the environment the brain operates in.

## What Is This?

A cognitive architecture built around a frozen LLM that enables behavioral development over extended interaction — without ever changing a single weight.

The system wraps a stateless language model in a stateful shell: persistent memory, dynamic context assembly, recursive self-reflection, and human feedback integration. The model behaves as if it's learning, adapting to the user over time through memory alone.

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                  INTERACTION LAYER                   │
│            (User ↔ System Interface)                 │
├──────────┬──────────┬──────────────┬────────────────┤
│  MEMORY  │ CONTEXT  │ REFLECTION   │   FEEDBACK     │
│  STORE   │ ASSEMBLER│ ENGINE       │   INTEGRATOR   │
│          │          │              │                │
│ Semantic │ Dynamic  │ Post-hoc     │ Human signal   │
│ vectors +│ window   │ recursive    │ → salience     │
│ metadata │ packing  │ analysis     │ adjustments    │
├──────────┴──────────┴──────────────┴────────────────┤
│              BASE LLM (Stateless Inference)          │
└─────────────────────────────────────────────────────┘
```

## Key Features

- **Memory Store** — ChromaDB-backed semantic memory with salience scoring and exponential decay
- **Context Assembler** — Token-budgeted prompt packing that prioritizes accumulated knowledge over conversation history
- **Reflection Engine** — 4-depth recursive self-analysis generating behavioral directives
- **Feedback System** — Human ratings and corrections translated into salience adjustments
- **Memory Consolidation** — Distills fading episodic memories into durable semantic memories
- **Cold Storage** — Archives decayed memories with semantic resurfacing on strong query match
- **Blended Retrieval** — Ranks memories by combining semantic similarity (60%) with salience (40%)

## Tech Stack

| Component | Details |
|-----------|---------|
| Base Model | Mistral 7B Instruct v0.3 |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| Vector Database | ChromaDB |
| GPU | NVIDIA RTX 5090 (34 GB VRAM) |
| Framework | PyTorch 2.9.1, Transformers 4.57.1 |
| UI | Gradio |
| Environment | Docker (Ubuntu 24.04) |

## Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| Episodic | Raw interaction records | "User asked about ML basics" |
| Semantic | Distilled facts and preferences | "User prefers concise answers" |
| Procedural | Behavioral directives | "Ground explanations in practical examples" |
| Reflective | Self-analysis observations | "Responses on technical topics tend to be verbose" |

## Design Philosophy

**Memories outrank conversation history.** Most chatbots drop old memories to keep recent turns. This system does the opposite — it will forget what was just said before it forgets what it has learned about you. Accumulated knowledge is more valuable than short-term context for longitudinal development.

## Research Questions

1. Can a frozen model exhibit measurable behavioral evolution through inference-time architecture alone?
2. Does the reflection engine produce better outcomes than feedback alone?
3. Where does inference-time learning hit its ceiling, and what causes it?
4. Does memory consolidation improve retrieval quality over time?

## Project Status

- [x] Phase 1: Memory + Context + Interaction Loop
- [x] Phase 1.5: Dedicated Embeddings + Blended Retrieval + Gradio UI
- [x] Phase 2: Reflection Engine
- [x] Phase 2: Memory Consolidation
- [x] Phase 2: Cold Storage
- [ ] Phase 3: Longitudinal Evaluation (100+ interactions)
- [ ] Phase 3: A/B Testing (framework vs vanilla model)

## Getting Started
```python
from engine.inference import InferenceEngine
from core.memory_store import MemoryStore
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop

engine = InferenceEngine("/path/to/Mistral-7B-Instruct-v0.3")
engine.load()

memory = MemoryStore(persist_dir="./data/memory_db", engine=engine)
assembler = ContextAssembler(engine=engine, memory_store=memory)
loop = InteractionLoop(engine=engine, memory_store=memory, assembler=assembler)

response = loop.chat("Hello, what do you remember about me?")
loop.feedback(rating=5, correction="Remember to be concise")
```

## License

MIT

## Author

**b1tr0n1n** — Systems-level AI engineering, honest research, open-source.
