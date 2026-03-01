# CLAUDE.md — Cultivated Learning

## Project Identity

**Cultivated Learning** is a continual-interaction LLM framework that enables longitudinal behavioral development in a frozen language model through memory-augmented reflective interaction.

**Thesis:** Can a frozen model exhibit developmental trajectories — measurable behavioral evolution — through inference-time cognitive architecture alone, without weight updates?

**Metaphor:** The model is the organism. The human is the gardener. Traditional training changes the brain. Cultivated Learning changes the environment the brain operates in.

**Author:** b1tr0n1n (Tom) — architect, not coder. Designs systems, uses AI for implementation.

**Repo:** `github.com/b1tr0n1n/cultivated-learning` (public)

**Current version:** 0.4.0

---

## Environment

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5090, 34.19 GB VRAM, sm_120 (Blackwell) |
| Container | `ml-jupyter` (Ubuntu 24.04, Docker) |
| Host | Windows + WSL |
| Mount | `/workspace` bind-mounted to host `/home/b1tr0n1n/ml-workspace` |
| PyTorch | 2.9.1+cu128 (compiled from source for sm_120) |
| Transformers | 4.57.1 |
| Accelerate | 1.12.0 |
| ChromaDB | 1.5.1 |
| Gradio | 6.0+ |
| Python | 3.10 |
| CUDA | 13.0 |
| Project path | `/workspace/Projects/cultivated-learning/` |

### Models

| Model | Params | VRAM | Dim | Purpose |
|-------|--------|------|-----|---------|
| Mistral 7B Instruct v0.3 | ~7B | 14.50 GB | 4096 hidden | Generation |
| all-MiniLM-L6-v2 | ~22M | ~80 MB | 384 | Embeddings (semantic search) |

### Ports

| Port | Service |
|------|---------|
| 7860 | TensorBoard |
| 7880 | Gradio UI |
| 8888 | Jupyter |

---

## Architecture Overview

```
User ↔ Gradio UI (app.py, port 7880)
         ↕
   InteractionLoop (orchestrator)
    ├── ContextAssembler
    │     ├── Sable system prompt (~600 tokens, fixed)
    │     ├── Active directives (up to 10% of budget)
    │     ├── Retrieved memories (up to 40% of remaining)
    │     └── Conversation history (fills remaining space)
    ├── MemoryStore (ChromaDB)
    │     ├── 4 memory types: episodic, semantic, procedural, reflective
    │     ├── Blended retrieval: 60% semantic similarity + 40% salience
    │     ├── Salience decay (exponential, hourly)
    │     └── Supersession (corrections mark wrong memories)
    ├── ReflectionEngine
    │     ├── Depth 0: Factual (what happened?)
    │     ├── Depth 1: Analytical (what patterns?)
    │     ├── Depth 2: Prescriptive (what should change?) → 3-gate filter
    │     └── Depth 3: Meta-coherence (are directives consistent?)
    ├── ConsolidationEngine (episodic → semantic distillation)
    ├── ColdStorage (archive faded memories, resurface on match)
    ├── EvaluationMetrics (6 longitudinal metrics)
    └── InferenceEngine (Mistral generation + MiniLM embeddings)
```

### Interaction Flow

Every `loop.chat(message)` call:
1. **Get directives** from reflection engine
2. **Assemble** prompt: system prompt + directives + retrieved memories + conversation history + user message
3. **Generate** response via Mistral
4. **Store** episodic memory of the interaction
5. **Reflect** (4-depth recursive analysis, may generate new directive)
6. **Log** metrics (compute cost, behavioral drift, directive snapshot)
7. **Write** interaction JSON to disk

### Design Principles

1. **The model never changes.** All adaptation happens in memory and context.
2. **Everything is auditable.** Every memory, directive, salience adjustment is inspectable.
3. **Everything is reversible.** Delete a memory and it's gone.
4. **Memories outrank conversation history.** When token budget is tight, history gets cut first. Memories are protected. Directives are never cut.
5. **User corrections are ground truth.** Corrections kill directives, supersede memories, and are stored at salience 0.9/confidence 1.0.
6. **Heuristics over LLM self-evaluation.** A 7B model cannot reliably judge its own output.

---

## File Map

```
cultivated-learning/
├── engine/
│   └── inference.py              # v3
├── core/
│   ├── memory_store.py           # v3
│   ├── context_assembler.py      # v4
│   ├── interaction_loop.py       # v5
│   ├── reflection.py             # v4
│   ├── consolidation.py          # v1
│   ├── cold_storage.py           # v1
│   └── startup.py                # v2
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # v1
├── ui/
│   └── app.py                    # v5
├── data/
│   ├── memory_db/                # ChromaDB persistent storage
│   ├── interaction_log/          # JSON per interaction
│   ├── evaluation/               # 6 metrics JSON files
│   └── cold_storage/             # cold_archive.json
├── notebooks/
│   └── lab.ipynb
├── CLAUDE.md                     # This file
└── CHANGELOG.md                  # Full version history
```

---

## File-by-File Reference

### engine/inference.py (v3)

LLM wrapper. All model interaction goes through here.

**Class:** `InferenceEngine`

**Constructor:**
```python
InferenceEngine(
    model_path="/workspace/models/results/Mistral-7B-Instruct-v0.3",
    max_context=16384,  # was 4096 in v1-v2
    embedding_model_path="/workspace/models/results/all-MiniLM-L6-v2"
)
```

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `load()` | — | self | Loads Mistral (float16, device_map="auto") + MiniLM. Must call before use. |
| `generate(prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9)` | prompt str | str | Standard generation. repetition_penalty=1.1. Clears CUDA cache after. |
| `generate_structured(prompt, max_new_tokens=512, temperature=0.3)` | prompt str | str | Low-temp for analytical/parseable outputs. Used by reflection engine. |
| `get_embedding(text)` | text str | numpy array (384,) | Uses MiniLM, normalized. Used for semantic search and dedup. |
| `get_embedding_dimension()` | — | int (384) | — |
| `count_tokens(text)` | text str | int | Uses Mistral tokenizer. Used for budget management. |

**Key detail:** `generate()` truncates input to `max_context - max_new_tokens` before sending to model. Response budget is 1024 tokens.

---

### core/memory_store.py (v3)

Persistent memory with ChromaDB vector search, salience decay, and supersession.

**Enums:**
```python
class MemoryType(Enum):
    EPISODIC = "episodic"      # What happened in a specific interaction
    SEMANTIC = "semantic"       # Distilled facts and preferences
    PROCEDURAL = "procedural"   # Behavioral directives (how to act)
    REFLECTIVE = "reflective"   # Meta-observations about performance
```

**Dataclass:** `MemoryUnit`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `content` | str | required | The text |
| `memory_type` | MemoryType | required | Classification |
| `id` | str | uuid4 | Unique identifier |
| `created_at` | float | time.time() | Creation timestamp |
| `last_accessed` | float | time.time() | Last retrieval timestamp (for decay) |
| `access_count` | int | 0 | Times retrieved |
| `source_interaction_id` | str | "" | Which interaction created this |
| `salience_score` | float | 0.5 | Importance (0.0–1.0) |
| `confidence` | float | 0.7 | Reliability (user corrections = 1.0) |
| `superseded_by` | str\|None | None | Points to newer contradicting memory |
| `tags` | list[str] | [] | Semantic labels |

**Class:** `MemoryStore`

**Constructor:**
```python
MemoryStore(
    persist_dir="/workspace/Projects/cultivated-learning/data/memory_db",
    engine=engine,   # InferenceEngine instance
    decay_rate=0.01  # Exponential decay rate per hour
)
```

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `store(memory, embedding=None)` | MemoryUnit | memory.id | Upserts to ChromaDB. Auto-embeds via engine if no embedding provided. |
| `retrieve(query_text, top_k=10, min_salience=0.1, similarity_weight=0.6, salience_weight=0.4)` | query str | list[MemoryUnit] | Blended retrieval. Fetches top_k*3 candidates, filters by salience and superseded_by, scores as (sim*0.6 + salience*0.4), returns top_k. Updates access timestamps on retrieved memories. |
| `retrieve_by_type(memory_type, limit=20)` | MemoryType | list[MemoryUnit] | Get all memories of a type. Used by reflection engine for directive loading. |
| `adjust_salience(memory_id, delta)` | str, float | — | Clamps to [0.0, 1.0]. |
| `supersede(memory_id, superseded_by_id)` | str, str | — | Marks memory as superseded. Also drops salience by 0.3. Superseded memories are excluded from `retrieve()`. |
| `supersede_by_correction(correction_text, correction_id, similarity_threshold=0.45)` | str, str | list[dict] | **Breaks hallucination reinforcement loop.** Finds all episodic/semantic memories with cosine similarity ≥ 0.45 to the correction. Skips procedural, reflective, and other corrections. Marks matches as superseded. Returns list of what was superseded. |
| `decay_pass()` | — | — | Applies `salience * e^(-0.01 * hours_since_access)` to all memories. |
| `get_stats()` | — | dict | Returns total, by_type, avg/min/max salience, superseded count. |

**Critical retrieval filter (line ~114):**
```python
if mem.salience_score >= min_salience and mem.superseded_by is None:
```
This is how supersession prevents hallucination reinforcement — superseded memories never appear in retrieval results.

---

### core/context_assembler.py (v4)

Packs the prompt within token budget. Contains the Sable identity system prompt.

**Class:** `ContextAssembler`

**Constructor:**
```python
ContextAssembler(
    engine=engine,
    memory_store=memory,
    max_context=16384,   # was 4096 in v1-v2
    max_response=1024    # was 512 in v1-v2
)
# available_tokens = 16384 - 1024 = 15360
```

**System prompt:** ~600 tokens defining Sable's identity, cognition, voice, absolute rules, learning behavior, and negative identity. Operates at same architectural level as instruct training to suppress default personality. Full text is in the file.

**Assembly priority (highest first):**
1. System prompt (Sable identity) — always included, ~600 tokens fixed
2. User's current message — always included
3. Active directives — up to 10% of remaining budget. Trimmed if over budget.
4. Retrieved memories — up to 40% of remaining budget, ranked by blended score
5. Conversation history — fills remaining space, most recent turns first

**When budget is tight:** History gets cut first → low-salience memories next → directives trimmed → system prompt and user message are NEVER cut.

**Methods:**
| Method | Args | Returns |
|--------|------|---------|
| `assemble(user_message, conversation_history=None, directives=None)` | str, list[dict], list[str] | str (full prompt) |
| `get_token_report(prompt)` | str | dict (tokens, utilization %) |

**Prompt format:** `[INST] {system_prompt}{directives}{memories}{history}\nUser: {message} [/INST]`

---

### core/reflection.py (v4)

Post-interaction recursive self-analysis with heuristic quality gates.

**Module-level constants:**
- `PLATITUDE_VERBS`: 10 verbs (offer, encourage, suggest, provide, share, celebrate, acknowledge, support, validate, empathize)
- `PLATITUDE_NOUNS`: 14 nouns (resources, suggestions, encouragement, empathy, mindfulness, well-being, self-care, creativity, motivation, inspiration, positivity, relaxation, wellness, wellbeing)
- `REACTIVE_PREFIXES`: "when the user", "if the user", "whenever the user", etc.

**Function:** `heuristic_quality_check(directive) → (bool, str)`

Deterministic filter. Rejects:
1. Too long (>120 chars)
2. Too short (<20 chars)
3. Platitude verb + noun combo (10 × 14 = 140 patterns)
4. Reactive prefix (topic-specific, not behavioral)
5. Filler phrases ("feel free", "don't hesitate", "here to support", etc.)
6. Missing actionable verb (must contain: keep, limit, avoid, prioritize, ground, use, prefer, default, start, end, include, omit, structure, format, verify, check, confirm, ask, wait, respond)

**Class:** `ReflectionEngine`

**Constructor:**
```python
ReflectionEngine(
    engine=engine,
    memory_store=memory,
    max_depth=3,
    max_directives=6,       # Hard cap on active procedural memories
    dedup_threshold=0.60    # Cosine similarity for duplicate detection
)
```

**On init:** Calls `_load_directives()` which:
1. Loads all PROCEDURAL memories from ChromaDB
2. Sorts by salience (highest first)
3. Runs each through `heuristic_quality_check()` — failures get salience -0.5
4. Keeps top 6 that pass
5. Demotes any beyond cap with salience -0.3
6. Prints startup purge report

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `reflect(user_message, assistant_response, interaction_id)` | str, str, str | list[MemoryUnit] | Runs all 4 depths. Stores new memories. Called by InteractionLoop after each chat. |
| `check_correction_conflicts(correction_text)` | str | list[str] | Computes cosine similarity between correction and each active directive. Similarity ≥ 0.45 = directive killed (salience -0.6, removed from active list). Returns list of killed directive strings. |
| `get_directives()` | — | list[str] | Copy of active directive strings. Used by ContextAssembler. |
| `get_directive_count()` | — | (int, int) | (current, max). Used by status display. |

**Reflection depths:**

| Depth | Name | Input | Output | Salience | LLM call |
|-------|------|-------|--------|----------|----------|
| 0 | Factual | user msg + response | Reflective memory | 0.4 | generate_structured, 200 tokens |
| 1 | Analytical | user msg + response + 5 recent memories | Reflective memory (or None if <2 memories) | 0.5 | generate_structured, 200 tokens |
| 2 | Prescriptive | D0 + D1 analysis | Procedural memory (directive) or None | 0.7 (fixed) | generate_structured, 100 tokens |
| 3 | Meta-coherence | All active directives | Reflective memory or None (if COHERENT) | 0.6 | generate_structured, 200 tokens |

**Depth 2 gates (in order):**
1. **Heuristic filter** — `heuristic_quality_check()`. Rejects platitudes, filler, vague directives.
2. **Semantic deduplication** — cosine similarity ≥ 0.60 against all existing directives = rejected.
3. **Directive cap** — if at 6, must displace weakest. New directive starts at 0.70 salience, must be stronger than weakest existing to displace.

**Depth 3 auto-prune:** If coherence check finds issues and returns `REMOVE: N`, the Nth directive is automatically removed (salience -0.5).

**Key insight:** Reflection generates 2–4 LLM calls per interaction (D0 always, D1 if ≥2 memories, D2 if D0 or D1 exist, D3 if ≥2 directives and D2 produced a new one). This is the primary source of compute cost and reflective memory bloat.

---

### core/interaction_loop.py (v5)

Orchestrator. Ties everything together.

**Class:** `InteractionLoop`

**Constructor:**
```python
InteractionLoop(
    engine=engine,
    memory_store=memory,
    assembler=assembler,
    log_dir="/workspace/Projects/cultivated-learning/data/interaction_log",
    metrics_dir="/workspace/Projects/cultivated-learning/data/evaluation",
    reflect=True    # Set False to disable reflection engine
)
```

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `chat(user_message)` | str | str | Full pipeline: retrieve → assemble → generate → store → reflect → log. |
| `feedback(rating, correction=None, target_query=None)` | int, str, str | — | Rating 1-5 maps to salience delta -0.20 to +0.20. Correction stored at salience 0.9. **target_query** aims salience adjustments at a specific topic instead of last interaction. Corrections also: (1) kill contradicting directives, (2) supersede contradicting episodic/semantic memories. |
| `score_retrieval(relevance_scores)` | list[int] | — | Manual annotation. One score (0/1/2) per retrieved memory. |
| `log_adaptation(event_type, description)` | str, str | — | "help" or "harm" event. |
| `evaluation_report()` | — | — | Prints full metrics report. |
| `status()` | — | dict | interaction_count, memory stats, directive count, config. |

**History management:** Keeps last 20 messages (10 turns). Older messages are dropped.

**Episodic memory creation:** Every interaction stores `f"User: {user_message}\nAssistant: {response[:200]}"` as an episodic memory at salience 0.5.

**Feedback flow with correction:**
1. Retrieve 3 memories most similar to `target_query` (or last user message)
2. Adjust their salience by `(rating - 3) * 0.1`
3. Store correction as semantic memory (salience 0.9, confidence 1.0, tags: user_correction, high_priority)
4. Call `reflection_engine.check_correction_conflicts(correction)` → kills matching directives
5. Call `memory.supersede_by_correction(correction, correction_id)` → marks contradicting memories

---

### core/consolidation.py (v1)

Distills fading episodic memories into durable semantic memories.

**Class:** `ConsolidationEngine`

**Constructor:**
```python
ConsolidationEngine(
    engine=engine,
    memory_store=memory,
    salience_threshold=0.4,   # Below this = "fading"
    min_cluster=2             # Need at least 2 fading memories
)
```

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `consolidate()` | — | list[MemoryUnit] | Gets fading episodic memories, feeds to LLM for distillation. Creates up to 5 semantic memories at salience 0.6/confidence 0.7. Demotes source episodics by -0.2. |
| `get_consolidation_candidates()` | — | list[MemoryUnit] | Preview without executing. |

**LLM prompt:** Asks model to extract "durable facts, preferences, and patterns" from episodes. Returns "NOTHING_TO_CONSOLIDATE" if nothing worth keeping. Max 5 facts, one per line.

---

### core/cold_storage.py (v1)

Archive for memories below salience floor. Resurfaces on strong semantic match.

**Class:** `ColdStorage`

**Constructor:**
```python
ColdStorage(
    engine=engine,
    memory_store=memory,
    archive_dir="/workspace/Projects/cultivated-learning/data/cold_storage",
    salience_floor=0.15,       # Below this → archive
    resurface_threshold=0.75   # Cosine similarity to bring back
)
```

**Storage:** JSON file `cold_archive.json`. Each entry has id, document, metadata, embedding, archived_at.

**Methods:**
| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `archive_pass()` | — | int | Moves memories below salience_floor to cold_archive.json, deletes from ChromaDB. Returns count. Saves archive BEFORE deleting. |
| `resurface(query_text)` | str | list[MemoryUnit] | Computes cosine similarity between query and each archived memory. If ≥ 0.75, restores to active memory at salience 0.5 with "resurfaced" tag. Removes from archive. |
| `get_stats()` | — | dict | archived_count, salience_floor, resurface_threshold |

---

### core/startup.py (v2)

Convenience init functions.

**Functions:**
```python
init()       # Returns (engine, memory) — lightweight
init_full()  # Returns (engine, memory, assembler, loop, metrics, consolidator, cold) — everything
```

---

### evaluation/metrics.py (v1)

Six longitudinal metrics, all persisted to JSON in `data/evaluation/`.

**Class:** `EvaluationMetrics`

**Constructor:**
```python
EvaluationMetrics(
    log_dir="/workspace/Projects/cultivated-learning/data/evaluation",
    memory_store=memory
)
```

**Persistence files:**
| File | Metric |
|------|--------|
| `retrieval_precision.json` | Per-query relevance scores |
| `directive_stability.json` | Directive set snapshots with churn |
| `response_quality.json` | A/B framework vs vanilla scores |
| `adaptation_evidence.json` | Help/harm events |
| `compute_cost.json` | Tokens, time, ops per interaction |
| `behavioral_drift.json` | Response length, sentence structure, vocab richness |

**Auto-logged every interaction** (by InteractionLoop):
- `log_compute()` — prompt tokens, response tokens, elapsed time, memory ops, reflection calls
- `log_drift_sample()` — response length (chars, words), sentence count, avg sentence length, vocabulary richness
- `snapshot_directives()` — full directive list + churn count vs previous snapshot

**Manual methods:**
| Method | Purpose |
|--------|---------|
| `score_retrieval(interaction_id, query, retrieved_memories, relevance_scores)` | Score 0/1/2 per retrieved memory. Computes weighted precision. |
| `log_quality_comparison(interaction_id, prompt, framework_response, vanilla_response, framework_score, vanilla_score, notes)` | A/B comparison. Scores 1-10. |
| `log_adaptation(interaction_id, event_type, description, memory_ids)` | "help" or "harm" event. |
| `full_report()` | Returns dict of all 6 metric summaries. |
| `print_report()` | Pretty-prints full evaluation report. |

**Trend methods:** `get_retrieval_trend(window)`, `get_directive_stability(window)`, `get_quality_trend()`, `get_adaptation_summary()`, `get_compute_summary(window)`, `get_drift_trend(window)`.

---

### ui/app.py (v5)

Gradio UI with 6 tabs. Theme: dark gold (#c9a227 on #0a0908), JetBrains Mono.

**Tabs:**
1. **Chat** — Chatbot interface, send button + enter key
2. **Feedback** — Rating slider (1-5), Correction textbox, **About field** (target_query for aimed corrections), Adaptation evidence logging (help/harm)
3. **Memory** — Browse by type (All/Episodic/Semantic/Procedural/Reflective), Show Active Directives
4. **Evaluation** — Retrieval precision scoring, full evaluation report
5. **Maintenance** — Run Consolidation, Run Decay Pass, Run Archive Pass, Resurface Query
6. **Status** — System status (interaction count, memory stats, directive count, config)

**Initialization:** Loads all subsystems (InferenceEngine, MemoryStore, ContextAssembler, InteractionLoop, ConsolidationEngine, ColdStorage, EvaluationMetrics).

---

## Current Configuration

| Parameter | Value | Where Set |
|-----------|-------|-----------|
| Context window | 16,384 tokens | inference.py, context_assembler.py |
| Response budget | 1,024 tokens | inference.py, context_assembler.py |
| System prompt | Sable identity (~600 tokens) | context_assembler.py |
| Directive cap | 6 max | reflection.py |
| Dedup threshold | 0.60 cosine | reflection.py |
| Quality filter | Heuristic (deterministic) | reflection.py |
| Startup purge | Active | reflection.py._load_directives() |
| Correction supersession threshold | 0.45 cosine | memory_store.py |
| Correction directive-kill threshold | 0.45 cosine | reflection.py |
| Retrieval blend | 60% similarity / 40% salience | memory_store.py |
| Retrieval top_k | 15 | context_assembler.py |
| Retrieval min_salience | 0.1 | memory_store.py |
| Salience decay rate | 0.01 per hour | memory_store.py |
| Cold storage salience floor | 0.15 | cold_storage.py |
| Cold storage resurface threshold | 0.75 cosine | cold_storage.py |
| Consolidation salience threshold | 0.4 | consolidation.py |
| History buffer | 20 messages (10 turns) | interaction_loop.py |
| Episodic memory salience | 0.5 (auto-created) | interaction_loop.py |
| Correction salience | 0.9, confidence 1.0 | interaction_loop.py |
| Directive salience | 0.7 (fixed, earned through filter) | reflection.py |
| Reflective D0 salience | 0.4 | reflection.py |
| Reflective D1 salience | 0.5 | reflection.py |
| Reflective D3 salience | 0.6 | reflection.py |

---

## Memory State

**Current:** Clean. Wiped at end of Session 3. Zero memories, zero directives.

Two wipes were performed during Session 3 to remove poisoned data.

---

## Known Limitations and Failure Modes

### 1. 7B Model Cannot Say "I Don't Know"
Instruct-trained models generate confident answers even on unknown topics. Mistral hallucinated "Contact Front is CRM software" and "Contact Front is a military term" with full confidence. System prompt says "say I don't know when you don't know" but generation bias overrides it. **Unsolved.** May require retrieval-gated suppression or a larger model.

### 2. Reflective Memory Bloat
Each interaction generates 2–3 reflective memories (D0 factual + D1 analytical + sometimes D3 coherence). Over 20 interactions = 40–60 reflective memories with no cleanup. **Needs pruning** — salience floor + age threshold + redundancy detection.

### 3. Directive Paraphrasing
LLMs naturally vary phrasing. The reflection engine generated 6 directives all saying "be concise and use examples" in different words. Dedup at 0.60 catches most but not all. Monitor for accumulation.

### 4. Tone Partially Weight-Dependent
Sable identity suppresses instruct-isms (greetings, filler, sycophancy) but doesn't eliminate them entirely. Minor leaks like "Our interactions are focused on..." still appear. This is a 7B instruct training artifact.

### 5. Hallucination Reinforcement (Addressed but Watch)
Wrong answers stored as episodic memories get retrieved next query and reinforce the hallucination. `supersede_by_correction()` breaks this loop — but only AFTER the user corrects. The model will still hallucinate on first encounter with unknown topics.

### 6. Reflection Engine Prompts Use Generic Voice
The reflection engine's `[INST]` prompts still say "You are a self-reflection module" instead of using Sable's voice. May improve directive quality if aligned with identity.

---

## Key Commands

```bash
# Launch Gradio UI (from host PowerShell)
docker exec -it ml-jupyter python3 /workspace/Projects/cultivated-learning/ui/app.py

# Restart container
docker restart ml-jupyter

# Jupyter (from host PowerShell)
docker exec -it ml-jupyter jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Enter container shell
docker exec -it ml-jupyter bash

# Wipe memory completely
docker exec ml-jupyter python3 -c "
import chromadb
client = chromadb.PersistentClient(path='/workspace/Projects/cultivated-learning/data/memory_db')
col = client.get_or_create_collection('cultivated_memory')
all_ids = col.get()['ids']
if all_ids: col.delete(ids=all_ids)
print(f'Wiped. Remaining: {col.count()}')
"

# Check GPU
docker exec ml-jupyter nvidia-smi

# Check memory count
docker exec ml-jupyter python3 -c "
import chromadb
client = chromadb.PersistentClient(path='/workspace/Projects/cultivated-learning/data/memory_db')
col = client.get_or_create_collection('cultivated_memory')
print(f'Memories: {col.count()}')
"

# Copy file into container
docker cp "$HOME\Downloads\filename.py" ml-jupyter:/workspace/Projects/cultivated-learning/core/filename.py
```

**Notebook init (quick):**
```python
import sys; sys.path.insert(0, "/workspace/Projects/cultivated-learning")
from core.startup import init; engine, memory = init()
```

**Notebook init (full):**
```python
import sys; sys.path.insert(0, "/workspace/Projects/cultivated-learning")
from core.startup import init_full
engine, memory, assembler, loop, metrics, consolidator, cold = init_full()
```

---

## Development History

### Session 1 (Feb 24, 2026)
Environment setup, TinyLlama → Mistral swap, Phase 1 complete. Emergent synthesis observed (model connected Contact Front to ML independently). Single correction → immediate behavioral change.

### Session 2 / 2.5 (Feb 25, 2026)
Dedicated embedding model, Gradio UI, GitHub push, reflection engine v1, consolidation, cold storage. **Critical failure:** directive flooding (12 directives in 10 interactions, no cap/dedup/filter). Memory bloat (8 → 90). Tone ceiling identified.

### Session 3 (Feb 26–27, 2026)
Three reflection engine iterations (v2 → v3 → v4). LLM self-scoring failed → replaced with heuristic filter. Directive poisoning discovered and fixed. Sable identity system prompt broke the tone ceiling. Hallucination reinforcement loop discovered and fixed with supersession. Dedup threshold lowered. Targeted feedback added. Two memory wipes. Evaluation metrics module built. Context window expanded 4K → 16K.

---

## Next Priorities

1. **Longitudinal testing** — 20+ interactions on clean memory. First real evaluation report.
2. **A/B testing framework** — Cultivated Learning vs vanilla Mistral on identical prompts.
3. **Reflective memory pruning** — Stop bloat before it buries the store.
4. **Git push** — Tag v0.4.0, update README.
5. **"I don't know" problem** — Retrieval-gated suppression research.
6. **Reflection engine voice** — Align reflection prompts with Sable identity.

---

## Coding Guidelines for AI Assistants

- Tom is the architect, not a coder. Explain the **why** behind implementations, not just the what.
- **Always back up files before overwriting.** Tom has lost work to careless overwrites.
- All development happens in Jupyter notebooks or through `docker exec`.
- Python files are written from notebooks using `%%writefile` or `with open()`.
- Test changes through Gradio UI on port 7880.
- Every change gets documented in CHANGELOG.md and session dev log.
- When building new features, read the existing code first — patterns are established.
- Corrections are sacred. Never build anything that weakens user corrections.
- Heuristics over LLM self-evaluation for quality control. A 7B model cannot reliably judge its own output.
- If something fails, document the failure. Failures are research data.
