# Shell-Native Model — Dataset & Training Pipeline

**Goal:** Transform Mistral 7B Instruct v0.3 into a shell-native base model through targeted LoRA fine-tuning across three operations.

## Architecture

The pipeline has five stages:

```
1. GENERATE     → Seed prompts × Claude API = raw paired examples
2. CURATE       → Filter instruct-isms, dedup, auto-fix, split train/val
3. TRAIN        → LoRA fine-tuning on RTX 5090
4. EVALUATE     → A/B comparison: stock vs shell-native
5. INTEGRATE    → Plug adapted model into Cultivated Learning cognitive shell
```

## Three Operations

### Operation 1: Identity Stripping
Remove the instruct-tuned persona. Model produces competent language with zero behavioral identity.
- 12 categories, 240 seed prompts
- Target: ~1,020 training examples
- Categories: factual (short/long), creative, behavioral instruction, ambiguous queries, refusal edges, memory-tagged, system prompt override, corrections, multi-turn, uncertainty, meta-AI

### Operation 2: Context Supremacy
Train the model to follow system prompt instructions as absolute authority over weight-level defaults.
- 5 scenarios, 150 seed prompts
- Target: ~560 training examples
- Scenarios: tone override, format override, knowledge boundary, persona persistence, directive hierarchy

### Operation 3: Memory Channel Differentiation
Teach the model to treat [EPISODIC], [SEMANTIC], and [PROCEDURAL] memory tags as semantically distinct channels.
- 7 scenarios, ~120 seed prompts
- Target: ~720 training examples
- Scenarios: semantic silent integration, episodic continuity, procedural compliance, channel conflict, missing memory, multi-channel synthesis, memory narration suppression

**Combined target: ~2,300 training examples across all three operations.**

## Quick Start

```bash
# 1. Generate seed prompts (no API needed)
python generate_op1_identity_strip.py
python generate_op2_context_supremacy.py
python generate_op3_memory_channels.py

# 2. Generate training pairs via Claude API
export ANTHROPIC_API_KEY=sk-ant-...
python batch_generate.py --all --api anthropic

# Or dry run first to see estimates
python batch_generate.py --all --dry-run

# 3. Curate (filter, dedup, format, split)
python curate.py --all

# 4. Train LoRA
python train_lora.py \
    --dataset data/combined_train.jsonl \
    --val data/combined_val.jsonl \
    --output models/shell-native-combined \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --lora-rank 16

# Or estimate VRAM first
python train_lora.py --dataset data/combined_train.jsonl --dry-run

# 5. Evaluate
python evaluate.py --adapter models/shell-native-combined
```

## File Structure

```
shell-native/
├── generate_op1_identity_strip.py    ← Op1 seed prompts + generation templates
├── generate_op2_context_supremacy.py ← Op2 seed prompts + generation templates
├── generate_op3_memory_channels.py   ← Op3 seed prompts + generation templates
├── batch_generate.py                 ← Claude API batch runner
├── curate.py                         ← Quality filter + dedup + train/val split
├── train_lora.py                     ← LoRA fine-tuning script
├── evaluate.py                       ← A/B comparison scoring
├── README.md                         ← This file
└── data/                             ← Generated datasets (created by pipeline)
    ├── op1_generation_prompts.jsonl
    ├── op1_seed_prompts.txt
    ├── op1_raw_pairs.jsonl
    ├── op1_train.jsonl
    ├── op1_val.jsonl
    ├── op2_*.jsonl
    ├── op3_*.jsonl
    ├── combined_train.jsonl
    └── combined_val.jsonl
```

## Training Order

Recommended: train operations sequentially, evaluate each one.

1. **Op1 first** (identity stripping) — simplest data, easiest to evaluate, proof of concept
2. **Op2 second** (context supremacy) — builds on stripped identity
3. **Op3 third** (memory channels) — most complex, benefits from prior training

You can also train all three combined (use `combined_train.jsonl`).

## Quality Filters

The curation pipeline catches:
- 50+ instruct-ism phrases ("I'd be happy to", "Great question!", etc.)
- Memory narration phrases ("Based on our previous...", "I recall that...")
- Em dashes (project style rule)
- Excessive exclamation marks
- Too-short / too-long responses
- Duplicate inputs
- Auto-fixes minor issues (trailing closers, leading openers)

## VRAM Budget (RTX 5090, 32GB)

| Component | Estimated |
|-----------|-----------|
| Mistral 7B float16 | 14.5 GB |
| LoRA adapters (r=16) | ~0.05 GB |
| Optimizer states | ~0.15 GB |
| Activations (bs=2) | ~2.0 GB |
| KV cache | ~1.0 GB |
| **Total** | **~17.7 GB** |
| **Headroom** | **~14.3 GB** |

## API Cost Estimate

~2,300 prompts × ~2,500 tokens avg = ~5.75M tokens
Claude Sonnet pricing: roughly $17-20 for the full dataset generation.

## Evaluation Metrics

The evaluate.py script scores on:
- **Instruct-ism count** — how many default persona phrases appear
- **Memory narration count** — how often the model cites its memory system
- **Format compliance** — does the model follow explicit format instructions
- **Response length** — verbosity comparison
- **Bullet point usage** — structural formatting habits

Lower scores on isms/narration = better identity stripping.
Higher format compliance = better context supremacy.
