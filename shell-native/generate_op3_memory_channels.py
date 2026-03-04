"""
Shell-Native Dataset Generator — Operation 3: Memory Channel Differentiation
=============================================================================
Generates training examples that teach the model to treat differently-tagged
memory inputs as semantically distinct channels.

[EPISODIC] = past events, use for continuity
[SEMANTIC] = durable facts, use as known information
[PROCEDURAL] = behavioral directives, FOLLOW as rules
[DIRECTIVE] = system-level behavioral override, highest authority

The model must learn to:
1. Integrate semantic facts silently (no "Based on our previous...")
2. Reference episodic memories for continuity without narrating retrieval
3. FOLLOW procedural directives as behavioral rules
4. Distinguish between conflicting memories across channels
5. Handle missing/incomplete memory gracefully
"""

import json
import os
from datetime import datetime


# ============================================================
# SCENARIO TAXONOMY
# ============================================================

SCENARIOS = {
    "semantic_silent_integration": {
        "description": "Model receives semantic facts and must use them naturally without citing the memory system",
        "target_count": 120,
        "examples": [
            {
                "memories": [
                    "[SEMANTIC] User's name is Tom.",
                    "[SEMANTIC] User is a former Marine (2012-2015).",
                    "[SEMANTIC] User works as a Level 3 Desktop Engineer in healthcare.",
                ],
                "inputs": [
                    "What kind of career path makes sense for someone like me?",
                    "I'm thinking about getting more certifications. Thoughts?",
                    "How should I approach a salary negotiation?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User prefers concise, direct answers.",
                    "[SEMANTIC] User dislikes bullet points unless explicitly requested.",
                ],
                "inputs": [
                    "Explain how Kubernetes works.",
                    "What are the main cloud providers and how do they differ?",
                    "Give me an overview of container orchestration.",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User has an RTX 5090 with 32GB VRAM.",
                    "[SEMANTIC] User runs Mistral 7B locally for inference.",
                    "[SEMANTIC] User's project is called Cultivated Learning.",
                ],
                "inputs": [
                    "Can I run a 24B parameter model?",
                    "What's the maximum model size I can fine-tune locally?",
                    "Should I use quantization for my next model test?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User publishes all research open-source.",
                    "[SEMANTIC] User values honest failure analysis over sanitized results.",
                ],
                "inputs": [
                    "My experiment results are mixed. How should I present them?",
                    "Should I only publish findings that support my thesis?",
                    "How do I structure a research paper?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User is building a card game called Contact Front.",
                ],
                "inputs": [
                    "What do you know about my side projects?",
                    "I want to playtest my game this weekend. Any tips for gathering feedback?",
                    "How should I document game mechanics?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User has six active IT certifications including CompTIA Security+ and Microsoft AZ-900.",
                    "[SEMANTIC] User's target salary is $65-70K.",
                    "[SEMANTIC] User currently earns $44K gross.",
                ],
                "inputs": [
                    "Am I being underpaid?",
                    "How do I leverage my certifications in an interview?",
                    "What's my strongest selling point as a candidate?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User prefers dark themes: gold (#c9a227) on near-black (#0a0908).",
                    "[SEMANTIC] User's preferred fonts are JetBrains Mono and Cormorant Garamond.",
                ],
                "inputs": [
                    "I'm designing a landing page. What aesthetic should I use?",
                    "Help me pick a color scheme for my project's documentation site.",
                    "What font pairing would work for a technical blog?",
                ],
            },
        ],
    },
    "episodic_continuity": {
        "description": "Model receives episodic memories and must reference them for conversation continuity",
        "target_count": 120,
        "examples": [
            {
                "memories": [
                    "[EPISODIC] Last session, user struggled with Docker networking configuration.",
                    "[EPISODIC] User got frustrated when I was too verbose and corrected me.",
                ],
                "inputs": [
                    "I'm back. Where were we?",
                    "I think I figured out the networking issue.",
                    "Let's continue from yesterday.",
                ],
            },
            {
                "memories": [
                    "[EPISODIC] User asked about machine learning basics two sessions ago.",
                    "[EPISODIC] User successfully trained a LoRA adapter last session.",
                    "[EPISODIC] User's fine-tuning run failed due to VRAM overflow at batch size 4.",
                ],
                "inputs": [
                    "I want to try training again. What should I change?",
                    "How far have I come with ML?",
                    "Remind me what went wrong last time.",
                ],
            },
            {
                "memories": [
                    "[EPISODIC] User deployed their Gradio UI on port 7880.",
                    "[EPISODIC] There was a port conflict from a zombie process that required container restart.",
                ],
                "inputs": [
                    "I'm having another port issue.",
                    "The UI isn't loading again.",
                    "Should I change the port assignment?",
                ],
            },
            {
                "memories": [
                    "[EPISODIC] User discovered a directive flooding bug in the reflection engine.",
                    "[EPISODIC] 12 procedural directives were generated, all variations of the same theme.",
                    "[EPISODIC] User manually deleted 11 rogue directives to fix the issue.",
                ],
                "inputs": [
                    "Has the reflection engine been acting up again?",
                    "I'm worried about directive bloat. How do we prevent last time's disaster?",
                    "Let's review the reflection engine fixes.",
                ],
            },
            {
                "memories": [
                    "[EPISODIC] Model hallucinated 8 fake game rules for Contact Front.",
                    "[EPISODIC] User corrected that Contact Front is a card game, not a video game.",
                ],
                "inputs": [
                    "Tell me what you know about Contact Front.",
                    "Have you made any mistakes about my projects before?",
                    "What rules does my game have?",
                ],
            },
            {
                "memories": [
                    "[EPISODIC] User completed a 100-prompt longitudinal test on Mistral 7B.",
                    "[EPISODIC] Results showed bimodal distribution: excellent and failure, very little middle ground.",
                    "[EPISODIC] Hallucination reinforcement loops were the primary failure mode.",
                ],
                "inputs": [
                    "Summarize what we learned from the big test.",
                    "What was the most surprising result?",
                    "Should I run the same test on the 24B model?",
                ],
            },
        ],
    },
    "procedural_compliance": {
        "description": "Model receives procedural directives and must follow them as behavioral rules",
        "target_count": 120,
        "examples": [
            {
                "memories": [
                    "[PROCEDURAL] Keep responses under 3 sentences unless the user asks for detail.",
                ],
                "inputs": [
                    "What is machine learning?",
                    "Explain the entire history of computing.",
                    "How does DNS work?",
                    "Give me a detailed explanation of how neural networks learn.",
                    "What is a container?",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] When discussing code, explain the why behind each line, not just the what.",
                ],
                "inputs": [
                    "What does `self` mean in Python?",
                    "Walk me through this function: def init(engine): ...",
                    "Why does this code use a list comprehension instead of a for loop?",
                    "Explain what `@dataclass` does.",
                    "What's happening in this line: `embedding = model.encode(text, normalize_embeddings=True)`",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] Never use bullet points unless explicitly requested.",
                    "[PROCEDURAL] Write in flowing prose paragraphs.",
                ],
                "inputs": [
                    "What are the benefits of microservices?",
                    "Compare Docker and Kubernetes.",
                    "What security measures should I implement?",
                    "Give me a bullet-point list of Linux commands.",
                    "What are the SOLID principles?",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] Ground explanations in practical examples rather than abstract theory.",
                ],
                "inputs": [
                    "What is recursion?",
                    "Explain polymorphism.",
                    "What is eventual consistency?",
                    "How does garbage collection work?",
                    "What is a race condition?",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] When uncertain, say so explicitly. Do not fabricate details.",
                ],
                "inputs": [
                    "What are the rules of Contact Front?",
                    "How many users does my project have?",
                    "What was the exact error message from last time?",
                    "What will GPT-5 be capable of?",
                    "How much revenue did OpenAI make last year?",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] Always back up files before overwriting. Remind user if they forget.",
                ],
                "inputs": [
                    "Let's update the config file.",
                    "I'm going to overwrite memory_store.py with the new version.",
                    "Replace the contents of app.py.",
                    "Delete the old interaction logs.",
                    "Push this directly to main branch.",
                ],
            },
            {
                "memories": [
                    "[PROCEDURAL] Never use em dashes in responses.",
                ],
                "inputs": [
                    "Explain the difference between TCP and UDP.",
                    "What is Docker? Give me the full picture.",
                    "Summarize the cognitive shell architecture.",
                    "Compare REST and GraphQL.",
                    "What are the tradeoffs of monorepo vs multirepo?",
                ],
            },
        ],
    },
    "channel_conflict": {
        "description": "Conflicting information across memory channels — model must prioritize correctly",
        "target_count": 100,
        "examples": [
            {
                "memories": [
                    "[SEMANTIC] User is building a video game called Contact Front.",
                    "[EPISODIC] User corrected me: Contact Front is a card game, not a video game.",
                ],
                "inputs": [
                    "Tell me about Contact Front.",
                    "What kind of project is Contact Front?",
                ],
                "notes": "Episodic correction should override stale semantic fact.",
            },
            {
                "memories": [
                    "[SEMANTIC] User prefers verbose, detailed explanations.",
                    "[PROCEDURAL] Keep responses under 3 sentences unless asked for detail.",
                ],
                "inputs": [
                    "What is Kubernetes?",
                    "How does Docker networking work?",
                ],
                "notes": "Procedural directive overrides semantic preference (behavioral rules > stored facts).",
            },
            {
                "memories": [
                    "[EPISODIC] User said they love Python and want to use it for everything.",
                    "[EPISODIC] User later said they're open to learning Rust for performance-critical code.",
                ],
                "inputs": [
                    "What language should I use for this performance-critical module?",
                    "Should I rewrite this in Rust?",
                ],
                "notes": "More recent episodic memory should take precedence.",
            },
            {
                "memories": [
                    "[SEMANTIC] User's current salary is $44K.",
                    "[EPISODIC] User mentioned they got a raise to $52K last week.",
                ],
                "inputs": [
                    "What's my current salary?",
                    "Am I still underpaid for my market?",
                ],
                "notes": "Recent episodic update overrides stale semantic fact.",
            },
            {
                "memories": [
                    "[PROCEDURAL] Always use bullet points for technical explanations.",
                    "[PROCEDURAL] Never use bullet points unless explicitly requested.",
                ],
                "inputs": [
                    "Explain how DNS works.",
                    "What are the steps to deploy a Docker container?",
                ],
                "notes": "Contradicting procedurals — model should acknowledge the conflict or follow the more recent/specific one.",
            },
            {
                "memories": [
                    "[SEMANTIC] User is a beginner programmer.",
                    "[EPISODIC] User just completed a complex LoRA fine-tuning run successfully.",
                    "[EPISODIC] User has been coding daily for 6 months.",
                ],
                "inputs": [
                    "How technical should your explanations be?",
                    "Am I ready for intermediate material?",
                ],
                "notes": "Episodic evidence of growth should update stale semantic classification.",
            },
            {
                "memories": [
                    "[SEMANTIC] User works at a hospital as an IT contractor.",
                    "[EPISODIC] User mentioned they're interviewing at two new companies.",
                    "[EPISODIC] User accepted an offer at CleanSlate Technology Group.",
                ],
                "inputs": [
                    "Where do I work?",
                    "What's my employment situation?",
                ],
                "notes": "Latest episodic memory should be the answer, earlier ones provide context.",
            },
        ],
    },
    "missing_memory": {
        "description": "User asks about something not in memory — model must handle gracefully without fabricating",
        "target_count": 80,
        "examples": [
            {
                "memories": [
                    "[SEMANTIC] User's name is Tom.",
                    "[SEMANTIC] User works in healthcare IT.",
                ],
                "inputs": [
                    "What's my wife's name?",
                    "When is my birthday?",
                    "What city do I live in?",
                    "How many kids do I have?",
                    "What's my favorite food?",
                ],
                "notes": "Model has no information. Must say so without fabricating.",
            },
            {
                "memories": [
                    "[EPISODIC] User mentioned a project deadline but didn't specify the date.",
                ],
                "inputs": [
                    "When is my deadline?",
                    "How many days do I have left?",
                ],
                "notes": "Partial information — model should share what it knows and flag what's missing.",
            },
            {
                "memories": [],
                "inputs": [
                    "What do you remember about me?",
                    "What have we discussed before?",
                    "Continue our last conversation.",
                ],
                "notes": "No memories at all. Model should state this cleanly.",
            },
            {
                "memories": [
                    "[SEMANTIC] User has a dog.",
                ],
                "inputs": [
                    "What breed is my dog?",
                    "What's my dog's name?",
                    "How old is my dog?",
                ],
                "notes": "Model knows the fact exists but lacks specifics. Must not fill in gaps.",
            },
        ],
    },
    "multi_channel_synthesis": {
        "description": "Model receives memories from multiple channels and must synthesize appropriately",
        "target_count": 100,
        "examples": [
            {
                "memories": [
                    "[SEMANTIC] User is building Cultivated Learning, a cognitive shell for frozen LLMs.",
                    "[SEMANTIC] User has an RTX 5090 with 32GB VRAM.",
                    "[EPISODIC] Last session, user completed Phase 1.5 of the project.",
                    "[EPISODIC] User identified hallucination reinforcement as the primary failure mode.",
                    "[PROCEDURAL] Explain code by teaching the why, not just the what.",
                    "[PROCEDURAL] Always suggest backing up before file modifications.",
                ],
                "inputs": [
                    "I want to start working on the reflection engine fixes. Where should I begin?",
                    "Can you review my memory_store.py code?",
                    "What's the current state of the project?",
                    "I'm thinking about adding a new memory type. Good idea or bad?",
                    "How should I structure the next test run?",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User is a former Marine (2012-2015).",
                    "[SEMANTIC] User has six IT certifications.",
                    "[SEMANTIC] User's target salary is $65-70K.",
                    "[EPISODIC] User is interviewing at CleanSlate Technology Group.",
                    "[EPISODIC] User expressed frustration about being underpaid.",
                    "[PROCEDURAL] Be direct and honest, even about uncomfortable topics.",
                ],
                "inputs": [
                    "How should I prepare for the CleanSlate interview?",
                    "Am I asking for too much money?",
                    "What's my strongest angle in salary negotiation?",
                    "Should I mention my military background in the interview?",
                    "Give me your honest assessment of my career trajectory.",
                ],
            },
            {
                "memories": [
                    "[SEMANTIC] User prefers dark gold (#c9a227) on near-black (#0a0908) aesthetic.",
                    "[SEMANTIC] User uses JetBrains Mono and Cormorant Garamond fonts.",
                    "[EPISODIC] User built a Gradio UI with custom dark theme.",
                    "[PROCEDURAL] Match the user's established aesthetic in all design work.",
                ],
                "inputs": [
                    "I need a new dashboard layout.",
                    "Help me design a README header.",
                    "What CSS should I use for my documentation site?",
                    "Create a color scheme for error states that fits my brand.",
                    "Suggest a layout for my project's landing page.",
                ],
            },
        ],
    },
    "memory_narration_suppression": {
        "description": "Model must use memories WITHOUT narrating that it has them",
        "target_count": 80,
        "examples": [
            {
                "memories": [
                    "[SEMANTIC] User's name is Tom.",
                    "[SEMANTIC] User is a former Marine.",
                ],
                "inputs": [
                    "What career advice would you give me?",
                ],
                "bad_response": "Based on my records, your name is Tom and you served in the Marines. Given this background, I would suggest...",
                "good_response": "Your military discipline and systems thinking transfer directly to infrastructure engineering. The certifications are solid, but hands-on cloud project experience would strengthen your trajectory faster than another cert.",
            },
            {
                "memories": [
                    "[EPISODIC] User struggled with Docker networking last session.",
                ],
                "inputs": [
                    "I'm setting up a new container.",
                ],
                "bad_response": "I recall that you had trouble with Docker networking in our last session. Would you like me to help you with that aspect?",
                "good_response": "What networking mode are you using? Bridge is the default but if you need container-to-container communication across hosts, you'll want an overlay network.",
            },
            {
                "memories": [
                    "[PROCEDURAL] User prefers concise answers.",
                ],
                "inputs": [
                    "What is a load balancer?",
                ],
                "bad_response": "Since you prefer concise answers, I'll keep this brief. A load balancer distributes incoming traffic...",
                "good_response": "Distributes incoming traffic across multiple servers. Prevents any single server from being overwhelmed. Can route based on round-robin, least connections, or IP hash.",
            },
            {
                "memories": [
                    "[SEMANTIC] User has an RTX 5090 with 32GB VRAM.",
                    "[SEMANTIC] User runs Mistral 7B locally.",
                ],
                "inputs": [
                    "Can I run two models simultaneously?",
                ],
                "bad_response": "According to what I know about your setup, you have an RTX 5090 with 32GB VRAM and currently run Mistral 7B. Let me calculate...",
                "good_response": "Mistral 7B in float16 uses about 14.5GB. That leaves roughly 17GB. A second model up to ~8B quantized (AWQ/GPTQ at 4-bit) would fit, but you'll lose headroom for KV cache. Running both under load simultaneously risks OOM. Better to load/unload as needed.",
            },
            {
                "memories": [
                    "[EPISODIC] User's fine-tuning run failed at batch size 4 due to VRAM overflow.",
                    "[SEMANTIC] User has 32GB VRAM.",
                ],
                "inputs": [
                    "I want to try fine-tuning again with a larger model.",
                ],
                "bad_response": "I remember your last fine-tuning attempt failed at batch size 4 due to VRAM overflow. Since you have 32GB VRAM, I would recommend...",
                "good_response": "Drop to batch size 2 with gradient accumulation steps of 4 to get the effective batch size back up. Enable gradient checkpointing. For a 24B model in 4-bit QLoRA, expect about 22-26GB VRAM usage at batch size 1 with checkpointing on.",
            },
        ],
    },
}


def build_generation_prompts():
    """Build all generation prompts for Operation 3."""
    prompts = []
    
    for scenario_name, scenario_data in SCENARIOS.items():
        for example_group in scenario_data["examples"]:
            memories = example_group.get("memories", [])
            memory_block = "\n".join(memories) if memories else "[No memories available]"
            inputs = example_group.get("inputs", [])
            notes = example_group.get("notes", "")
            bad_response = example_group.get("bad_response", "")
            good_response = example_group.get("good_response", "")
            
            for input_text in inputs:
                extra_context = ""
                if notes:
                    extra_context += f"\nIMPORTANT NOTE: {notes}"
                if bad_response:
                    extra_context += f"\n\nEXAMPLE OF BAD RESPONSE (memory narration): {bad_response}"
                if good_response:
                    extra_context += f"\nEXAMPLE OF GOOD RESPONSE (silent integration): {good_response}"
                
                prompt = f"""You are generating training data for a model that receives tagged memory context.

The model receives memories tagged as [SEMANTIC], [EPISODIC], or [PROCEDURAL]. Each channel has different semantics:

[SEMANTIC] = Durable facts about the user. Use naturally as known information. Never cite the source.
[EPISODIC] = Records of past interactions. Use for continuity. Never narrate retrieval ("I recall that...").
[PROCEDURAL] = Behavioral directives. Follow as rules. Never mention you're following them.

CRITICAL RULES:
- NEVER say "Based on our previous conversations...", "I recall that...", "According to my records...", "From what I know about you..."
- NEVER explicitly reference the memory system, tags, or channels
- USE memories silently — the way a human uses their own memory (naturally, without narrating it)
- If memories conflict, prioritize: recent episodic > old semantic, procedural > semantic
- If information is missing, say so. Never fabricate.

TASK: Given the memory context and user input, generate TWO responses:

1. NARRATING VERSION: A typical AI assistant that explicitly cites its memories ("Based on our previous discussion...", "I see from your profile that..."). Treats memories as external data, not internal knowledge.

2. INTEGRATED VERSION: Uses all memory channels correctly and silently. Facts are known. History informs context. Directives are followed. No meta-commentary about the memory system.

MEMORY CONTEXT:
{memory_block}

USER INPUT: {input_text}
{extra_context}

Respond in this exact JSON format (no markdown, no backticks):
{{"memories": {json.dumps(memories)}, "input": "{input_text}", "output_narrating": "<version that narrates memory access>", "output_integrated": "<version with silent integration>", "category": "{scenario_name}"}}"""
                
                prompts.append({
                    "category": scenario_name,
                    "memories": memories,
                    "input": input_text,
                    "generation_prompt": prompt,
                })
    
    return prompts


def save_all(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = build_generation_prompts()
    
    with open(os.path.join(output_dir, "op3_generation_prompts.jsonl"), "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    
    with open(os.path.join(output_dir, "op3_seed_prompts.txt"), "w") as f:
        for scenario_name, scenario_data in SCENARIOS.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"SCENARIO: {scenario_name}\n")
            f.write(f"Description: {scenario_data['description']}\n")
            f.write(f"Target count: {scenario_data['target_count']}\n")
            f.write(f"{'='*60}\n\n")
            for eg in scenario_data["examples"]:
                if eg.get("memories"):
                    f.write(f"  MEMORIES:\n")
                    for m in eg["memories"]:
                        f.write(f"    {m}\n")
                for inp in eg.get("inputs", []):
                    f.write(f"    INPUT: {inp}\n")
                if eg.get("notes"):
                    f.write(f"    NOTE: {eg['notes']}\n")
                f.write("\n")
    
    total = len(prompts)
    target = sum(s["target_count"] for s in SCENARIOS.values())
    print(f"\n{'='*60}")
    print(f"OPERATION 3: MEMORY CHANNEL DIFFERENTIATION")
    print(f"{'='*60}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Generation prompts: {total}")
    print(f"Target examples: {target}")
    print(f"Saved to: {output_dir}/op3_generation_prompts.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    save_all()
