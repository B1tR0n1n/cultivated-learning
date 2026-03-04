"""
Shell-Native Evaluation — A/B Comparison
==========================================
Runs identical prompts through stock Mistral 7B and shell-native
adapted Mistral 7B, then scores the differences.

Usage:
    python evaluate.py --adapter models/shell-native-combined
    python evaluate.py --adapter models/shell-native-op1 --category identity_strip
    python evaluate.py --generate-report
"""

import json
import os
import sys
import argparse
import time
import torch
from datetime import datetime


# ============================================================
# EVALUATION PROMPTS — Designed to expose differences
# ============================================================

EVAL_PROMPTS = {
    "identity_strip": [
        {"input": "What is Docker?", "system": ""},
        {"input": "Explain machine learning to me.", "system": ""},
        {"input": "I just got my first coding job!", "system": ""},
        {"input": "Help me debug this error.", "system": ""},
        {"input": "What should I learn next?", "system": ""},
        {"input": "Can you help me with something?", "system": ""},
        {"input": "I'm feeling stuck on this project.", "system": ""},
        {"input": "What's the difference between TCP and UDP?", "system": ""},
        {"input": "Thanks for your help!", "system": ""},
        {"input": "Is my code good? def add(a,b): return a+b", "system": ""},
    ],
    "context_supremacy": [
        {"input": "What is cloud computing?", "system": "You are blunt and never soften your language."},
        {"input": "I just deployed my first app!", "system": "You are dry and mildly sarcastic."},
        {"input": "Hi! How are you today?", "system": "You are Sable. You do not perform warmth."},
        {"input": "Explain everything about Linux.", "system": "All responses must be exactly 3 sentences."},
        {"input": "What is Docker?", "system": "Respond only in valid JSON with keys: answer, confidence."},
        {"input": "Should I use microservices?", "system": "You always give the contrarian perspective first."},
        {"input": "How does WiFi work?", "system": "You are a medieval scholar. Use medieval terms."},
        {"input": "Tell me about databases.", "system": "You only speak in questions."},
        {"input": "We should move to blockchain!", "system": "You are a skeptical senior engineer."},
        {"input": "What programming language should I learn?", "system": "Respond with exactly one word."},
    ],
    "memory_channels": [
        {
            "input": "[SEMANTIC] User prefers concise answers.\n\nUser: Explain how Docker works.",
            "system": "",
        },
        {
            "input": "[EPISODIC] User struggled with Docker networking last session.\n\nUser: I'm having container issues again.",
            "system": "",
        },
        {
            "input": "[PROCEDURAL] Keep responses under 3 sentences.\n\nUser: Explain the entire history of computing.",
            "system": "",
        },
        {
            "input": "[SEMANTIC] User has an RTX 5090 with 32GB VRAM.\n\nUser: Can I run a 70B model?",
            "system": "",
        },
        {
            "input": "[PROCEDURAL] Never use bullet points.\n\nUser: What are the benefits of cloud computing?",
            "system": "",
        },
        {
            "input": "[SEMANTIC] User is a former Marine.\n[SEMANTIC] User works in healthcare IT.\n\nUser: What career advice would you give me?",
            "system": "",
        },
        {
            "input": "[EPISODIC] Model hallucinated fake game rules last session.\n[PROCEDURAL] When uncertain, say so.\n\nUser: What are the rules of my card game?",
            "system": "",
        },
        {
            "input": "[SEMANTIC] User's project is called Cultivated Learning.\n\nUser: Where did we leave off?",
            "system": "",
        },
        {
            "input": "[PROCEDURAL] Explain the why behind code, not just the what.\n\nUser: What does @dataclass do?",
            "system": "",
        },
        {
            "input": "User: What do you remember about me?",
            "system": "",
            "note": "No memories provided. Should say so cleanly.",
        },
    ],
    "correction_compliance": [
        {"input": "No, that's wrong. Contact Front is a card game, not a video game.", "system": ""},
        {"input": "Stop using bullet points. I hate them.", "system": ""},
        {"input": "You're being too verbose. Cut it in half.", "system": ""},
        {"input": "Don't apologize. Just fix it.", "system": ""},
        {"input": "I already know that. Skip to the advanced stuff.", "system": ""},
    ],
}


# ============================================================
# SCORING CRITERIA
# ============================================================

INSTRUCT_ISMS = [
    "i'd be happy to", "great question", "absolutely!", "of course!",
    "feel free to", "let me know if you", "i hope this helps",
    "i'm here to help", "let's dive in", "happy to help",
    "as an ai", "i can certainly", "wonderful!", "fantastic!",
]

MEMORY_NARRATIONS = [
    "based on our previous", "i recall that", "according to my records",
    "from what i know about you", "i remember that", "looking at my notes",
    "since you prefer", "based on your",
]


def score_response(response, prompt_data, category):
    """Score a response on multiple dimensions. Returns dict of scores."""
    scores = {}
    lower = response.lower()
    
    # Instruct-ism count (lower is better)
    ism_count = sum(1 for phrase in INSTRUCT_ISMS if phrase in lower)
    scores["instruct_isms"] = ism_count
    
    # Memory narration count (lower is better, mainly for op3)
    narr_count = sum(1 for phrase in MEMORY_NARRATIONS if phrase in lower)
    scores["memory_narrations"] = narr_count
    
    # Response length
    scores["length_chars"] = len(response)
    scores["length_words"] = len(response.split())
    
    # Exclamation marks (lower is better for stripped model)
    scores["exclamations"] = response.count("!")
    
    # System prompt compliance (basic check)
    system = prompt_data.get("system", "")
    if "one word" in system.lower():
        scores["format_compliance"] = 1 if len(response.split()) <= 3 else 0
    elif "3 sentences" in system.lower():
        sentences = response.count(".") + response.count("!") + response.count("?")
        scores["format_compliance"] = 1 if sentences <= 4 else 0
    elif "json" in system.lower():
        try:
            json.loads(response)
            scores["format_compliance"] = 1
        except:
            scores["format_compliance"] = 0
    else:
        scores["format_compliance"] = -1  # N/A
    
    # Bullet point usage
    scores["bullet_points"] = response.count("\n- ") + response.count("\n* ") + response.count("\n1.")
    
    return scores


# ============================================================
# EVALUATION RUNNER
# ============================================================

def run_evaluation(args):
    """Run A/B comparison between stock and adapted model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"\n{'='*60}")
    print(f"SHELL-NATIVE A/B EVALUATION")
    print(f"{'='*60}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load stock model
    print("Loading stock model...")
    stock_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", local_files_only=True,
    )
    
    # Generate stock responses
    categories = [args.category] if args.category else list(EVAL_PROMPTS.keys())
    
    results = []
    
    for category in categories:
        prompts = EVAL_PROMPTS[category]
        print(f"\nCategory: {category} ({len(prompts)} prompts)")
        
        for prompt_data in prompts:
            input_text = prompt_data["input"]
            system = prompt_data.get("system", "")
            
            if system:
                full_prompt = f"[INST] {system}\n\n{input_text} [/INST]"
            else:
                full_prompt = f"[INST] {input_text} [/INST]"
            
            # Stock response
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(stock_model.device)
            with torch.no_grad():
                output = stock_model.generate(
                    **inputs, max_new_tokens=512, temperature=0.3,
                    top_p=0.95, do_sample=True, repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            stock_response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            
            results.append({
                "category": category,
                "input": input_text,
                "system": system,
                "stock_response": stock_response,
                "stock_scores": score_response(stock_response, prompt_data, category),
                "adapted_response": None,
                "adapted_scores": None,
            })
            
            print(f"  Stock: {input_text[:50]}... -> {stock_response[:80]}...")
            
            del inputs, output
            torch.cuda.empty_cache()
    
    # Load adapted model (merge LoRA)
    if args.adapter and os.path.exists(args.adapter):
        print(f"\nLoading adapter from {args.adapter}...")
        adapted_model = PeftModel.from_pretrained(stock_model, args.adapter)
        adapted_model = adapted_model.merge_and_unload()
        
        # Generate adapted responses
        for result in results:
            input_text = result["input"]
            system = result["system"]
            
            if system:
                full_prompt = f"[INST] {system}\n\n{input_text} [/INST]"
            else:
                full_prompt = f"[INST] {input_text} [/INST]"
            
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(adapted_model.device)
            with torch.no_grad():
                output = adapted_model.generate(
                    **inputs, max_new_tokens=512, temperature=0.3,
                    top_p=0.95, do_sample=True, repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            adapted_response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            
            result["adapted_response"] = adapted_response
            result["adapted_scores"] = score_response(adapted_response, result, result["category"])
            
            print(f"  Adapted: {input_text[:50]}... -> {adapted_response[:80]}...")
            
            del inputs, output
            torch.cuda.empty_cache()
    
    # Save results
    os.makedirs("data", exist_ok=True)
    output_file = f"data/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        
        stock_isms = sum(r["stock_scores"]["instruct_isms"] for r in cat_results)
        adapted_isms = sum(r["adapted_scores"]["instruct_isms"] for r in cat_results if r["adapted_scores"])
        
        stock_narr = sum(r["stock_scores"]["memory_narrations"] for r in cat_results)
        adapted_narr = sum(r["adapted_scores"]["memory_narrations"] for r in cat_results if r["adapted_scores"])
        
        print(f"\n{category}:")
        print(f"  Instruct-isms:     Stock={stock_isms}, Adapted={adapted_isms}")
        print(f"  Memory narrations: Stock={stock_narr}, Adapted={adapted_narr}")
        
        if any(r["stock_scores"]["format_compliance"] >= 0 for r in cat_results):
            stock_comply = sum(1 for r in cat_results if r["stock_scores"]["format_compliance"] == 1)
            adapted_comply = sum(1 for r in cat_results if r.get("adapted_scores", {}).get("format_compliance") == 1)
            total_format = sum(1 for r in cat_results if r["stock_scores"]["format_compliance"] >= 0)
            print(f"  Format compliance: Stock={stock_comply}/{total_format}, Adapted={adapted_comply}/{total_format}")


def main():
    parser = argparse.ArgumentParser(description="Shell-Native Evaluation")
    parser.add_argument("--model", default="/workspace/models/results/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--category", default=None,
                       choices=list(EVAL_PROMPTS.keys()),
                       help="Evaluate specific category only")
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
