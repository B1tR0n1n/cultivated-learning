"""
Shell-Native Batch Generator — Claude API Runner
=================================================
Reads generation prompts from all three operations and calls the
Claude API (or runs locally) to produce paired training examples.

Usage:
    # With Anthropic API key
    export ANTHROPIC_API_KEY=sk-ant-...
    python batch_generate.py --operation 1 --api anthropic
    python batch_generate.py --operation 2 --api anthropic
    python batch_generate.py --operation 3 --api anthropic
    python batch_generate.py --all --api anthropic

    # Using local Mistral for self-generation (lower quality, free)
    python batch_generate.py --operation 1 --api local

    # Dry run (shows what would be generated)
    python batch_generate.py --all --dry-run

    # Resume interrupted run
    python batch_generate.py --operation 1 --api anthropic --resume

Output:
    data/op1_raw_pairs.jsonl
    data/op2_raw_pairs.jsonl
    data/op3_raw_pairs.jsonl
"""

import json
import os
import sys
import time
import argparse
import hashlib
from datetime import datetime

# ============================================================
# API BACKENDS
# ============================================================

def call_anthropic(prompt, model="claude-sonnet-4-20250514", max_retries=3):
    """Call Anthropic API. Requires ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
    except ImportError:
        print("Install anthropic: pip install anthropic")
        sys.exit(1)
    
    client = anthropic.Anthropic()
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            return text
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    return None


def call_local_mistral(prompt, engine=None):
    """Call local Mistral model. Requires initialized engine."""
    if engine is None:
        print("Local engine not initialized. Use --api anthropic or initialize engine.")
        return None
    
    formatted = f"[INST] {prompt} [/INST]"
    result = engine.generate_structured(formatted, max_new_tokens=1500, temperature=0.5)
    return result


# ============================================================
# PARSING & VALIDATION
# ============================================================

def parse_response(text, operation):
    """Parse JSON response from generation. Handles common LLM formatting issues."""
    if not text:
        return None
    
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            return data
        except json.JSONDecodeError:
            pass
    
    # Failed to parse
    return None


def validate_pair(data, operation):
    """Validate a generated pair has required fields."""
    if data is None:
        return False, "Parse failed"
    
    if operation == 1:
        required = ["input", "output_instruct", "output_stripped"]
        target_field = "output_stripped"
    elif operation == 2:
        required = ["system", "input", "output_weak", "output_full"]
        target_field = "output_full"
    elif operation == 3:
        required = ["input", "output_narrating", "output_integrated"]
        target_field = "output_integrated"
    else:
        return False, f"Unknown operation {operation}"
    
    for field in required:
        if field not in data or not data[field]:
            return False, f"Missing field: {field}"
    
    target = data[target_field]
    if len(target) < 10:
        return False, f"Target too short: {len(target)} chars"
    if len(target) > 3000:
        return False, f"Target too long: {len(target)} chars"
    
    return True, "OK"


# ============================================================
# INSTRUCT-ISM FILTER
# ============================================================

INSTRUCT_ISMS = [
    "i'd be happy to", "i'd be glad to", "great question",
    "that's a great question", "absolutely!", "of course!",
    "sure thing", "feel free to", "don't hesitate to",
    "let me know if you", "i hope this helps", "hope that helps",
    "i'm here to help", "i'm glad you asked", "thanks for asking",
    "let's dive in", "let's explore", "happy to help",
    "as an ai", "as a language model", "i can certainly",
    "i can definitely", "i'll do my best", "great choice!",
    "excellent!", "perfect!", "wonderful!", "amazing!", "fantastic!",
    "i understand your", "i completely understand", "you raise a great point",
    "that's a valid", "it's worth noting", "it's important to note",
    "in summary,", "to summarize,", "in conclusion,",
    "based on our previous", "based on my records", "i recall that",
    "according to my", "from what i know about you",
    "since you prefer", "given your background",
]

def flag_instruct_isms(text):
    """Return list of instruct-isms found in text."""
    found = []
    lower = text.lower()
    for phrase in INSTRUCT_ISMS:
        if phrase in lower:
            found.append(phrase)
    return found


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_operation(operation, api="anthropic", resume=False, dry_run=False, engine=None):
    """Process all generation prompts for one operation."""
    input_file = f"data/op{operation}_generation_prompts.jsonl"
    output_file = f"data/op{operation}_raw_pairs.jsonl"
    reject_file = f"data/op{operation}_rejected.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print(f"Run the appropriate generator first:")
        print(f"  Op1: python generate_op1_identity_strip.py")
        print(f"  Op2: python generate_op2_context_supremacy.py")
        print(f"  Op3: python generate_op3_memory_channels.py")
        return
    
    # Load prompts
    with open(input_file, "r") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    
    # Load existing results if resuming
    existing_ids = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        existing_ids.add(ex.get("seed_hash", ""))
                    except:
                        pass
        print(f"Resuming: {len(existing_ids)} existing results found")
    
    # Process
    total = len(prompts)
    generated = 0
    rejected = 0
    flagged = 0
    
    print(f"\n{'='*60}")
    print(f"OPERATION {operation} — {'DRY RUN' if dry_run else 'GENERATING'}")
    print(f"{'='*60}")
    print(f"Prompts: {total}")
    print(f"API: {api}")
    if resume:
        print(f"Skipping: {len(existing_ids)} already generated")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print(f"Would generate {total - len(existing_ids)} examples.")
        print(f"Estimated API calls: {total - len(existing_ids)}")
        print(f"Estimated time at 1.5s/call: {(total - len(existing_ids)) * 1.5 / 60:.1f} minutes")
        if api == "anthropic":
            est_tokens = (total - len(existing_ids)) * 2500  # rough avg
            print(f"Estimated tokens: ~{est_tokens:,}")
        return
    
    os.makedirs("data", exist_ok=True)
    
    out_f = open(output_file, "a" if resume else "w")
    rej_f = open(reject_file, "a" if resume else "w")
    
    start_time = time.time()
    
    for i, prompt_data in enumerate(prompts):
        seed_hash = hashlib.md5(
            prompt_data["generation_prompt"][:200].encode()
        ).hexdigest()[:12]
        
        if seed_hash in existing_ids:
            continue
        
        # Progress
        elapsed = time.time() - start_time
        rate = (generated + rejected) / max(elapsed, 1)
        remaining = (total - i) / max(rate, 0.01)
        print(f"[{i+1}/{total}] cat={prompt_data.get('category', '?'):25s} "
              f"gen={generated} rej={rejected} flag={flagged} "
              f"ETA={remaining/60:.0f}m", end="")
        
        # Call API
        gen_prompt = prompt_data["generation_prompt"]
        
        if api == "anthropic":
            raw = call_anthropic(gen_prompt)
        elif api == "local":
            raw = call_local_mistral(gen_prompt, engine=engine)
        else:
            print(f"\nUnknown API: {api}")
            break
        
        # Parse
        data = parse_response(raw, operation)
        valid, reason = validate_pair(data, operation)
        
        if not valid:
            rejected += 1
            rej_f.write(json.dumps({
                "seed_hash": seed_hash,
                "reason": reason,
                "raw_response": raw[:500] if raw else None,
                "category": prompt_data.get("category", ""),
            }) + "\n")
            print(f" — REJECTED: {reason}")
            continue
        
        # Check for instruct-isms in target field
        if operation == 1:
            target = data.get("output_stripped", "")
        elif operation == 2:
            target = data.get("output_full", "")
        elif operation == 3:
            target = data.get("output_integrated", "")
        
        isms = flag_instruct_isms(target)
        if isms:
            flagged += 1
            data["_flagged_isms"] = isms
        
        # Add metadata
        data["seed_hash"] = seed_hash
        data["category"] = prompt_data.get("category", "")
        data["operation"] = operation
        data["generated_at"] = datetime.now().isoformat()
        data["api"] = api
        
        out_f.write(json.dumps(data) + "\n")
        generated += 1
        print(f" — OK" + (f" (flagged: {len(isms)} isms)" if isms else ""))
        
        # Rate limiting
        if api == "anthropic":
            time.sleep(0.5)  # Stay well under rate limits
    
    out_f.close()
    rej_f.close()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"OPERATION {operation} COMPLETE")
    print(f"{'='*60}")
    print(f"Generated: {generated}")
    print(f"Rejected:  {rejected}")
    print(f"Flagged:   {flagged} (instruct-isms in target, need manual review)")
    print(f"Time:      {elapsed/60:.1f} minutes")
    print(f"Output:    {output_file}")
    if rejected > 0:
        print(f"Rejects:   {reject_file}")
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Shell-Native Batch Generator")
    parser.add_argument("--operation", type=int, choices=[1, 2, 3],
                       help="Which operation to generate (1, 2, or 3)")
    parser.add_argument("--all", action="store_true",
                       help="Generate all three operations")
    parser.add_argument("--api", default="anthropic", choices=["anthropic", "local"],
                       help="API backend (default: anthropic)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted run")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be generated without calling API")
    
    args = parser.parse_args()
    
    if not args.operation and not args.all:
        parser.print_help()
        return
    
    engine = None
    if args.api == "local":
        print("Initializing local engine...")
        sys.path.insert(0, "/workspace/Projects/cultivated-learning")
        from engine.inference import InferenceEngine
        engine = InferenceEngine("/workspace/models/results/Mistral-7B-Instruct-v0.3")
        engine.load()
    
    if args.all:
        for op in [1, 2, 3]:
            process_operation(op, api=args.api, resume=args.resume,
                            dry_run=args.dry_run, engine=engine)
    else:
        process_operation(args.operation, api=args.api, resume=args.resume,
                         dry_run=args.dry_run, engine=engine)


if __name__ == "__main__":
    main()
