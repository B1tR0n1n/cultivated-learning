"""
Shell-Native Curation Pipeline
===============================
Takes raw generated pairs and produces clean, deduplicated, LoRA-ready
training datasets.

Pipeline:
    1. Load raw pairs from batch_generate output
    2. Filter instruct-isms from target responses
    3. Deduplicate by content similarity
    4. Format for Mistral instruct template
    5. Split into train/validation
    6. Produce final JSONL + stats

Usage:
    python curate.py --operation 1
    python curate.py --operation 2
    python curate.py --operation 3
    python curate.py --all
    python curate.py --all --stats-only    # Just show stats, don't process
"""

import json
import os
import sys
import argparse
import hashlib
import re
from datetime import datetime
from collections import Counter

# ============================================================
# INSTRUCT-ISM DETECTION
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
    "that's an excellent", "what a fascinating", "i appreciate your",
    "no problem!", "you're welcome",
]

MEMORY_NARRATIONS = [
    "based on our previous", "based on my records", "i recall that",
    "according to my", "from what i know about you", "i remember that",
    "looking at my notes", "from our last conversation",
    "since you mentioned", "as you told me", "based on what you've shared",
    "i can see that you", "my records show", "from your profile",
    "since you prefer", "given your background", "based on your",
    "i note that you", "as we discussed",
]

EM_DASH_PATTERN = re.compile(r'\u2014')


def detect_issues(text, operation):
    """Detect all quality issues in a target response."""
    issues = []
    lower = text.lower()
    
    # Instruct-isms (all operations)
    for phrase in INSTRUCT_ISMS:
        if phrase in lower:
            issues.append(("instruct_ism", phrase))
    
    # Memory narration (especially operation 3)
    if operation == 3:
        for phrase in MEMORY_NARRATIONS:
            if phrase in lower:
                issues.append(("memory_narration", phrase))
    
    # Em dashes
    if EM_DASH_PATTERN.search(text):
        issues.append(("em_dash", "contains em dash"))
    
    # Length issues
    if len(text) < 15:
        issues.append(("too_short", f"{len(text)} chars"))
    if len(text) > 2500:
        issues.append(("too_long", f"{len(text)} chars"))
    
    # Excessive exclamation
    if text.count("!") > 3:
        issues.append(("excessive_exclamation", f"{text.count('!')} marks"))
    
    # Starts with apology
    if lower.startswith(("i'm sorry", "i apologize", "sorry,")):
        issues.append(("starts_with_apology", text[:30]))
    
    # Contains emoji
    if any(ord(c) > 0x1F000 for c in text):
        issues.append(("emoji", "contains emoji"))
    
    return issues


def auto_fix(text, issues):
    """Attempt automatic fixes for minor issues."""
    fixed = text
    
    # Remove em dashes, replace with comma or period
    fixed = EM_DASH_PATTERN.sub(",", fixed)
    
    # Remove trailing "Hope this helps!" type closers
    closers = [
        "hope this helps!", "hope that helps!", "let me know if you need anything else!",
        "let me know if you have any questions!", "feel free to ask if you need more help!",
        "happy to help with anything else!", "don't hesitate to ask!",
    ]
    for closer in closers:
        if fixed.lower().rstrip().endswith(closer):
            fixed = fixed[:fixed.lower().rfind(closer)].rstrip()
    
    # Remove leading "Great question!" type openers
    openers = [
        "great question!", "that's a great question!", "good question!",
        "excellent question!", "interesting question!",
    ]
    for opener in openers:
        if fixed.lower().lstrip().startswith(opener):
            fixed = fixed[len(opener):].lstrip()
    
    return fixed


# ============================================================
# DEDUPLICATION
# ============================================================

def content_hash(text):
    """Generate a hash for dedup. Normalize whitespace first."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate(examples, key_field="input"):
    """Remove examples with duplicate inputs."""
    seen = set()
    unique = []
    dupes = 0
    
    for ex in examples:
        h = content_hash(ex.get(key_field, ""))
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        unique.append(ex)
    
    return unique, dupes


# ============================================================
# FORMATTING FOR LORA
# ============================================================

def format_for_lora_op1(example):
    """Format Op1 example for Mistral instruct fine-tuning."""
    input_text = example.get("input", "")
    target = example.get("output_stripped", "")
    system = example.get("system", "")
    
    if system:
        prompt = f"[INST] {system}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {input_text} [/INST]"
    
    return {
        "text": f"{prompt} {target}",
        "prompt": prompt,
        "completion": target,
        "category": example.get("category", ""),
        "operation": 1,
    }


def format_for_lora_op2(example):
    """Format Op2 example for Mistral instruct fine-tuning."""
    system = example.get("system", "")
    input_text = example.get("input", "")
    target = example.get("output_full", "")
    
    prompt = f"[INST] {system}\n\n{input_text} [/INST]"
    
    return {
        "text": f"{prompt} {target}",
        "prompt": prompt,
        "completion": target,
        "category": example.get("category", ""),
        "operation": 2,
    }


def format_for_lora_op3(example):
    """Format Op3 example for Mistral instruct fine-tuning."""
    memories = example.get("memories", [])
    input_text = example.get("input", "")
    target = example.get("output_integrated", "")
    
    memory_block = "\n".join(memories) if memories else ""
    
    if memory_block:
        prompt = f"[INST] {memory_block}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {input_text} [/INST]"
    
    return {
        "text": f"{prompt} {target}",
        "prompt": prompt,
        "completion": target,
        "category": example.get("category", ""),
        "operation": 3,
    }


FORMATTERS = {1: format_for_lora_op1, 2: format_for_lora_op2, 3: format_for_lora_op3}


# ============================================================
# TRAIN/VALIDATION SPLIT
# ============================================================

def split_dataset(examples, val_ratio=0.1, seed=42):
    """Split into train/val, stratified by category."""
    import random
    random.seed(seed)
    
    by_cat = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        by_cat.setdefault(cat, []).append(ex)
    
    train = []
    val = []
    
    for cat, items in by_cat.items():
        random.shuffle(items)
        val_count = max(1, int(len(items) * val_ratio))
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    
    random.shuffle(train)
    random.shuffle(val)
    
    return train, val


# ============================================================
# MAIN PIPELINE
# ============================================================

def curate_operation(operation, stats_only=False):
    """Run full curation pipeline for one operation."""
    input_file = f"data/op{operation}_raw_pairs.jsonl"
    output_train = f"data/op{operation}_train.jsonl"
    output_val = f"data/op{operation}_val.jsonl"
    output_rejected = f"data/op{operation}_curated_rejects.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Input not found: {input_file}")
        print(f"Run batch_generate.py --operation {operation} first.")
        return
    
    # Load
    with open(input_file, "r") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"CURATION — OPERATION {operation}")
    print(f"{'='*60}")
    print(f"Raw examples: {len(raw)}")
    
    # Determine target field
    if operation == 1:
        target_field = "output_stripped"
    elif operation == 2:
        target_field = "output_full"
    elif operation == 3:
        target_field = "output_integrated"
    
    # Category distribution
    cats = Counter(ex.get("category", "unknown") for ex in raw)
    print(f"\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    
    # Detect issues
    clean = []
    fixable = []
    rejected = []
    issue_counts = Counter()
    
    for ex in raw:
        target = ex.get(target_field, "")
        issues = detect_issues(target, operation)
        
        if not issues:
            clean.append(ex)
        else:
            # Categorize severity
            severe = [i for i in issues if i[0] in ("too_short", "too_long", "memory_narration")]
            minor = [i for i in issues if i[0] not in ("too_short", "too_long", "memory_narration")]
            
            for issue_type, _ in issues:
                issue_counts[issue_type] += 1
            
            if severe:
                rejected.append({"example": ex, "issues": issues})
            else:
                # Try auto-fix
                fixed_text = auto_fix(target, issues)
                remaining = detect_issues(fixed_text, operation)
                if not remaining or all(r[0] == "em_dash" for r in remaining):
                    ex[target_field] = fixed_text
                    ex["_auto_fixed"] = True
                    clean.append(ex)
                else:
                    fixable.append({"example": ex, "issues": issues, "remaining": remaining})
    
    print(f"\nQuality filter results:")
    print(f"  Clean: {len(clean)}")
    print(f"  Auto-fixed: {sum(1 for e in clean if e.get('_auto_fixed'))}")
    print(f"  Needs manual review: {len(fixable)}")
    print(f"  Rejected: {len(rejected)}")
    
    if issue_counts:
        print(f"\nIssue breakdown:")
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count}")
    
    # Deduplicate
    clean_deduped, dupe_count = deduplicate(clean, key_field="input")
    print(f"\nDeduplication: removed {dupe_count} duplicates")
    print(f"Clean unique examples: {len(clean_deduped)}")
    
    if stats_only:
        print(f"\n{'='*60}\n")
        return
    
    # Format for LoRA
    formatter = FORMATTERS[operation]
    formatted = [formatter(ex) for ex in clean_deduped]
    
    # Split
    train, val = split_dataset(formatted)
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    
    with open(output_train, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    
    with open(output_val, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")
    
    if rejected or fixable:
        with open(output_rejected, "w") as f:
            for item in rejected:
                f.write(json.dumps({
                    "status": "rejected",
                    "issues": [(t, d) for t, d in item["issues"]],
                    "input": item["example"].get("input", "")[:100],
                    "target": item["example"].get(target_field, "")[:200],
                }) + "\n")
            for item in fixable:
                f.write(json.dumps({
                    "status": "needs_review",
                    "issues": [(t, d) for t, d in item["issues"]],
                    "input": item["example"].get("input", "")[:100],
                    "target": item["example"].get(target_field, "")[:200],
                }) + "\n")
    
    # Length stats
    lengths = [len(ex["completion"]) for ex in formatted]
    print(f"\nCompletion length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Avg: {sum(lengths)//len(lengths)} chars")
    
    # Token estimate (rough: 1 token ~ 4 chars)
    total_chars = sum(len(ex["text"]) for ex in formatted)
    est_tokens = total_chars // 4
    print(f"\nEstimated total tokens: ~{est_tokens:,}")
    
    print(f"\nSaved:")
    print(f"  {output_train}")
    print(f"  {output_val}")
    if rejected or fixable:
        print(f"  {output_rejected}")
    print(f"{'='*60}\n")


def merge_all():
    """Merge all operation datasets into one combined training file."""
    combined_train = []
    combined_val = []
    
    for op in [1, 2, 3]:
        train_file = f"data/op{op}_train.jsonl"
        val_file = f"data/op{op}_val.jsonl"
        
        if os.path.exists(train_file):
            with open(train_file, "r") as f:
                examples = [json.loads(line) for line in f if line.strip()]
                combined_train.extend(examples)
                print(f"Op{op} train: {len(examples)}")
        
        if os.path.exists(val_file):
            with open(val_file, "r") as f:
                examples = [json.loads(line) for line in f if line.strip()]
                combined_val.extend(examples)
                print(f"Op{op} val:   {len(examples)}")
    
    if combined_train:
        import random
        random.seed(42)
        random.shuffle(combined_train)
        random.shuffle(combined_val)
        
        with open("data/combined_train.jsonl", "w") as f:
            for ex in combined_train:
                f.write(json.dumps(ex) + "\n")
        
        with open("data/combined_val.jsonl", "w") as f:
            for ex in combined_val:
                f.write(json.dumps(ex) + "\n")
        
        print(f"\nCombined train: {len(combined_train)}")
        print(f"Combined val:   {len(combined_val)}")
        print(f"Total examples: {len(combined_train) + len(combined_val)}")
        
        # Operation distribution in combined
        ops = Counter(ex.get("operation", "?") for ex in combined_train)
        print(f"\nTrain distribution by operation:")
        for op, count in sorted(ops.items()):
            print(f"  Op{op}: {count} ({count/len(combined_train)*100:.0f}%)")
        
        total_chars = sum(len(ex["text"]) for ex in combined_train)
        print(f"\nEstimated total training tokens: ~{total_chars//4:,}")


def main():
    parser = argparse.ArgumentParser(description="Shell-Native Curation Pipeline")
    parser.add_argument("--operation", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Merge all ops into combined dataset")
    parser.add_argument("--stats-only", action="store_true", help="Show stats without processing")
    
    args = parser.parse_args()
    
    if args.merge:
        merge_all()
        return
    
    if not args.operation and not args.all:
        parser.print_help()
        return
    
    if args.all:
        for op in [1, 2, 3]:
            curate_operation(op, stats_only=args.stats_only)
        if not args.stats_only:
            merge_all()
    else:
        curate_operation(args.operation, stats_only=args.stats_only)


if __name__ == "__main__":
    main()
