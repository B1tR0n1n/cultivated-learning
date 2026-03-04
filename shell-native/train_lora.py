"""
Shell-Native LoRA Fine-Tuning Script
=====================================
Fine-tunes Mistral 7B Instruct v0.3 using LoRA on the curated
dataset from the shell-native pipeline.

Usage:
    # Train on Operation 1 only
    python train_lora.py --dataset data/op1_train.jsonl --val data/op1_val.jsonl --output models/shell-native-op1

    # Train on combined dataset (all operations)
    python train_lora.py --dataset data/combined_train.jsonl --val data/combined_val.jsonl --output models/shell-native-combined

    # Dry run (compute VRAM estimate without training)
    python train_lora.py --dataset data/combined_train.jsonl --dry-run

Requirements:
    pip install peft bitsandbytes trl datasets --break-system-packages
"""

import json
import os
import sys
import argparse
import torch
from datetime import datetime


def estimate_vram(model_path, dataset_path, batch_size=2, lora_rank=16):
    """Estimate VRAM usage without loading the model."""
    # Mistral 7B float16 base
    base_vram_gb = 14.5
    
    # LoRA overhead (rank 16, target modules: q_proj, k_proj, v_proj, o_proj)
    # Each adapter: rank * hidden_dim * 2 (A and B matrices) * 4 target modules
    # Mistral hidden_dim = 4096, float16 = 2 bytes
    lora_params = lora_rank * 4096 * 2 * 4 * 2  # bytes
    lora_vram_gb = lora_params / 1e9
    
    # Optimizer states (AdamW: 2x param memory for momentum + variance)
    optimizer_vram_gb = lora_vram_gb * 3
    
    # Gradient checkpointing reduces activation memory
    # Rough estimate: batch_size * seq_len * hidden_dim * num_layers * 2 bytes / checkpoint_ratio
    # With checkpointing: ~2GB for batch_size=2
    activation_vram_gb = batch_size * 1.0  # rough
    
    # KV cache for training
    kv_vram_gb = 0.5 * batch_size
    
    total = base_vram_gb + lora_vram_gb + optimizer_vram_gb + activation_vram_gb + kv_vram_gb
    
    # Dataset stats
    with open(dataset_path, "r") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    total_chars = sum(len(ex.get("text", "")) for ex in examples)
    avg_len = total_chars // max(len(examples), 1)
    
    print(f"\n{'='*60}")
    print(f"VRAM ESTIMATE")
    print(f"{'='*60}")
    print(f"Base model (float16):   {base_vram_gb:.1f} GB")
    print(f"LoRA adapters (r={lora_rank}):  {lora_vram_gb:.2f} GB")
    print(f"Optimizer states:       {optimizer_vram_gb:.2f} GB")
    print(f"Activations (bs={batch_size}):   {activation_vram_gb:.1f} GB")
    print(f"KV cache:               {kv_vram_gb:.1f} GB")
    print(f"{'='*40}")
    print(f"ESTIMATED TOTAL:        {total:.1f} GB")
    print(f"Available (RTX 5090):   32.0 GB")
    print(f"Headroom:               {32.0 - total:.1f} GB")
    print(f"\nDataset: {len(examples)} examples")
    print(f"Avg length: {avg_len} chars (~{avg_len//4} tokens)")
    print(f"{'='*60}\n")
    
    if total > 30:
        print("WARNING: Tight on VRAM. Consider batch_size=1 with gradient_accumulation_steps=4.")
    
    return total


def train(args):
    """Run LoRA fine-tuning."""
    # Lazy imports (heavy)
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    print(f"\n{'='*60}")
    print(f"SHELL-NATIVE LORA TRAINING")
    print(f"{'='*60}")
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    with open(args.dataset, "r") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    
    val_data = []
    if args.val and os.path.exists(args.val):
        with open(args.val, "r") as f:
            val_data = [json.loads(line) for line in f if line.strip()]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    
    print(f"Train examples: {len(train_data)}")
    print(f"Val examples:   {len(val_data)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    
    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"Model loaded. VRAM: {vram:.2f} GB")
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,  # Standard: alpha = 2 * rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    # Training config
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else None,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output, "logs"),
        max_length=args.max_seq_length,
        dataset_text_field="text",
        seed=42,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Max seq length: {args.max_seq_length}")
    
    start = datetime.now()
    result = trainer.train()
    elapsed = (datetime.now() - start).total_seconds()
    
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Final loss: {result.training_loss:.4f}")
    
    # Save
    print(f"Saving adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Save training metadata
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "val_dataset": args.val,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "max_seq_length": args.max_seq_length,
        "training_loss": result.training_loss,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    vram_final = torch.cuda.memory_allocated(0) / 1e9
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
    print(f"\nDone. Adapter saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Shell-Native LoRA Training")
    parser.add_argument("--model", default="/workspace/models/results/Mistral-7B-Instruct-v0.3",
                       help="Base model path")
    parser.add_argument("--dataset", required=True, help="Training JSONL file")
    parser.add_argument("--val", default=None, help="Validation JSONL file")
    parser.add_argument("--output", default="models/shell-native",
                       help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true",
                       help="Estimate VRAM without training")
    
    args = parser.parse_args()
    
    if args.dry_run:
        estimate_vram(args.model, args.dataset, args.batch_size, args.lora_rank)
        return
    
    train(args)


if __name__ == "__main__":
    main()
