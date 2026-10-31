"""
Fine-tuning script for BART and T5 on CNN/DailyMail using Seq2SeqTrainer.

Features:
  - Resumable training (auto-detects last checkpoint).
  - Checkpoints saved every eval_steps; best kept by eval_loss.
  - --smoke flag for quick Mac tests (~50 examples).
  - Auto-detects device; disables fp16 on MPS.
  - All hyperparameters from YAML configs (no magic numbers).

Usage:
    python src/train.py --model bart               # full BART training
    python src/train.py --model t5                  # full T5 training
    python src/train.py --model bart --smoke        # smoke test on Mac
    python src/train.py --model bart --resume       # resume from checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import (
    build_tokenize_function,
    load_cnn_dailymail,
    load_indices,
    get_subsets,
)
from src.utils import (
    Timer,
    get_device,
    get_library_versions,
    get_peak_memory_mb,
    is_mac_mps,
    load_config,
    save_run_record,
    set_seed,
)


def find_last_checkpoint(output_dir: str):
    """Find the last checkpoint in output_dir, if any.

    Returns:
        Path to the last checkpoint directory, or None.
    """
    from transformers.trainer_utils import get_last_checkpoint

    if os.path.isdir(output_dir):
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is not None:
            print(f"[train] Found checkpoint: {last_ckpt}")
        return last_ckpt
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BART or T5 on CNN/DailyMail."
    )
    parser.add_argument(
        "--model",
        choices=["bart", "t5"],
        required=True,
        help="Which model to fine-tune.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny subsets (~50 examples) for a quick pipeline test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (if one exists).",
    )
    args = parser.parse_args()

    # ---- Load configs ----
    model_config_file = f"configs/{args.model}.yaml"
    cfg = load_config("configs/base.yaml", model_config_file)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_name = cfg["model_name"]
    source_prefix = cfg.get("source_prefix", "")
    output_dir = cfg["output_dir"]

    # ---- Device ----
    device = get_device()
    on_mps = is_mac_mps(device)

    # Force fp16 off on MPS
    use_fp16 = cfg.get("fp16", False) and not on_mps
    if on_mps:
        print("[train] MPS detected — forcing fp16=False")

    # ---- Load data ----
    print("[train] Loading dataset...")
    with Timer("Dataset loading"):
        ds = load_cnn_dailymail(
            dataset_name=cfg.get("dataset_name", "abisee/cnn_dailymail"),
            dataset_config=cfg.get("dataset_config", "3.0.0"),
        )

    indices = load_indices(cfg.get("data_dir", "data"))
    subsets = get_subsets(ds, indices)

    # Apply smoke mode
    if args.smoke:
        smoke_size = cfg.get("smoke_subset_size", 50)
        for split in subsets:
            n = min(smoke_size, len(subsets[split]))
            subsets[split] = subsets[split].select(range(n))
        print(f"[train] SMOKE MODE: {smoke_size} examples per split")

    # ---- Tokenizer + Model ----
    print(f"[train] Loading model: {model_name}")
    with Timer("Model loading"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ---- Tokenize datasets ----
    print("[train] Tokenizing datasets...")
    tokenize_fn = build_tokenize_function(
        tokenizer=tokenizer,
        max_source_length=cfg["max_source_length"],
        max_target_length=cfg["max_target_length"],
        source_prefix=source_prefix,
    )

    tokenized = {}
    for split_name, split_ds in subsets.items():
        with Timer(f"Tokenizing {split_name}"):
            tokenized[split_name] = split_ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=split_ds.column_names,
                desc=f"Tokenizing {split_name}",
            )
    print(f"[train] Tokenized: train={len(tokenized['train'])}, "
          f"val={len(tokenized['validation'])}")

    # ---- Data collator ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ---- Training arguments ----
    # Adjust steps for smoke mode
    eval_steps = cfg.get("eval_steps", 100)
    save_steps = cfg.get("save_steps", 100)
    logging_steps = cfg.get("logging_steps", 10)

    if args.smoke:
        eval_steps = 5
        save_steps = 5
        logging_steps = 2

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=cfg.get("warmup_steps", 37) if not args.smoke else 2,
        fp16=use_fp16,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        predict_with_generate=False,  # too slow during training validation
        report_to="none",  # no wandb etc.
        seed=seed,
        dataloader_num_workers=0 if on_mps else 2,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ---- Train ----
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_last_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"[train] Resuming from {resume_checkpoint}")
        else:
            print("[train] No checkpoint found, starting from scratch")

    print(f"\n[train] Starting {'smoke ' if args.smoke else ''}fine-tuning of {model_name}")
    print(f"[train] Output dir: {output_dir}")
    print(f"[train] fp16: {use_fp16}")
    print(f"[train] Effective batch size: "
          f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    with Timer("Training"):
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ---- Save final model ----
    print("[train] Saving best model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---- Log metrics ----
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Eval on validation set
    print("[train] Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized["validation"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ---- Save run record ----
    mode = "smoke" if args.smoke else "full"
    run_record = {
        "experiment": f"finetune_{args.model}",
        "mode": mode,
        "model_name": model_name,
        "train_samples": len(tokenized["train"]),
        "val_samples": len(tokenized["validation"]),
        "training_args": {
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "effective_batch_size": (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            ),
            "fp16": use_fp16,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
        },
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "output_dir": output_dir,
        "peak_memory_mb": get_peak_memory_mb(),
        "library_versions": get_library_versions(),
        "seed": seed,
    }
    save_run_record(
        run_record,
        results_dir=cfg.get("results_dir", "results"),
        filename=f"finetune_{args.model}_{mode}_record.json",
    )

    print(f"\n[train] Done. Best model saved to {output_dir}")
    print(f"[train] Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"[train] Eval loss:  {eval_metrics.get('eval_loss', 'N/A')}")


if __name__ == "__main__":
    main()
