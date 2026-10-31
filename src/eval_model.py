"""
Evaluation script for fine-tuned BART and T5 models.

Generates summaries on the test subset using the fixed decoding config,
computes ROUGE-1/2/L, measures inference latency and memory, and saves
all outputs.

Usage:
    python src/evaluate.py --model bart              # evaluate fine-tuned BART
    python src/evaluate.py --model t5                # evaluate fine-tuned T5
    python src/evaluate.py --model bart --smoke      # quick test
    python src/evaluate.py --model bart --checkpoint path/to/checkpoint  # custom checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Prevent transformers from attempting to import TensorFlow (avoids protobuf conflicts in Python 3.13)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import time
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_cnn_dailymail, load_indices, get_subsets
from src.utils import (
    Timer,
    get_device,
    get_library_versions,
    get_peak_memory_mb,
    is_mac_mps,
    load_config,
    load_decoding_config,
    save_run_record,
    set_seed,
)


def generate_with_timing(
    model,
    tokenizer,
    articles: list[str],
    decoding_cfg: dict,
    source_prefix: str,
    max_source_length: int,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[list[str], list[float], list[int]]:
    """Generate summaries and record per-batch timing.

    Returns:
        Tuple of (predictions, per_article_latencies, generated_lengths_in_tokens).
    """
    model.eval()
    predictions = []
    latencies = []
    gen_lengths = []

    for i in tqdm(range(0, len(articles), batch_size), desc="Generating"):
        batch_articles = articles[i : i + batch_size]
        inputs_text = [source_prefix + a for a in batch_articles]

        inputs = tokenizer(
            inputs_text,
            max_length=max_source_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(device)

        # Sync CUDA before timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=decoding_cfg["num_beams"],
                max_length=decoding_cfg["max_length"],
                min_length=decoding_cfg["min_length"],
                length_penalty=decoding_cfg["length_penalty"],
                no_repeat_ngram_size=decoding_cfg["no_repeat_ngram_size"],
                early_stopping=decoding_cfg["early_stopping"],
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        batch_latency = t1 - t0

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

        # Per-article latency (approximate: divide batch time evenly)
        per_article = batch_latency / len(batch_articles)
        latencies.extend([per_article] * len(batch_articles))

        # Token lengths of generated outputs
        for ids in output_ids:
            # Count non-pad tokens
            non_pad = (ids != tokenizer.pad_token_id).sum().item()
            gen_lengths.append(non_pad)

    return predictions, latencies, gen_lengths


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (F-measure, stemming on)."""
    rouge = hf_evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned BART or T5 on CNN/DailyMail test set."
    )
    parser.add_argument(
        "--model",
        choices=["bart", "t5"],
        required=True,
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint. Defaults to the model's output_dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny subset for a quick test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        help="Number of warm-up articles to skip when computing mean latency.",
    )
    parser.add_argument(
        "--latency-sample",
        type=int,
        default=100,
        help="Number of articles to use for latency measurement.",
    )
    args = parser.parse_args()

    # ---- Load configs ----
    model_config_file = f"configs/{args.model}.yaml"
    cfg = load_config("configs/base.yaml", model_config_file)
    decoding_cfg = load_decoding_config()
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_name = cfg["model_name"]
    source_prefix = cfg.get("source_prefix", "")
    max_source_length = cfg["max_source_length"]
    checkpoint_path = args.checkpoint or cfg["output_dir"]

    # ---- Device ----
    device = get_device()

    # ---- Load data ----
    with Timer("Dataset loading"):
        ds = load_cnn_dailymail(
            dataset_name=cfg.get("dataset_name", "abisee/cnn_dailymail"),
            dataset_config=cfg.get("dataset_config", "3.0.0"),
        )

    indices = load_indices(cfg.get("data_dir", "data"))
    subsets = get_subsets(ds, indices)

    if args.smoke:
        smoke_size = cfg.get("smoke_subset_size", 50)
        test_data = subsets["test"].select(range(min(smoke_size, len(subsets["test"]))))
        print(f"[eval] SMOKE MODE: {len(test_data)} test examples")
    else:
        test_data = subsets["test"]
        print(f"[eval] Using {len(test_data)} test examples")

    articles = test_data["article"]
    references = test_data["highlights"]

    # ---- Load fine-tuned model ----
    print(f"[eval] Loading fine-tuned model from: {checkpoint_path}")
    with Timer("Model loading"):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model.to(device)

    # Handle precision
    on_mps = is_mac_mps(device)
    use_fp16 = cfg.get("fp16", False) and not on_mps
    if use_fp16:
        model.half()
        print("[eval] Using fp16")
    else:
        print("[eval] Using fp32")

    # ---- Generate summaries ----
    print(f"[eval] Generating summaries for {len(articles)} articles...")
    with Timer("Full generation"):
        predictions, latencies, gen_lengths = generate_with_timing(
            model=model,
            tokenizer=tokenizer,
            articles=articles,
            decoding_cfg=decoding_cfg,
            source_prefix=source_prefix,
            max_source_length=max_source_length,
            device=device,
            batch_size=args.batch_size,
        )

    # ---- ROUGE scores ----
    print("[eval] Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    print("[eval] ROUGE scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.4f}")

    # ---- Efficiency stats ----
    # Skip warm-up articles for latency
    warmup = min(args.latency_warmup, len(latencies) // 2)
    sample_size = min(args.latency_sample, len(latencies) - warmup)
    latency_sample = latencies[warmup : warmup + sample_size]

    efficiency = {
        "mean_latency_seconds": float(np.mean(latency_sample)) if latency_sample else 0,
        "std_latency_seconds": float(np.std(latency_sample)) if latency_sample else 0,
        "latency_sample_size": len(latency_sample),
        "warmup_skipped": warmup,
        "mean_generated_length_tokens": float(np.mean(gen_lengths)),
        "peak_memory_mb": get_peak_memory_mb(),
    }
    print(f"[eval] Mean latency: {efficiency['mean_latency_seconds']:.3f}s "
          f"(±{efficiency['std_latency_seconds']:.3f}s, n={efficiency['latency_sample_size']})")
    print(f"[eval] Mean generated length: {efficiency['mean_generated_length_tokens']:.1f} tokens")
    print(f"[eval] Peak memory: {efficiency['peak_memory_mb']:.0f} MB")

    # ---- Save results ----
    results_dir = cfg.get("results_dir", "results")
    mode = "smoke" if args.smoke else "full"

    # Per-example outputs
    per_example = []
    for i, (art, ref, pred, lat, glen) in enumerate(
        zip(articles, references, predictions, latencies, gen_lengths)
    ):
        per_example.append({
            "index": i,
            "article_preview": art[:200],
            "reference": ref,
            "prediction": pred,
            "latency_seconds": round(lat, 4),
            "generated_length_tokens": glen,
        })

    per_example_path = Path(results_dir) / f"eval_{args.model}_{mode}_outputs.json"
    per_example_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_example_path, "w") as f:
        json.dump(per_example, f, indent=2)
    print(f"[eval] Per-example outputs saved to {per_example_path}")

    # Run record
    run_record = {
        "experiment": f"eval_{args.model}_finetuned",
        "mode": mode,
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "num_test_examples": len(articles),
        "source_prefix": source_prefix,
        "decoding_config": decoding_cfg,
        "rouge_scores": rouge_scores,
        "efficiency": efficiency,
        "library_versions": get_library_versions(),
        "seed": seed,
    }
    save_run_record(
        run_record,
        results_dir=results_dir,
        filename=f"eval_{args.model}_{mode}_record.json",
    )

    print(f"\n[eval] Done. {args.model.upper()} evaluation complete.")


if __name__ == "__main__":
    main()
