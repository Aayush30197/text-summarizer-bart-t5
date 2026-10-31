"""
Zero-shot baseline evaluation for pretrained BART and T5.

Generates summaries from the *pretrained* (not fine-tuned) checkpoints and
computes ROUGE scores against reference highlights.  This measures what
fine-tuning adds.

Usage:
    python src/baseline.py --model bart          # BART baseline
    python src/baseline.py --model t5            # T5 baseline
    python src/baseline.py --model bart --smoke   # quick 50-example test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Prevent transformers from attempting to import TensorFlow (avoids protobuf conflicts in Python 3.13)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path

import evaluate
import torch
from torch.utils.data import DataLoader
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


def generate_summaries(
    model,
    tokenizer,
    articles: list[str],
    decoding_cfg: dict,
    source_prefix: str,
    max_source_length: int,
    device: torch.device,
    batch_size: int = 4,
) -> list[str]:
    """Generate summaries for a list of articles using the given model.

    Args:
        model: A HuggingFace seq2seq model.
        tokenizer: Corresponding tokenizer.
        articles: List of raw article strings.
        decoding_cfg: Decoding hyperparameters (num_beams, max_length, etc.).
        source_prefix: Prefix for T5 (empty string for BART).
        max_source_length: Max tokens for the input.
        device: Torch device.
        batch_size: Batch size for generation.

    Returns:
        List of generated summary strings.
    """
    model.eval()
    summaries = []

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

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(decoded)

    return summaries


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L using HuggingFace evaluate.

    Returns dict with keys rouge1, rouge2, rougeL (F-measure, stemming on).
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot baseline evaluation on pretrained models."
    )
    parser.add_argument(
        "--model",
        choices=["bart", "t5"],
        required=True,
        help="Which model to evaluate (bart or t5).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny subset (~50 examples) for a quick test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (reduce if OOM).",
    )
    args = parser.parse_args()

    # ---- Load configs ----
    model_config_file = f"configs/{args.model if args.model == 't5' else 'bart'}.yaml"
    cfg = load_config("configs/base.yaml", model_config_file)
    decoding_cfg = load_decoding_config()
    seed = cfg.get("seed", 42)
    set_seed(seed)

    model_name = cfg["model_name"]
    source_prefix = cfg.get("source_prefix", "")
    max_source_length = cfg["max_source_length"]

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

    # Use test subset (or smoke subset)
    if args.smoke:
        smoke_size = cfg.get("smoke_subset_size", 50)
        test_data = subsets["test"].select(range(min(smoke_size, len(subsets["test"]))))
        print(f"[baseline] SMOKE MODE: using {len(test_data)} test examples")
    else:
        test_data = subsets["test"]
        print(f"[baseline] Using {len(test_data)} test examples")

    articles = test_data["article"]
    references = test_data["highlights"]

    # ---- Load model ----
    print(f"[baseline] Loading pretrained model: {model_name}")
    with Timer("Model loading"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)

    # Disable fp16 on MPS
    use_fp16 = cfg.get("fp16", False) and not is_mac_mps(device)
    if use_fp16:
        model.half()
        print("[baseline] Using fp16")
    else:
        print("[baseline] Using fp32")

    # ---- Generate ----
    print(f"[baseline] Generating summaries with {model_name}...")
    with Timer("Baseline generation"):
        predictions = generate_summaries(
            model=model,
            tokenizer=tokenizer,
            articles=articles,
            decoding_cfg=decoding_cfg,
            source_prefix=source_prefix,
            max_source_length=max_source_length,
            device=device,
            batch_size=args.batch_size,
        )

    # ---- Evaluate ----
    print("[baseline] Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)
    print(f"[baseline] ROUGE scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.4f}")

    # ---- Save results ----
    results_dir = cfg.get("results_dir", "results")
    mode = "smoke" if args.smoke else "full"

    # Per-example outputs
    per_example = []
    for i, (art, ref, pred) in enumerate(zip(articles, references, predictions)):
        per_example.append({
            "index": i,
            "article_preview": art[:200],
            "reference": ref,
            "prediction": pred,
        })

    per_example_path = Path(results_dir) / f"baseline_{args.model}_{mode}_outputs.json"
    per_example_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_example_path, "w") as f:
        json.dump(per_example, f, indent=2)
    print(f"[baseline] Per-example outputs saved to {per_example_path}")

    # Run record
    run_record = {
        "experiment": f"baseline_{args.model}",
        "mode": mode,
        "model_name": model_name,
        "num_examples": len(articles),
        "source_prefix": source_prefix,
        "decoding_config": decoding_cfg,
        "rouge_scores": rouge_scores,
        "peak_memory_mb": get_peak_memory_mb(),
        "library_versions": get_library_versions(),
        "seed": seed,
    }
    save_run_record(
        run_record,
        results_dir=results_dir,
        filename=f"baseline_{args.model}_{mode}_record.json",
    )

    print(f"\n[baseline] Done. {args.model.upper()} baseline complete.")


if __name__ == "__main__":
    main()
